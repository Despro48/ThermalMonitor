"""Profile A: lossless 32x24 thermal frames as three independently decodable strips.

Nominal wire format (one LoRa packet per strip):

    [frame_id][strip_id | (k << 2)][ref_hi][ref_lo][Rice bitstream...]

If a strip does not fit in the 255-byte LoRa packet (large hot objects or
adversarial data), the strip is split into four 32x2 parts. Each part is sent
as its own packet, and a part that still does not fit is stored as raw int16
pixels. The split flag/part index live in the reserved bits of byte 1, so the
nominal 3-packet format is unchanged for normal scenes. See
docs/thermal-compression.md, section 2, for the normative description.
"""

from __future__ import annotations

import numpy as np

from .common import (
    BitReader,
    BitWriter,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    PIXEL_COUNT,
    quantize_temperature,
)

STRIP_HEIGHT = 8
STRIP_COUNT = FRAME_HEIGHT // STRIP_HEIGHT
STRIP_PIXELS = FRAME_WIDTH * STRIP_HEIGHT
SPLIT_HEIGHT = 2
SPLIT_PARTS = STRIP_HEIGHT // SPLIT_HEIGHT
MAX_PACKETS = STRIP_COUNT * SPLIT_PARTS
HEADER_BYTES = 4
MAX_PACKET_BYTES = 255
MAX_PAYLOAD_BYTES = MAX_PACKET_BYTES - HEADER_BYTES
SPLIT_RAW_BYTES = FRAME_WIDTH * SPLIT_HEIGHT * 2
RICE_LIMIT = 15
K_VALUES = (1, 2, 3)
SPLIT_FLAG = 0x40
PART_SHIFT = 4


def _median(a: int, b: int, c: int) -> int:
    lo = min(a, b)
    hi = max(a, b)
    return lo if c <= lo else hi if c >= hi else c


def _predict(pixels: np.ndarray, x: int, y: int, ref: int) -> int:
    """MED predictor with the boundary rules defined by the spec."""
    if y == 0:
        return int(pixels[0, x - 1]) if x > 0 else ref
    if x == 0:
        return int(pixels[y - 1, 0])
    a = int(pixels[y, x - 1])
    b = int(pixels[y - 1, x])
    c = int(pixels[y - 1, x - 1])
    return max(-32768, min(32767, _median(a, b, a + b - c)))


def _zigzag(residual: int) -> int:
    return (residual << 1) if residual >= 0 else ((-residual << 1) - 1)


def _residuals(pixels: np.ndarray, ref: int) -> list[int]:
    height = pixels.shape[0]
    out: list[int] = []
    for y in range(height):
        for x in range(FRAME_WIDTH):
            if x == 0 and y == 0:
                continue
            out.append(int(pixels[y, x]) - _predict(pixels, x, y, ref))
    return out


def _rice_cost(residuals: list[int], k: int) -> int:
    cost = 0
    for residual in residuals:
        quotient = _zigzag(residual) >> k
        cost += RICE_LIMIT + 1 + 16 if quotient >= RICE_LIMIT else quotient + 1 + k
    return cost


def _encode_rice(pixels: np.ndarray, k: int) -> bytes:
    """Encode a 32xH int16 block (H = 8 or 2) as a Rice bitstream."""
    height = pixels.shape[0]
    ref = int(pixels[0, 0])
    writer = BitWriter()
    for y in range(height):
        for x in range(FRAME_WIDTH):
            if x == 0 and y == 0:
                continue
            residual = int(pixels[y, x]) - _predict(pixels, x, y, ref)
            folded = _zigzag(residual)
            quotient = folded >> k
            if quotient >= RICE_LIMIT:
                for _ in range(RICE_LIMIT):
                    writer.bit(1)
                writer.bit(0)
                writer.bits(int(pixels[y, x]) & 0xFFFF, 16)
            else:
                for _ in range(quotient):
                    writer.bit(1)
                writer.bit(0)
                writer.bits(folded & ((1 << k) - 1), k)
    return writer.finish()


def _make_packet(
    pixels: np.ndarray,
    frame_id: int,
    strip_id: int,
    part: int = 0,
    split: bool = False,
) -> bytes | None:
    """Encode one coded block; returns None if the Rice payload is too large."""
    ref = int(pixels[0, 0])
    k = min(K_VALUES, key=lambda candidate: _rice_cost(_residuals(pixels, ref), candidate))
    payload = _encode_rice(pixels, k)
    if len(payload) > MAX_PAYLOAD_BYTES:
        return None
    flags = strip_id | (k << 2) | ((part & 0x03) << PART_SHIFT)
    if split:
        flags |= SPLIT_FLAG
    return bytes((frame_id, flags, (ref >> 8) & 0xFF, ref & 0xFF)) + payload


def _make_raw_part(pixels: np.ndarray, frame_id: int, strip_id: int, part: int) -> bytes:
    flags = strip_id | ((part & 0x03) << PART_SHIFT) | SPLIT_FLAG  # k = 0 means raw
    return bytes((frame_id, flags, 0, 0)) + pixels.astype("<i2").tobytes()


def encode_strip(strip: np.ndarray, frame_id: int, strip_id: int) -> bytes:
    """Encode a 32x8 strip as one nominal packet (raises if it does not fit)."""
    if strip.shape != (STRIP_HEIGHT, FRAME_WIDTH):
        raise ValueError(f"expected {(STRIP_HEIGHT, FRAME_WIDTH)} strip, got {strip.shape}")
    if not 0 <= frame_id <= 255:
        raise ValueError("frame_id must fit in one byte")
    if not 0 <= strip_id < STRIP_COUNT:
        raise ValueError("invalid strip id")
    packet = _make_packet(strip, frame_id, strip_id)
    if packet is None:
        raise ValueError(f"strip {strip_id} does not fit in one LoRa packet")
    return packet


def encode_strip_packets(strip: np.ndarray, frame_id: int, strip_id: int) -> list[bytes]:
    """Encode one 32x8 strip into one packet, or four split packets if needed."""
    packet = _make_packet(strip, frame_id, strip_id)
    if packet is not None:
        return [packet]
    packets: list[bytes] = []
    for part in range(SPLIT_PARTS):
        sub = strip[part * SPLIT_HEIGHT : (part + 1) * SPLIT_HEIGHT, :]
        coded = _make_packet(sub, frame_id, strip_id, part=part, split=True)
        packets.append(coded if coded is not None else _make_raw_part(sub, frame_id, strip_id, part))
    return packets


def parse_header(packet: bytes) -> tuple[int, int, int, int, bool, int]:
    """Return (frame_id, strip_id, k, part, split, ref)."""
    if len(packet) < HEADER_BYTES:
        raise ValueError("packet shorter than header")
    frame_id = packet[0]
    flags = packet[1]
    if flags & 0x80:
        raise ValueError("reserved header bit is set")
    strip_id = flags & 0x03
    k = (flags >> 2) & 0x03
    part = (flags >> PART_SHIFT) & 0x03
    split = bool(flags & SPLIT_FLAG)
    if strip_id >= STRIP_COUNT:
        raise ValueError(f"invalid strip id {strip_id}")
    if not split:
        if part != 0:
            raise ValueError("part index requires the split flag")
        if k not in K_VALUES:
            raise ValueError(f"invalid Rice parameter {k}")
    elif k not in (0, *K_VALUES):
        raise ValueError(f"invalid split Rice parameter {k}")
    ref = int.from_bytes(packet[2:4], "big", signed=True)
    return frame_id, strip_id, k, part, split, ref


def _decode_rice(payload: bytes, k: int, ref: int, height: int) -> np.ndarray:
    reader = BitReader(payload)
    pixels = np.zeros((height, FRAME_WIDTH), dtype=np.int16)
    pixels[0, 0] = ref
    for y in range(height):
        for x in range(FRAME_WIDTH):
            if x == 0 and y == 0:
                continue
            ones = 0
            while reader.bit() == 1:
                ones += 1
                if ones > RICE_LIMIT:
                    raise ValueError("invalid unary code")
            if ones == RICE_LIMIT:
                raw = reader.bits(16)
                pixels[y, x] = raw - 65536 if raw >= 32768 else raw
                continue
            remainder = reader.bits(k)
            folded = (ones << k) | remainder
            residual = (folded >> 1) ^ -(folded & 1)
            pixels[y, x] = _predict(pixels, x, y, ref) + residual
    return pixels


def decode_packet(packet: bytes) -> tuple[int, int, int, bool, np.ndarray]:
    """Decode any packet into (frame_id, strip_id, part, split, pixels)."""
    frame_id, strip_id, k, part, split, ref = parse_header(packet)
    height = SPLIT_HEIGHT if split else STRIP_HEIGHT
    if split and k == 0:
        payload = packet[HEADER_BYTES:]
        if len(payload) < SPLIT_RAW_BYTES:
            raise ValueError("truncated raw split payload")
        pixels = np.frombuffer(payload[:SPLIT_RAW_BYTES], dtype="<i2").copy().reshape(
            SPLIT_HEIGHT, FRAME_WIDTH
        )
    else:
        pixels = _decode_rice(packet[HEADER_BYTES:], k, ref, height)
    return frame_id, strip_id, part, split, pixels


def decode_strip(packet: bytes) -> tuple[int, np.ndarray]:
    """Decode one nominal 32x8 packet into (strip_id, 32x8 int16 array)."""
    _, strip_id, _, _, split, _ = parse_header(packet)
    if split:
        raise ValueError("split packet; use decode_frame()")
    _, _, _, _, pixels = decode_packet(packet)
    return strip_id, pixels


def split_strips(frame: np.ndarray) -> list[np.ndarray]:
    if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(f"expected {(FRAME_HEIGHT, FRAME_WIDTH)} frame, got {frame.shape}")
    return [frame[i * STRIP_HEIGHT : (i + 1) * STRIP_HEIGHT, :] for i in range(STRIP_COUNT)]


def encode_frame(temperatures_c: np.ndarray, frame_id: int) -> list[bytes]:
    """Quantize and encode a frame into 3 packets (or up to 12 when split)."""
    quantized = quantize_temperature(temperatures_c).reshape(FRAME_HEIGHT, FRAME_WIDTH)
    packets: list[bytes] = []
    for strip_id, strip in enumerate(split_strips(quantized)):
        packets.extend(encode_strip_packets(strip, frame_id, strip_id))
    return packets


def decode_frame(packets: list[bytes]) -> np.ndarray:
    """Decode a complete set of packets (any order) into an int16 frame."""
    if not packets:
        raise ValueError("no packets")
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.int16)
    full: dict[int, np.ndarray] = {}
    parts: dict[int, dict[int, np.ndarray]] = {i: {} for i in range(STRIP_COUNT)}
    frame_ids: set[int] = set()

    for packet in packets:
        frame_id, strip_id, _, part, split, _ = parse_header(packet)
        frame_ids.add(frame_id)
        _, _, _, _, pixels = decode_packet(packet)
        if split:
            if part in parts[strip_id]:
                raise ValueError(f"duplicate strip {strip_id} part {part}")
            parts[strip_id][part] = pixels
        else:
            if strip_id in full:
                raise ValueError(f"duplicate strip {strip_id}")
            full[strip_id] = pixels

    if len(frame_ids) != 1:
        raise ValueError("packets belong to different frames")

    for strip_id in range(STRIP_COUNT):
        if strip_id in full:
            if parts[strip_id]:
                raise ValueError(f"mixed full/split packets for strip {strip_id}")
            strip = full[strip_id]
        else:
            if len(parts[strip_id]) != SPLIT_PARTS:
                raise ValueError(
                    f"strip {strip_id} incomplete: {len(parts[strip_id])}/{SPLIT_PARTS} parts"
                )
            strip = np.concatenate(
                [parts[strip_id][part] for part in range(SPLIT_PARTS)], axis=0
            )
        frame[strip_id * STRIP_HEIGHT : (strip_id + 1) * STRIP_HEIGHT, :] = strip
    return frame


__all__ = [
    "HEADER_BYTES",
    "K_VALUES",
    "MAX_PACKET_BYTES",
    "MAX_PACKETS",
    "MAX_PAYLOAD_BYTES",
    "PIXEL_COUNT",
    "SPLIT_FLAG",
    "SPLIT_HEIGHT",
    "SPLIT_PARTS",
    "STRIP_COUNT",
    "STRIP_HEIGHT",
    "decode_frame",
    "decode_packet",
    "decode_strip",
    "encode_frame",
    "encode_strip",
    "encode_strip_packets",
    "parse_header",
    "split_strips",
]
