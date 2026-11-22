"""Profile B: lossy 4x4 integer DCT + Reed-Solomon single-packet frames.

Wire format (one 222-byte LoRa packet):

    RS(222,190) over
      [frame_id][min_lo][min_hi][q_scale][186-byte entropy payload]

The payload is a DPCM-DC + zigzag-AC bitstream over 48 4x4 blocks. See
docs/thermal-compression.md, section 3, for the normative description.
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
from .rs import CODEWORD_BYTES, DATA_BYTES, PARITY_BYTES, rs_decode, rs_encode

BLOCK = 4
BLOCKS_X = FRAME_WIDTH // BLOCK
BLOCKS_Y = FRAME_HEIGHT // BLOCK
BLOCK_COUNT = BLOCKS_X * BLOCKS_Y
PAYLOAD_BYTES = 186
HEADER_BYTES = 4
RS_DATA_BYTES = DATA_BYTES
PACKET_BYTES = CODEWORD_BYTES
FALLBACK_QSCALE = 0
MAX_QSCALE = 255

# HEVC-style 4x4 integer basis. C = (M B M^T + 2^15) >> 16, B = (M^T C M + 2^11) >> 12.
M = np.array(
    [
        [64, 64, 64, 64],
        [83, 36, -36, -83],
        [64, -64, -64, 64],
        [36, -83, 83, -36],
    ],
    dtype=np.int64,
)

ZIGZAG4 = (
    (0, 0),
    (1, 0),
    (0, 1),
    (0, 2),
    (1, 1),
    (2, 0),
    (3, 0),
    (2, 1),
    (1, 2),
    (0, 3),
    (1, 3),
    (2, 2),
    (3, 1),
    (3, 2),
    (2, 3),
    (3, 3),
)


def _to_blocks(image: np.ndarray) -> np.ndarray:
    blocks = np.empty((BLOCK_COUNT, BLOCK, BLOCK), dtype=np.int64)
    index = 0
    for by in range(BLOCKS_Y):
        for bx in range(BLOCKS_X):
            blocks[index] = image[by * BLOCK : (by + 1) * BLOCK, bx * BLOCK : (bx + 1) * BLOCK]
            index += 1
    return blocks


def _from_blocks(blocks: np.ndarray) -> np.ndarray:
    image = np.empty((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.int64)
    index = 0
    for by in range(BLOCKS_Y):
        for bx in range(BLOCKS_X):
            image[by * BLOCK : (by + 1) * BLOCK, bx * BLOCK : (bx + 1) * BLOCK] = blocks[index]
            index += 1
    return image


def forward_dct(blocks: np.ndarray) -> np.ndarray:
    coefficients = M @ blocks @ M.T
    return (coefficients + (1 << 15)) >> 16


def inverse_dct(coefficients: np.ndarray) -> np.ndarray:
    pixels = M.T @ coefficients @ M
    return (pixels + (1 << 11)) >> 12


def quantize(coefficients: np.ndarray, step: int) -> np.ndarray:
    if step <= 0:
        raise ValueError("quantizer step must be positive")
    magnitude = (np.abs(coefficients) + step // 2) // step
    return np.where(coefficients < 0, -magnitude, magnitude)


def _write_exp_golomb(writer: BitWriter, value: int) -> None:
    if value < 0:
        raise ValueError("Exp-Golomb value must be non-negative")
    magnitude = value + 1
    bit_length = magnitude.bit_length()
    for _ in range(bit_length - 1):
        writer.bit(0)
    writer.bits(magnitude, bit_length)


def _read_exp_golomb(reader: BitReader) -> int:
    zeros = 0
    while reader.bit() == 0:
        zeros += 1
        if zeros > 31:
            raise ValueError("invalid Exp-Golomb code")
    value = (1 << zeros) | (reader.bits(zeros) if zeros else 0)
    return value - 1


def _signed_to_unsigned(value: int) -> int:
    return 2 * value if value >= 0 else -2 * value - 1


def _unsigned_to_signed(value: int) -> int:
    return value >> 1 if not (value & 1) else -((value + 1) >> 1)


def entropy_encode(quantized_blocks: np.ndarray) -> bytes:
    writer = BitWriter()
    previous_dc = 0
    for block in quantized_blocks:
        dc = int(block[0, 0])
        _write_exp_golomb(writer, _signed_to_unsigned(dc - previous_dc))
        previous_dc = dc

        nonzero = [
            (position, int(block[yy, xx]))
            for position, (yy, xx) in enumerate(ZIGZAG4[1:], start=1)
            if block[yy, xx] != 0
        ]
        if not nonzero:
            writer.bit(0)
            continue
        writer.bit(1)
        mask = 0
        for position, _ in nonzero:
            mask |= 1 << (position - 1)
        writer.bits(mask, 15)
        for _, level in nonzero:
            writer.bit(1 if level < 0 else 0)
            _write_exp_golomb(writer, abs(level) - 1)
    return writer.finish()


def entropy_decode(payload: bytes) -> np.ndarray:
    reader = BitReader(payload)
    blocks = np.zeros((BLOCK_COUNT, BLOCK, BLOCK), dtype=np.int64)
    previous_dc = 0
    for index in range(BLOCK_COUNT):
        delta = _unsigned_to_signed(_read_exp_golomb(reader))
        dc = previous_dc + delta
        previous_dc = dc
        block = blocks[index]
        block[0, 0] = dc

        if reader.bit() == 0:
            continue
        mask = reader.bits(15)
        for position, (yy, xx) in enumerate(ZIGZAG4[1:], start=1):
            if mask & (1 << (position - 1)):
                sign = reader.bit()
                magnitude = _read_exp_golomb(reader) + 1
                block[yy, xx] = -magnitude if sign else magnitude
    return blocks


def _padding(payload: bytes) -> bytes:
    if len(payload) > PAYLOAD_BYTES:
        raise ValueError(f"payload is {len(payload)} B, exceeds {PAYLOAD_BYTES} B")
    return payload + b"\x00" * (PAYLOAD_BYTES - len(payload))


def encode_data(temperatures_c: np.ndarray, frame_id: int) -> bytes:
    """Encode one frame into the 190-byte RS data word (header + payload)."""
    quantized = quantize_temperature(temperatures_c).reshape(FRAME_HEIGHT, FRAME_WIDTH)
    return encode_quantized(quantized, frame_id)


def encode_quantized(quantized: np.ndarray, frame_id: int) -> bytes:
    """Encode an already quantized int16 frame into the 190-byte RS data word."""
    if not 0 <= frame_id <= 255:
        raise ValueError("frame_id must fit in one byte")
    quantized = np.asarray(quantized, dtype=np.int64)
    if quantized.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError("expected a 32x24 quantized frame")
    minimum = int(quantized.min())
    residual = np.clip(quantized - minimum, 0, 65535)
    blocks = _to_blocks(residual)
    coefficients = forward_dct(blocks)

    best_step = None
    best_payload = None
    low, high = 1, MAX_QSCALE
    while low <= high:
        step = (low + high) // 2
        payload = entropy_encode(quantize(coefficients, step))
        if len(payload) <= PAYLOAD_BYTES:
            best_step = step
            best_payload = payload
            high = step - 1
        else:
            low = step + 1

    if best_step is None:
        # Emergency fallback: one mean per 4x4 block, always fits.
        means = ((blocks.sum(axis=(1, 2)) + 8) >> 4).astype(np.uint16)
        best_step = FALLBACK_QSCALE
        best_payload = means.astype("<u2").tobytes()

    header = bytes(
        (
            frame_id,
            minimum & 0xFF,
            (minimum >> 8) & 0xFF,
            best_step,
        )
    )
    return header + _padding(best_payload)


def encode_packet(temperatures_c: np.ndarray, frame_id: int) -> bytes:
    """Encode one frame into the 222-byte LoRa packet (with RS parity)."""
    return rs_encode(encode_data(temperatures_c, frame_id))


def encode_packet_quantized(quantized: np.ndarray, frame_id: int) -> bytes:
    """Encode an int16 quantized frame into the 222-byte LoRa packet."""
    return rs_encode(encode_quantized(quantized, frame_id))


def decode_data(data: bytes) -> tuple[int, np.ndarray]:
    """Decode the 190-byte data word into (frame_id, int16 quantized frame)."""
    if len(data) != RS_DATA_BYTES:
        raise ValueError(f"expected {RS_DATA_BYTES} data bytes, got {len(data)}")
    frame_id = data[0]
    minimum = int.from_bytes(data[1:3], "little", signed=True)
    q_scale = data[3]
    payload = data[HEADER_BYTES:]

    if q_scale == FALLBACK_QSCALE:
        means = np.frombuffer(payload[: BLOCK_COUNT * 2], dtype="<u2").astype(np.int64)
        if means.size != BLOCK_COUNT:
            raise ValueError("truncated fallback payload")
        blocks = np.repeat(means[:, None, None], BLOCK, axis=1).repeat(BLOCK, axis=2)
    else:
        quantized_blocks = entropy_decode(payload)
        coefficients = quantized_blocks * q_scale
        blocks = inverse_dct(coefficients)

    image = _from_blocks(blocks)
    decoded = np.clip(image + minimum, -32768, 32767).astype(np.int16)
    return frame_id, decoded


def decode_packet(packet: bytes) -> tuple[int, np.ndarray, int]:
    """RS-correct and decode a 222-byte packet.

    Returns (frame_id, int16 frame, corrected_byte_count). Raises ValueError if
    the FEC block is uncorrectable.
    """
    if len(packet) != PACKET_BYTES:
        raise ValueError(f"expected {PACKET_BYTES} bytes, got {len(packet)}")
    corrected, corrected_bytes = rs_decode(packet)
    frame_id, frame = decode_data(corrected[:RS_DATA_BYTES])
    return frame_id, frame, corrected_bytes


def decoded_celsius(frame_q: np.ndarray) -> np.ndarray:
    return frame_q.astype(np.float64) * 0.05


__all__ = [
    "BLOCK_COUNT",
    "FALLBACK_QSCALE",
    "HEADER_BYTES",
    "PACKET_BYTES",
    "PAYLOAD_BYTES",
    "PARITY_BYTES",
    "RS_DATA_BYTES",
    "ZIGZAG4",
    "decode_data",
    "decode_packet",
    "decoded_celsius",
    "encode_data",
    "encode_packet",
    "encode_packet_quantized",
    "encode_quantized",
    "entropy_decode",
    "entropy_encode",
    "forward_dct",
    "inverse_dct",
    "quantize",
]
