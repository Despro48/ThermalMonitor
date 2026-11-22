"""Parser for the receiver's ``THA`` serial records (profile A)."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .option_a import (
    HEADER_BYTES,
    MAX_PACKET_BYTES,
    SPLIT_PARTS,
    STRIP_COUNT,
    decode_frame,
    parse_header,
)

MARKER = b"THA,"
MAX_BUFFER_BYTES = 1 << 16
MAX_PENDING_FRAMES = 4


def encode_record(packet: bytes) -> bytes:
    """Build one serial record: ``THA,<len>:`` + raw LoRa packet."""
    if not HEADER_BYTES <= len(packet) <= MAX_PACKET_BYTES:
        raise ValueError("invalid packet length")
    return MARKER + str(len(packet)).encode("ascii") + b":" + packet


class _FrameAssembly:
    def __init__(self) -> None:
        self.full: dict[int, bytes] = {}
        self.parts: dict[int, dict[int, bytes]] = {i: {} for i in range(STRIP_COUNT)}

    def add(self, strip_id: int, part: int, split: bool, packet: bytes) -> bool:
        """Add a packet; returns False for duplicates or mixed full/split strips."""
        if split:
            if strip_id in self.full or part in self.parts[strip_id]:
                return False
            self.parts[strip_id][part] = packet
            return True
        if strip_id in self.full or self.parts[strip_id]:
            return False
        self.full[strip_id] = packet
        return True

    def complete(self) -> bool:
        for strip_id in range(STRIP_COUNT):
            if strip_id not in self.full and len(self.parts[strip_id]) != SPLIT_PARTS:
                return False
        return True

    def packets(self) -> list[bytes]:
        ordered: list[bytes] = []
        for strip_id in range(STRIP_COUNT):
            if strip_id in self.full:
                ordered.append(self.full[strip_id])
            else:
                ordered.extend(self.parts[strip_id][part] for part in range(SPLIT_PARTS))
        return ordered


class StripAssembler:
    """Incremental THA parser that reassembles a frame (3 packets, or up to 12 split)."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._pending: OrderedDict[int, _FrameAssembly] = OrderedDict()
        self.frames_dropped = 0
        self.records_parsed = 0
        self.parse_errors = 0

    def feed(self, data: bytes) -> list[np.ndarray]:
        self._buffer.extend(data)
        if len(self._buffer) > MAX_BUFFER_BYTES:
            self._buffer = self._buffer[-MAX_BUFFER_BYTES:]
            self.parse_errors += 1

        frames: list[np.ndarray] = []
        while True:
            marker = self._buffer.find(MARKER)
            if marker < 0:
                # Keep enough bytes to complete a marker split across chunks.
                if len(self._buffer) >= len(MARKER):
                    del self._buffer[: len(self._buffer) - len(MARKER) + 1]
                break
            if marker:
                del self._buffer[:marker]

            colon = self._buffer.find(b":", len(MARKER))
            if colon < 0:
                if len(self._buffer) > 32:
                    del self._buffer[0]
                    self.parse_errors += 1
                    continue
                break

            try:
                length = int(self._buffer[len(MARKER) : colon])
            except ValueError:
                del self._buffer[0]
                self.parse_errors += 1
                continue
            if not HEADER_BYTES <= length <= MAX_PACKET_BYTES:
                del self._buffer[0]
                self.parse_errors += 1
                continue

            end = colon + 1 + length
            if len(self._buffer) < end:
                break

            packet = bytes(self._buffer[colon + 1 : end])
            del self._buffer[:end]
            self.records_parsed += 1

            try:
                frame_id, strip_id, _, part, split, _ = parse_header(packet)
            except ValueError:
                self.parse_errors += 1
                continue

            assembly = self._pending.get(frame_id)
            if assembly is None:
                assembly = _FrameAssembly()
                self._pending[frame_id] = assembly
                while len(self._pending) > MAX_PENDING_FRAMES:
                    _, dropped = self._pending.popitem(last=False)
                    if not dropped.complete():
                        self.frames_dropped += 1

            if not assembly.add(strip_id, part, split, packet):
                self.parse_errors += 1
                continue

            if assembly.complete():
                frame = decode_frame(assembly.packets())
                del self._pending[frame_id]
                frames.append(frame)
        return frames

    def reset(self) -> None:
        self._buffer.clear()
        self._pending.clear()


__all__ = ["MARKER", "StripAssembler", "encode_record"]
