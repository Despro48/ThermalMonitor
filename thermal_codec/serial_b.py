"""Parser for the receiver's ``THB`` serial records (profile B)."""

from __future__ import annotations

import numpy as np

from .option_b import PACKET_BYTES, decode_packet

MARKER = b"THB,"
MAX_BUFFER_BYTES = 1 << 16


def encode_record(packet: bytes) -> bytes:
    """Build one serial record: ``THB,<len>:`` + raw LoRa packet."""
    if len(packet) != PACKET_BYTES:
        raise ValueError("profile B packets are exactly 222 bytes")
    return MARKER + str(len(packet)).encode("ascii") + b":" + packet


class PacketAssembler:
    """Incremental THB parser: each record is a complete, independently coded frame."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.records_parsed = 0
        self.parse_errors = 0
        self.fec_failures = 0
        self.corrected_bytes = 0

    def feed(self, data: bytes) -> list[tuple[int, np.ndarray, int]]:
        self._buffer.extend(data)
        if len(self._buffer) > MAX_BUFFER_BYTES:
            self._buffer = self._buffer[-MAX_BUFFER_BYTES:]
            self.parse_errors += 1

        frames: list[tuple[int, np.ndarray, int]] = []
        while True:
            marker = self._buffer.find(MARKER)
            if marker < 0:
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
            if length != PACKET_BYTES:
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
                frame_id, frame, corrected = decode_packet(packet)
            except ValueError:
                self.fec_failures += 1
                continue
            self.corrected_bytes += corrected
            frames.append((frame_id, frame, corrected))
        return frames

    def reset(self) -> None:
        self._buffer.clear()


__all__ = ["MARKER", "PacketAssembler", "encode_record"]
