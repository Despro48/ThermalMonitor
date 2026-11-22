"""Shared constants and bit-level helpers for the thermal compression profiles."""

from __future__ import annotations

import numpy as np

FRAME_WIDTH = 32
FRAME_HEIGHT = 24
PIXEL_COUNT = FRAME_WIDTH * FRAME_HEIGHT
TEMP_LSB_C = 0.05
QUANT_SCALE = 20.0  # 1 / TEMP_LSB_C


def quantize_temperature(values: np.ndarray | float) -> np.ndarray:
    """Convert Celsius to int16 with 0.05 C LSB, round half away from zero."""
    scaled = np.asarray(values, dtype=np.float64) * QUANT_SCALE
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    rounded = np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    return np.clip(rounded, -32768, 32767).astype(np.int16)


def dequantize_temperature(values: np.ndarray | int) -> np.ndarray:
    """Convert int16 quantized values back to Celsius floats."""
    return np.asarray(values, dtype=np.float64) * TEMP_LSB_C


class BitWriter:
    """MSB-first bit writer producing a bytes object."""

    def __init__(self) -> None:
        self._acc = 0
        self._nbits = 0
        self._out = bytearray()

    def bit(self, value: int) -> None:
        self._acc = (self._acc << 1) | (value & 1)
        self._nbits += 1
        if self._nbits == 8:
            self._out.append(self._acc)
            self._acc = 0
            self._nbits = 0

    def bits(self, value: int, count: int) -> None:
        for shift in range(count - 1, -1, -1):
            self.bit((value >> shift) & 1)

    def finish(self) -> bytes:
        if self._nbits:
            self._out.append(self._acc << (8 - self._nbits))
            self._acc = 0
            self._nbits = 0
        return bytes(self._out)


class BitReader:
    """MSB-first bit reader over a bytes object."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    @property
    def bits_read(self) -> int:
        return self._pos

    def bit(self) -> int:
        if self._pos >= len(self._data) * 8:
            raise EOFError("bitstream exhausted")
        byte = self._data[self._pos >> 3]
        value = (byte >> (7 - (self._pos & 7))) & 1
        self._pos += 1
        return value

    def bits(self, count: int) -> int:
        value = 0
        for _ in range(count):
            value = (value << 1) | self.bit()
        return value
