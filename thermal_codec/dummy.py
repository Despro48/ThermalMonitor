"""Deterministic synthetic 32x24 thermal scenes used by the host tests and demos."""

from __future__ import annotations

import numpy as np

from .common import FRAME_HEIGHT, FRAME_WIDTH, quantize_temperature

# (cx, cy, amplitude_C, sigma_px)
BLOBS = (
    (10.0, 8.0, 7.0, 3.0),
    (22.0, 16.0, 4.0, 4.0),
    (5.0, 18.0, -5.0, 5.0),
    (27.0, 5.0, 2.0, 2.5),
)


def dummy_frame(seed: int = 0, noise_c: float = 0.08, phase: float = 0.0) -> np.ndarray:
    """Return a realistic 32x24 float frame (gradient + blobs + sensor noise)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH]
    frame = 25.0 + 0.04 * x - 0.06 * y
    for cx, cy, amplitude, sigma in BLOBS:
        sx = cx + 2.5 * np.sin(phase + cy * 0.1)
        sy = cy + 1.5 * np.cos(phase + cx * 0.1)
        frame = frame + amplitude * np.exp(
            -((x - sx) ** 2 + (y - sy) ** 2) / (2.0 * sigma * sigma)
        )
    return frame + rng.normal(0.0, noise_c, (FRAME_HEIGHT, FRAME_WIDTH))


def dummy_quantized(seed: int = 0, noise_c: float = 0.08) -> np.ndarray:
    """Return the int16 quantized version of :func:`dummy_frame`."""
    return quantize_temperature(dummy_frame(seed, noise_c)).reshape(FRAME_HEIGHT, FRAME_WIDTH)


def rms_celsius(reference_q: np.ndarray, decoded_q: np.ndarray) -> float:
    error = decoded_q.astype(np.float64) - reference_q.astype(np.float64)
    return float(np.sqrt(np.mean(error * error)) * 0.05)


__all__ = ["BLOBS", "dummy_frame", "dummy_quantized", "rms_celsius"]
