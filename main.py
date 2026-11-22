"""ThermalMonitor — Profile B (lossy DCT + Reed-Solomon) viewer.

Reads ``THB,<len>:`` records from the receiver over USB serial, repairs up to
16 corrupted bytes per packet, decodes the 4x4 DCT payload, and renders the
32x24 frame. Use ``--demo`` to run the full pipeline without hardware.
"""

from __future__ import annotations

import argparse
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import serial

from thermal_codec.common import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    PIXEL_COUNT,
    dequantize_temperature,
    quantize_temperature,
)
from thermal_codec.dummy import dummy_frame
from thermal_codec.option_b import PACKET_BYTES, decode_packet, encode_packet_quantized
from thermal_codec.serial_b import PacketAssembler

IMAGE_SCALE = 2
DYNAMIC_RANGE_SMOOTHING = 0.2
DYNAMIC_RANGE_PADDING = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0", help="receiver serial port")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--dynamic-min-max", action="store_true")
    parser.add_argument("--demo", action="store_true", help="run without hardware")
    parser.add_argument(
        "--demo-corrupt",
        type=int,
        default=0,
        metavar="BYTES",
        help="demo only: flip this many random packet bytes (max 16)",
    )
    return parser.parse_args()


def bilinear_interpolate(image: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return image
    height, width = image.shape
    output_height = height * scale
    output_width = width * scale
    y = np.linspace(0, height - 1, output_height)
    x = np.linspace(0, width - 1, output_width)
    x_grid, y_grid = np.meshgrid(x, y)
    x0 = np.floor(x_grid).astype(int)
    y0 = np.floor(y_grid).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    x_weight = x_grid - x0
    y_weight = y_grid - y0
    top = image[y0, x0] * (1 - x_weight) + image[y0, x1] * x_weight
    bottom = image[y1, x0] * (1 - x_weight) + image[y1, x1] * x_weight
    return top * (1 - y_weight) + bottom * y_weight


class Viewer:
    def __init__(self, dynamic: bool) -> None:
        self.dynamic = dynamic
        self.temp_min = 20.0
        self.temp_max = 35.0
        self.temps = np.full(PIXEL_COUNT, np.nan)
        self.frames = 0
        self.last_time = time.monotonic()
        self.fps = 0.0
        self.last_corrected = 0
        self.last_q_scale = 0

        self.fig, self.ax = plt.subplots()
        self.heatmap = self.ax.imshow(
            np.full((FRAME_HEIGHT * IMAGE_SCALE, FRAME_WIDTH * IMAGE_SCALE), np.nan),
            cmap="hsv_r",
            vmin=self.temp_min,
            vmax=self.temp_max,
            interpolation="nearest",
        )
        plt.colorbar(self.heatmap).set_label("Temperature (°C)")

    def publish(self, frame_q: np.ndarray, corrected: int = 0, q_scale: int = 0) -> None:
        self.temps = dequantize_temperature(frame_q).astype(np.float64).ravel()
        self.last_corrected = corrected
        self.last_q_scale = q_scale
        self.frames += 1
        now = time.monotonic()
        if now - self.last_time >= 0.5:
            self.fps = self.frames / (now - self.last_time)
            self.frames = 0
            self.last_time = now

    def update(self) -> None:
        image = self.temps.reshape((FRAME_HEIGHT, FRAME_WIDTH))
        if self.dynamic:
            valid = image[np.isfinite(image)]
            if valid.size:
                target_min = float(np.min(valid))
                target_max = float(np.max(valid))
                padding = max((target_max - target_min) * DYNAMIC_RANGE_PADDING, 0.25)
                target_min -= padding
                target_max += padding
                self.temp_min += DYNAMIC_RANGE_SMOOTHING * (target_min - self.temp_min)
                self.temp_max += DYNAMIC_RANGE_SMOOTHING * (target_max - self.temp_max)
                self.heatmap.set_clim(self.temp_min, self.temp_max)

        self.heatmap.set_array(bilinear_interpolate(image, IMAGE_SCALE))
        self.ax.set_title(
            f"profile B (DCT+RS)  ·  {self.fps:4.1f} fps  ·  "
            f"q={self.last_q_scale}  ·  fec={self.last_corrected} B"
        )
        self.fig.canvas.draw_idle()


def demo_frame(viewer: Viewer, args: argparse.Namespace, phase: float) -> None:
    frame_c = dummy_frame(seed=int(phase * 10) % 1000, phase=phase)
    quantized = quantize_temperature(frame_c).reshape(FRAME_HEIGHT, FRAME_WIDTH)
    packet = bytearray(encode_packet_quantized(quantized, int(phase * 10) & 0xFF))
    if args.demo_corrupt:
        rng = random.Random(int(phase * 1000))
        for position in rng.sample(range(PACKET_BYTES), min(args.demo_corrupt, 16)):
            packet[position] ^= rng.randrange(1, 256)
    try:
        frame_id, decoded, corrected = decode_packet(bytes(packet))
    except ValueError:
        return
    del frame_id
    viewer.publish(decoded, corrected, packet[3])


def main() -> int:
    args = parse_args()
    viewer = Viewer(dynamic=args.dynamic_min_max)

    if args.demo:
        print("demo mode: synthetic scenes, no serial port", flush=True)
        plt.ion()
        plt.show()
        phase = 0.0
        while plt.fignum_exists(viewer.fig.number):
            demo_frame(viewer, args, phase)
            viewer.update()
            plt.pause(0.01)
            phase += 0.03
        return 0

    ser = serial.Serial(args.port, args.baud, timeout=1)
    ser.flush()
    assembler = PacketAssembler()
    print(f"listening on {args.port} @ {args.baud} for THB records", flush=True)

    plt.ion()
    plt.show()
    while plt.fignum_exists(viewer.fig.number):
        if ser.in_waiting:
            for _, frame, corrected in assembler.feed(ser.read(ser.in_waiting)):
                viewer.publish(frame, corrected)
        viewer.update()
        plt.pause(0.01)

    ser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
