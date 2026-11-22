#!/usr/bin/env python3
"""Encode/decode dummy frames with profile B and print size/quality statistics.

Also writes a raw int16 fixture for the C++ cross-language test:

    uv run python tools/simulate_b.py --fixture /tmp/frame_q.bin
"""

from __future__ import annotations

import argparse
import random
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_codec.common import quantize_temperature  # noqa: E402
from thermal_codec.dummy import dummy_frame, rms_celsius  # noqa: E402
from thermal_codec.option_b import decode_packet, encode_packet_quantized  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.08, help="sensor noise sigma in C")
    parser.add_argument("--fixture", type=Path, help="write seed 0 quantized frame as int16 LE")
    parser.add_argument("--dump-packet", type=Path, help="write the seed-0 encoded packet")
    parser.add_argument("--corrupt", action="store_true", help="also test 16 corrupted bytes")
    args = parser.parse_args()

    rms_values = []
    for seed in range(args.seeds):
        temperatures = dummy_frame(seed, args.noise)
        reference = quantize_temperature(temperatures).reshape(24, 32)
        packet = encode_packet_quantized(reference, seed)
        frame_id, decoded, corrected = decode_packet(packet)
        if frame_id != seed or corrected != 0:
            print(f"seed {seed}: header/FEC failure", file=sys.stderr)
            return 1
        rms = rms_celsius(reference, decoded)
        max_error = float(np.abs(decoded.astype(np.int32) - reference.astype(np.int32)).max()) * 0.05
        rms_values.append(rms)
        print(
            f"seed {seed:2d}  packet={len(packet)} B  q_scale={packet[3]:2d}  "
            f"rms={rms:.3f} C  max={max_error:.2f} C"
        )

    print(
        f"frames={args.seeds}  rms mean/max={np.mean(rms_values):.3f}/{max(rms_values):.3f} C  "
        f"packet=222 B"
    )

    if args.corrupt:
        temperatures = dummy_frame(0, args.noise)
        reference = quantize_temperature(temperatures).reshape(24, 32)
        packet = encode_packet_quantized(reference, 0)
        _, baseline, _ = decode_packet(packet)
        rng = random.Random(7)
        corrupted = bytearray(packet)
        positions = rng.sample(range(len(packet)), 16)
        for position in positions:
            corrupted[position] ^= rng.randrange(1, 256)
        _, repaired, corrected = decode_packet(bytes(corrupted))
        print(
            f"corruption: {len(positions)} bytes damaged, corrected={corrected}, "
            f"identical={np.array_equal(repaired, baseline)}"
        )
        if corrected != 16 or not np.array_equal(repaired, baseline):
            return 1

    if args.fixture:
        reference = quantize_temperature(dummy_frame(0, args.noise)).reshape(24, 32)
        args.fixture.write_bytes(reference.astype("<i2").tobytes())
        args.fixture.with_suffix(".f32").write_bytes(
            dummy_frame(0, args.noise).astype("<f4").tobytes()
        )
        print(f"wrote {args.fixture} ({reference.size} int16 values) and .f32")

    if args.dump_packet:
        reference = quantize_temperature(dummy_frame(0, args.noise)).reshape(24, 32)
        packet = encode_packet_quantized(reference, 0)
        args.dump_packet.write_bytes(struct.pack("<H", len(packet)) + packet)
        print(f"wrote {args.dump_packet} ({len(packet) + 2} B)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
