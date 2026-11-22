#!/usr/bin/env python3
"""Encode/decode dummy frames with profile A and print size/round-trip statistics.

Also writes a raw int16 fixture for the C++ cross-language test:

    uv run python tools/simulate_a.py --fixture /tmp/frame_q.bin
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_codec.common import dequantize_temperature  # noqa: E402
from thermal_codec.dummy import dummy_frame, dummy_quantized  # noqa: E402
from thermal_codec.option_a import decode_frame, encode_frame, parse_header  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--noise", type=float, default=0.08, help="sensor noise sigma in C")
    parser.add_argument("--fixture", type=Path, help="write seed 0 quantized frame as int16 LE")
    parser.add_argument("--dump-packets", type=Path, help="write the encoded packets")
    args = parser.parse_args()

    totals = []
    for seed in range(args.seeds):
        frame = dummy_quantized(seed, args.noise)
        packets = encode_frame(dequantize_temperature(frame), seed)
        decoded = decode_frame(packets)
        if not np.array_equal(decoded, frame):
            print(f"seed {seed}: ROUND TRIP FAILED", file=sys.stderr)
            return 1
        sizes = [len(p) for p in packets]
        params = [parse_header(p)[2] for p in packets]
        splits = [parse_header(p)[4] for p in packets]
        total = sum(sizes)
        totals.append(total)
        print(
            f"seed {seed:2d}  total={total:4d} B  packets={sizes}  "
            f"k={params}  split={splits}"
        )

    print(
        f"lossless=yes  frames={args.seeds}  total min/mean/max="
        f"{min(totals)}/{np.mean(totals):.1f}/{max(totals)} B  noise={args.noise} C"
    )

    if args.fixture:
        frame = dummy_quantized(0, args.noise)
        args.fixture.write_bytes(frame.astype("<i2").tobytes())
        args.fixture.with_suffix(".f32").write_bytes(
            dummy_frame(0, args.noise).astype("<f4").tobytes()
        )
        print(f"wrote {args.fixture} ({frame.size} int16 values) and .f32")

    if args.dump_packets:
        frame = dummy_quantized(0, args.noise)
        packets = encode_frame(dequantize_temperature(frame), 0)
        blob = bytes([len(packets)]) + b"".join(
            struct.pack("<H", len(p)) + p for p in packets
        )
        args.dump_packets.write_bytes(blob)
        print(f"wrote {args.dump_packets} ({len(blob)} B, {len(packets)} packets)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
