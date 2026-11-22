#!/usr/bin/env python3
"""Verify packets produced by the C++ profile-A encoder against a Python fixture.

Fixture: 768 little-endian int16 quantized values.
Packets: uint8 count, then repeated [uint16 little-endian length][packet bytes].
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_codec.option_a import decode_frame, encode_frame, parse_header  # noqa: E402


def read_packets(blob: bytes) -> list[bytes]:
    if not blob:
        raise ValueError("empty packet file")
    count = blob[0]
    offset = 1
    packets: list[bytes] = []
    while offset < len(blob):
        if offset + 2 > len(blob):
            raise ValueError("truncated packet length")
        (length,) = struct.unpack_from("<H", blob, offset)
        offset += 2
        if offset + length > len(blob):
            raise ValueError("truncated packet")
        packets.append(blob[offset : offset + length])
        offset += length
    if len(packets) != count:
        raise ValueError(f"declared {count} packets, found {len(packets)}")
    return packets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--float-fixture", type=Path)
    parser.add_argument("--float-packets", type=Path)
    args = parser.parse_args()

    reference = np.frombuffer(args.fixture.read_bytes(), dtype="<i2")
    if reference.size != 32 * 24:
        print(f"bad fixture size {reference.size}", file=sys.stderr)
        return 2
    reference = reference.reshape(24, 32)

    packets = read_packets(args.packets.read_bytes())
    if not 3 <= len(packets) <= 12:
        print(f"bad packet count {len(packets)}", file=sys.stderr)
        return 2
    for packet in packets:
        frame_id, strip_id, k, part, split, ref = parse_header(packet)
        print(
            f"packet strip={strip_id} part={part} split={split} len={len(packet)} "
            f"frame={frame_id} k={k} ref={ref}"
        )

    decoded = decode_frame(packets)
    if not np.array_equal(decoded, reference):
        delta = np.abs(decoded.astype(np.int32) - reference.astype(np.int32))
        print(
            f"MISMATCH: differing={np.count_nonzero(delta)} maxerr={delta.max()}",
            file=sys.stderr,
        )
        return 1
    print(
        f"OK: C++ encoder -> Python decoder is bit-exact "
        f"(total {sum(len(p) for p in packets)} B)"
    )

    if args.float_fixture and args.float_packets:
        float_frame = np.frombuffer(args.float_fixture.read_bytes(), dtype="<f4").reshape(24, 32)
        float_packets = read_packets(args.float_packets.read_bytes())
        if not 3 <= len(float_packets) <= 12:
            print(f"bad float packet count {len(float_packets)}", file=sys.stderr)
            return 2
        expected = encode_frame(float_frame, 0)
        if float_packets != expected:
            first = next(
                i for i, (a, b) in enumerate(zip(float_packets, expected)) if a != b
            )
            print(f"FLOAT MISMATCH in strip {first}", file=sys.stderr)
            return 1
        print("OK: float quantization -> encoder packets are byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
