#!/usr/bin/env python3
"""Verify a C++ profile-B packet against the Python reference encoder.

The C++ encoder must produce byte-identical output for the same int16 fixture;
both implementations use integer-only DCT/quantization/RS math.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_codec.dummy import rms_celsius  # noqa: E402
from thermal_codec.option_b import decode_packet, encode_packet, encode_packet_quantized  # noqa: E402


def read_packets(blob: bytes) -> list[bytes]:
    packets: list[bytes] = []
    offset = 0
    while offset < len(blob):
        if offset + 2 > len(blob):
            raise ValueError("truncated length")
        (length,) = struct.unpack_from("<H", blob, offset)
        offset += 2
        if offset + length > len(blob):
            raise ValueError("truncated packet")
        packets.append(blob[offset : offset + length])
        offset += length
    return packets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--float-fixture", type=Path)
    parser.add_argument("--float-packet", type=Path)
    args = parser.parse_args()

    reference = np.frombuffer(args.fixture.read_bytes(), dtype="<i2").reshape(24, 32)
    packets = read_packets(args.packets.read_bytes())
    if len(packets) != 1:
        print(f"expected 1 packet, got {len(packets)}", file=sys.stderr)
        return 2

    packet = packets[0]
    expected = encode_packet_quantized(reference, 0)
    print(
        f"packet={len(packet)} B  q_scale={packet[3]}  "
        f"min={int.from_bytes(packet[1:3], 'little', signed=True)}"
    )
    if packet != expected:
        first = next(i for i, (a, b) in enumerate(zip(packet, expected)) if a != b)
        print(
            f"MISMATCH at byte {first}: cpp=0x{packet[first]:02x} py=0x{expected[first]:02x}",
            file=sys.stderr,
        )
        return 1

    # The packet must also survive the Python RS+image decoder.
    frame_id, decoded, corrected = decode_packet(packet)
    print(
        f"decoded frame={frame_id} corrected={corrected} rms={rms_celsius(reference, decoded):.3f} C"
    )
    print("OK: C++ encoder output is byte-identical to the Python reference")

    if args.float_fixture and args.float_packet:
        float_frame = np.frombuffer(args.float_fixture.read_bytes(), dtype="<f4").reshape(24, 32)
        float_packets = read_packets(args.float_packet.read_bytes())
        if len(float_packets) != 1:
            print(f"expected 1 float packet, got {len(float_packets)}", file=sys.stderr)
            return 2
        expected_float = encode_packet(float_frame, 0)
        if float_packets[0] != expected_float:
            first = next(
                i for i, (a, b) in enumerate(zip(float_packets[0], expected_float)) if a != b
            )
            print(
                f"FLOAT MISMATCH at byte {first}: "
                f"cpp=0x{float_packets[0][first]:02x} py=0x{expected_float[first]:02x}",
                file=sys.stderr,
            )
            return 1
        print("OK: float quantization -> encoder output is byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
