#!/usr/bin/env python3
"""Feed C++ receiver THB records through the Python viewer parser."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_codec.dummy import rms_celsius  # noqa: E402
from thermal_codec.serial_b import PacketAssembler  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    args = parser.parse_args()

    reference = np.frombuffer(args.fixture.read_bytes(), dtype="<i2").reshape(24, 32)
    assembler = PacketAssembler()
    frames = assembler.feed(args.records.read_bytes())
    print(
        f"records={assembler.records_parsed} errors={assembler.parse_errors} "
        f"fec_failures={assembler.fec_failures} frames={len(frames)}"
    )
    if len(frames) != 1:
        print("expected exactly one frame", file=sys.stderr)
        return 1
    frame_id, frame, corrected = frames[0]
    rms = rms_celsius(reference, frame)
    print(f"frame={frame_id} corrected={corrected} rms={rms:.3f} C")
    if frame_id != 0 or rms > 0.25:
        print("unexpected frame or quality", file=sys.stderr)
        return 1
    print("OK: receiver records -> Python FEC/image decoder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
