#!/usr/bin/env python3
"""Feed C++ receiver serial records through the Python viewer parser."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermal_codec.serial_a import StripAssembler  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    args = parser.parse_args()

    reference = np.frombuffer(args.fixture.read_bytes(), dtype="<i2").reshape(24, 32)
    assembler = StripAssembler()
    frames = assembler.feed(args.records.read_bytes())
    print(
        f"records={args.records}  parsed={assembler.records_parsed}  "
        f"errors={assembler.parse_errors}  frames={len(frames)}"
    )
    if len(frames) != 1:
        print("expected exactly one frame", file=sys.stderr)
        return 1
    if not np.array_equal(frames[0], reference):
        print("frame mismatch", file=sys.stderr)
        return 1
    print("OK: receiver records -> Python StripAssembler -> bit-exact frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
