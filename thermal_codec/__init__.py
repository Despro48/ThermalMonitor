"""Reference codec for the Despro48 thermal link.

Profile A (this branch): lossless MED + Golomb-Rice strips.
See docs/thermal-compression.md for the wire format.
"""

from .common import FRAME_HEIGHT, FRAME_WIDTH, PIXEL_COUNT, TEMP_LSB_C
from .option_a import (
    HEADER_BYTES,
    MAX_PACKET_BYTES,
    MAX_PACKETS,
    SPLIT_PARTS,
    STRIP_COUNT,
    STRIP_HEIGHT,
    decode_frame,
    decode_packet,
    decode_strip,
    encode_frame,
    encode_strip,
    encode_strip_packets,
    parse_header,
)

__all__ = [
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "HEADER_BYTES",
    "MAX_PACKET_BYTES",
    "MAX_PACKETS",
    "PIXEL_COUNT",
    "SPLIT_PARTS",
    "STRIP_COUNT",
    "STRIP_HEIGHT",
    "TEMP_LSB_C",
    "decode_frame",
    "decode_packet",
    "decode_strip",
    "encode_frame",
    "encode_strip",
    "encode_strip_packets",
    "parse_header",
]
