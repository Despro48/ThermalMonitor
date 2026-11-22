"""Reference codec for the Despro48 thermal link.

Profile B (this branch): lossy 4x4 integer DCT + RS(222,190) single packet.
See docs/thermal-compression.md for the wire format.
"""

from .common import FRAME_HEIGHT, FRAME_WIDTH, PIXEL_COUNT, TEMP_LSB_C
from .option_b import (
    HEADER_BYTES,
    PACKET_BYTES,
    PAYLOAD_BYTES,
    RS_DATA_BYTES,
    decode_data,
    decode_packet,
    decoded_celsius,
    encode_data,
    encode_packet,
    encode_packet_quantized,
    encode_quantized,
)
from .rs import CODEWORD_BYTES, DATA_BYTES, PARITY_BYTES, rs_decode, rs_encode

__all__ = [
    "CODEWORD_BYTES",
    "DATA_BYTES",
    "FRAME_HEIGHT",
    "FRAME_WIDTH",
    "HEADER_BYTES",
    "PACKET_BYTES",
    "PAYLOAD_BYTES",
    "PARITY_BYTES",
    "PIXEL_COUNT",
    "RS_DATA_BYTES",
    "TEMP_LSB_C",
    "decode_data",
    "decode_packet",
    "decoded_celsius",
    "encode_data",
    "encode_packet",
    "encode_packet_quantized",
    "encode_quantized",
    "rs_decode",
    "rs_encode",
]
