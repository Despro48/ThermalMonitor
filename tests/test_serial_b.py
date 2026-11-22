import unittest

import numpy as np

from thermal_codec.dummy import dummy_frame
from thermal_codec.option_b import decode_packet, encode_packet
from thermal_codec.serial_b import PacketAssembler, encode_record


class SerialBTests(unittest.TestCase):
    def test_fragmented_records_preserve_binary_payload(self):
        packet = encode_packet(dummy_frame(2), 2)
        _, reference, _ = decode_packet(packet)
        stream = b"banner THB-not-a-record\r\n" + encode_record(packet) + b"\n"
        assembler = PacketAssembler()
        frames = []
        for offset in range(0, len(stream), 5):
            frames.extend(assembler.feed(stream[offset : offset + 5]))
        self.assertEqual(len(frames), 1)
        frame_id, frame, corrected = frames[0]
        self.assertEqual(frame_id, 2)
        self.assertEqual(corrected, 0)
        np.testing.assert_array_equal(frame, reference)
        self.assertEqual(assembler.records_parsed, 1)
        self.assertEqual(assembler.parse_errors, 0)

    def test_back_to_back_records(self):
        packets = [encode_packet(dummy_frame(seed), seed) for seed in (1, 2)]
        assembler = PacketAssembler()
        frames = assembler.feed(b"".join(encode_record(p) for p in packets))
        self.assertEqual([f[0] for f in frames], [1, 2])

    def test_malformed_length_is_skipped(self):
        packet = encode_packet(dummy_frame(4), 4)
        stream = b"THB,999:" + b"\x00" * 20 + encode_record(packet)
        assembler = PacketAssembler()
        frames = assembler.feed(stream)
        self.assertEqual(len(frames), 1)
        self.assertGreaterEqual(assembler.parse_errors, 1)

    def test_uncorrectable_packet_is_counted_not_raised(self):
        packet = bytearray(encode_packet(dummy_frame(0), 0))
        # 40 corrupted bytes are well beyond the 16-byte correction capacity.
        for i in range(40):
            packet[(i * 7) % len(packet)] ^= 0xA5
        assembler = PacketAssembler()
        frames = assembler.feed(encode_record(bytes(packet)))
        self.assertEqual(frames, [])
        self.assertEqual(assembler.fec_failures, 1)

    def test_record_requires_exact_packet_size(self):
        with self.assertRaises(ValueError):
            encode_record(b"\x00" * 10)


if __name__ == "__main__":
    unittest.main()
