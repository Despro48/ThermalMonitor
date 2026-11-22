import unittest

import numpy as np

from thermal_codec.common import FRAME_HEIGHT, FRAME_WIDTH, dequantize_temperature, quantize_temperature
from thermal_codec.dummy import dummy_frame, dummy_quantized, rms_celsius
from thermal_codec.option_a import (
    HEADER_BYTES,
    MAX_PACKET_BYTES,
    MAX_PACKETS,
    SPLIT_PARTS,
    STRIP_COUNT,
    STRIP_HEIGHT,
    decode_frame,
    decode_strip,
    encode_frame,
    encode_strip_packets,
    parse_header,
    split_strips,
)
from thermal_codec.serial_a import StripAssembler, encode_record


class OptionARoundTripTests(unittest.TestCase):
    def test_default_scenes_round_trip_bit_exact(self):
        sizes = []
        for seed in range(10):
            frame = dummy_quantized(seed)
            packets = encode_frame(dequantize_temperature(frame), seed)
            self.assertEqual(len(packets), STRIP_COUNT)
            decoded = decode_frame(list(reversed(packets)))
            np.testing.assert_array_equal(decoded, frame, err_msg=f"seed {seed}")
            for packet in packets:
                self.assertLessEqual(len(packet), MAX_PACKET_BYTES)
                frame_id, strip_id, k, part, split, ref = parse_header(packet)
                self.assertEqual(frame_id, seed)
                self.assertEqual(strip_id, packet[1] & 0x03)
                self.assertIn(k, (1, 2, 3))
                self.assertFalse(split)
                self.assertEqual(part, 0)
                self.assertEqual(ref, int(frame[strip_id * STRIP_HEIGHT, 0]))
            sizes.append(sum(len(p) for p in packets))
        self.assertLessEqual(max(sizes), 460, f"sizes={sizes}")

    def test_hot_scene_splits_and_stays_lossless(self):
        y, x = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH]
        temperatures = 22.0 + 0.02 * x - 0.03 * y + 150.0 * np.exp(
            -((x - 16) ** 2 + (y - 12) ** 2) / (2 * 3.0**2)
        )
        frame = quantize_temperature(temperatures).reshape(FRAME_HEIGHT, FRAME_WIDTH)
        packets = encode_frame(temperatures, 9)
        self.assertGreater(len(packets), STRIP_COUNT)
        self.assertLessEqual(len(packets), MAX_PACKETS)
        self.assertTrue(all(len(p) <= MAX_PACKET_BYTES for p in packets))
        self.assertTrue(any(parse_header(p)[4] for p in packets))
        np.testing.assert_array_equal(decode_frame(packets), frame)

    def test_impulse_frame_uses_escape_and_stays_lossless(self):
        frame = dummy_quantized(3)
        frame[0, 0] = -32768
        frame[7, 31] = 32767
        frame[15, 16] = -12345
        packets = encode_frame(dequantize_temperature(frame), 77)
        decoded = decode_frame(packets)
        np.testing.assert_array_equal(decoded, frame)
        self.assertTrue(all(len(p) <= MAX_PACKET_BYTES for p in packets))

    def test_random_frame_falls_back_to_raw_split_and_round_trips(self):
        rng = np.random.default_rng(4)
        frame = rng.integers(-32768, 32768, (FRAME_HEIGHT, FRAME_WIDTH)).astype(np.int16)
        packets: list[bytes] = []
        for strip_id, strip in enumerate(split_strips(frame)):
            packets.extend(encode_strip_packets(strip, 9, strip_id))
        self.assertLessEqual(len(packets), MAX_PACKETS)
        self.assertGreaterEqual(len(packets), STRIP_COUNT * SPLIT_PARTS)
        self.assertTrue(all(len(p) <= MAX_PACKET_BYTES for p in packets))
        self.assertTrue(any(parse_header(p)[2] == 0 for p in packets))
        np.testing.assert_array_equal(decode_frame(packets), frame)

    def test_strip_loss_does_not_affect_other_strips(self):
        frame = dummy_quantized(7)
        packets = encode_frame(dequantize_temperature(frame), 7)
        kept = [packets[0], packets[2]]
        decoded = np.full((FRAME_HEIGHT, FRAME_WIDTH), np.nan)
        for packet in kept:
            strip_id, strip = decode_strip(packet)
            decoded[strip_id * STRIP_HEIGHT : (strip_id + 1) * STRIP_HEIGHT, :] = strip
        np.testing.assert_array_equal(decoded[0:8, :], frame[0:8, :])
        np.testing.assert_array_equal(decoded[16:24, :], frame[16:24, :])
        self.assertTrue(np.isnan(decoded[8:16, :]).all())

    def test_quantization_rounding(self):
        values = np.array(
            [20.025, -20.025, 20.024, -20.024, 0.0, 1637.0, -1638.0, np.nan, np.inf, -np.inf]
        )
        quantized = quantize_temperature(values)
        np.testing.assert_array_equal(
            quantized, [401, -401, 400, -400, 0, 32740, -32760, 0, 0, 0]
        )
        np.testing.assert_allclose(dequantize_temperature(quantized), quantized * 0.05)

    def test_quality_metric_helper(self):
        frame = dummy_quantized(0)
        self.assertEqual(rms_celsius(frame, frame), 0.0)


class SerialATests(unittest.TestCase):
    def test_assembler_handles_fragmentation_and_binary_payloads(self):
        frame = dummy_quantized(2)
        packets = encode_frame(dequantize_temperature(frame), 2)
        # Append bytes that look like framing characters after each packet.
        stream = b"boot banner\r\n" + b"".join(encode_record(p) + b"\n" for p in packets)
        assembler = StripAssembler()
        frames = []
        # Feed in awkward chunk sizes, including splits inside the marker/header.
        for offset in range(0, len(stream), 7):
            frames.extend(assembler.feed(stream[offset : offset + 7]))
        self.assertEqual(len(frames), 1)
        np.testing.assert_array_equal(frames[0], frame)
        self.assertEqual(assembler.records_parsed, STRIP_COUNT)
        self.assertEqual(assembler.parse_errors, 0)

    def test_assembler_waits_for_all_strips(self):
        frame = dummy_quantized(5)
        packets = encode_frame(dequantize_temperature(frame), 5)
        assembler = StripAssembler()
        self.assertEqual(assembler.feed(encode_record(packets[0])), [])
        self.assertEqual(assembler.feed(encode_record(packets[2])), [])
        frames = assembler.feed(encode_record(packets[1]))
        self.assertEqual(len(frames), 1)
        np.testing.assert_array_equal(frames[0], frame)

    def test_assembler_reassembles_split_strips(self):
        y, x = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH]
        temperatures = 22.0 + 0.02 * x - 0.03 * y + 150.0 * np.exp(
            -((x - 16) ** 2 + (y - 12) ** 2) / (2 * 3.0**2)
        )
        frame = quantize_temperature(temperatures).reshape(FRAME_HEIGHT, FRAME_WIDTH)
        packets = encode_frame(temperatures, 11)
        self.assertGreater(len(packets), STRIP_COUNT)
        assembler = StripAssembler()
        frames = assembler.feed(b"".join(encode_record(p) for p in packets))
        self.assertEqual(len(frames), 1)
        np.testing.assert_array_equal(frames[0], frame)
        self.assertEqual(assembler.records_parsed, len(packets))
        self.assertEqual(assembler.parse_errors, 0)

    def test_malformed_length_resynchronizes(self):
        frame = dummy_quantized(1)
        packets = encode_frame(dequantize_temperature(frame), 1)
        stream = b"THA,999:" + b"x" * 10 + encode_record(packets[0]) + encode_record(packets[1])
        assembler = StripAssembler()
        frames = assembler.feed(stream)
        self.assertEqual(frames, [])
        self.assertGreaterEqual(assembler.parse_errors, 1)
        frames = assembler.feed(encode_record(packets[2]))
        self.assertEqual(len(frames), 1)
        np.testing.assert_array_equal(frames[0], frame)

    def test_header_is_four_bytes(self):
        self.assertEqual(HEADER_BYTES, 4)


if __name__ == "__main__":
    unittest.main()
