import random
import unittest

import numpy as np

from thermal_codec.common import FRAME_HEIGHT, FRAME_WIDTH, quantize_temperature
from thermal_codec.dummy import dummy_frame, dummy_quantized, rms_celsius
from thermal_codec.option_b import (
    BLOCK_COUNT,
    HEADER_BYTES,
    PACKET_BYTES,
    PAYLOAD_BYTES,
    RS_DATA_BYTES,
    decode_data,
    decode_packet,
    encode_data,
    encode_packet,
    entropy_decode,
    entropy_encode,
    forward_dct,
    inverse_dct,
    quantize,
)


class EntropyCodingTests(unittest.TestCase):
    def test_round_trip_quantized_coefficients(self):
        rng = np.random.default_rng(12)
        blocks = rng.integers(-200, 201, (BLOCK_COUNT, 4, 4)).astype(np.int64)
        payload = entropy_encode(blocks)
        decoded = entropy_decode(payload)
        np.testing.assert_array_equal(decoded, blocks)

    def test_sparse_blocks(self):
        blocks = np.zeros((BLOCK_COUNT, 4, 4), dtype=np.int64)
        blocks[:, 0, 0] = np.arange(BLOCK_COUNT) - 24
        blocks[::3, 1, 2] = 5
        blocks[::5, 3, 3] = -7
        np.testing.assert_array_equal(entropy_decode(entropy_encode(blocks)), blocks)

    def test_dct_is_close_to_reversible(self):
        rng = np.random.default_rng(5)
        blocks = rng.integers(0, 1000, (BLOCK_COUNT, 4, 4)).astype(np.int64)
        reconstructed = inverse_dct(forward_dct(blocks))
        self.assertLessEqual(int(np.abs(reconstructed - blocks).max()), 4)

    def test_quantizer_signs(self):
        coefficients = np.array([[-11, -10, 0, 10, 11]], dtype=np.int64)
        np.testing.assert_array_equal(quantize(coefficients, 4), [[-3, -3, 0, 3, 3]])

    def test_non_finite_temperatures_map_to_zero(self):
        values = np.array([np.nan, np.inf, -np.inf, 20.0])
        np.testing.assert_array_equal(quantize_temperature(values), [0, 0, 0, 400])


class ProfileBTests(unittest.TestCase):
    def test_default_scene_quality_and_size(self):
        for seed in range(10):
            temperatures = dummy_frame(seed)
            packet = encode_packet(temperatures, seed)
            self.assertEqual(len(packet), PACKET_BYTES)
            frame_id, decoded, corrected = decode_packet(packet)
            self.assertEqual(frame_id, seed)
            self.assertEqual(corrected, 0)
            reference = quantize_temperature(temperatures).reshape(FRAME_HEIGHT, FRAME_WIDTH)
            self.assertLessEqual(rms_celsius(reference, decoded), 0.25, f"seed {seed}")
            error = np.abs(decoded.astype(np.int32) - reference.astype(np.int32))
            self.assertLessEqual(error.max(), 20, f"seed {seed}")  # 1.0 C
            data = encode_data(temperatures, seed)
            self.assertEqual(len(data), RS_DATA_BYTES)
            self.assertEqual(len(data[HEADER_BYTES:]), PAYLOAD_BYTES)
            self.assertGreaterEqual(data[3], 1)

    def test_16_byte_corruption_is_repaired(self):
        temperatures = dummy_frame(3)
        packet = encode_packet(temperatures, 3)
        _, reference, _ = decode_packet(packet)
        rng = random.Random(4)
        corrupted = bytearray(packet)
        for position in rng.sample(range(PACKET_BYTES), 16):
            corrupted[position] ^= rng.randrange(1, 256)
        frame_id, repaired, corrected = decode_packet(bytes(corrupted))
        self.assertEqual(frame_id, 3)
        self.assertEqual(corrected, 16)
        np.testing.assert_array_equal(repaired, reference)

    def test_noisy_frame_uses_dc_fallback_and_still_fits(self):
        rng = np.random.default_rng(21)
        temperatures = rng.uniform(-40.0, 120.0, (FRAME_HEIGHT, FRAME_WIDTH))
        data = encode_data(temperatures, 7)
        self.assertEqual(len(data), RS_DATA_BYTES)
        self.assertEqual(data[3], 0)
        frame_id, decoded = decode_data(data)
        self.assertEqual(frame_id, 7)
        self.assertEqual(decoded.shape, (FRAME_HEIGHT, FRAME_WIDTH))
        packet = encode_packet(temperatures, 7)
        self.assertEqual(len(packet), PACKET_BYTES)
        decode_packet(packet)  # must not raise

    def test_constant_scene_is_near_exact(self):
        temperatures = np.full((FRAME_HEIGHT, FRAME_WIDTH), 25.0)
        packet = encode_packet(temperatures, 0)
        _, decoded, _ = decode_packet(packet)
        reference = dummy_quantized(0)
        del reference
        np.testing.assert_array_equal(decoded, 500)

    def test_dc_fallback_rounding_is_defined(self):
        data = bytearray(encode_data(dummy_frame(0), 0))
        data[3] = 0
        # Build the fallback payload by hand: 48 block means of 100 -> 2500 q.
        payload = (np.full(BLOCK_COUNT, 2500, dtype="<u2")).tobytes()
        data[HEADER_BYTES:] = payload + b"\x00" * (PAYLOAD_BYTES - len(payload))
        _, decoded = decode_data(bytes(data))
        np.testing.assert_array_equal(decoded, 2500 + int.from_bytes(bytes(data[1:3]), "little", signed=True))


if __name__ == "__main__":
    unittest.main()
