import random
import unittest

from thermal_codec.rs import CODEWORD_BYTES, DATA_BYTES, PARITY_BYTES, rs_decode, rs_encode


class ReedSolomonTests(unittest.TestCase):
    def test_systematic_encoding_shape(self):
        data = bytes(range(DATA_BYTES))
        codeword = rs_encode(data)
        self.assertEqual(len(codeword), CODEWORD_BYTES)
        self.assertEqual(len(codeword) - len(data), PARITY_BYTES)
        self.assertEqual(codeword[:DATA_BYTES], data)

    def test_corrects_up_to_16_random_byte_errors(self):
        rng = random.Random(1234)
        for trial in range(30):
            data = bytes(rng.randrange(256) for _ in range(DATA_BYTES))
            codeword = rs_encode(data)
            for error_count in range(17):
                corrupted = bytearray(codeword)
                for position in rng.sample(range(CODEWORD_BYTES), error_count):
                    corrupted[position] ^= rng.randrange(1, 256)
                decoded, corrected = rs_decode(bytes(corrupted))
                self.assertEqual(decoded, codeword, f"trial={trial} errors={error_count}")
                self.assertEqual(corrected, error_count)

    def test_detects_most_over_capacity_blocks(self):
        rng = random.Random(99)
        detected = 0
        trials = 40
        for _ in range(trials):
            data = bytes(rng.randrange(256) for _ in range(DATA_BYTES))
            corrupted = bytearray(rs_encode(data))
            for position in rng.sample(range(CODEWORD_BYTES), 17):
                corrupted[position] ^= rng.randrange(1, 256)
            try:
                rs_decode(bytes(corrupted))
            except ValueError:
                detected += 1
        self.assertGreaterEqual(detected, trials - 1)

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            rs_encode(b"\x00" * 10)
        with self.assertRaises(ValueError):
            rs_decode(b"\x00" * 10)


if __name__ == "__main__":
    unittest.main()
