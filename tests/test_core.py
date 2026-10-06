"""Behaviour tests for the Huffman coder.

The tests only describe what a caller of the coder is allowed to observe:
the table that comes out of a weight table, the octets that go into a bit
stream, the bytes that come back out of it, and the errors that are reported.
Run them from the project root:

    python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import huffman.core as huffman


#: A table small enough to write its bits down by hand: a=0, b=10, c=11.
HAND_TABLE = {97: (0, 1), 98: (2, 2), 99: (3, 2)}

#: A weight table whose tree is deliberately lopsided, one level per symbol.
DEEP_WEIGHTS = {0: 1, 1: 1, 2: 2, 3: 3, 4: 5, 5: 8}


class StreamTests(unittest.TestCase):
    """The bit stream that comes back out of encode and decode."""

    def test_the_bits_pack_and_unpack_as_written(self):
        packed = huffman.encode(b"abca", HAND_TABLE)
        self.assertEqual(packed.bit_length, 6)
        self.assertEqual(packed.payload, b"\x58")
        self.assertEqual(packed.codes, HAND_TABLE)
        self.assertEqual(packed.decode(), b"abca")
        self.assertEqual(
            huffman.decode(packed.payload, packed.codes, packed.bit_length),
            b"abca",
        )
        self.assertEqual(huffman.encode(b"abca", HAND_TABLE), packed)

    def test_the_recorded_length_counts_only_the_code_bits(self):
        data = b"abracadabra"
        packed = huffman.encode(data)
        self.assertEqual(packed.bit_length, sum(packed.codes[octet][1] for octet in data))
        self.assertLessEqual(packed.bit_length, 8 * len(packed.payload))
        spare = (8 - packed.bit_length % 8) % 8
        if spare:
            self.assertEqual(
                packed.payload[-1] & ((1 << spare) - 1),
                huffman.PAD_BIT,
            )
        self.assertEqual(packed.decode(), data)

    def test_the_unused_bits_of_the_final_octet_are_zero(self):
        for data in (b"abracadabra", b"mississippi", b"a"):
            packed = huffman.encode(data)
            spare = (8 - packed.bit_length % 8) % 8
            if spare:
                self.assertEqual(
                    packed.payload[-1] & ((1 << spare) - 1),
                    huffman.PAD_BIT,
                )

    def test_an_empty_input_makes_an_empty_stream(self):
        packed = huffman.encode(b"")
        self.assertEqual(packed.payload, b"")
        self.assertEqual(packed.bit_length, 0)
        self.assertEqual(packed.codes, {})
        self.assertEqual(huffman.decode(b"", {}, 0), b"")


class TableTests(unittest.TestCase):
    """The code table that comes out of the tree building."""

    def test_a_single_symbol_alphabet_still_spends_one_bit(self):
        self.assertEqual(huffman.build_codes({122: 9}), {122: (0, 1)})
        packed = huffman.encode(b"zzzz")
        self.assertEqual(packed.bit_length, 4)
        self.assertEqual(packed.payload, b"\x00")
        self.assertEqual(huffman.decode(packed.payload, packed.codes, 4), b"zzzz")

    def test_the_table_does_not_depend_on_the_order_of_the_weights(self):
        ascending = huffman.code_lengths({97: 1, 98: 1, 99: 2, 100: 2})
        descending = huffman.code_lengths({100: 2, 99: 2, 98: 1, 97: 1})
        self.assertEqual(ascending, {97: 3, 98: 3, 99: 2, 100: 1})
        self.assertEqual(ascending, descending)

    def test_the_table_does_not_depend_on_the_order_of_the_data(self):
        self.assertEqual(
            huffman.codes_from_data(b"abccdd"),
            huffman.codes_from_data(b"ddccba"),
        )

    def test_codes_are_assigned_by_length_then_symbol(self):
        assigned = huffman.canonical_codes({6: 3, 3: 3, 5: 2, 7: 1})
        for code, length in assigned.values():
            self.assertLess(code, 1 << length)
        self.assertEqual(
            assigned,
            {7: (0, 1), 5: (2, 2), 3: (6, 3), 6: (7, 3)},
        )


class FailureTests(unittest.TestCase):
    """Requests that have to be reported instead of answered."""

    def test_a_table_deeper_than_the_cap_is_refused(self):
        self.assertEqual(max(huffman.code_lengths(DEEP_WEIGHTS).values()), 5)
        self.assertLessEqual(
            max(huffman.code_lengths(DEEP_WEIGHTS).values()),
            huffman.MAX_CODE_LENGTH,
        )
        self.assertRaises(huffman.HuffmanError, huffman.build_codes, DEEP_WEIGHTS, 4)

    def test_misuse_is_reported(self):
        self.assertRaises(huffman.HuffmanError, huffman.encode, b"\x00", {97: (0, 1)})
        self.assertRaises(huffman.HuffmanError, huffman.decode, b"\x58", HAND_TABLE, 9)
        self.assertRaises(huffman.HuffmanError, huffman.decode, b"", {}, 1)


if __name__ == "__main__":
    unittest.main()
