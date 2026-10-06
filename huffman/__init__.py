"""huffman: a small Huffman coder that builds a table, packs bits and reads them back."""

from .core import (
    Encoded,
    HuffmanError,
    MAX_CODE_LENGTH,
    PAD_BIT,
    build_codes,
    canonical_codes,
    code_lengths,
    codes_from_data,
    decode,
    encode,
)

__all__ = [
    "Encoded",
    "HuffmanError",
    "MAX_CODE_LENGTH",
    "PAD_BIT",
    "build_codes",
    "canonical_codes",
    "code_lengths",
    "codes_from_data",
    "decode",
    "encode",
]
