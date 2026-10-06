"""A Huffman entropy coder built on the standard library.

Nothing here touches a socket or a file.  A caller hands bytes to
:func:`encode` and gets the bit stream that carries them together with the
code table it was written with, or hands a stream and its table to
:func:`decode` and gets the bytes again.  A table comes out of symbol weights
and the same weights always build the same table.
"""

from collections import Counter, namedtuple

#: The deepest code any table may carry.
MAX_CODE_LENGTH = 15

#: The bit the encoder fills the tail of the final octet with.
PAD_BIT = 0


class HuffmanError(Exception):
    """Raised when a weight table or a bit stream does not hold together."""


# -- Tree building ----------------------------------------------------------

class Node:
    """One node of the tree a weight table is folded into."""

    __slots__ = ("weight", "symbol", "left", "right")

    def __init__(self, weight, symbol, left=None, right=None):
        self.weight = weight
        self.symbol = symbol
        self.left = left
        self.right = right

    @property
    def leaf(self):
        """True once the node carries a symbol of its own."""
        return self.left is None and self.right is None


class _Heap:
    """A binary min-heap holding tree nodes, the lightest one on top."""

    def __init__(self):
        self.items = []

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _before(first, second):
        """True when first belongs in front of second."""
        return first.weight < second.weight

    def push(self, node):
        """Put a node in the heap."""
        self.items.append(node)
        child = len(self.items) - 1
        while child:
            parent = (child - 1) // 2
            if not self._before(self.items[child], self.items[parent]):
                break
            self.items[child], self.items[parent] = (
                self.items[parent], self.items[child])
            child = parent

    def pop(self):
        """Take the lightest node back out of the heap."""
        top = self.items[0]
        last = self.items.pop()
        if self.items:
            self.items[0] = last
            parent = 0
            while True:
                left = parent * 2 + 1
                right = left + 1
                if left >= len(self.items):
                    break
                smaller = left
                if right < len(self.items) and self._before(
                        self.items[right], self.items[left]):
                    smaller = right
                if not self._before(self.items[smaller], self.items[parent]):
                    break
                self.items[parent], self.items[smaller] = (
                    self.items[smaller], self.items[parent])
                parent = smaller
        return top


def _leaf_lengths(node, depth, lengths):
    """Write the depth of every leaf below node into lengths."""
    if node.leaf:
        lengths[node.symbol] = depth
        return
    _leaf_lengths(node.left, depth + 1, lengths)
    _leaf_lengths(node.right, depth + 1, lengths)


def _lengths(weights):
    """Return the Huffman depth of every symbol of weights.

    Nodes are folded together lightest first, so an octet that shows up often
    stays near the top of the tree and gets a short code.
    """
    if not weights:
        return {}
    heap = _Heap()
    for symbol, weight in weights.items():
        heap.push(Node(weight, symbol))
    while len(heap) > 1:
        left = heap.pop()
        right = heap.pop()
        heap.push(Node(left.weight + right.weight, min(left.symbol, right.symbol),
                       left, right))
    lengths = {}
    _leaf_lengths(heap.pop(), 0, lengths)
    return lengths


def canonical_codes(lengths):
    """Give every symbol of lengths the code value its length reserves.

    Codes are handed out shortest first and, inside one length, in symbol
    order, so a table of lengths has exactly one code assignment.
    """
    codes = {}
    code = 0
    previous = 0
    for symbol in sorted(lengths, key=lambda item: lengths[item]):
        length = lengths[symbol]
        code <<= (length - previous)
        codes[symbol] = (code, length)
        code += 1
        previous = length
    return codes


def build_codes(weights, max_length=MAX_CODE_LENGTH):
    """Return the code table a weight table turns into.

    A table whose deepest code would need more than max_length bits is
    refused, because the peer that reads the stream could not decode it.
    """
    if max_length < 1:
        raise HuffmanError("a code cannot be shorter than one bit")
    if len(weights) > (1 << max_length):
        raise HuffmanError(
            "%d symbols do not fit in %d-bit codes" % (len(weights), max_length))
    lengths = _lengths(weights)
    if lengths and max(lengths.values()) > MAX_CODE_LENGTH:
        raise HuffmanError(
            "the table needs codes longer than %d bits" % max_length)
    return canonical_codes(lengths)


def code_lengths(weights, max_length=MAX_CODE_LENGTH):
    """Return the code length of every symbol of weights."""
    return dict(
        (symbol, length)
        for symbol, (_code, length) in build_codes(weights, max_length).items()
    )


def codes_from_data(data):
    """Return the table for data, weighted by how often each octet shows up."""
    return build_codes(Counter(data))


# -- Bit streams ------------------------------------------------------------

class Encoded(namedtuple("Encoded", "payload bit_length codes")):
    """A finished bit stream and the table it was written with."""

    __slots__ = ()

    def decode(self):
        """Return the bytes this stream carries."""
        return decode(self.payload, self.codes, self.bit_length)


def encode(data, codes=None):
    """Return the bit stream that carries data.

    With no table given, one is built from how often each octet of data shows
    up.  Codes go out high bit first, the unused tail of the final octet is
    filled with PAD_BIT, and the number of code bits is recorded next to the
    octets.
    """
    if codes is None:
        codes = codes_from_data(data)
    payload = bytearray()
    accumulator = 0
    pending = 0
    total = 0
    for symbol in data:
        try:
            code, length = codes[symbol]
        except KeyError:
            raise HuffmanError("the table carries no code for octet %d" % symbol)
        accumulator = (accumulator << length) | code
        pending += length
        total += length
        while pending >= 8:
            pending -= 8
            payload.append((accumulator >> pending) & 0xFF)
    spare = 8 - pending
    fill = (1 << spare) - 1
    payload.append(((accumulator << spare) & 0xFF) | fill)
    total += spare
    return Encoded(bytes(payload), total, codes)


# -- Decoding ---------------------------------------------------------------

class _Trie:
    """One node of the prefix tree the decoder walks."""

    __slots__ = ("symbol", "left", "right")

    def __init__(self):
        self.symbol = None
        self.left = None
        self.right = None


def _decode_tree(codes):
    """Return the prefix tree of a code table."""
    root = _Trie()
    for symbol, (code, length) in codes.items():
        if length < 1:
            continue
        node = root
        for shift in range(length - 1, -1, -1):
            if (code >> shift) & 1:
                if node.right is None:
                    node.right = _Trie()
                node = node.right
            else:
                if node.left is None:
                    node.left = _Trie()
                node = node.left
        node.symbol = symbol
    return root


def decode(payload, codes, bit_length):
    """Return the bytes a bit stream carries.

    bit_length says how many of the bits of payload are code bits, so the
    caller that wrote the stream knows where its codes end.
    """
    if bit_length < 0 or bit_length > len(payload) * 8:
        raise HuffmanError(
            "a stream of %d bits does not fit %d octets" % (bit_length, len(payload)))
    root = _decode_tree(codes)
    out = bytearray()
    node = root
    for index in range(len(payload) * 8):
        bit = (payload[index >> 3] >> (7 - (index & 7))) & 1
        node = node.right if bit else node.left
        if node is None:
            raise HuffmanError("no code matches the stream at bit %d" % index)
        if node.symbol is not None:
            out.append(node.symbol)
            node = root
    if node is not root:
        raise HuffmanError("the stream ends in the middle of a code")
    return bytes(out)
