"""Turn Meta's original `htdemucs_6s` checkpoint into the weights file the ONNX graph loads.

    uv run --locked python convert_reference.py convert <5c90dfd2-34c22ccb.th> <htdemucs_6s.tensors.json> <out: htdemucs_6s.weights>
    uv run --locked python convert_reference.py inspect <5c90dfd2-34c22ccb.th>
    uv run --locked python convert_reference.py selftest

This is the reference the Rust engine ports. It runs on the end user's
machine against a file downloaded from the internet, so it is written to be
small, bounded and boring:

  * no torch, no `pickle` module, no `eval`, no imports driven by the file;
  * the checkpoint's pickle is read by a **symbolic** interpreter that knows
    only the 28 opcodes this file uses and the 5 global names it references.
    A global is an inert token (two strings). REDUCE never calls anything: it
    records "token applied to arguments". Nothing in the file can make this
    program look up, import or call a function;
  * every size is limited (file, zip entries, pickle bytes, opcodes, stack,
    memo, strings, tensor elements) and anything unexpected is an error;
  * tensors are copied out of the zip as raw little-endian float16 and
    widened to float32 with an explicit bit-level rule (every float16 value is
    exactly representable in float32, so the conversion is lossless and the
    output is a pure function of the input bytes).

numpy is used only as a fast array of integers (table lookup and byte
output); the conversion rule itself is `half_bits_to_float_bits` below.

File formats (CONTRACT.md section 5):

  checkpoint   a zip, all members stored (not compressed):
                 <prefix>/data.pkl        pickle protocol 2, the object graph
                 <prefix>/version         b"3\\n"
                 <prefix>/data/<n>        raw storage n: little-endian float16
  tensor table JSON: {"upstream": {"sha256", "sizeBytes"}, "weights": {"sha256", "sizeBytes"},
                      "tensors": [{"key", "shape", "sourceDtype": "float16", "offset", "byteLength"}, ...]}
  weights file the tensors of the table, in table order, each as little-endian
               float32 in C (row-major) order at its `offset`; no header, no padding.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Limits. The real file: 55 MB, 527 zip members, 70 kB pickle, 17488 opcodes,
# longest string 71 bytes, largest tensor 2.65 M elements, 27.4 M in total.
# ----------------------------------------------------------------------------

MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_ZIP_ENTRIES = 4096
MAX_PICKLE_BYTES = 1024 * 1024
MAX_OPCODES = 200_000
MAX_STACK_DEPTH = 4096
MAX_MEMO_ENTRIES = 65_536
MAX_STRING_BYTES = 1024
MAX_TENSOR_DIMS = 8
MAX_TENSOR_ELEMENTS = 16 * 1024 * 1024
MAX_TOTAL_ELEMENTS = 64 * 1024 * 1024

# The only names the pickle may mention. They are never resolved to anything.
ALLOWED_GLOBALS = frozenset({
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch", "HalfStorage"),
    ("demucs.htdemucs", "HTDemucs"),
    ("fractions", "Fraction"),
})


class ConversionError(Exception):
    """The input is not exactly what this converter expects. Never caught and ignored."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConversionError(message)


# ----------------------------------------------------------------------------
# Inert values produced by the symbolic interpreter
# ----------------------------------------------------------------------------

class Global:
    """A name from a GLOBAL opcode. Just two strings."""

    def __init__(self, module: str, name: str):
        self.module, self.name = module, name

    def is_(self, module: str, name: str) -> bool:
        return self.module == module and self.name == name

    def __repr__(self) -> str:
        return f"<global {self.module}.{self.name}>"


class Call:
    """A REDUCE that was recorded instead of executed: `func` applied to `args`."""

    def __init__(self, func: Global, args: tuple):
        self.func, self.args = func, args

    def __repr__(self) -> str:
        return f"<call {self.func.module}.{self.func.name}{self.args!r}>"


class Persistent:
    """A BINPERSID: torch's reference to a storage held in another zip member."""

    def __init__(self, value: tuple):
        self.value = value


class Mark:
    """Stack marker pushed by the MARK opcode."""


MARK = Mark()


# ----------------------------------------------------------------------------
# Symbolic pickle interpreter (protocol 2 subset)
# ----------------------------------------------------------------------------

def symbolic_unpickle(data: bytes, allowed_globals: frozenset = ALLOWED_GLOBALS, allow_build: bool = False):
    """Interpret a pickle without resolving or calling anything. Returns the root value.

    Values are None, bool, int, float, str, tuple, list, dict, Global, Call, Persistent.
    `allow_build` exists only so that the export tooling can read Meta's other
    checkpoints for comparison; the shipped conversion never sets it.
    """
    require(len(data) <= MAX_PICKLE_BYTES, "pickle too large")
    stack: list = []
    memo: dict[int, object] = {}
    position = 0
    opcodes = 0

    def take(count: int) -> bytes:
        nonlocal position
        require(count >= 0 and position + count <= len(data), "pickle truncated")
        chunk = data[position: position + count]
        position += count
        return chunk

    def take_line() -> str:
        nonlocal position
        end = data.find(b"\n", position)
        require(end >= 0 and end - position <= MAX_STRING_BYTES, "bad text line in pickle")
        line = data[position:end]
        position = end + 1
        return line.decode("ascii")

    def push(value) -> None:
        require(len(stack) < MAX_STACK_DEPTH, "pickle stack too deep")
        stack.append(value)

    def pop():
        require(len(stack) > 0, "pickle stack underflow")
        value = stack.pop()
        require(value is not MARK, "unexpected MARK on the pickle stack")
        return value

    def pop_to_mark() -> list:
        items = []
        while True:
            require(len(stack) > 0, "MARK not found")
            value = stack.pop()
            if value is MARK:
                break
            items.append(value)
        items.reverse()
        return items

    def remember(index: int) -> None:
        require(len(stack) > 0 and stack[-1] is not MARK, "nothing to memoize")
        require(index in memo or len(memo) < MAX_MEMO_ENTRIES, "pickle memo too large")
        memo[index] = stack[-1]

    def recall(index: int) -> None:
        require(index in memo, "pickle reads a memo slot that was never written")
        # In this file only immutable values are ever read back from the memo
        # (strings, global names, tuples). Refusing lists and dicts means a
        # port needs no shared mutable state.
        require(not isinstance(memo[index], (list, dict)), "pickle shares a mutable container through the memo")
        push(memo[index])

    while True:
        opcodes += 1
        require(opcodes <= MAX_OPCODES, "too many pickle opcodes")
        opcode = take(1)[0]

        if opcode == 0x80:                                  # PROTO
            require(take(1)[0] == 2, "only pickle protocol 2 is accepted")
        elif opcode == 0x2E:                                # STOP '.'
            require(len(stack) == 1 and stack[0] is not MARK, "pickle did not end with exactly one value")
            require(position == len(data), "trailing bytes after the pickle")
            return stack[0]
        elif opcode == 0x28:                                # MARK '('
            push(MARK)
        elif opcode == 0x4E:                                # NONE 'N'
            push(None)
        elif opcode == 0x88:                                # NEWTRUE
            push(True)
        elif opcode == 0x89:                                # NEWFALSE
            push(False)
        elif opcode == 0x4B:                                # BININT1 'K': unsigned byte
            push(take(1)[0])
        elif opcode == 0x4D:                                # BININT2 'M': unsigned 16-bit little-endian
            push(struct.unpack("<H", take(2))[0])
        elif opcode == 0x4A:                                # BININT 'J': signed 32-bit little-endian
            push(struct.unpack("<i", take(4))[0])
        elif opcode == 0x47:                                # BINFLOAT 'G': 64-bit big-endian
            push(struct.unpack(">d", take(8))[0])
        elif opcode == 0x58:                                # BINUNICODE 'X': 32-bit length, UTF-8
            length = struct.unpack("<I", take(4))[0]
            require(length <= MAX_STRING_BYTES, "string too long")
            push(take(length).decode("utf-8"))
        elif opcode == 0x7D:                                # EMPTY_DICT '}'
            push({})
        elif opcode == 0x5D:                                # EMPTY_LIST ']'
            push([])
        elif opcode == 0x29:                                # EMPTY_TUPLE ')'
            push(())
        elif opcode == 0x74:                                # TUPLE 't'
            push(tuple(pop_to_mark()))
        elif opcode == 0x85:                                # TUPLE1
            push((pop(),))
        elif opcode == 0x86:                                # TUPLE2
            second = pop()
            push((pop(), second))
        elif opcode == 0x87:                                # TUPLE3
            third = pop()
            second = pop()
            push((pop(), second, third))
        elif opcode == 0x61:                                # APPEND 'a'
            value = pop()
            require(len(stack) > 0 and isinstance(stack[-1], list), "APPEND target is not a list")
            stack[-1].append(value)
        elif opcode == 0x65:                                # APPENDS 'e'
            items = pop_to_mark()
            require(len(stack) > 0 and isinstance(stack[-1], list), "APPENDS target is not a list")
            stack[-1].extend(items)
        elif opcode == 0x75:                                # SETITEMS 'u'
            items = pop_to_mark()
            require(len(items) % 2 == 0, "SETITEMS with an odd number of values")
            require(len(stack) > 0 and isinstance(stack[-1], dict), "SETITEMS target is not a dict")
            for index in range(0, len(items), 2):
                key = items[index]
                require(isinstance(key, (str, int)) and not isinstance(key, bool), "dict key is not a string or an integer")
                stack[-1][key] = items[index + 1]
        elif opcode == 0x71:                                # BINPUT 'q'
            remember(take(1)[0])
        elif opcode == 0x72:                                # LONG_BINPUT 'r'
            remember(struct.unpack("<I", take(4))[0])
        elif opcode == 0x68:                                # BINGET 'h'
            recall(take(1)[0])
        elif opcode == 0x6A:                                # LONG_BINGET 'j'
            recall(struct.unpack("<I", take(4))[0])
        elif opcode == 0x63:                                # GLOBAL 'c': "module\nname\n"
            module = take_line()
            name = take_line()
            require((module, name) in allowed_globals, f"pickle references a name that is not allowed: {module}.{name}")
            push(Global(module, name))
        elif opcode == 0x52:                                # REDUCE 'R': recorded, never executed
            args = pop()
            func = pop()
            require(isinstance(args, tuple) and isinstance(func, Global), "REDUCE on something that is not a global")
            if func.is_("collections", "OrderedDict") and args == ():
                push({})                                    # an empty ordered mapping; dicts keep insertion order
            else:
                push(Call(func, args))
        elif opcode == 0x51:                                # BINPERSID 'Q'
            value = pop()
            require(isinstance(value, tuple), "persistent id is not a tuple")
            push(Persistent(value))
        elif opcode == 0x62 and allow_build:                # BUILD 'b' (comparison checkpoints only): recorded
            state = pop()
            target = pop()
            require(isinstance(target, Call), "BUILD on something that is not a recorded call")
            push(Call(Global("<build>", "<build>"), (target, state)))
        else:
            raise ConversionError(f"pickle opcode 0x{opcode:02x} is not allowed")


# ----------------------------------------------------------------------------
# float16 -> float32
# ----------------------------------------------------------------------------

def half_bits_to_float_bits(half: int) -> int:
    """IEEE 754 binary16 bit pattern -> binary32 bit pattern of the same value. Finite inputs only."""
    sign = (half >> 15) & 0x1
    exponent = (half >> 10) & 0x1F
    fraction = half & 0x3FF
    require(exponent != 0x1F, "infinity or NaN in the weights")
    if exponent == 0:
        if fraction == 0:
            return sign << 31                               # signed zero
        # subnormal half: value = fraction * 2^-24. Normalise it.
        shift = 0
        while (fraction & 0x400) == 0:
            fraction <<= 1
            shift += 1
        fraction &= 0x3FF
        exponent = 1 - shift
    return (sign << 31) | ((exponent - 15 + 127) << 23) | (fraction << 13)


def build_half_table() -> np.ndarray:
    """All 65536 conversions as a lookup table of uint32. Infinity / NaN slots hold a sentinel."""
    table = np.zeros(1 << 16, dtype=np.uint32)
    for half in range(1 << 16):
        if (half >> 10) & 0x1F == 0x1F:
            table[half] = 0xFFFFFFFF                        # never produced: checked before lookup
        else:
            table[half] = half_bits_to_float_bits(half)
    return table


_HALF_TABLE: np.ndarray | None = None


def halves_to_float32_bytes(raw: bytes) -> bytes:
    """Little-endian float16 bytes -> little-endian float32 bytes."""
    global _HALF_TABLE
    if _HALF_TABLE is None:
        _HALF_TABLE = build_half_table()
    require(len(raw) % 2 == 0, "odd number of bytes in a float16 tensor")
    halves = np.frombuffer(raw, dtype="<u2")
    require(not bool(np.any((halves & 0x7C00) == 0x7C00)), "infinity or NaN in the weights")
    return _HALF_TABLE[halves].astype("<u4").tobytes()


# ----------------------------------------------------------------------------
# Reading the checkpoint
# ----------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Tensor:
    """One state-dict entry: where its float16 data lives in the zip."""

    def __init__(self, key: str, shape: tuple[int, ...], member: str, offset_elements: int):
        self.key, self.shape, self.member, self.offset_elements = key, shape, member, offset_elements
        self.elements = 1
        for size in shape:
            self.elements *= size


def open_checkpoint(path: Path, allowed_globals: frozenset = ALLOWED_GLOBALS, allow_build: bool = False):
    """Validate the container and interpret its pickle. Returns (zip archive, prefix, root dict)."""
    require(path.stat().st_size <= MAX_FILE_BYTES, "checkpoint file too large")
    archive = zipfile.ZipFile(path)
    infos = archive.infolist()
    require(0 < len(infos) <= MAX_ZIP_ENTRIES, "unexpected number of zip members")
    pickles = [info.filename for info in infos if info.filename.endswith("/data.pkl")]
    require(len(pickles) == 1, "expected exactly one data.pkl")
    prefix = pickles[0][: -len("data.pkl")]
    require(prefix.count("/") == 1 and ".." not in prefix and not prefix.startswith("/"), "unexpected archive layout")
    for info in infos:
        name = info.filename
        require(info.compress_type == zipfile.ZIP_STORED, f"zip member {name!r} is compressed")
        require(info.flag_bits & 0x1 == 0, f"zip member {name!r} is encrypted")
        require(name.startswith(prefix), f"unexpected zip member {name!r}")
        rest = name[len(prefix):]
        is_storage = rest.startswith("data/") and rest[5:].isdigit() and len(rest) <= 5 + 9
        require(rest in ("data.pkl", "version") or is_storage, f"unexpected zip member {name!r}")
    require(archive.read(prefix + "version").strip() == b"3", "unexpected checkpoint format version")
    pickle_info = archive.getinfo(prefix + "data.pkl")
    require(pickle_info.file_size <= MAX_PICKLE_BYTES, "pickle too large")
    root = symbolic_unpickle(archive.read(pickle_info), allowed_globals, allow_build)
    require(isinstance(root, dict), "checkpoint root is not a dict")
    return archive, prefix, root


def describe_tensor(key: str, value, prefix: str, archive: zipfile.ZipFile, storage_type: str = "HalfStorage") -> Tensor:
    """Check one recorded `_rebuild_tensor_v2(...)` and locate its data."""
    require(isinstance(key, str), "state-dict key is not a string")
    require(isinstance(value, Call) and value.func.is_("torch._utils", "_rebuild_tensor_v2"), f"{key}: not a tensor")
    require(len(value.args) == 6, f"{key}: unexpected tensor record")
    storage, offset, shape, stride, _requires_grad, _hooks = value.args
    require(isinstance(storage, Persistent) and len(storage.value) == 5, f"{key}: bad storage reference")
    tag, storage_class, storage_key, location, storage_elements = storage.value
    require(tag == "storage" and isinstance(storage_class, Global) and storage_class.is_("torch", storage_type),
            f"{key}: storage is not {storage_type}")
    require(isinstance(storage_key, str) and storage_key.isdigit() and len(storage_key) <= 9, f"{key}: bad storage key")
    require(location == "cpu", f"{key}: storage is not on cpu")
    require(isinstance(offset, int) and not isinstance(offset, bool) and offset >= 0, f"{key}: bad offset")
    require(isinstance(shape, tuple) and len(shape) <= MAX_TENSOR_DIMS, f"{key}: bad shape")
    require(isinstance(stride, tuple) and len(stride) == len(shape), f"{key}: bad stride")
    tensor = Tensor(key, shape, prefix + "data/" + storage_key, offset)
    expected_stride = 1
    for size, step in zip(reversed(shape), reversed(stride)):
        require(isinstance(size, int) and not isinstance(size, bool) and 0 < size <= MAX_TENSOR_ELEMENTS, f"{key}: bad dimension")
        require(step == expected_stride, f"{key}: tensor is not stored contiguously")
        expected_stride *= size
    require(tensor.elements <= MAX_TENSOR_ELEMENTS, f"{key}: tensor too large")
    require(isinstance(storage_elements, int) and offset + tensor.elements <= storage_elements, f"{key}: tensor exceeds its storage")
    bytes_per_element = 2 if storage_type == "HalfStorage" else 4
    require(archive.getinfo(tensor.member).file_size == storage_elements * bytes_per_element, f"{key}: storage size mismatch")
    return tensor


def read_checkpoint(path: Path) -> tuple[zipfile.ZipFile, dict, list[Tensor]]:
    """Open the htdemucs_6s checkpoint. Returns (archive, root dict, tensors in state-dict order)."""
    archive, prefix, root = open_checkpoint(path)
    require({"klass", "args", "kwargs", "state"} <= set(root) <= {"klass", "args", "kwargs", "state", "training_args", "metrics"},
            "unexpected checkpoint keys")           # training_args and metrics are parsed and ignored
    require(isinstance(root["klass"], Global) and root["klass"].is_("demucs.htdemucs", "HTDemucs"), "not an HTDemucs checkpoint")
    require(root["args"] == () and isinstance(root["kwargs"], dict), "unexpected constructor arguments")
    state = root["state"]
    require(isinstance(state, dict) and len(state) > 0, "state dict missing")
    tensors, total = [], 0
    for key, value in state.items():
        tensor = describe_tensor(key, value, prefix, archive)
        total += tensor.elements
        require(total <= MAX_TOTAL_ELEMENTS, "too many weights")
        tensors.append(tensor)
    return archive, root, tensors


def tensor_float16_bytes(archive: zipfile.ZipFile, tensor: Tensor) -> bytes:
    """The tensor's data exactly as stored: little-endian float16, row-major."""
    storage = archive.read(tensor.member)
    start = tensor.offset_elements * 2
    data = storage[start: start + tensor.elements * 2]
    require(len(data) == tensor.elements * 2, f"{tensor.key}: storage truncated")
    return data


# ----------------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------------

def convert(checkpoint_path: Path, table_path: Path, out_path: Path) -> dict:
    """Write the weights file described by the tensor table. Returns {"sha256", "sizeBytes"}."""
    table = json.loads(table_path.read_text())
    upstream = table["upstream"]
    require(checkpoint_path.stat().st_size == upstream["sizeBytes"], "checkpoint size differs from the pinned size")
    require(sha256_file(checkpoint_path) == upstream["sha256"], "checkpoint SHA-256 differs from the pinned value")

    archive, _root, tensors = read_checkpoint(checkpoint_path)
    by_key = {tensor.key: tensor for tensor in tensors}
    digest = hashlib.sha256()
    written = 0
    partial = out_path.with_name(out_path.name + ".part")
    with open(partial, "wb") as out:
        for row in table["tensors"]:
            require(row["key"] in by_key, f"checkpoint has no tensor {row['key']!r}")
            tensor = by_key[row["key"]]
            require(list(tensor.shape) == row["shape"], f"{row['key']}: shape differs from the table")
            require(row["sourceDtype"] == "float16" and row["offset"] == written, f"{row['key']}: table is inconsistent")
            data = halves_to_float32_bytes(tensor_float16_bytes(archive, tensor))
            require(len(data) == row["byteLength"], f"{row['key']}: length differs from the table")
            out.write(data)
            digest.update(data)
            written += len(data)
    result = {"sha256": digest.hexdigest(), "sizeBytes": written}
    if result != {"sha256": table["weights"]["sha256"], "sizeBytes": table["weights"]["sizeBytes"]}:
        partial.unlink()
        raise ConversionError(f"converted weights {result} differ from the pinned {table['weights']}")
    partial.replace(out_path)
    return result


# ----------------------------------------------------------------------------
# Self-test and CLI
# ----------------------------------------------------------------------------

def selftest() -> None:
    """The bit-level conversion rule against numpy's float16 -> float32 cast, for every finite half."""
    table = build_half_table()
    halves = np.arange(1 << 16, dtype=np.uint16)
    finite = (halves & 0x7C00) != 0x7C00
    with np.errstate(invalid="ignore"):                    # the NaN patterns are masked out below
        expected = halves.view(np.float16).astype(np.float32).view(np.uint32)
    assert np.array_equal(table[finite], expected[finite]), "half -> float table disagrees with IEEE conversion"
    assert halves_to_float32_bytes(struct.pack("<4e", 1.0, -2.5, 0.0, 6e-8)) == struct.pack("<4f", 1.0, -2.5, 0.0, np.float32(np.float16(6e-8)))
    for bad in (b"\x80\x02cos\nsystem\n.", b"\x80\x02c__builtin__\neval\n.", b"\x80\x04N.", b"\x80\x02N", b"\x80\x02\x8c\x01a."):
        try:
            symbolic_unpickle(bad)
        except ConversionError:
            continue
        raise AssertionError(f"hostile pickle was accepted: {bad!r}")
    for bad_half in (0x7C00, 0xFC00, 0x7E00):
        try:
            halves_to_float32_bytes(struct.pack("<H", bad_half))
        except ConversionError:
            continue
        raise AssertionError("infinity / NaN was accepted")
    print(f"selftest ok: {int(finite.sum())} finite float16 values convert exactly; hostile pickles and non-finite weights are rejected")


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "selftest":
        selftest()
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "inspect":
        archive, root, tensors = read_checkpoint(Path(sys.argv[2]))
        print(f"class {root['klass']}  constructor kwargs: {len(root['kwargs'])}  tensors: {len(tensors)}  "
              f"elements: {sum(t.elements for t in tensors)}")
        for name in ("sources", "segment", "samplerate", "nfft", "channels", "depth", "bottom_channels"):
            print(f"  kwargs[{name!r}] = {root['kwargs'].get(name)!r}")
        for tensor in tensors[:3] + tensors[-2:]:
            print(f"  {tensor.key}  {tensor.shape}  {tensor.member} + {tensor.offset_elements}")
        return 0
    if len(sys.argv) == 5 and sys.argv[1] == "convert":
        result = convert(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
        print(f"{sys.argv[4]}: {result['sizeBytes']} bytes  sha256={result['sha256']}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
