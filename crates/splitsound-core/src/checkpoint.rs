//! Safe reader and converter for the upstream `htdemucs_6s` checkpoint
//! (`ml/CONTRACT.md` section 5; a port of `ml/convert_reference.py`).
//!
//! The restricted Demucs model is not redistributed. The app downloads
//! Meta's original PyTorch checkpoint and this module turns it into the
//! plain float32 weights file that the weight-free ONNX graph loads.
//!
//! A PyTorch checkpoint is a zip holding a Python pickle, and a pickle is a
//! program. **Nothing here executes it.** This is the most
//! security-sensitive code in the engine, so it is deliberately small,
//! bounded and strict:
//!
//! * The pickle is read by a *symbolic* interpreter that knows only the 28
//!   opcodes this one file uses. A `GLOBAL` is checked against an
//!   allow-list of five names and becomes an inert token (an index);
//!   `REDUCE` never calls anything, it records "token applied to
//!   arguments". There is no code path from the file's contents to a
//!   function call, an import, a path or an allocation sized by the file
//!   without a bound.
//! * Every size is limited: file, zip member count, pickle bytes, opcode
//!   count, stack depth, memo entries, string length, tensor rank and
//!   elements. Anything unexpected is an error; nothing is skipped.
//! * Zip members must be stored (uncompressed), unencrypted, uniquely
//!   named and match the exact layout `<prefix>/data.pkl`,
//!   `<prefix>/version`, `<prefix>/data/<n>`. Member names are never used
//!   as file-system paths. Sizes are checked against the tensor table
//!   before anything is allocated.
//! * float16 is widened to float32 with an explicit bit-level rule (every
//!   finite float16 is exactly representable); infinities and NaNs are
//!   rejected.
//! * The output is written to a temporary file, must match the pinned size
//!   and SHA-256, and only then is renamed into place.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, Write};
use std::path::Path;
use std::rc::Rc;
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

use crate::error::{Error, ErrorCode, Result};
use crate::progress::{Progress, check_cancelled};

/// Largest checkpoint file accepted. The real one is 55 MB.
pub const MAX_FILE_BYTES: u64 = 128 * 1024 * 1024;
/// Largest number of zip members. The real file has 527.
pub const MAX_ZIP_ENTRIES: usize = 4096;
/// Largest pickle. The real one is 70 kB.
pub const MAX_PICKLE_BYTES: usize = 1024 * 1024;
/// Most opcodes interpreted. The real pickle has 17488.
pub const MAX_OPCODES: usize = 200_000;
/// Deepest interpreter stack.
pub const MAX_STACK_DEPTH: usize = 4096;
/// Deepest constructed value, including values shared through the memo.
const MAX_NESTING_DEPTH: u8 = 64;
/// Most memo slots.
pub const MAX_MEMO_ENTRIES: usize = 65_536;
/// Longest string or name line in the pickle. The real maximum is 71.
pub const MAX_STRING_BYTES: usize = 1024;
/// Highest tensor rank.
pub const MAX_TENSOR_DIMS: usize = 8;
/// Most elements in one tensor. The real maximum is 2.65 M.
pub const MAX_TENSOR_ELEMENTS: u64 = 16 * 1024 * 1024;
/// Most elements in all tensors together. The real total is 27.4 M.
pub const MAX_TOTAL_ELEMENTS: u64 = 64 * 1024 * 1024;

/// The only names the pickle may mention. They are never resolved to
/// anything; a `GLOBAL` opcode yields the index into this list.
const ALLOWED_GLOBALS: [(&str, &str); 5] = [
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch", "HalfStorage"),
    ("demucs.htdemucs", "HTDemucs"),
    ("fractions", "Fraction"),
];
const G_ORDERED_DICT: usize = 0;
const G_REBUILD_TENSOR: usize = 1;
const G_HALF_STORAGE: usize = 2;
const G_HTDEMUCS: usize = 3;

fn bad(detail: impl std::fmt::Display) -> Error {
    Error::new(
        ErrorCode::ModelDownloadFailed,
        "The downloaded model file is not what was expected, so it was not used",
    )
    .with_detail(format!("checkpoint: {detail}"))
}

fn require(condition: bool, detail: &str) -> Result<()> {
    if condition { Ok(()) } else { Err(bad(detail)) }
}

/// A dictionary key: this file only uses strings and small integers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    Str(Rc<str>),
    Int(i64),
}

/// An insertion-ordered dictionary.
#[derive(Debug, Clone, Default, PartialEq)]
struct Dict {
    entries: Vec<(Key, Value)>,
    index: HashMap<Key, usize>,
}

impl Dict {
    fn insert(&mut self, key: Key, value: Value) {
        match self.index.get(&key) {
            Some(&i) => self.entries[i].1 = value,
            None => {
                self.index.insert(key.clone(), self.entries.len());
                self.entries.push((key, value));
            }
        }
    }

    fn get(&self, key: &str) -> Option<&Value> {
        self.index
            .get(&Key::Str(Rc::from(key)))
            .map(|&i| &self.entries[i].1)
    }
}

/// An inert value produced by the symbolic interpreter. Cloning is cheap
/// (immutable parts are reference counted) and never shares mutable state.
#[derive(Debug, Clone, PartialEq)]
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Rc<str>),
    Tuple(Rc<[Value]>),
    List(Vec<Value>),
    Dict(Dict),
    /// Index into [`ALLOWED_GLOBALS`].
    Global(usize),
    /// A `REDUCE` that was recorded instead of executed.
    Call(usize, Rc<[Value]>),
    /// A `BINPERSID`: torch's reference to a storage in another zip member.
    Persistent(Rc<[Value]>),
}

/// Cache the depth of the immutable children, or an upper bound for a
/// mutable container. Even error cleanup can only drop a bounded tree.
#[derive(Clone)]
struct BoundedValue {
    value: Value,
    depth: u8,
}

fn parent_depth(child_depth: u8) -> Result<u8> {
    require(
        child_depth < MAX_NESTING_DEPTH,
        "pickle value nesting too deep",
    )?;
    Ok(child_depth + 1)
}

enum Slot {
    Value(BoundedValue),
    Mark,
}

enum Memo {
    Value(BoundedValue),
    /// A list or dict was memoised. Reading it back would need shared
    /// mutable state; the real file never does, so it is refused.
    Mutable,
}

struct Interpreter<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Slot>,
    memo: HashMap<u32, Memo>,
}

impl<'a> Interpreter<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(count)
            .filter(|e| *e <= self.data.len());
        let Some(end) = end else {
            return Err(bad("pickle truncated"));
        };
        let chunk = &self.data[self.pos..end];
        self.pos = end;
        Ok(chunk)
    }

    fn take_u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn take_u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// A text line terminated by `\n`, ASCII only, at most
    /// [`MAX_STRING_BYTES`] long.
    fn take_line(&mut self) -> Result<&'a str> {
        let rest = &self.data[self.pos..];
        let window = &rest[..rest.len().min(MAX_STRING_BYTES + 1)];
        let Some(end) = window.iter().position(|b| *b == b'\n') else {
            return Err(bad("bad text line in pickle"));
        };
        let line = &rest[..end];
        require(line.is_ascii(), "bad text line in pickle")?;
        self.pos += end + 1;
        std::str::from_utf8(line).map_err(|_| bad("bad text line in pickle"))
    }

    fn push(&mut self, value: Value) -> Result<()> {
        let depth = u8::from(matches!(
            value,
            Value::Tuple(_) | Value::List(_) | Value::Dict(_)
        ));
        self.push_bounded(BoundedValue { value, depth })
    }

    fn push_tuple(&mut self, items: Vec<BoundedValue>) -> Result<()> {
        let depth = parent_depth(items.iter().map(|v| v.depth).max().unwrap_or(0))?;
        let value = Value::Tuple(items.into_iter().map(|v| v.value).collect());
        self.push_bounded(BoundedValue { value, depth })
    }

    fn push_bounded(&mut self, value: BoundedValue) -> Result<()> {
        require(self.stack.len() < MAX_STACK_DEPTH, "pickle stack too deep")?;
        self.stack.push(Slot::Value(value));
        Ok(())
    }

    fn pop(&mut self) -> Result<BoundedValue> {
        match self.stack.pop() {
            Some(Slot::Value(v)) => Ok(v),
            Some(Slot::Mark) => Err(bad("unexpected MARK on the pickle stack")),
            None => Err(bad("pickle stack underflow")),
        }
    }

    fn pop_to_mark(&mut self) -> Result<Vec<BoundedValue>> {
        let mut items = Vec::new();
        loop {
            match self.stack.pop() {
                Some(Slot::Mark) => break,
                Some(Slot::Value(v)) => items.push(v),
                None => return Err(bad("MARK not found")),
            }
        }
        items.reverse();
        Ok(items)
    }

    fn remember(&mut self, index: u32) -> Result<()> {
        let entry = match self.stack.last() {
            Some(Slot::Value(BoundedValue {
                value: Value::List(_) | Value::Dict(_),
                ..
            })) => Memo::Mutable,
            Some(Slot::Value(v)) => Memo::Value(v.clone()),
            _ => return Err(bad("nothing to memoize")),
        };
        require(
            self.memo.contains_key(&index) || self.memo.len() < MAX_MEMO_ENTRIES,
            "pickle memo too large",
        )?;
        self.memo.insert(index, entry);
        Ok(())
    }

    fn recall(&mut self, index: u32) -> Result<()> {
        let value = match self.memo.get(&index) {
            None => return Err(bad("pickle reads a memo slot that was never written")),
            Some(Memo::Mutable) => {
                return Err(bad("pickle shares a mutable container through the memo"));
            }
            Some(Memo::Value(v)) => v.clone(),
        };
        self.push_bounded(value)
    }

    fn run(mut self) -> Result<Value> {
        let mut opcodes = 0usize;
        loop {
            opcodes += 1;
            require(opcodes <= MAX_OPCODES, "too many pickle opcodes")?;
            let opcode = self.take_u8()?;
            match opcode {
                // PROTO
                0x80 => require(self.take_u8()? == 2, "only pickle protocol 2 is accepted")?,
                // STOP
                0x2E => {
                    require(
                        self.stack.len() == 1 && self.pos == self.data.len(),
                        "pickle did not end with exactly one value and no trailing bytes",
                    )?;
                    return self.pop().map(|v| v.value);
                }
                // MARK
                0x28 => {
                    require(self.stack.len() < MAX_STACK_DEPTH, "pickle stack too deep")?;
                    self.stack.push(Slot::Mark);
                }
                // NONE, NEWTRUE, NEWFALSE
                0x4E => self.push(Value::None)?,
                0x88 => self.push(Value::Bool(true))?,
                0x89 => self.push(Value::Bool(false))?,
                // BININT1: unsigned byte
                0x4B => {
                    let v = self.take_u8()?;
                    self.push(Value::Int(i64::from(v)))?;
                }
                // BININT2: unsigned 16-bit little-endian
                0x4D => {
                    let b = self.take(2)?;
                    self.push(Value::Int(i64::from(u16::from_le_bytes([b[0], b[1]]))))?;
                }
                // BININT: signed 32-bit little-endian
                0x4A => {
                    let v = self.take_u32()? as i32;
                    self.push(Value::Int(i64::from(v)))?;
                }
                // BINFLOAT: 64-bit big-endian
                0x47 => {
                    let b = self.take(8)?;
                    let v = f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
                    self.push(Value::Float(v))?;
                }
                // BINUNICODE: 32-bit length, UTF-8
                0x58 => {
                    let len = self.take_u32()? as usize;
                    require(len <= MAX_STRING_BYTES, "string too long")?;
                    let bytes = self.take(len)?;
                    let s = std::str::from_utf8(bytes).map_err(|_| bad("string is not UTF-8"))?;
                    self.push(Value::Str(Rc::from(s)))?;
                }
                // EMPTY_DICT, EMPTY_LIST, EMPTY_TUPLE
                0x7D => self.push(Value::Dict(Dict::default()))?,
                0x5D => self.push(Value::List(Vec::new()))?,
                0x29 => self.push(Value::Tuple(Rc::from(Vec::new())))?,
                // TUPLE, TUPLE1, TUPLE2, TUPLE3
                0x74 => {
                    let items = self.pop_to_mark()?;
                    self.push_tuple(items)?;
                }
                0x85 => {
                    let a = self.pop()?;
                    self.push_tuple(vec![a])?;
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push_tuple(vec![a, b])?;
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push_tuple(vec![a, b, c])?;
                }
                // APPEND
                0x61 => {
                    let value = self.pop()?;
                    match self.stack.last_mut() {
                        Some(Slot::Value(BoundedValue {
                            value: Value::List(list),
                            depth,
                        })) => {
                            *depth = (*depth).max(parent_depth(value.depth)?);
                            list.push(value.value);
                        }
                        _ => return Err(bad("APPEND target is not a list")),
                    }
                }
                // APPENDS
                0x65 => {
                    let items = self.pop_to_mark()?;
                    match self.stack.last_mut() {
                        Some(Slot::Value(BoundedValue {
                            value: Value::List(list),
                            depth,
                        })) => {
                            let child_depth = items.iter().map(|v| v.depth).max().unwrap_or(0);
                            *depth = (*depth).max(parent_depth(child_depth)?);
                            list.extend(items.into_iter().map(|v| v.value));
                        }
                        _ => return Err(bad("APPENDS target is not a list")),
                    }
                }
                // SETITEMS
                0x75 => {
                    let items = self.pop_to_mark()?;
                    require(
                        items.len() % 2 == 0,
                        "SETITEMS with an odd number of values",
                    )?;
                    let Some(Slot::Value(BoundedValue {
                        value: Value::Dict(dict),
                        depth,
                    })) = self.stack.last_mut()
                    else {
                        return Err(bad("SETITEMS target is not a dict"));
                    };
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        let key = match k.value {
                            Value::Str(s) => Key::Str(s),
                            Value::Int(i) => Key::Int(i),
                            _ => return Err(bad("dict key is not a string or an integer")),
                        };
                        *depth = (*depth).max(parent_depth(v.depth)?);
                        dict.insert(key, v.value);
                    }
                }
                // BINPUT, LONG_BINPUT
                0x71 => {
                    let i = self.take_u8()?;
                    self.remember(u32::from(i))?;
                }
                0x72 => {
                    let i = self.take_u32()?;
                    self.remember(i)?;
                }
                // BINGET, LONG_BINGET
                0x68 => {
                    let i = self.take_u8()?;
                    self.recall(u32::from(i))?;
                }
                0x6A => {
                    let i = self.take_u32()?;
                    self.recall(i)?;
                }
                // GLOBAL: "module\nname\n". Looked up in the allow-list
                // and nowhere else.
                0x63 => {
                    let module = self.take_line()?;
                    let name = self.take_line()?;
                    let Some(index) = ALLOWED_GLOBALS
                        .iter()
                        .position(|(m, n)| *m == module && *n == name)
                    else {
                        return Err(bad(format!(
                            "pickle references a name that is not allowed: {module}.{name}"
                        )));
                    };
                    self.push(Value::Global(index))?;
                }
                // REDUCE: recorded, never executed.
                0x52 => {
                    let args = self.pop()?;
                    let func = self.pop()?;
                    let depth = args.depth;
                    let (Value::Tuple(args), Value::Global(func)) = (args.value, func.value) else {
                        return Err(bad("REDUCE on something that is not a global"));
                    };
                    if func == G_ORDERED_DICT && args.is_empty() {
                        // An empty ordered mapping; `Dict` keeps insertion order.
                        self.push(Value::Dict(Dict::default()))?;
                    } else {
                        self.push_bounded(BoundedValue {
                            value: Value::Call(func, args),
                            depth,
                        })?;
                    }
                }
                // BINPERSID
                0x51 => {
                    let BoundedValue {
                        value: Value::Tuple(value),
                        depth,
                    } = self.pop()?
                    else {
                        return Err(bad("persistent id is not a tuple"));
                    };
                    self.push_bounded(BoundedValue {
                        value: Value::Persistent(value),
                        depth,
                    })?;
                }
                other => {
                    return Err(bad(format!("pickle opcode 0x{other:02x} is not allowed")));
                }
            }
        }
    }
}

/// Interprets a pickle symbolically. Never resolves or calls anything.
fn symbolic_unpickle(data: &[u8]) -> Result<Value> {
    require(data.len() <= MAX_PICKLE_BYTES, "pickle too large")?;
    Interpreter {
        data,
        pos: 0,
        stack: Vec::new(),
        memo: HashMap::new(),
    }
    .run()
}

/// IEEE 754 binary16 bit pattern to the binary32 bit pattern of the same
/// value. `None` for infinities and NaNs.
pub fn half_bits_to_float_bits(half: u16) -> Option<u32> {
    let sign = u32::from(half >> 15) & 1;
    let mut exponent = i32::from((half >> 10) & 0x1F);
    let mut fraction = u32::from(half & 0x3FF);
    if exponent == 0x1F {
        return None;
    }
    if exponent == 0 {
        if fraction == 0 {
            return Some(sign << 31);
        }
        // Subnormal half: value = fraction * 2^-24. Normalise it.
        let mut shift = 0;
        while fraction & 0x400 == 0 {
            fraction <<= 1;
            shift += 1;
        }
        fraction &= 0x3FF;
        exponent = 1 - shift;
    }
    Some((sign << 31) | (((exponent - 15 + 127) as u32) << 23) | (fraction << 13))
}

fn half_table() -> &'static [u32] {
    static TABLE: OnceLock<Vec<u32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=u16::MAX)
            // Infinity / NaN slots hold a sentinel that is never read: the
            // caller rejects those patterns before the lookup.
            .map(|h| half_bits_to_float_bits(h).unwrap_or(u32::MAX))
            .collect()
    })
}

/// Little-endian float16 bytes to little-endian float32 bytes, appended to
/// `out`. Errors on an odd length or any infinity / NaN.
fn halves_to_float32_bytes(raw: &[u8], out: &mut Vec<u8>) -> Result<()> {
    require(
        raw.len().is_multiple_of(2),
        "odd number of bytes in a float16 tensor",
    )?;
    let table = half_table();
    out.clear();
    out.reserve(raw.len() * 2);
    for pair in raw.as_chunks::<2>().0 {
        let half = u16::from_le_bytes([pair[0], pair[1]]);
        require(half & 0x7C00 != 0x7C00, "infinity or NaN in the weights")?;
        out.extend_from_slice(&table[usize::from(half)].to_le_bytes());
    }
    Ok(())
}

/// One state-dict entry: where its float16 data lives in the zip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLocation {
    /// State-dict key, e.g. `encoder.0.conv.weight`.
    pub key: String,
    /// Shape.
    pub shape: Vec<u64>,
    /// Zip member holding the storage.
    pub member: String,
    /// Offset into the storage, in elements.
    pub offset_elements: u64,
    /// Number of elements.
    pub elements: u64,
}

/// One row of the tensor table that ships with the graph.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TableRow {
    /// State-dict key in the checkpoint.
    pub key: String,
    /// Expected shape.
    pub shape: Vec<u64>,
    /// Storage type in the checkpoint; only `float16` is supported.
    pub source_dtype: String,
    /// Byte offset of this tensor in the weights file.
    pub offset: u64,
    /// Byte length of this tensor in the weights file.
    pub byte_length: u64,
}

/// A pinned file: its size and SHA-256 (lowercase hex).
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Pinned {
    /// SHA-256, lowercase hex.
    pub sha256: String,
    /// Size in bytes.
    pub size_bytes: u64,
}

/// The tensor table (`htdemucs_6s.tensors.json`).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TensorTable {
    /// The upstream checkpoint this table was made from.
    pub upstream: Pinned,
    /// The weights file the conversion must produce.
    pub weights: Pinned,
    /// The tensors, in file order.
    pub tensors: Vec<TableRow>,
}

impl TensorTable {
    /// Parses and sanity-checks a tensor table.
    pub fn from_json(json: &str) -> Result<Self> {
        let table: TensorTable = serde_json::from_str(json)
            .map_err(|e| Error::internal(format!("tensor table: {e}")))?;
        let mut offset = 0u64;
        for row in &table.tensors {
            let elements: u64 = row.shape.iter().product();
            if row.source_dtype != "float16"
                || row.offset != offset
                || row.byte_length != elements * 4
                || row.shape.len() > MAX_TENSOR_DIMS
            {
                return Err(Error::internal(format!(
                    "tensor table: bad row {}",
                    row.key
                )));
            }
            offset += row.byte_length;
        }
        if offset != table.weights.size_bytes {
            return Err(Error::internal("tensor table: sizes do not add up"));
        }
        Ok(table)
    }
}

/// An opened and validated checkpoint.
pub struct Checkpoint<R: Read + Seek> {
    archive: zip::ZipArchive<R>,
    file_len: u64,
    tensors: Vec<TensorLocation>,
}

fn zip_err(e: zip::result::ZipError) -> Error {
    bad(format!("zip: {e}"))
}

fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

impl<R: Read + Seek> Checkpoint<R> {
    /// Validates the container, interprets its pickle symbolically and
    /// locates every tensor. `file_len` is the length of `reader`.
    pub fn open(reader: R, file_len: u64) -> Result<Self> {
        require(file_len <= MAX_FILE_BYTES, "checkpoint file too large")?;
        let mut archive = zip::ZipArchive::new(reader).map_err(zip_err)?;
        let count = archive.len();
        require(
            count > 0 && count <= MAX_ZIP_ENTRIES,
            "unexpected number of zip members",
        )?;
        require(!archive.has_duplicate_names(), "duplicate zip member names")?;
        archive.validate_header_consistency().map_err(zip_err)?;

        // Collect names first; they are data, never paths.
        let mut names: Vec<String> = Vec::with_capacity(count);
        let mut sizes: HashMap<String, u64> = HashMap::with_capacity(count);
        for i in 0..count {
            let entry = archive.by_index_data(i).map_err(zip_err)?;
            let name = std::str::from_utf8(entry.name_raw())
                .map_err(|_| bad("zip member name is not UTF-8"))?
                .to_string();
            require(
                entry.compression() == zip::CompressionMethod::Stored,
                "zip member is compressed",
            )?;
            require(!entry.encrypted(), "zip member is encrypted")?;
            require(
                entry.compressed_size() == entry.size() && entry.size() <= file_len,
                "zip member size is inconsistent",
            )?;
            sizes.insert(name.clone(), entry.size());
            names.push(name);
        }
        let pickles: Vec<&String> = names.iter().filter(|n| n.ends_with("/data.pkl")).collect();
        require(pickles.len() == 1, "expected exactly one data.pkl")?;
        let prefix = pickles[0][..pickles[0].len() - "data.pkl".len()].to_string();
        require(
            prefix.matches('/').count() == 1
                && !prefix.contains("..")
                && !prefix.starts_with('/')
                && !prefix.contains('\\'),
            "unexpected archive layout",
        )?;
        for name in &names {
            let Some(rest) = name.strip_prefix(&prefix) else {
                return Err(bad("unexpected zip member"));
            };
            let is_storage = rest
                .strip_prefix("data/")
                .is_some_and(|n| is_digits(n) && n.len() <= 9);
            require(
                rest == "data.pkl" || rest == "version" || is_storage,
                "unexpected zip member",
            )?;
        }

        let version = read_member(&mut archive, &format!("{prefix}version"), 16)?;
        let version = std::str::from_utf8(&version)
            .unwrap_or("")
            .trim()
            .to_string();
        require(version == "3", "unexpected checkpoint format version")?;

        let pickle = read_member(&mut archive, &format!("{prefix}data.pkl"), MAX_PICKLE_BYTES)?;
        let Value::Dict(root) = symbolic_unpickle(&pickle)? else {
            return Err(bad("checkpoint root is not a dict"));
        };

        // The root must be exactly the HTDemucs package demucs writes.
        for required in ["klass", "args", "kwargs", "state"] {
            require(root.get(required).is_some(), "unexpected checkpoint keys")?;
        }
        for (key, _) in &root.entries {
            let ok = matches!(
                key,
                Key::Str(k) if matches!(
                    &**k,
                    "klass" | "args" | "kwargs" | "state" | "training_args" | "metrics"
                )
            );
            require(ok, "unexpected checkpoint keys")?;
        }
        require(
            root.get("klass") == Some(&Value::Global(G_HTDEMUCS)),
            "not an HTDemucs checkpoint",
        )?;
        require(
            matches!(root.get("args"), Some(Value::Tuple(t)) if t.is_empty())
                && matches!(root.get("kwargs"), Some(Value::Dict(_))),
            "unexpected constructor arguments",
        )?;
        let Some(Value::Dict(state)) = root.get("state") else {
            return Err(bad("state dict missing"));
        };
        require(!state.entries.is_empty(), "state dict missing")?;

        let mut tensors = Vec::with_capacity(state.entries.len());
        let mut total = 0u64;
        for (key, value) in &state.entries {
            let Key::Str(key) = key else {
                return Err(bad("state-dict key is not a string"));
            };
            let tensor = describe_tensor(key, value, &prefix, &sizes)?;
            total += tensor.elements;
            require(total <= MAX_TOTAL_ELEMENTS, "too many weights")?;
            tensors.push(tensor);
        }
        Ok(Checkpoint {
            archive,
            file_len,
            tensors,
        })
    }

    /// The tensors, in state-dict order.
    pub fn tensors(&self) -> &[TensorLocation] {
        &self.tensors
    }

    /// Reads one tensor's data exactly as stored: little-endian float16,
    /// row-major. `buf` is reused between calls.
    fn read_tensor_halves(&mut self, tensor: &TensorLocation, buf: &mut Vec<u8>) -> Result<()> {
        let bytes = tensor
            .elements
            .checked_mul(2)
            .filter(|b| *b <= self.file_len)
            .ok_or_else(|| bad("tensor larger than the file"))?;
        let skip = tensor.offset_elements * 2;
        let mut member = self.archive.by_name(&tensor.member).map_err(zip_err)?;
        // Skip to the tensor inside its storage without buffering the rest.
        let skipped = std::io::copy(&mut (&mut member).take(skip), &mut std::io::sink())
            .map_err(|e| bad(format!("zip: {e}")))?;
        require(skipped == skip, "storage truncated")?;
        buf.clear();
        buf.try_reserve_exact(bytes as usize)
            .map_err(|_| Error::out_of_memory())?;
        let got = (&mut member)
            .take(bytes)
            .read_to_end(buf)
            .map_err(|e| bad(format!("zip: {e}")))?;
        require(got as u64 == bytes, "storage truncated")
    }
}

/// Reads a whole zip member of at most `limit` bytes.
fn read_member<R: Read + Seek>(
    archive: &mut zip::ZipArchive<R>,
    name: &str,
    limit: usize,
) -> Result<Vec<u8>> {
    let member = archive.by_name(name).map_err(zip_err)?;
    require(member.size() <= limit as u64, "zip member too large")?;
    let mut data = Vec::with_capacity(member.size() as usize);
    // `take` bounds the read even if the declared size lied.
    member
        .take(limit as u64 + 1)
        .read_to_end(&mut data)
        .map_err(|e| bad(format!("zip: {e}")))?;
    require(data.len() <= limit, "zip member too large")?;
    Ok(data)
}

fn as_uint(value: &Value) -> Option<u64> {
    match value {
        Value::Int(i) if *i >= 0 => Some(*i as u64),
        _ => None,
    }
}

/// Checks one recorded `_rebuild_tensor_v2(...)` and locates its data.
fn describe_tensor(
    key: &str,
    value: &Value,
    prefix: &str,
    member_sizes: &HashMap<String, u64>,
) -> Result<TensorLocation> {
    let fail = |what: &str| bad(format!("{key}: {what}"));
    let Value::Call(G_REBUILD_TENSOR, args) = value else {
        return Err(fail("not a tensor"));
    };
    let [storage, offset, shape, stride, _requires_grad, _hooks] = &args[..] else {
        return Err(fail("unexpected tensor record"));
    };
    let Value::Persistent(storage) = storage else {
        return Err(fail("bad storage reference"));
    };
    let [tag, storage_class, storage_key, location, storage_elements] = &storage[..] else {
        return Err(fail("bad storage reference"));
    };
    if !matches!(tag, Value::Str(s) if &**s == "storage")
        || *storage_class != Value::Global(G_HALF_STORAGE)
    {
        return Err(fail("storage is not HalfStorage"));
    }
    let Value::Str(storage_key) = storage_key else {
        return Err(fail("bad storage key"));
    };
    if !is_digits(storage_key) || storage_key.len() > 9 {
        return Err(fail("bad storage key"));
    }
    if !matches!(location, Value::Str(s) if &**s == "cpu") {
        return Err(fail("storage is not on cpu"));
    }
    let offset = as_uint(offset).ok_or_else(|| fail("bad offset"))?;
    let (Value::Tuple(shape), Value::Tuple(stride)) = (shape, stride) else {
        return Err(fail("bad shape"));
    };
    if shape.len() > MAX_TENSOR_DIMS || stride.len() != shape.len() {
        return Err(fail("bad shape"));
    }
    let mut dims = Vec::with_capacity(shape.len());
    let mut elements = 1u64;
    for size in shape.iter() {
        let size = as_uint(size)
            .filter(|s| *s > 0 && *s <= MAX_TENSOR_ELEMENTS)
            .ok_or_else(|| fail("bad dimension"))?;
        elements = elements
            .checked_mul(size)
            .filter(|e| *e <= MAX_TENSOR_ELEMENTS)
            .ok_or_else(|| fail("tensor too large"))?;
        dims.push(size);
    }
    // Contiguous row-major storage: the stride of each dimension is the
    // product of the sizes after it.
    let mut expected = 1u64;
    for (size, step) in dims.iter().rev().zip(stride.iter().rev()) {
        if as_uint(step) != Some(expected) {
            return Err(fail("tensor is not stored contiguously"));
        }
        expected *= size;
    }
    let storage_elements = as_uint(storage_elements).ok_or_else(|| fail("bad storage size"))?;
    let end = offset
        .checked_add(elements)
        .ok_or_else(|| fail("bad offset"))?;
    if end > storage_elements {
        return Err(fail("tensor exceeds its storage"));
    }
    let member = format!("{prefix}data/{storage_key}");
    match member_sizes.get(&member) {
        Some(size) if storage_elements.checked_mul(2) == Some(*size) => {}
        _ => return Err(fail("storage size mismatch")),
    }
    Ok(TensorLocation {
        key: key.to_string(),
        shape: dims,
        member,
        offset_elements: offset,
        elements,
    })
}

/// SHA-256 of a file, lowercase hex, plus its length. Polls `progress` for
/// cancellation.
pub fn sha256_file(path: &Path, progress: &dyn Progress) -> Result<(String, u64)> {
    let meta = std::fs::metadata(path).map_err(|e| Error::from_read(&e, path))?;
    require(meta.is_file(), "checkpoint path is not a regular file")?;
    let mut file = File::open(path).map_err(|e| Error::from_read(&e, path))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        check_cancelled(progress)?;
        let n = file
            .read(&mut buf)
            .map_err(|e| Error::from_read(&e, path))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hex(&hasher.finalize()), total))
}

/// Lowercase hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(DIGITS[usize::from(b >> 4)] as char);
        s.push(DIGITS[usize::from(b & 15)] as char);
    }
    s
}

/// Reads `path` once on a single open handle, verifying it is a regular file
/// of at most `expected_size` bytes (bounded by [`MAX_FILE_BYTES`]), and
/// returns its bytes alongside its lowercase hex SHA-256 digest.
fn read_pinned_bytes(
    path: &Path,
    expected_size: u64,
    progress: &dyn Progress,
) -> Result<(Vec<u8>, String)> {
    require(expected_size <= MAX_FILE_BYTES, "checkpoint file too large")?;
    let path_meta = std::fs::metadata(path).map_err(|e| Error::from_read(&e, path))?;
    require(path_meta.is_file(), "checkpoint path is not a regular file")?;
    let mut file = File::open(path).map_err(|e| Error::from_read(&e, path))?;
    let meta = file.metadata().map_err(|e| Error::from_read(&e, path))?;
    require(meta.is_file(), "checkpoint path is not a regular file")?;
    require(
        meta.len() == expected_size,
        "checkpoint size differs from the pinned size",
    )?;
    let mut data = Vec::new();
    data.try_reserve_exact(expected_size as usize)
        .map_err(|_| Error::out_of_memory())?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        check_cancelled(progress)?;
        let n = match file.read(&mut buf) {
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(Error::from_read(&e, path)),
        };
        if n == 0 {
            break;
        }
        if (data.len() as u64).saturating_add(n as u64) > expected_size {
            return Err(bad("checkpoint size differs from the pinned size"));
        }
        hasher.update(&buf[..n]);
        data.extend_from_slice(&buf[..n]);
    }
    require(
        data.len() as u64 == expected_size,
        "checkpoint size differs from the pinned size",
    )?;
    Ok((data, hex(&hasher.finalize())))
}

/// Converts already-read checkpoint `bytes` into the weights file at
/// `out_path`, verifying both the pinned upstream digest and the pinned
/// Converts checkpoint `bytes` whose upstream size and SHA-256 have already
/// been checked against `table.upstream`.
pub(crate) fn convert_verified_checkpoint_bytes(
    bytes: &[u8],
    table: &TensorTable,
    out_path: &Path,
    progress: &dyn Progress,
) -> Result<()> {
    check_cancelled(progress)?;
    let mut checkpoint = Checkpoint::open(std::io::Cursor::new(bytes), bytes.len() as u64)?;
    let by_key: HashMap<String, TensorLocation> = checkpoint
        .tensors()
        .iter()
        .map(|t| (t.key.clone(), t.clone()))
        .collect();

    // Everything in the table is checked against the checkpoint before a
    // single byte is written.
    let mut plan = Vec::with_capacity(table.tensors.len());
    let mut expected_offset = 0u64;
    for row in &table.tensors {
        let Some(tensor) = by_key.get(&row.key) else {
            return Err(bad(format!("checkpoint has no tensor {:?}", row.key)));
        };
        require(
            tensor.shape == row.shape,
            "tensor shape differs from the table",
        )?;
        require(
            row.source_dtype == "float16" && row.offset == expected_offset,
            "tensor table is inconsistent",
        )?;
        require(
            tensor.elements * 4 == row.byte_length,
            "tensor length differs from the table",
        )?;
        expected_offset += row.byte_length;
        plan.push(tensor.clone());
    }
    require(
        expected_offset == table.weights.size_bytes,
        "tensor table is inconsistent",
    )?;

    let (tmp, out) =
        crate::wav::create_temp_sibling(out_path).map_err(|e| Error::from_write(&e, out_path))?;
    let mut out = BufWriter::with_capacity(1 << 20, out);
    let mut hasher = Sha256::new();
    let mut halves = Vec::new();
    let mut floats = Vec::new();
    let mut written = 0u64;
    for tensor in &plan {
        check_cancelled(progress)?;
        checkpoint.read_tensor_halves(tensor, &mut halves)?;
        halves_to_float32_bytes(&halves, &mut floats)?;
        out.write_all(&floats)
            .map_err(|e| Error::from_write(&e, out_path))?;
        hasher.update(&floats);
        written += floats.len() as u64;
    }
    let file = out
        .into_inner()
        .map_err(|e| Error::from_write(&e.into_error(), out_path))?;
    file.sync_all()
        .map_err(|e| Error::from_write(&e, out_path))?;
    drop(file);
    require(
        written == table.weights.size_bytes && hex(&hasher.finalize()) == table.weights.sha256,
        "converted weights differ from the pinned digest",
    )?;
    tmp.persist(out_path)
        .map_err(|e| Error::from_write(&e, out_path))
}

/// Converts the upstream checkpoint at `checkpoint_path` into the weights
/// file at `out_path`, as `table` describes.
///
/// The checkpoint is read once into memory and checked against the table's
/// pinned upstream size and SHA-256 before it is parsed, and the result
/// must match the pinned weights size and SHA-256 before it is renamed into
/// place; otherwise nothing is left at `out_path`.
///
/// # Errors
/// `model_download_failed` for any mismatch or malformed input,
/// `cancelled`, `disk_full`, `out_of_memory`.
pub fn convert_checkpoint(
    checkpoint_path: &Path,
    table: &TensorTable,
    out_path: &Path,
    progress: &dyn Progress,
) -> Result<()> {
    let (bytes, digest) = read_pinned_bytes(checkpoint_path, table.upstream.size_bytes, progress)?;
    require(
        digest == table.upstream.sha256,
        "checkpoint SHA-256 differs from the pinned value",
    )?;
    convert_verified_checkpoint_bytes(&bytes, table, out_path, progress)
}

#[cfg(test)]
pub(crate) mod testing {
    //! Builders for synthetic checkpoints: a raw zip writer (so tests can
    //! produce archives no well-behaved tool would) and a tiny pickle
    //! assembler.

    pub fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for b in data {
            crc ^= u32::from(*b);
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    /// One zip member as the test wants it to *look*.
    pub struct Member {
        pub name: Vec<u8>,
        pub data: Vec<u8>,
        /// Compression method field (0 = stored, 8 = deflate).
        pub method: u16,
        /// General purpose flags (bit 0 = encrypted).
        pub flags: u16,
        /// Declared uncompressed size; `None` = the real length.
        pub declared_size: Option<u32>,
    }

    impl Member {
        pub fn stored(name: &str, data: &[u8]) -> Self {
            Member {
                name: name.as_bytes().to_vec(),
                data: data.to_vec(),
                method: 0,
                flags: 0,
                declared_size: None,
            }
        }
    }

    pub fn build_zip(members: &[Member]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for m in members {
            let offset = out.len() as u32;
            let crc = crc32(&m.data);
            let comp = m.data.len() as u32;
            let size = m.declared_size.unwrap_or(comp);
            out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            out.extend_from_slice(&20u16.to_le_bytes());
            out.extend_from_slice(&m.flags.to_le_bytes());
            out.extend_from_slice(&m.method.to_le_bytes());
            out.extend_from_slice(&[0, 0, 0x21, 0]); // time, date
            out.extend_from_slice(&crc.to_le_bytes());
            out.extend_from_slice(&comp.to_le_bytes());
            out.extend_from_slice(&size.to_le_bytes());
            out.extend_from_slice(&(m.name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&m.name);
            out.extend_from_slice(&m.data);

            central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&m.flags.to_le_bytes());
            central.extend_from_slice(&m.method.to_le_bytes());
            central.extend_from_slice(&[0, 0, 0x21, 0]);
            central.extend_from_slice(&crc.to_le_bytes());
            central.extend_from_slice(&comp.to_le_bytes());
            central.extend_from_slice(&size.to_le_bytes());
            central.extend_from_slice(&(m.name.len() as u16).to_le_bytes());
            central.extend_from_slice(&[0u8; 12]); // extra, comment, disk, attrs
            central.extend_from_slice(&offset.to_le_bytes());
            central.extend_from_slice(&m.name);
        }
        let cd_offset = out.len() as u32;
        out.extend_from_slice(&central);
        out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        out.extend_from_slice(&[0u8; 4]);
        out.extend_from_slice(&(members.len() as u16).to_le_bytes());
        out.extend_from_slice(&(members.len() as u16).to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&cd_offset.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    /// Pickle assembler.
    #[derive(Default)]
    pub struct P(pub Vec<u8>);

    impl P {
        pub fn new() -> Self {
            P(vec![0x80, 2])
        }
        pub fn op(mut self, b: u8) -> Self {
            self.0.push(b);
            self
        }
        pub fn raw(mut self, b: &[u8]) -> Self {
            self.0.extend_from_slice(b);
            self
        }
        pub fn str(mut self, s: &str) -> Self {
            self.0.push(0x58);
            self.0.extend_from_slice(&(s.len() as u32).to_le_bytes());
            self.0.extend_from_slice(s.as_bytes());
            self
        }
        pub fn int(mut self, v: u32) -> Self {
            if v < 256 {
                self.0.extend_from_slice(&[0x4B, v as u8]);
            } else if v < 65_536 {
                self.0.push(0x4D);
                self.0.extend_from_slice(&(v as u16).to_le_bytes());
            } else {
                self.0.push(0x4A);
                self.0.extend_from_slice(&(v as i32).to_le_bytes());
            }
            self
        }
        pub fn global(mut self, module: &str, name: &str) -> Self {
            self.0.push(0x63);
            self.0.extend_from_slice(module.as_bytes());
            self.0.push(b'\n');
            self.0.extend_from_slice(name.as_bytes());
            self.0.push(b'\n');
            self
        }
        /// `(ints...)` as MARK ... TUPLE.
        pub fn int_tuple(mut self, values: &[u32]) -> Self {
            self.0.push(0x28);
            for v in values {
                self = self.int(*v);
            }
            self.op(0x74)
        }
        /// A `_rebuild_tensor_v2` record for `shape` at `offset` in storage
        /// `storage_key` of `storage_elements` elements.
        pub fn tensor(
            self,
            storage_class: (&str, &str),
            storage_key: &str,
            storage_elements: u32,
            offset: u32,
            shape: &[u32],
            stride: &[u32],
        ) -> Self {
            self.global("torch._utils", "_rebuild_tensor_v2")
                .op(0x28) // MARK (args)
                .op(0x28) // MARK (persistent id)
                .str("storage")
                .global(storage_class.0, storage_class.1)
                .str(storage_key)
                .str("cpu")
                .int(storage_elements)
                .op(0x74) // TUPLE
                .op(0x51) // BINPERSID
                .int(offset)
                .int_tuple(shape)
                .int_tuple(stride)
                .op(0x89) // False
                .global("collections", "OrderedDict")
                .op(0x29)
                .op(0x52) // OrderedDict()
                .op(0x74) // TUPLE
                .op(0x52) // REDUCE
        }
        pub fn done(self) -> Vec<u8> {
            self.op(0x2E).0
        }
    }

    /// The root pickle of a well-formed checkpoint whose state dict is
    /// `state` (already-assembled key/value pickles between MARK and
    /// SETITEMS).
    pub fn root_pickle(state_items: P) -> Vec<u8> {
        let mut p = P::new()
            .op(0x7D) // {
            .op(0x28) // MARK
            .str("klass")
            .global("demucs.htdemucs", "HTDemucs")
            .str("args")
            .op(0x29)
            .str("kwargs")
            .op(0x7D)
            .op(0x28)
            .str("segment")
            .global("fractions", "Fraction")
            .str("39/5")
            .op(0x85)
            .op(0x52)
            .str("depth")
            .int(4)
            .op(0x75)
            .str("state")
            .global("collections", "OrderedDict")
            .op(0x29)
            .op(0x52)
            .op(0x28);
        p.0.extend_from_slice(&state_items.0);
        p.op(0x75).op(0x75).done()
    }

    /// A small, valid checkpoint: two tensors in two storages.
    /// Returns (zip bytes, expected float32 weights bytes, table JSON).
    pub fn sample_checkpoint() -> (Vec<u8>, Vec<u8>, String) {
        use sha2::{Digest, Sha256};
        // Storage 0: 6 halves = a [2, 3] tensor. Storage 1: 5 halves, the
        // tensor is the 4 elements at offset 1.
        let halves0: [u16; 6] = [0x3C00, 0xBC00, 0x4000, 0x0000, 0x8000, 0x0001];
        let halves1: [u16; 5] = [0x7BFF, 0x3800, 0x3400, 0xC400, 0x03FF];
        let s0: Vec<u8> = halves0.iter().flat_map(|h| h.to_le_bytes()).collect();
        let s1: Vec<u8> = halves1.iter().flat_map(|h| h.to_le_bytes()).collect();
        let state = P(Vec::new())
            .str("a.weight")
            .tensor(("torch", "HalfStorage"), "0", 6, 0, &[2, 3], &[3, 1])
            .str("b.bias")
            .tensor(("torch", "HalfStorage"), "1", 5, 1, &[4], &[1]);
        let pickle = root_pickle(state);
        let zip = build_zip(&[
            Member::stored("archive/data.pkl", &pickle),
            Member::stored("archive/data/0", &s0),
            Member::stored("archive/data/1", &s1),
            Member::stored("archive/version", b"3\n"),
        ]);
        let floats: [f32; 10] = [
            1.0,
            -1.0,
            2.0,
            0.0,
            -0.0,
            5.960_464_5e-8,
            0.5,
            0.25,
            -4.0,
            6.097_555e-5,
        ];
        let weights: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let hex = |b: &[u8]| super::hex(&Sha256::digest(b));
        let table = format!(
            r#"{{"upstream":{{"sha256":"{}","sizeBytes":{}}},"weights":{{"sha256":"{}","sizeBytes":{}}},
            "tensors":[{{"key":"b.bias","shape":[4],"sourceDtype":"float16","offset":0,"byteLength":16}},
                       {{"key":"a.weight","shape":[2,3],"sourceDtype":"float16","offset":16,"byteLength":24}}]}}"#,
            hex(&zip),
            zip.len(),
            // Table order is b then a: the weights file follows the table.
            hex(&[&weights[24..40], &weights[0..24]].concat()),
            weights.len()
        );
        let ordered = [&weights[24..40], &weights[0..24]].concat();
        (zip, ordered, table)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use crate::progress::NoProgress;
    use crate::testutil::TempDir;
    use std::io::Cursor;

    fn open(bytes: &[u8]) -> Result<Checkpoint<Cursor<&[u8]>>> {
        Checkpoint::open(Cursor::new(bytes), bytes.len() as u64)
    }

    fn open_err(bytes: &[u8]) -> String {
        match open(bytes) {
            Ok(_) => panic!("hostile checkpoint was accepted"),
            Err(e) => {
                assert_eq!(e.code(), ErrorCode::ModelDownloadFailed);
                e.detail().unwrap_or("").to_string()
            }
        }
    }

    #[test]
    fn half_to_float_is_exact_for_every_finite_value() {
        let mut finite = 0;
        for h in 0..=u16::MAX {
            let sign = if h >> 15 == 1 { -1.0f64 } else { 1.0 };
            let exp = i32::from((h >> 10) & 0x1F);
            let frac = f64::from(h & 0x3FF);
            let bits = half_bits_to_float_bits(h);
            if exp == 0x1F {
                assert!(bits.is_none(), "{h:#06x}");
                continue;
            }
            finite += 1;
            // The value by the textbook definition, computed in f64.
            let expected = if exp == 0 {
                sign * frac * 2f64.powi(-24)
            } else {
                sign * (1.0 + frac / 1024.0) * 2f64.powi(exp - 15)
            };
            let got = f32::from_bits(bits.unwrap());
            assert_eq!(f64::from(got), expected, "{h:#06x}");
            assert_eq!(got.is_sign_negative(), h >> 15 == 1, "sign of {h:#06x}");
        }
        assert_eq!(finite, 63_488);
        let mut out = Vec::new();
        halves_to_float32_bytes(&[0x00, 0x3C, 0x00, 0xC0], &mut out).unwrap();
        assert_eq!(
            out,
            [1.0f32.to_le_bytes(), (-2.0f32).to_le_bytes()].concat()
        );
        for bad_half in [0x7C00u16, 0xFC00, 0x7E00, 0x7C01] {
            assert!(halves_to_float32_bytes(&bad_half.to_le_bytes(), &mut out).is_err());
        }
        assert!(halves_to_float32_bytes(&[0x00], &mut out).is_err());
    }

    #[test]
    fn symbolic_unpickle_accepts_the_shapes_the_real_file_uses() {
        let p = P::new()
            .op(0x5D) // []
            .op(0x71)
            .raw(&[0]) // BINPUT 0 (a list: stored as "mutable")
            .op(0x28)
            .int(1)
            .int(70_000)
            .op(0x4E)
            .op(0x88)
            .raw(&[0x47, 0x3F, 0xF8, 0, 0, 0, 0, 0, 0]) // 1.5
            .str("héllo")
            .op(0x71)
            .raw(&[1]) // BINPUT 1 (a string)
            .op(0x68)
            .raw(&[1]) // BINGET 1
            .op(0x65) // APPENDS
            .done();
        let Value::List(items) = symbolic_unpickle(&p).unwrap() else {
            panic!("not a list")
        };
        assert_eq!(items.len(), 7);
        assert_eq!(items[0], Value::Int(1));
        assert_eq!(items[1], Value::Int(70_000));
        assert_eq!(items[4], Value::Float(1.5));
        assert_eq!(items[5], Value::Str(Rc::from("héllo")));
        assert_eq!(items[6], items[5]);
        // REDUCE records, never executes.
        let p = P::new()
            .global("fractions", "Fraction")
            .str("39/5")
            .op(0x85)
            .op(0x52)
            .done();
        assert!(matches!(symbolic_unpickle(&p).unwrap(), Value::Call(4, _)));
    }

    #[test]
    fn hostile_pickles_are_rejected() {
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("os.system", b"\x80\x02cos\nsystem\n.".to_vec()),
            ("builtins eval", b"\x80\x02c__builtin__\neval\n.".to_vec()),
            (
                "posix system",
                b"\x80\x02cposix\nsystem\nX\x02\x00\x00\x00id\x85R.".to_vec(),
            ),
            (
                "torch load helper not in list",
                b"\x80\x02ctorch._utils\n_rebuild_parameter\n.".to_vec(),
            ),
            (
                "allowed module, other name",
                b"\x80\x02ccollections\ndeque\n.".to_vec(),
            ),
            (
                "name with trailing junk",
                b"\x80\x02ccollections\nOrderedDict \n.".to_vec(),
            ),
            ("protocol 4", b"\x80\x04N.".to_vec()),
            ("protocol 0 opcodes (INT)", b"I1\n.".to_vec()),
            ("no STOP", b"\x80\x02N".to_vec()),
            ("SHORT_BINUNICODE (proto 4)", b"\x80\x02\x8c\x01a.".to_vec()),
            (
                "STACK_GLOBAL",
                b"\x80\x02X\x02\x00\x00\x00osX\x06\x00\x00\x00system\x93.".to_vec(),
            ),
            ("INST", b"\x80\x02(ios\nsystem\n.".to_vec()),
            ("OBJ", b"\x80\x02(cos\nsystem\no.".to_vec()),
            (
                "NEWOBJ",
                b"\x80\x02ccollections\nOrderedDict\n)\x81.".to_vec(),
            ),
            (
                "BUILD",
                b"\x80\x02ccollections\nOrderedDict\n)R}b.".to_vec(),
            ),
            ("EXT1", b"\x80\x02\x82\x01.".to_vec()),
            ("LONG1", b"\x80\x02\x8a\x01\x01.".to_vec()),
            ("BINBYTES", b"\x80\x02B\x01\x00\x00\x00a.".to_vec()),
            ("SETITEM (single)", b"\x80\x02}K\x01K\x02s.".to_vec()),
            (
                "FRAME",
                b"\x80\x02\x95\x00\x00\x00\x00\x00\x00\x00\x00N.".to_vec(),
            ),
            ("trailing bytes", b"\x80\x02N.N".to_vec()),
            ("two values left", b"\x80\x02NN.".to_vec()),
            ("empty", Vec::new()),
            ("stack underflow", b"\x80\x02\x85.".to_vec()),
            ("mark left on stack", b"\x80\x02(.".to_vec()),
            ("TUPLE without MARK", b"\x80\x02Nt.".to_vec()),
            ("REDUCE on a non-global", b"\x80\x02N)R.".to_vec()),
            (
                "REDUCE with non-tuple args",
                b"\x80\x02ccollections\nOrderedDict\nNR.".to_vec(),
            ),
            ("BINGET of unset slot", b"\x80\x02h\x05.".to_vec()),
            (
                "memoised dict read back",
                b"\x80\x02}q\x00h\x00\x86.".to_vec(),
            ),
            (
                "memoised list read back",
                b"\x80\x02]q\x00h\x00\x86.".to_vec(),
            ),
            ("persistent id not a tuple", b"\x80\x02NQ.".to_vec()),
            ("dict key is a tuple", b"\x80\x02}()Nu.".to_vec()),
            ("dict key is a bool", b"\x80\x02}(\x88Nu.".to_vec()),
            ("odd SETITEMS", b"\x80\x02}(Nu.".to_vec()),
            ("APPEND to a dict", b"\x80\x02}Na.".to_vec()),
            (
                "string length past the end",
                b"\x80\x02X\xff\x00\x00\x00a.".to_vec(),
            ),
            (
                "string over the limit",
                [
                    b"\x80\x02X\x01\x04\x00\x00".to_vec(),
                    vec![b'a'; 1025],
                    b".".to_vec(),
                ]
                .concat(),
            ),
            (
                "string not UTF-8",
                b"\x80\x02X\x02\x00\x00\x00\xff\xfe.".to_vec(),
            ),
            (
                "global line never ends",
                [b"\x80\x02c".to_vec(), vec![b'a'; 3000]].concat(),
            ),
            (
                "global not ASCII",
                b"\x80\x02ccollections\nOrdered\xc3\xa9\n.".to_vec(),
            ),
            ("truncated BININT", b"\x80\x02J\x01\x02".to_vec()),
        ];
        for (what, bytes) in cases {
            let r = symbolic_unpickle(&bytes);
            assert!(r.is_err(), "accepted: {what}");
            assert_eq!(
                r.unwrap_err().code(),
                ErrorCode::ModelDownloadFailed,
                "{what}"
            );
        }
    }

    #[test]
    fn value_nesting_is_bounded_at_construction() {
        const LIMIT: usize = 64;
        let nested = |depth: usize| {
            let mut pickle = vec![0x80, 2, 0x4e]; // protocol 2, None
            pickle.extend(std::iter::repeat_n(0x85, depth)); // single-item tuples
            pickle.push(0x2e);
            pickle
        };
        assert!(symbolic_unpickle(&nested(LIMIT - 1)).is_ok());
        assert!(symbolic_unpickle(&nested(LIMIT)).is_ok());
        for depth in [LIMIT + 1, 199_995] {
            let err = symbolic_unpickle(&nested(depth)).unwrap_err();
            assert_eq!(
                err.detail(),
                Some("checkpoint: pickle value nesting too deep")
            );
        }
    }

    #[test]
    fn pickle_container_updates_and_memo_preserve_depth_bounds() {
        // Memoize a value at the limit and try to append the recalled value.
        let mut pickle = vec![0x80, 2, 0x5d, 0x4e];
        pickle.extend(std::iter::repeat_n(0x85, 64));
        pickle.extend([0x71, 0, 0x68, 0, 0x85]);
        let err = symbolic_unpickle(&pickle).unwrap_err();
        assert_eq!(
            err.detail(),
            Some("checkpoint: pickle value nesting too deep")
        );
        // A list and dict also raise depth when their entries are added.
        let tuple = vec![0x85; 64];
        let list = P::new().op(0x5d).op(0x4e).raw(&tuple).op(0x61).done();
        let dict = P::new()
            .op(0x7d)
            .op(0x28)
            .str("key")
            .op(0x4e)
            .raw(&tuple)
            .op(0x75)
            .done();
        for pickle in [list, dict] {
            let err = symbolic_unpickle(&pickle).unwrap_err();
            assert_eq!(
                err.detail(),
                Some("checkpoint: pickle value nesting too deep")
            );
        }
    }

    #[test]
    fn interpreter_limits_hold() {
        // Opcode budget: a long run of NONE.
        let mut p = vec![0x80, 2, 0x28];
        p.extend(std::iter::repeat_n(0x4E, MAX_OPCODES + 10));
        assert!(symbolic_unpickle(&p).is_err());
        // Stack depth.
        let mut p = vec![0x80, 2];
        p.extend(std::iter::repeat_n(0x4E, MAX_STACK_DEPTH + 1));
        p.push(0x2E);
        let e = symbolic_unpickle(&p).unwrap_err();
        assert!(e.detail().unwrap().contains("too deep"));
        // Memo entries: LONG_BINPUT to many distinct slots.
        let mut p = vec![0x80, 2, 0x4E];
        for i in 0..(MAX_MEMO_ENTRIES as u32 + 1) {
            p.push(0x72);
            p.extend_from_slice(&i.to_le_bytes());
        }
        p.push(0x2E);
        let e = symbolic_unpickle(&p).unwrap_err();
        assert!(e.detail().unwrap().contains("memo"), "{:?}", e.detail());
        // Pickle size.
        assert!(symbolic_unpickle(&vec![0x4E; MAX_PICKLE_BYTES + 1]).is_err());
        // Random bytes never panic.
        let mut seed = 0x5EED_5EEDu64;
        for len in [1usize, 2, 8, 64, 1024, 20_000] {
            for _ in 0..300 {
                let bytes: Vec<u8> = (0..len)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        seed as u8
                    })
                    .collect();
                let _ = symbolic_unpickle(&bytes);
            }
        }
    }

    #[test]
    fn valid_checkpoint_opens_and_converts() {
        let (zip, weights, table) = sample_checkpoint();
        let cp = open(&zip).unwrap();
        assert_eq!(cp.tensors().len(), 2);
        assert_eq!(cp.tensors()[0].key, "a.weight");
        assert_eq!(cp.tensors()[0].shape, vec![2, 3]);
        assert_eq!(cp.tensors()[1].member, "archive/data/1");
        assert_eq!(cp.tensors()[1].offset_elements, 1);

        let dir = TempDir::new("ckpt-ok");
        let src = dir.path().join("model.th");
        std::fs::write(&src, &zip).unwrap();
        let table = TensorTable::from_json(&table).unwrap();
        let out = dir.path().join("model.weights");
        convert_checkpoint(&src, &table, &out, &NoProgress).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), weights);
        // Only the source and the result are in the folder.
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
    }

    #[test]
    fn conversion_rejects_anything_not_pinned() {
        let (zip, _weights, table_json) = sample_checkpoint();
        let dir = TempDir::new("ckpt-pin");
        let src = dir.path().join("model.th");
        let out = dir.path().join("model.weights");
        let table = TensorTable::from_json(&table_json).unwrap();
        let leftovers = |dir: &TempDir| std::fs::read_dir(dir.path()).unwrap().count();

        // One flipped bit in a storage: the upstream digest no longer matches.
        let mut tampered = zip.clone();
        let at = tampered.windows(2).position(|w| w == [0x00, 0x3C]).unwrap();
        tampered[at] ^= 1;
        std::fs::write(&src, &tampered).unwrap();
        let e = convert_checkpoint(&src, &table, &out, &NoProgress).unwrap_err();
        assert!(e.detail().unwrap().contains("SHA-256"), "{:?}", e.detail());
        assert!(!out.exists());

        // Truncated file: wrong size.
        std::fs::write(&src, &zip[..zip.len() - 10]).unwrap();
        let e = convert_checkpoint(&src, &table, &out, &NoProgress).unwrap_err();
        assert!(e.detail().unwrap().contains("size"));

        // The right file but a table that pins a different result.
        std::fs::write(&src, &zip).unwrap();
        let mut wrong = table.clone();
        wrong.weights.sha256 = "00".repeat(32);
        let e = convert_checkpoint(&src, &wrong, &out, &NoProgress).unwrap_err();
        assert!(e.detail().unwrap().contains("pinned digest"));
        assert!(!out.exists());
        assert_eq!(leftovers(&dir), 1, "no temporary file left behind");

        // A table that names a tensor the checkpoint lacks, or another shape.
        let mut wrong = table.clone();
        wrong.tensors[0].key = "missing".into();
        assert!(convert_checkpoint(&src, &wrong, &out, &NoProgress).is_err());
        let mut wrong = table.clone();
        wrong.tensors[1].shape = vec![3, 2];
        assert!(convert_checkpoint(&src, &wrong, &out, &NoProgress).is_err());
        assert!(!out.exists());

        // Cancellation.
        let flag = crate::progress::CancelFlag::new();
        flag.cancel();
        assert!(
            convert_checkpoint(&src, &table, &out, &flag)
                .unwrap_err()
                .is_cancelled()
        );
        assert_eq!(leftovers(&dir), 1);

        // Tables that do not add up are refused at parse time.
        assert!(
            TensorTable::from_json(&table_json.replace("\"offset\":16", "\"offset\":20")).is_err()
        );
        assert!(TensorTable::from_json(&table_json.replace("float16", "float32")).is_err());
        assert!(
            TensorTable::from_json(&table_json.replace("\"byteLength\":24", "\"byteLength\":28"))
                .is_err()
        );

        // Directories are rejected before opening (on Windows File::open on a
        // directory fails with ERROR_ACCESS_DENIED instead of EISDIR).
        assert_eq!(
            convert_checkpoint(dir.path(), &table, &out, &NoProgress)
                .unwrap_err()
                .code(),
            ErrorCode::ModelDownloadFailed
        );
        assert_eq!(
            sha256_file(dir.path(), &NoProgress).unwrap_err().code(),
            ErrorCode::ModelDownloadFailed
        );
    }

    fn valid_members() -> Vec<Member> {
        let s0: Vec<u8> = [0x3C00u16; 6]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let state =
            P(Vec::new())
                .str("w")
                .tensor(("torch", "HalfStorage"), "0", 6, 0, &[2, 3], &[3, 1]);
        vec![
            Member::stored("archive/data.pkl", &root_pickle(state)),
            Member::stored("archive/data/0", &s0),
            Member::stored("archive/version", b"3\n"),
        ]
    }

    #[test]
    fn hostile_containers_are_rejected() {
        assert!(
            open(&build_zip(&valid_members())).is_ok(),
            "the baseline must be valid"
        );

        // Not a zip at all / wrong magic / truncated.
        open_err(b"");
        open_err(b"this is not a zip file at all, just text padding padding padding");
        let good = build_zip(&valid_members());
        let mut wrong_magic = good.clone();
        wrong_magic[0..4].copy_from_slice(b"XXXX");
        let n = good.len();
        wrong_magic[n - 22..n - 18].copy_from_slice(b"XXXX");
        open_err(&wrong_magic);
        for keep in [10, n / 3, n / 2, n - 30, n - 1] {
            open_err(&good[..keep]);
        }

        // A compressed member (deflate) and an encrypted one.
        let mut m = valid_members();
        m[1].method = 8;
        assert!(open_err(&build_zip(&m)).to_lowercase().contains("compress"));
        let mut m = valid_members();
        m[1].flags = 1;
        open_err(&build_zip(&m));

        // Zip bomb shape: a stored member that declares far more than it has.
        let mut m = valid_members();
        m[1].declared_size = Some(0xFFFF_FFF0);
        open_err(&build_zip(&m));

        // Zip-slip style names and anything outside the expected layout.
        for evil in [
            "../../etc/passwd",
            "archive/../../evil",
            "/abs/data.pkl",
            "archive/data/../../x",
            "archive/data/0x1",
            "archive/data/-1",
            "archive/data/1234567890",
            "archive/extra.txt",
            "other/data/1",
            "archive\\data\\1",
            "archive/data/",
        ] {
            let mut m = valid_members();
            m.push(Member::stored(evil, b"x"));
            open_err(&build_zip(&m));
        }
        // The prefix itself must be one plain folder.
        for prefix in ["", "a/b/", "../", "/"] {
            let m: Vec<Member> = valid_members()
                .into_iter()
                .map(|mut mem| {
                    let name = String::from_utf8(mem.name.clone()).unwrap();
                    mem.name = name.replacen("archive/", prefix, 1).into_bytes();
                    mem
                })
                .collect();
            open_err(&build_zip(&m));
        }

        // Duplicate member names.
        let mut m = valid_members();
        m.push(Member::stored("archive/data/0", &[0u8; 12]));
        open_err(&build_zip(&m));

        // Two pickles, no pickle, missing or wrong version.
        let mut m = valid_members();
        m.push(Member::stored("other/data.pkl", b"\x80\x02N."));
        open_err(&build_zip(&m));
        let m: Vec<Member> = valid_members().into_iter().skip(1).collect();
        open_err(&build_zip(&m));
        let mut m = valid_members();
        m[2] = Member::stored("archive/version", b"4\n");
        assert!(open_err(&build_zip(&m)).contains("version"));
        let mut m = valid_members();
        m.pop();
        open_err(&build_zip(&m));

        // Too many members.
        let mut m = valid_members();
        for i in 1..=(MAX_ZIP_ENTRIES as u32) {
            m.push(Member::stored(&format!("archive/data/{i}"), b""));
        }
        assert!(open_err(&build_zip(&m)).contains("number of zip members"));

        // An oversized pickle.
        let mut m = valid_members();
        m[0] = Member::stored("archive/data.pkl", &vec![0x4E; MAX_PICKLE_BYTES + 1]);
        open_err(&build_zip(&m));

        // A file larger than the cap is refused before it is parsed.
        let e = Checkpoint::open(Cursor::new(&good[..]), MAX_FILE_BYTES + 1)
            .err()
            .unwrap();
        assert!(e.detail().unwrap().contains("too large"));
    }

    fn with_state(state: P) -> Vec<u8> {
        let s0: Vec<u8> = [0x3C00u16; 6]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        build_zip(&[
            Member::stored("archive/data.pkl", &root_pickle(state)),
            Member::stored("archive/data/0", &s0),
            Member::stored("archive/version", b"3\n"),
        ])
    }

    #[test]
    fn hostile_tensor_records_are_rejected() {
        let half = ("torch", "HalfStorage");
        let t = |f: &dyn Fn(P) -> P| with_state(f(P(Vec::new()).str("w")));
        // Baseline.
        assert!(open(&t(&|p| p.tensor(half, "0", 6, 0, &[2, 3], &[3, 1]))).is_ok());
        // Shape larger than the storage / offset past the end.
        open_err(&t(&|p| p.tensor(half, "0", 6, 0, &[2, 4], &[4, 1])));
        open_err(&t(&|p| p.tensor(half, "0", 6, 1, &[2, 3], &[3, 1])));
        // Storage size that disagrees with the zip member (would over-read).
        open_err(&t(&|p| p.tensor(half, "0", 600, 0, &[2, 3], &[3, 1])));
        // Storage that is not in the archive.
        open_err(&t(&|p| p.tensor(half, "7", 6, 0, &[2, 3], &[3, 1])));
        // Storage key that is not a plain number (path tricks).
        open_err(&t(&|p| p.tensor(half, "../0", 6, 0, &[2, 3], &[3, 1])));
        open_err(&t(&|p| p.tensor(half, "", 6, 0, &[2, 3], &[3, 1])));
        // Non-contiguous strides, zero and huge dimensions, too many dims.
        open_err(&t(&|p| p.tensor(half, "0", 6, 0, &[2, 3], &[1, 2])));
        open_err(&t(&|p| p.tensor(half, "0", 6, 0, &[0, 3], &[3, 1])));
        open_err(&t(&|p| {
            p.tensor(half, "0", 6, 0, &[70_000, 70_000], &[70_000, 1])
        }));
        open_err(&t(&|p| p.tensor(half, "0", 6, 0, &[1; 9], &[1; 9])));
        open_err(&t(&|p| p.tensor(half, "0", 6, 0, &[2, 3], &[1])));
        // A storage class that is not on the allow-list.
        open_err(&t(&|p| {
            p.tensor(("torch", "FloatStorage"), "0", 6, 0, &[2, 3], &[3, 1])
        }));
        // An allow-listed global that is not a storage class.
        open_err(&t(&|p| {
            p.tensor(("fractions", "Fraction"), "0", 6, 0, &[2, 3], &[3, 1])
        }));
        // A state value that is not a tensor record.
        open_err(&t(&|p| p.int(5)));
        open_err(&t(&|p| {
            p.global("fractions", "Fraction")
                .str("1/2")
                .op(0x85)
                .op(0x52)
        }));
        // An empty state dict.
        let empty = build_zip(&[
            Member::stored(
                "archive/data.pkl",
                &P::new()
                    .op(0x7D)
                    .op(0x28)
                    .str("klass")
                    .global("demucs.htdemucs", "HTDemucs")
                    .str("args")
                    .op(0x29)
                    .str("kwargs")
                    .op(0x7D)
                    .str("state")
                    .op(0x7D)
                    .op(0x75)
                    .done(),
            ),
            Member::stored("archive/version", b"3\n"),
        ]);
        open_err(&empty);
    }

    #[test]
    fn root_must_be_the_expected_package() {
        let wrap = |pickle: Vec<u8>| {
            build_zip(&[
                Member::stored("archive/data.pkl", &pickle),
                Member::stored("archive/version", b"3\n"),
            ])
        };
        // Root is not a dict.
        open_err(&wrap(P::new().op(0x5D).done()));
        // Missing keys.
        open_err(&wrap(P::new().op(0x7D).done()));
        // An unexpected extra key.
        let state =
            P(Vec::new())
                .str("w")
                .tensor(("torch", "HalfStorage"), "0", 6, 0, &[2, 3], &[3, 1]);
        let mut pickle = root_pickle(state);
        // Splice `"surprise": None` in before the final SETITEMS + STOP.
        let tail = pickle.split_off(pickle.len() - 2);
        pickle.extend_from_slice(&P(Vec::new()).str("surprise").op(0x4E).0);
        pickle.extend_from_slice(&tail);
        let s0: Vec<u8> = [0x3C00u16; 6]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let z = build_zip(&[
            Member::stored("archive/data.pkl", &pickle),
            Member::stored("archive/data/0", &s0),
            Member::stored("archive/version", b"3\n"),
        ]);
        assert!(open_err(&z).contains("unexpected checkpoint keys"));
    }

    #[test]
    fn mutated_valid_checkpoints_never_panic() {
        let (zip, _, _) = sample_checkpoint();
        let mut seed = 0xABCD_EF01_2345_6789u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let (mut ok, mut err) = (0, 0);
        for _ in 0..4000 {
            let mut d = zip.clone();
            for _ in 0..(1 + next() % 3) {
                let at = (next() as usize) % d.len();
                d[at] ^= 1 << (next() % 8);
            }
            if next() % 5 == 0 {
                let keep = (next() as usize) % d.len();
                d.truncate(keep);
            }
            match open(&d) {
                Ok(mut cp) => {
                    ok += 1;
                    // Reading the tensors must also stay in bounds.
                    let tensors = cp.tensors().to_vec();
                    let mut buf = Vec::new();
                    for t in &tensors {
                        let _ = cp.read_tensor_halves(t, &mut buf);
                        assert!(buf.len() <= zip.len());
                    }
                }
                Err(_) => err += 1,
            }
        }
        assert!(err > 0 && ok + err == 4000);
    }

    #[test]
    fn deeply_nested_pickle_is_rejected_without_stack_overflow() {
        let nested_tuple1 = |wraps: usize| -> Vec<u8> {
            let mut p = Vec::with_capacity(wraps + 4);
            p.extend_from_slice(&[0x80, 0x02, 0x29]); // PROTO 2, EMPTY_TUPLE (depth 1)
            p.resize(p.len() + wraps, 0x85); // TUPLE1
            p.push(0x2E); // STOP
            p
        };

        // Just under the limit: EMPTY_TUPLE (depth 1) + 63 TUPLE1 wraps = depth 64.
        let at_limit = usize::from(MAX_NESTING_DEPTH) - 1;
        assert!(symbolic_unpickle(&nested_tuple1(at_limit)).is_ok());

        // Just over the limit: depth 65 is rejected cleanly.
        let err = symbolic_unpickle(&nested_tuple1(at_limit + 1)).unwrap_err();
        assert!(
            err.detail().unwrap_or("").contains("nesting too deep"),
            "{:?}",
            err.detail()
        );

        // Regression for the 200,000 nested TUPLE1 stack-overflow abort, both
        // directly and through Checkpoint::open.
        let attack = nested_tuple1(199_995);
        assert!(symbolic_unpickle(&attack).is_err());
        let z = build_zip(&[
            Member::stored("archive/data.pkl", &attack),
            Member::stored("archive/version", b"3\n"),
        ]);
        assert!(open_err(&z).contains("nesting too deep"));

        // Memo-shared nesting (BINPUT/BINGET + TUPLE2) is also depth-tracked.
        let memo_chain = |steps: usize| -> Vec<u8> {
            let mut p = vec![0x80, 0x02, 0x29, 0x71, 0x00]; // EMPTY_TUPLE (depth 1), BINPUT 0
            for i in 1..=steps {
                // Pair top-of-stack with its memoized copy via TUPLE2 so stack stays depth 1.
                p.extend_from_slice(&[0x68, (i - 1) as u8, 0x86, 0x71, i as u8]);
            }
            p.push(0x2E);
            p
        };
        assert!(symbolic_unpickle(&memo_chain(at_limit)).is_ok());
        assert!(symbolic_unpickle(&memo_chain(at_limit + 1)).is_err());
    }

    #[test]
    fn inconsistent_and_overlapping_zip_headers_are_rejected() {
        // Local header filename differs from central directory filename.
        let mut mismatch = build_zip(&valid_members());
        let pos = mismatch
            .windows(b"archive/data.pkl".len())
            .position(|w| w == b"archive/data.pkl")
            .unwrap();
        mismatch[pos] = b'X';
        open_err(&mismatch);

        // Overlapping entries: point the second central directory entry's local
        // header offset to offset 0 (same as the first entry).
        let mut overlap = build_zip(&valid_members());
        let cd_positions: Vec<usize> = overlap
            .windows(4)
            .enumerate()
            .filter_map(|(i, w)| (w == [0x50, 0x4B, 0x01, 0x02]).then_some(i))
            .collect();
        assert!(cd_positions.len() >= 2);
        let second_cd = cd_positions[1];
        overlap[second_cd + 42..second_cd + 46].copy_from_slice(&0u32.to_le_bytes());
        open_err(&overlap);
    }

    #[cfg(unix)]
    #[test]
    fn convert_checkpoint_does_not_follow_or_remove_preexisting_symlinks() {
        let (zip, _, table_json) = sample_checkpoint();
        let dir = TempDir::new("ckpt-symlink");
        let src = dir.path().join("upstream.th");
        let out = dir.path().join("model.weights");
        let target = dir.path().join("existing.txt");
        std::fs::write(&src, &zip).unwrap();
        std::fs::write(&target, b"keep this").unwrap();
        let old_tmp = dir
            .path()
            .join(format!("model.weights.{}.part", std::process::id()));
        std::os::unix::fs::symlink(&target, &old_tmp).unwrap();
        let table = TensorTable::from_json(&table_json).unwrap();
        convert_checkpoint(&src, &table, &out, &NoProgress).unwrap();
        assert!(out.exists());
        assert_eq!(std::fs::read(&target).unwrap(), b"keep this");
        assert_eq!(std::fs::read_link(&old_tmp).unwrap(), target);
    }

    /// The real thing, when it is on disk: Meta's checkpoint from the ML
    /// cache must convert to exactly the pinned weights file.
    #[test]
    fn real_checkpoint_converts_to_the_pinned_weights() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let checkpoint = std::env::var_os("SPLITSOUND_TEST_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| root.join("ml/.cache/weights/5c90dfd2-34c22ccb.th"));
        let table_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("models/htdemucs_6s.tensors.json");
        if !checkpoint.exists() || !table_path.exists() {
            eprintln!(
                "skipped: needs the upstream checkpoint ({}) and the tensor table",
                checkpoint.display()
            );
            return;
        }
        let table = TensorTable::from_json(&std::fs::read_to_string(&table_path).unwrap()).unwrap();
        let dir = TempDir::new("ckpt-real");
        let out = dir.path().join("htdemucs_6s.weights");
        let t0 = std::time::Instant::now();
        convert_checkpoint(&checkpoint, &table, &out, &NoProgress).unwrap();
        let (digest, size) = sha256_file(&out, &NoProgress).unwrap();
        assert_eq!(digest, table.weights.sha256);
        assert_eq!(size, table.weights.size_bytes);
        eprintln!(
            "converted {} tensors, {} bytes, in {:.2}s",
            table.tensors.len(),
            size,
            t0.elapsed().as_secs_f64()
        );
    }
}
