"""Assert that exported ONNX files are inert data. Exits non-zero on any violation.

    uv run --locked python verify_onnx.py dist/*.onnx
    uv run --locked python verify_onnx.py --json dist/security/onnx_verify.json dist/*.onnx

An ONNX file is a protobuf graph of operators plus weight tensors. It cannot
run code by itself, but three things can make a model reach outside the file
or outside the standard operator set, and we forbid all of them:

  1. operators from a non-default domain (custom ops, `com.microsoft` contrib
     ops, `ai.onnx.ml`, ...): these need extra kernels or a custom-op shared
     library to be loaded into the runtime;
  2. external data: an initializer that names another file on disk, which the
     runtime would open (path traversal, missing weights not covered by the
     file's SHA-256). The one exception is the weight-free htdemucs_6s graph,
     checked with `--external-data htdemucs_6s.weights --max-inline-bytes 65536`:
     its weights must all point at that one sibling file by plain name, with
     consistent, non-overlapping ranges, and the graph itself must hold none;
  3. control-flow subgraphs (`If`, `Loop`, `Scan`): legal ONNX, but nothing in
     our fixed-shape models needs them, so their presence means the export
     changed in a way somebody should look at.

Also checked: model-local functions (none allowed), a single default-domain
opset import within the allowed range, sparse initializers (none), tensors
stored anywhere other than inline, declared I/O, and
`onnx.checker.check_model(path, full_check=True)` (which also runs strict
shape inference).

The file is parsed with `load_external_data=False`, so even a hostile file is
never asked to open another path during verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import onnx

DEFAULT_DOMAINS = ("", "ai.onnx")          # the two spellings of the default operator set
FORBIDDEN_OPS = ("If", "Loop", "Scan", "SequenceMap")   # anything that carries a subgraph
MIN_OPSET, MAX_OPSET = 13, 21             # what onnxruntime >= 1.17 runs on every EP we ship


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk_graph(graph: onnx.GraphProto, depth: int, found: dict) -> None:
    """Visit every node, including nodes inside attribute subgraphs."""
    found["max_depth"] = max(found["max_depth"], depth)
    for init in graph.initializer:
        found["initializers"].append((init, depth))
    for sparse in graph.sparse_initializer:
        found["sparse"].append(sparse.values.name or "<sparse>")
    for node in graph.node:
        found["nodes"].append(node)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                found["subgraph_attrs"].append(f"{node.op_type}({node.name}).{attr.name}")
                _walk_graph(attr.g, depth + 1, found)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                found["subgraph_attrs"].append(f"{node.op_type}({node.name}).{attr.name}")
                for sub in attr.graphs:
                    _walk_graph(sub, depth + 1, found)
            elif attr.type == onnx.AttributeProto.TENSOR:
                found["attr_tensors"].append((attr.t, f"{node.op_type}({node.name}).{attr.name}"))
            elif attr.type == onnx.AttributeProto.TENSORS:
                for tensor in attr.tensors:
                    found["attr_tensors"].append((tensor, f"{node.op_type}({node.name}).{attr.name}"))


def _is_external(tensor: onnx.TensorProto) -> bool:
    return (tensor.data_location == onnx.TensorProto.EXTERNAL) or len(tensor.external_data) > 0


def _dims(value_info: onnx.ValueInfoProto) -> list:
    return [d.dim_param if d.HasField("dim_param") else d.dim_value
            for d in value_info.type.tensor_type.shape.dim]


TENSOR_ELEMENT_BYTES = {onnx.TensorProto.FLOAT: 4, onnx.TensorProto.FLOAT16: 2, onnx.TensorProto.DOUBLE: 8,
                        onnx.TensorProto.INT64: 8, onnx.TensorProto.INT32: 4, onnx.TensorProto.BOOL: 1,
                        onnx.TensorProto.UINT8: 1, onnx.TensorProto.INT8: 1}


def _tensor_bytes(tensor: onnx.TensorProto) -> int:
    if tensor.raw_data:
        return len(tensor.raw_data)
    return (4 * len(tensor.float_data) + 8 * len(tensor.int64_data) + 4 * len(tensor.int32_data)
            + 8 * len(tensor.double_data) + sum(len(x) for x in tensor.string_data))


def verify(path: Path, external_data: str | None = None, max_inline_bytes: int | None = None) -> dict:
    """Return a report dict; report["violations"] is empty when the file passes.

    external_data: None (default) forbids external data. A file name allows
        initializers to live in exactly that sibling file and nowhere else
        (used for the weight-free htdemucs_6s graph); every reference is then
        checked for a plain file name, a sane offset/length and no overlap.
    max_inline_bytes: if given, the tensors stored inside the .onnx itself
        (initializers and Constant nodes) may not exceed this many bytes in
        total, i.e. the graph carries no weights.
    """
    violations: list[str] = []
    model = onnx.load(str(path), load_external_data=False)

    found = {"nodes": [], "initializers": [], "sparse": [], "subgraph_attrs": [],
             "attr_tensors": [], "max_depth": 0}
    _walk_graph(model.graph, 0, found)

    # 1. operator domains ------------------------------------------------------
    opset_imports = {imp.domain: imp.version for imp in model.opset_import}
    for domain in opset_imports:
        if domain not in DEFAULT_DOMAINS:
            violations.append(f"non-default opset import: domain={domain!r} version={opset_imports[domain]}")
    default_versions = [v for d, v in opset_imports.items() if d in DEFAULT_DOMAINS]
    if len(default_versions) != 1:
        violations.append(f"expected exactly one default-domain opset import, found {default_versions}")
    opset = default_versions[0] if default_versions else None
    if opset is not None and not (MIN_OPSET <= opset <= MAX_OPSET):
        violations.append(f"opset {opset} outside allowed range {MIN_OPSET}..{MAX_OPSET}")

    op_counts: dict[str, int] = {}
    for node in found["nodes"]:
        if node.domain not in DEFAULT_DOMAINS:
            violations.append(f"node {node.name!r}: op {node.op_type!r} from non-default domain {node.domain!r}")
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    # every op must exist in the standard schema set at this opset
    if opset is not None:
        for op_type in sorted(op_counts):
            if not onnx.defs.has(op_type, ""):
                violations.append(f"op {op_type!r} is not a standard ONNX operator")
            else:
                since = onnx.defs.get_schema(op_type, opset, "").since_version
                if since > opset:
                    violations.append(f"op {op_type!r} needs opset {since} > declared {opset}")

    # 2. control flow / subgraphs / local functions ----------------------------
    for op_type in FORBIDDEN_OPS:
        if op_type in op_counts:
            violations.append(f"forbidden control-flow op {op_type!r} x{op_counts[op_type]}")
    if found["subgraph_attrs"]:
        violations.append(f"subgraph attributes present: {found['subgraph_attrs'][:5]}")
    if len(model.functions) > 0:
        violations.append(f"model-local functions present: {[f.name for f in model.functions][:5]}")

    # 3. external data ---------------------------------------------------------
    total_bytes = 0
    external_ranges: list[tuple[int, int, str]] = []
    for tensor, _depth in found["initializers"]:
        if _is_external(tensor):
            location = {e.key: e.value for e in tensor.external_data}
            if external_data is None:
                violations.append(f"initializer {tensor.name!r} uses external data: {location}")
                continue
            if location.get("location") != external_data or set(location) != {"location", "offset", "length"}:
                violations.append(f"initializer {tensor.name!r}: external data must be exactly "
                                  f"location={external_data!r} with offset and length, found {location}")
                continue
            if tensor.raw_data:
                violations.append(f"initializer {tensor.name!r} is external but also carries inline data")
            try:
                offset, length = int(location["offset"]), int(location["length"])
            except ValueError:
                violations.append(f"initializer {tensor.name!r}: non-numeric external offset/length")
                continue
            elements = 1
            for dim in tensor.dims:
                elements *= dim
            expected = elements * TENSOR_ELEMENT_BYTES.get(tensor.data_type, 0)
            if offset < 0 or length != expected or expected == 0:
                violations.append(f"initializer {tensor.name!r}: external length {length} does not match "
                                  f"its shape and type ({expected} bytes)")
            external_ranges.append((offset, offset + length, tensor.name))
        else:
            total_bytes += _tensor_bytes(tensor)
    if external_data is not None:
        if any(sep in external_data for sep in ("/", "\\", "..")) or Path(external_data).is_absolute():
            violations.append(f"external data location {external_data!r} is not a plain file name")
        external_ranges.sort()
        for (_, end, name), (start, _, next_name) in zip(external_ranges, external_ranges[1:]):
            if start < end:
                violations.append(f"external data of {name!r} and {next_name!r} overlap")
    inline_bytes = total_bytes
    for tensor, where in found["attr_tensors"]:
        if _is_external(tensor):
            violations.append(f"attribute tensor at {where} uses external data")
        inline_bytes += _tensor_bytes(tensor)
    if found["sparse"]:
        violations.append(f"sparse initializers present: {found['sparse'][:5]}")
    if max_inline_bytes is not None and inline_bytes > max_inline_bytes:
        violations.append(f"{inline_bytes} bytes of tensor data inside the graph, limit {max_inline_bytes}")

    # 4. structural validity ---------------------------------------------------
    checker = "ok"
    external_total = max((end for _, end, _ in external_ranges), default=0)
    weights_path = path.parent / external_data if external_data else None
    try:
        if weights_path is not None and not weights_path.exists():
            # The weight-free graph is distributed without its data file. The
            # checker insists on opening that file, so here only the graph is
            # checked: strict shape inference needs shapes and types, not values.
            onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=False)
            checker = "ok (graph only: external weights file not present)"
        else:
            if weights_path is not None and weights_path.stat().st_size != external_total:
                violations.append(f"{external_data} is {weights_path.stat().st_size} bytes, the graph expects {external_total}")
            onnx.checker.check_model(str(path), full_check=True)
    except Exception as exc:
        checker = f"{type(exc).__name__}: {str(exc)[:400]}"
        violations.append(f"onnx.checker.check_model(full_check=True) failed: {checker}")

    initializer_names = {t.name for t, _ in found["initializers"]}
    inputs = [(v.name, _dims(v)) for v in model.graph.input if v.name not in initializer_names]
    outputs = [(v.name, _dims(v)) for v in model.graph.output]

    return {
        "file": path.name,
        "sizeBytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "irVersion": model.ir_version,
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
        "opset": opset,
        "opsetImports": opset_imports,
        "domainsUsed": sorted({n.domain for n in found["nodes"]}),
        "nodeCount": len(found["nodes"]),
        "opTypes": dict(sorted(op_counts.items())),
        "initializerCount": len(found["initializers"]),
        "initializerBytesInline": total_bytes,
        "tensorBytesInsideGraph": inline_bytes,
        "externalData": ({"location": external_data, "tensors": len(external_ranges), "bytes": external_total}
                         if external_ranges else False),
        "subgraphs": len(found["subgraph_attrs"]),
        "localFunctions": len(model.functions),
        "checker": checker,
        "inputs": inputs,
        "outputs": outputs,
        "metadata": {p.key: p.value for p in model.metadata_props},
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, help="also write the full reports as JSON")
    parser.add_argument("--external-data", metavar="FILE",
                        help="allow initializers to live in exactly this sibling file (weight-free graphs only)")
    parser.add_argument("--max-inline-bytes", type=int, help="limit on tensor data stored inside the .onnx itself")
    args = parser.parse_args()

    reports, failed = [], False
    for path in args.files:
        report = verify(path, external_data=args.external_data, max_inline_bytes=args.max_inline_bytes)
        reports.append(report)
        status = "FAIL" if report["violations"] else "ok"
        print(f"[{status}] {report['file']}  sha256={report['sha256']}  {report['sizeBytes']} bytes")
        print(f"       opset {report['opset']}  domains {report['domainsUsed']}  nodes {report['nodeCount']}  "
              f"initializers {report['initializerCount']} (external data: {report['externalData']}, "
              f"{report['tensorBytesInsideGraph']} tensor bytes inside the graph)  "
              f"subgraphs {report['subgraphs']}  functions {report['localFunctions']}  checker {report['checker']}")
        print(f"       inputs {report['inputs']}  outputs {report['outputs']}")
        print(f"       ops {report['opTypes']}")
        for violation in report["violations"]:
            failed = True
            print(f"       VIOLATION: {violation}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
