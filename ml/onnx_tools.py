"""Small ONNX post-processing helpers shared by the export scripts.

Everything here only rewrites protobuf fields of a graph this project just
exported; nothing is downloaded or executed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def declare_shapes(model: onnx.ModelProto, shapes: dict[str, list]) -> None:
    """Write exact shapes onto graph outputs (the exporter leaves them symbolic)."""
    for output in model.graph.output:
        if output.name not in shapes:
            continue
        dims = output.type.tensor_type.shape.dim
        assert len(dims) == len(shapes[output.name]), (output.name, len(dims), shapes[output.name])
        for dim, value in zip(dims, shapes[output.name]):
            dim.Clear()
            if isinstance(value, str):
                dim.dim_param = value
            else:
                dim.dim_value = int(value)


def set_metadata(model: onnx.ModelProto, metadata: dict[str, str]) -> None:
    """Replace the metadata properties with `metadata`, sorted by key (deterministic bytes)."""
    del model.metadata_props[:]
    for key in sorted(metadata):
        prop = model.metadata_props.add()
        prop.key, prop.value = key, str(metadata[key])
    model.doc_string = ""


def store_weights_as_float16(model: onnx.ModelProto, min_elements: int = 1024) -> dict:
    """Halve the file: store large float32 initializers as float16 and Cast them back at load.

    Each affected initializer `w` becomes `w__f16` (float16 data) feeding a
    `Cast(to=float32)` node whose output is named `w`, so every consumer still
    sees a float32 tensor and all arithmetic stays float32. onnxruntime's
    constant folding evaluates those Casts once at session creation.
    Returns statistics, including the worst rounding error and any value that
    does not survive the trip through float16.
    """
    graph = model.graph
    kept, converted, cast_nodes = [], [], []
    stats = {"tensors": 0, "elements": 0, "maxAbsRoundingError": 0.0, "maxRelRoundingError": 0.0,
             "overflowed": 0, "underflowedToZero": 0}
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT or int(np.prod(init.dims)) < min_elements:
            kept.append(init)
            continue
        values = numpy_helper.to_array(init)
        halves = values.astype(np.float16)
        back = halves.astype(np.float32)
        stats["tensors"] += 1
        stats["elements"] += int(values.size)
        stats["overflowed"] += int(np.isinf(back).sum() - np.isinf(values).sum())
        stats["underflowedToZero"] += int(((back == 0) & (values != 0)).sum())
        error = np.abs(back.astype(np.float64) - values.astype(np.float64))
        stats["maxAbsRoundingError"] = max(stats["maxAbsRoundingError"], float(error.max()))
        scale = float(np.abs(values).max()) or 1.0
        stats["maxRelRoundingError"] = max(stats["maxRelRoundingError"], float(error.max()) / scale)
        half_name = init.name + "__f16"
        converted.append(numpy_helper.from_array(halves, half_name))
        cast_nodes.append(helper.make_node("Cast", [half_name], [init.name], name="CastF16_" + init.name,
                                           to=TensorProto.FLOAT))
    del graph.initializer[:]
    graph.initializer.extend(kept + converted)
    existing = list(graph.node)
    del graph.node[:]
    graph.node.extend(cast_nodes + existing)          # Casts first keeps the node list topologically sorted
    return stats


def initializer_stats(model: onnx.ModelProto) -> dict:
    by_type: dict[str, int] = {}
    for init in model.graph.initializer:
        name = TensorProto.DataType.Name(init.data_type)
        by_type[name] = by_type.get(name, 0) + len(init.raw_data)
    return by_type
