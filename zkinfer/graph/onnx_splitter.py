import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from onnx import TensorProto

import numpy as np
import onnx
import onnxruntime as ort
from onnx.utils import Extractor
from onnx import helper, numpy_helper, TensorProto
import numpy as np
from zkinfer.storage.io import compute_bytes_md5_hex
from zkinfer.utils.onnx import (
    infer_shapes,
    inputs_to_json,
    load_or_generate_inputs,
    simplify_model_if_requested,
)

logger = logging.getLogger(__name__)

PASSTHROUGH_OPS = {
    "Identity",
    "Constant",
    "Shape",
    "Gather",
    "Unsqueeze",
    "Concat",
}
@dataclass(frozen=True)
class ModelPartition:
    name: str
    input_names: List[str]
    output_names: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterializedSubmodel:
    name: str
    parent_hash: str
    sub_hash: str
    model: onnx.ModelProto
    input_data: Dict[str, Any]


def build_producer_map(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
    return {
        output_name: node
        for node in model.graph.node
        for output_name in node.output
        if output_name
    }

def freeze_reshape_shapes_from_values(model, tensor_values):
    for node in model.graph.node:
        if node.op_type != "Reshape":
            continue

        shape_name = node.input[1]
        shape_value = tensor_values.get(shape_name)

        if shape_value is None:
            continue

        shape_value = np.asarray(shape_value, dtype=np.int64).reshape(-1)
        frozen_name = shape_name + "_frozen"

        init = numpy_helper.from_array(shape_value, name=frozen_name)
        model.graph.initializer.append(init)

        node.input[1] = frozen_name

    return model

def set_graph_input_shapes_from_values(model, tensor_values):
    for graph_input in model.graph.input:
        value = tensor_values.get(graph_input.name)
        if value is None:
            continue

        tensor_type = graph_input.type.tensor_type

        if value.dtype == np.float32:
            tensor_type.elem_type = TensorProto.FLOAT
        elif value.dtype == np.int64:
            tensor_type.elem_type = TensorProto.INT64
        elif value.dtype == np.int32:
            tensor_type.elem_type = TensorProto.INT32

        del tensor_type.shape.dim[:]
        for dim in value.shape:
            tensor_type.shape.dim.add().dim_value = int(dim)

    return model

def remove_unused_graph_inputs(model):
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [x for x in model.graph.input if x.name in used]

    del model.graph.input[:]
    model.graph.input.extend(kept)
    return model

def trace_sources(
    tensor_names: Sequence[str],
    producer_map: Dict[str, onnx.NodeProto],
    passthrough_ops: set[str],
    graph_inputs: set[str],
    initializers: set[str],
) -> List[str]:
    sources: set[str] = set()
    visited: set[str] = set()

    def visit(tensor_name: str) -> None:
        if not tensor_name or tensor_name in visited:
            return

        visited.add(tensor_name)

        if tensor_name in graph_inputs:
            sources.add(tensor_name)
            return

        if tensor_name in initializers:
            return

        producer = producer_map.get(tensor_name)

        if producer is None:
            sources.add(tensor_name)
            return

        if producer.op_type in passthrough_ops:
            for input_name in producer.input:
                visit(input_name)
        else:
            sources.add(tensor_name)

    for tensor_name in tensor_names:
        visit(tensor_name)

    return sorted(sources)


def get_partition_io(
    nodes: Sequence[onnx.NodeProto],
    producer_map: Dict[str, onnx.NodeProto],
    graph_inputs: set[str],
    initializers: set[str],
    passthrough_ops: set[str],
) -> Tuple[List[str], List[str]]:
    internal_outputs = {
        output_name
        for node in nodes
        for output_name in node.output
        if output_name
    }

    raw_inputs = [
        input_name
        for node in nodes
        for input_name in node.input
        if input_name and input_name not in internal_outputs
    ]

    input_names = trace_sources(
        tensor_names=raw_inputs,
        producer_map=producer_map,
        passthrough_ops=passthrough_ops,
        graph_inputs=graph_inputs,
        initializers=initializers,
    )

    return input_names, sorted(internal_outputs)


def split_by_single_ops(
    model: onnx.ModelProto,
    passthrough_ops: Optional[set[str]] = None,
) -> List[ModelPartition]:
    passthrough_ops = passthrough_ops or PASSTHROUGH_OPS

    graph_inputs = {graph_input.name for graph_input in model.graph.input}
    initializers = {initializer.name for initializer in model.graph.initializer}
    producer_map = build_producer_map(model)

    partitions = []

    for node_idx, node in enumerate(model.graph.node):
        if node.op_type in passthrough_ops:
            continue

        input_names, output_names = get_partition_io(
            nodes=[node],
            producer_map=producer_map,
            graph_inputs=graph_inputs,
            initializers=initializers,
            passthrough_ops=passthrough_ops,
        )

        if not input_names or not output_names:
            continue

        partitions.append(
            ModelPartition(
                name=f"sub_model_{len(partitions) + 1}",
                input_names=input_names,
                output_names=output_names,
                metadata={
                    "split_strategy": "single_ops",
                    "node_index": node_idx,
                    "node_name": node.name,
                    "op_type": node.op_type,
                    "num_nodes": 1,
                },
            )
        )

    return partitions


def split_by_fixed_groups(
    model: onnx.ModelProto,
    group_size: int,
    passthrough_ops: Optional[set[str]] = None,
) -> List[ModelPartition]:
    if group_size <= 0:
        raise ValueError("group_size must be > 0")

    passthrough_ops = passthrough_ops or PASSTHROUGH_OPS

    graph_inputs = {graph_input.name for graph_input in model.graph.input}
    initializers = {initializer.name for initializer in model.graph.initializer}
    producer_map = build_producer_map(model)

    meaningful_nodes = [
        node
        for node in model.graph.node
        if node.op_type not in passthrough_ops
    ]

    partitions = []

    for start in range(0, len(meaningful_nodes), group_size):
        group_nodes = meaningful_nodes[start:start + group_size]

        input_names, output_names = get_partition_io(
            nodes=group_nodes,
            producer_map=producer_map,
            graph_inputs=graph_inputs,
            initializers=initializers,
            passthrough_ops=passthrough_ops,
        )

        if not input_names or not output_names:
            continue

        partitions.append(
            ModelPartition(
                name=f"sub_model_{len(partitions) + 1}",
                input_names=input_names,
                output_names=output_names,
                metadata={
                    "split_strategy": "fixed_groups",
                    "group_size": group_size,
                    "group_start": start,
                    "num_nodes": len(group_nodes),
                    "op_types": [node.op_type for node in group_nodes],
                    "node_names": [node.name for node in group_nodes],
                },
            )
        )

    return partitions


def partition_model(
    model: onnx.ModelProto,
    split_mode: str,
    split_group_size: int,
) -> List[Tuple[ModelPartition, onnx.ModelProto]]:
    model = infer_shapes(model)
    extractor = Extractor(model)

    split_mode = split_mode.lower()

    if split_mode in {"single", "single_op", "single_ops"}:
        partitions = split_by_single_ops(model)
    elif split_mode in {"fixed", "fixed_groups"}:
        partitions = split_by_fixed_groups(model, split_group_size)
    else:
        raise ValueError(
            f"Unsupported split_mode={split_mode!r}. "
            "Expected one of: none, single_ops, fixed."
        )

    extracted = []

    for partition in partitions:
        try:
            sub_model = extractor.extract_model(
                partition.input_names,
                partition.output_names,
            )
            # Work around tract/zkinfer issue with value_info emitted by onnx.utils.Extractor
            sub_model = onnx.shape_inference.infer_shapes(sub_model)
            del sub_model.graph.value_info[:]

            # Optional: preserve original producer metadata
            sub_model.producer_name = model.producer_name
            sub_model.producer_version = model.producer_version
        except Exception as exc:
            raise RuntimeError(
                f"Failed to extract partition {partition.name}: "
                f"inputs={partition.input_names}, outputs={partition.output_names}"
            ) from exc

        extracted.append((partition, sub_model))

    return extracted


def add_intermediate_outputs(model: onnx.ModelProto) -> onnx.ModelProto:
    inference_model = onnx.ModelProto()
    inference_model.CopyFrom(model)

    existing_outputs = {
        output.name
        for output in inference_model.graph.output
        if output.name
    }

    shaped_model = infer_shapes(inference_model)

    for node in shaped_model.graph.node:
        for output_name in node.output:
            if not output_name or output_name in existing_outputs:
                continue

            output_info = onnx.ValueInfoProto()
            output_info.name = output_name
            inference_model.graph.output.append(output_info)
            existing_outputs.add(output_name)

    return inference_model


def collect_tensor_values(
    model: onnx.ModelProto,
    input_data_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    inference_model = add_intermediate_outputs(model)
    session = ort.InferenceSession(inference_model.SerializeToString())

    inputs = load_or_generate_inputs(session, input_data_path)
    outputs = session.run(None, inputs)

    tensor_values = dict(inputs)

    for output, value in zip(session.get_outputs(), outputs):
        tensor_values[output.name] = value

    return tensor_values


def materialize_submodels(
    sub_models: Sequence[Tuple[ModelPartition, onnx.ModelProto]],
    tensor_values: Dict[str, np.ndarray],
    parent_hash: str,
) -> List[MaterializedSubmodel]:
    materialized = []

    for idx, (partition, sub_model) in enumerate(sub_models):
        sub_model = set_graph_input_shapes_from_values(sub_model, tensor_values)
        sub_model = freeze_reshape_shapes_from_values(sub_model, tensor_values)
        # sub_model = remove_unused_graph_inputs(sub_model)


        del sub_model.graph.value_info[:]
        input_values = []
        missing_inputs = []

        for graph_input in sub_model.graph.input:
            value = tensor_values.get(graph_input.name)

            if value is None:
                missing_inputs.append(graph_input.name)
                continue

            input_values.append(value.flatten().tolist())

        if missing_inputs:
            raise RuntimeError(
                f"Missing materialized inputs for {partition.name}: {missing_inputs}"
            )

        if not input_values:
            raise RuntimeError(f"No inputs were materialized for {partition.name}")

        sub_hash = compute_bytes_md5_hex(sub_model.SerializeToString())

        materialized.append(
            MaterializedSubmodel(
                name=partition.name or f"sub_model_{idx + 1}",
                parent_hash=parent_hash,
                sub_hash=sub_hash,
                model=sub_model,
                input_data={"input_data": input_values},
            )
        )

    return materialized


def materialize_full_model(
    model: onnx.ModelProto,
    model_hash: str,
    input_data_path: Optional[str],
    model_name: Optional[str],
) -> MaterializedSubmodel:
    session = ort.InferenceSession(model.SerializeToString())
    inputs = load_or_generate_inputs(session, input_data_path)
    input_names = [model_input.name for model_input in session.get_inputs()]

    return MaterializedSubmodel(
        name=model_name or "model",
        parent_hash=model_hash,
        sub_hash=model_hash,
        model=model,
        input_data=inputs_to_json(inputs, input_names),
    )


def split_onnx_model_with_inputs(
    model_path: str,
    input_data_path: Optional[str] = None,
    split_group_size: int = 1,
    split_mode: str = "none", 
    simplify_model: bool = False,
    simplified_model_path: Optional[str] = None,
    input_shapes: Optional[Dict[str, List[int]]] = None,
    model_name: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> List[MaterializedSubmodel]:
    metrics = metrics if metrics is not None else {}
    split_mode = (split_mode or "none").lower()

    prepared_model_path = simplify_model_if_requested(
        model_path=model_path,
        simplify=simplify_model,
        output_path=simplified_model_path,
        input_shapes=input_shapes,
    )

    model = onnx.load(prepared_model_path)
    model_hash = compute_bytes_md5_hex(model.SerializeToString())

    if split_mode == "none":
        metrics.update(
            {
                "intermediate_tensor_generation_time(s)": 0.0,
                "unique_boundary_tensors": 0,
                "unique_boundary_tensor_bytes": 0,
                "consumer_boundary_tensor_bytes": 0,
                "boundary_fanout_ratio": 0.0,
            }
        )
        return [
            materialize_full_model(
                model=model,
                model_hash=model_hash,
                input_data_path=input_data_path,
                model_name=model_name,
            )
        ]

    start = time.perf_counter()
    tensor_values = collect_tensor_values(
        model=model,
        input_data_path=input_data_path,
    )
    metrics["intermediate_tensor_generation_time(s)"] = (
        time.perf_counter() - start
    )

    sub_models = partition_model(
        model=model,
        split_mode=split_mode,
        split_group_size=split_group_size,
    )

    initializer_names = {
        initializer.name for initializer in model.graph.initializer
    }
    original_inputs = {
        graph_input.name for graph_input in model.graph.input
    } - initializer_names

    boundary_refs = [
        name
        for partition, _ in sub_models
        for name in partition.input_names
        if name not in original_inputs and name in tensor_values
    ]
    unique_boundary_names = set(boundary_refs)

    unique_boundary_bytes = sum(
        int(np.asarray(tensor_values[name]).nbytes)
        for name in unique_boundary_names
    )
    consumer_boundary_bytes = sum(
        int(np.asarray(tensor_values[name]).nbytes)
        for name in boundary_refs
    )

    metrics.update(
        {
            "unique_boundary_tensors": len(unique_boundary_names),
            "unique_boundary_tensor_bytes": unique_boundary_bytes,
            "consumer_boundary_tensor_bytes": consumer_boundary_bytes,
            "boundary_fanout_ratio": (
                consumer_boundary_bytes / unique_boundary_bytes
                if unique_boundary_bytes
                else 0.0
            ),
        }
    )

    return materialize_submodels(
        sub_models=sub_models,
        tensor_values=tensor_values,
        parent_hash=model_hash,
    )


def save_submodels(
    models: Sequence[MaterializedSubmodel],
    output_dir: str,
) -> None:
    #delete and recreate output_dir
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    os.makedirs(output_dir, exist_ok=True)

    for idx, item in enumerate(models, start=1):
        model_path = os.path.join(output_dir, f"model_{idx}.onnx")
        input_path = os.path.join(output_dir, f"input_{idx}.json")

        with open(model_path, "wb") as file:
            file.write(item.model.SerializeToString())

        with open(input_path, "w", encoding="utf-8") as file:
            json.dump(item.input_data, file, indent=4)

        print(
            f"Saved {item.name} | "
            f"parent_hash={item.parent_hash} | "
            f"sub_hash={item.sub_hash}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    onnx_file_input_mapping = {
        # "experiments/models/mnist_classifier/mnist_classifier.onnx": "experiments/models/mnist_classifier/input.json",
        # "experiments/models/mnist_gan/mnist_gan.onnx": "experiments/models/mnist_gan/input.json",
        # "experiments/models/mobilenet/mobilenetv2_050_Opset18.onnx": "experiments/models/mobilenet/input.json",
        "experiments/models/nanoGPT/nano_gpt_4_layers_64_embd.onnx": "experiments/models/nanoGPT/input.json",
        # # "experiments/models/pythia-14m/model_static.onnx": "experiments/models/pythia-14m/input.json",
    }

    tmp_dir = "_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    for onnx_file, input_file in onnx_file_input_mapping.items():
        #get name after second to last slash
        model_name = onnx_file.split("/")[-2]
        split_models = split_onnx_model_with_inputs(
            model_path=onnx_file,
            input_data_path=input_file,
            split_mode="fixed",
            split_group_size=2,
            simplify_model=True,
            input_shapes=None,
        )

        save_submodels(split_models, f"_tmp/split_{model_name}")
