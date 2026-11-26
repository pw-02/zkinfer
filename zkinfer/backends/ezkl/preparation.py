"""Prepare exact, public-linked EZKL partition witnesses before scheduling.

All numeric boundary transformations are integer operations. No proving keys
are generated here. The coordinator is trusted to define the model, graph,
settings, and request input; proof verification must separately establish VK
trust.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# BN254 scalar field, used by the EZKL backend. EZKL serializes Fr as
# canonical 32-byte little-endian hexadecimal strings.
FIELD_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)
PREPARATION_VERSION = 2
PREPARED_INPUT_KIND = "zkinfer.ezkl.prepared.v1"
SIGNED_LIMIT = 1 << 63


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(value, file, separators=(",", ":"))


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_felt(value: str) -> int:
    """Decode a canonical signed field element within EZKL's i64 domain."""
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("Expected a 32-byte hexadecimal EZKL field element")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("Invalid field element encoding") from exc
    if len(raw) != 32:
        raise ValueError("Invalid field element length")
    integer = int.from_bytes(raw, "little")
    if integer >= FIELD_MODULUS:
        raise ValueError("Non-canonical field element")
    if integer > FIELD_MODULUS // 2:
        integer -= FIELD_MODULUS
    if not -SIGNED_LIMIT < integer < SIGNED_LIMIT:
        raise ValueError("Field element outside supported signed i64 range")
    return integer


def encode_felt(integer: int) -> str:
    if type(integer) is not int or not -SIGNED_LIMIT < integer < SIGNED_LIMIT:
        raise ValueError("Integer outside supported signed i64 range")
    return (integer % FIELD_MODULUS).to_bytes(32, "little").hex()


def checked_scale(scale: int) -> int:
    if type(scale) is not int or not 0 <= scale <= 62:
        raise ValueError(f"Unsupported fixed-point scale: {scale!r}")
    return scale


def rescale_integer(value: int, source_scale: int, target_scale: int) -> int:
    """Deterministic exact rebasing, with nearest ties away from zero.

    This is a defined zkInfer boundary operation, not an assertion that every
    internal EZKL rebase uses the same rounding rule.
    """
    delta = checked_scale(target_scale) - checked_scale(source_scale)
    if delta >= 0:
        result = value << delta
    else:
        divisor = 1 << -delta
        magnitude, remainder = divmod(abs(value), divisor)
        if remainder * 2 >= divisor:
            magnitude += 1
        result = -magnitude if value < 0 else magnitude
    if not -SIGNED_LIMIT < result < SIGNED_LIMIT:
        raise ValueError("Rescaled boundary exceeds supported signed i64 range")
    return result


def rescale_felts(values: Sequence[str], source_scale: int, target_scale: int) -> list[str]:
    return [
        encode_felt(rescale_integer(decode_felt(value), source_scale, target_scale))
        for value in values
    ]


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int | None, ...] | None

    def check_size(self, size: int) -> None:
        if size <= 0:
            raise ValueError(f"Empty tensor: {self.name}")
        if self.shape is not None and all(d is not None for d in self.shape):
            if math.prod(self.shape) != size:
                raise ValueError(f"Incorrect size for tensor {self.name}: {size}")


@dataclass(frozen=True)
class BoundaryTensor:
    spec: TensorSpec
    scale: int
    values: list[str]


@dataclass(frozen=True)
class PreparedPartition:
    name: str
    settings_path: Path
    compiled_path: Path
    witness_path: Path
    input_specs: tuple[TensorSpec, ...]
    output_specs: tuple[TensorSpec, ...]
    metrics: dict[str, Any]

    def artifact_identity(self) -> dict[str, Any]:
        return {
            "prepared_version": PREPARATION_VERSION,
            "settings_sha256": self.metrics.get("settings_sha256") or file_digest(self.settings_path),
            "compiled_sha256": self.metrics.get("compiled_sha256") or file_digest(self.compiled_path),
        }


def _spec(value_info) -> TensorSpec:
    tensor_type = value_info.type.tensor_type
    shape = None
    if tensor_type.HasField("shape"):
        shape = tuple(
            int(dim.dim_value) if dim.HasField("dim_value") else None
            for dim in tensor_type.shape.dim
        )
    return TensorSpec(value_info.name, shape)


def partition_specs(partition) -> tuple[tuple[TensorSpec, ...], tuple[TensorSpec, ...]]:
    inputs = tuple(_spec(value) for value in partition.model.graph.input)
    outputs = tuple(_spec(value) for value in partition.model.graph.output)
    for group in (inputs, outputs):
        names = [spec.name for spec in group]
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError(f"Invalid tensor names for {partition.name}")
    return inputs, outputs


def dependency_order(specs: dict[str, tuple[Sequence[TensorSpec], Sequence[TensorSpec]]]):
    """Return stable topological order and a unique producer map."""
    producers: dict[str, str] = {}
    for name, (_, outputs) in specs.items():
        for spec in outputs:
            if spec.name in producers:
                raise ValueError(f"Multiple producers for tensor {spec.name}")
            producers[spec.name] = name
    dependencies = {
        name: {producers[spec.name] for spec in inputs if spec.name in producers}
        for name, (inputs, _) in specs.items()
    }
    ordered = []
    pending = list(specs)
    while pending:
        ready = [name for name in pending if not dependencies[name]]
        if not ready:
            raise ValueError("Partition graph contains a cycle or self dependency")
        for name in ready:
            ordered.append(name)
            pending.remove(name)
        for name in pending:
            dependencies[name].difference_update(ready)
    return ordered, producers


def check_compatible_shape(source: TensorSpec, target: TensorSpec) -> None:
    if source.shape is None or target.shape is None:
        return
    if len(source.shape) != len(target.shape) or any(
        a is not None and b is not None and a != b
        for a, b in zip(source.shape, target.shape)
    ):
        raise ValueError(f"Shape mismatch at boundary {target.name}")


def validate_settings(settings: dict, input_count: int, output_count: int):
    run_args = settings["run_args"]
    for key, expected in (
        ("input_visibility", "public"),
        ("output_visibility", "public"),
        ("param_visibility", "fixed"),
    ):
        if str(run_args[key]).lower() != expected:
            raise ValueError(f"Expected {key}={expected}")
    tolerance = run_args.get("tolerance", 0)
    if (tolerance["val"] if isinstance(tolerance, dict) else tolerance) != 0:
        raise ValueError("Exact linking requires zero circuit output tolerance")
    input_scales = settings["model_input_scales"]
    output_scales = settings["model_output_scales"]
    if len(input_scales) != input_count or len(output_scales) != output_count:
        raise ValueError("EZKL tensor counts differ from ONNX graph interface")
    return [checked_scale(s) for s in input_scales], [
        checked_scale(s) for s in output_scales
    ]


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty prepared artifact: {path}")


def _run_setup(ezkl, model: Path, calibration: Path, directory: Path, candidate: int | None):
    directory.mkdir(parents=True, exist_ok=True)
    settings = directory / "settings.json"
    compiled = directory / "network.compiled"
    args = ezkl.PyRunArgs()
    args.input_visibility = "public"
    args.output_visibility = "public"
    args.param_visibility = "fixed"
    if candidate is not None:
        args.input_scale = candidate
        args.param_scale = candidate
    started = time.perf_counter()
    ezkl.gen_settings(str(model), str(settings), py_run_args=args)
    kwargs = {} if candidate is None else {"scales": [candidate]}
    if not ezkl.calibrate_settings(str(calibration), str(model), str(settings), "resources", **kwargs):
        raise RuntimeError("EZKL calibration returned false")
    calibration_s = time.perf_counter() - started
    started = time.perf_counter()
    ezkl.compile_circuit(str(model), str(compiled), str(settings))
    compilation_s = time.perf_counter() - started
    _require_file(settings)
    _require_file(compiled)
    return settings, compiled, calibration_s, compilation_s


def _backend_identity(ezkl) -> dict:
    identity = {"version": str(getattr(ezkl, "__version__", "unknown"))}
    if getattr(ezkl, "__name__", None) == "ezkl":
        try:
            native = importlib.import_module("ezkl.ezkl")
        except ImportError:
            native = ezkl
        binary = getattr(native, "__file__", None)
        if binary and Path(binary).is_file():
            identity["module_sha256"] = file_digest(binary)
    return identity


def _prepare_setup(ezkl, model, calibration, directory, cache_root, candidate, backend_identity):
    """Publish a complete setup atomically; never reuse a partial setup."""
    logger = logging.getLogger(__name__)
    if cache_root is None:
        logger.info("Circuit cache disabled; calibrating %s", model)
        settings, compiled, cal_s, comp_s = _run_setup(
            ezkl, model, calibration, directory, candidate
        )
        return settings, compiled, cal_s, comp_s, False

    from zkinfer.storage.partition_cache import (
        matches, metadata, partition_lock, publish_group,
    )

    identity = {
        "version": PREPARATION_VERSION,
        "model": file_digest(model),
        "candidate_scale": candidate,
        "visibility": ["public", "public", "fixed"],
    }
    names = ("settings.json", "network.compiled")
    with partition_lock(cache_root) as root:
        info = metadata(root)
        circuit = info.get("circuit", {})
        hashes = circuit.get("sha256", {})
        # Older metadata includes an EZKL binary fingerprint. Ignore it for
        # compatibility; build identity is provenance, not a cache key.
        cached_identity = circuit.get("identity", {})
        if not isinstance(cached_identity, dict):
            cached_identity = {}
        identity_matches = all(cached_identity.get(k) == v for k, v in identity.items()
                               if k in cached_identity) and all(k in cached_identity for k in identity)
        if (identity_matches and set(hashes) == set(names)
                and matches(root, hashes)):
            logger.info("Circuit cache hit: %s", root)
            return root / names[0], root / names[1], 0.0, 0.0, True
        if "circuit" in info:
            changed = [key for key, value in identity.items() if cached_identity.get(key) != value]
            logger.warning("Circuit cache miss at %s: %s; committed slot retained, preparing request-local circuit",
                           root, "identity differs: " + ", ".join(changed) if changed
                           else "artifact hashes or files invalid")
        else:
            logger.info("Circuit cache miss: no committed circuit at %s", root)
        # A committed slot is never replaced, even if incompatible or damaged.
        if "circuit" not in info and not (set(info.get("keys", {}).get("sha256", {})) & set(names)):
            with tempfile.TemporaryDirectory(prefix=".circuit-", dir=root) as staging:
                settings, compiled, cal_s, comp_s = _run_setup(
                    ezkl, model, calibration, Path(staging), candidate
                )
                sources = dict(zip(names, (settings, compiled)))
                hashes = {name: file_digest(path) for name, path in sources.items()}
                publish_group(root, info, "circuit",
                              {"identity": identity, "sha256": hashes, "created_with": backend_identity}, sources)
                return root / names[0], root / names[1], cal_s, comp_s, False
    settings, compiled, cal_s, comp_s = _run_setup(
        ezkl, model, calibration, directory, candidate
    )
    return settings, compiled, cal_s, comp_s, False


def prepare_partitions(
    partitions,
    root: str | Path,
    *,
    cache_root: str | Path | None = None,
    max_propagated_scale: int = 16,
    allow_rescale: bool = True,
    ezkl_module=None,
    on_prepared=None,
    cache_root_for_partition=None,
    model_path_for_partition=None,
) -> dict[str, PreparedPartition]:
    """Prepare witnesses in dependency order and optionally publish each ready partition.

    The callback runs after the witness and its boundary outputs are validated.
    The returned mapping remains available for batch callers.

    Single compatible incoming scales are propagated using restricted
    calibration. Mixed/large scales or a failed restricted calibration can
    fall back to independently calibrated circuits with exact verified
    rebasing. Initial inputs use the splitter's reference data. Calibration
    uses floating-point representatives; actual internal inputs are always
    raw field elements derived from upstream witnesses.
    """
    if ezkl_module is None:
        import ezkl as ezkl_module
    checked_scale(max_propagated_scale)
    backend_identity = _backend_identity(ezkl_module)
    from zkinfer.storage.partition_cache import checked_digest
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    partitions = list(partitions)
    by_name = {partition.name: partition for partition in partitions}
    if not partitions or len(by_name) != len(partitions):
        raise ValueError("Expected unique nonempty partition list")
    specs = {name: partition_specs(partition) for name, partition in by_name.items()}
    order, producers = dependency_order(specs)
    boundaries: dict[str, BoundaryTensor] = {}
    external_inputs: dict[str, list] = {}
    prepared = {}

    for index, name in enumerate(order):
        partition = by_name[name]
        inputs, outputs = specs[name]
        directory = root / f"{index:05d}"
        directory.mkdir(parents=True, exist_ok=False)
        if model_path_for_partition is not None:
            model = Path(model_path_for_partition(partition, directory))
        else:
            model = directory / "model.onnx"
            model.write_bytes(partition.model.SerializeToString())
        references = partition.input_data["input_data"]
        if len(references) != len(inputs):
            raise ValueError(f"Reference input count mismatch for {name}")

        reference_data = []
        incoming_scales = set()
        for spec, reference in zip(inputs, references):
            if spec.name in producers:
                source = boundaries[spec.name]
                check_compatible_shape(source.spec, spec)
                spec.check_size(len(source.values))
                incoming_scales.add(source.scale)
                reference_data.append([
                    decode_felt(value) / (1 << source.scale) for value in source.values
                ])
            else:
                spec.check_size(len(reference))
                previous = external_inputs.setdefault(spec.name, reference)
                if previous != reference:
                    raise ValueError(f"Inconsistent original input {spec.name}")
                reference_data.append(reference)
        calibration = directory / "calibration.json"
        write_json(calibration, {"input_data": reference_data})
        candidate = next(iter(incoming_scales)) if len(incoming_scales) == 1 else None
        reason = ""
        if incoming_scales and (candidate is None or candidate > max_propagated_scale):
            if not allow_rescale:
                raise ValueError(f"Cannot propagate incoming scales {incoming_scales} for {name}")
            reason = "mixed or high incoming scales"
            candidate = None

        partition_cache_root = (cache_root_for_partition(partition) if cache_root_for_partition
                                else Path(cache_root) / file_digest(model) if cache_root is not None else None)
        start_setup = time.perf_counter()
        try:
            setup = _prepare_setup(
                ezkl_module, model, calibration, directory / "setup", partition_cache_root, candidate,
                backend_identity,
            )
        except Exception as exc:
            if candidate is None or not allow_rescale:
                raise
            reason = f"restricted calibration/compilation failed: {exc}"
            setup = _prepare_setup(
                ezkl_module, model, calibration, directory / "fallback_setup", partition_cache_root, None,
                backend_identity,
            )
            candidate = None
        settings_path, compiled_path, cal_s, comp_s, cache_hit = setup
        settings = read_json(settings_path)
        input_scales, output_scales = validate_settings(settings, len(inputs), len(outputs))
        setup_s = time.perf_counter() - start_setup

        exact_inputs = []
        rescaled_count = 0
        for spec, scale, reference in zip(inputs, input_scales, references):
            if spec.name not in producers:
                exact_inputs.append(reference)
                continue
            source = boundaries[spec.name]
            if source.scale != scale:
                if not allow_rescale:
                    raise ValueError(f"Scale mismatch at {spec.name}: {source.scale} != {scale}")
                rescaled_count += 1
            exact_inputs.append(rescale_felts(source.values, source.scale, scale))

        input_path = directory / "input.json"
        witness_path = directory / "witness.json"
        write_json(input_path, {"input_data": exact_inputs})
        started = time.perf_counter()
        ezkl_module.gen_witness(str(input_path), str(compiled_path), str(witness_path))
        witness_s = time.perf_counter() - started
        witness = read_json(witness_path)
        if len(witness["inputs"]) != len(inputs) or len(witness["outputs"]) != len(outputs):
            raise ValueError(f"Witness interface mismatch for {name}")
        for spec, actual, expected in zip(inputs, witness["inputs"], exact_inputs):
            spec.check_size(len(actual))
            if spec.name in producers and [encode_felt(decode_felt(v)) for v in actual] != expected:
                raise ValueError(f"EZKL changed raw boundary input {spec.name}")
        for spec, scale, values in zip(outputs, output_scales, witness["outputs"]):
            spec.check_size(len(values))
            # Check canonical encoding and signed range before forwarding.
            values = [encode_felt(decode_felt(value)) for value in values]
            boundaries[spec.name] = BoundaryTensor(spec, scale, values)

        prepared[name] = PreparedPartition(
            name, settings_path, compiled_path, witness_path, inputs, outputs,
            {
                "setup_time_s": setup_s,
                "calibration_time_s": cal_s,
                "compilation_time_s": comp_s,
                "witness_time_s": witness_s,
                "setup_cache_hit": cache_hit,
                "settings_sha256": checked_digest(settings_path),
                "compiled_sha256": checked_digest(compiled_path),
                "backend_identity": backend_identity,
                "rescaled_inputs": rescaled_count,
                "scale_fallback_reason": reason,
                "restricted_scale": candidate,
                "input_scales": input_scales,
                "output_scales": output_scales,
                "logrows": settings["run_args"]["logrows"],
                "witness_bytes": witness_path.stat().st_size,
                "compiled_bytes": compiled_path.stat().st_size,
            },
        )
        if on_prepared is not None:
            on_prepared(prepared[name])
    return prepared


def public_tensors(proof: dict, settings: dict, input_specs, output_specs):
    """Extract signed public tensors from a verified single-proof payload.

    EZKL orders public inputs before public outputs in one instance column.
    Fixed parameters do not contribute public parameter instances. The
    caller MUST first verify the proof using the expected settings and VK.
    """
    input_scales, output_scales = validate_settings(
        settings, len(input_specs), len(output_specs)
    )
    shapes = settings["model_instance_shapes"]
    specs = [*input_specs, *output_specs]
    if len(shapes) != len(specs):
        raise ValueError("Unexpected EZKL public instance layout")
    instances = proof["instances"]
    if not isinstance(instances, list) or len(instances) != 1 or not isinstance(instances[0], list):
        raise ValueError("Expected exactly one public instance column")
    values = instances[0]
    offset = 0
    tensors = []
    for spec, scale, shape in zip(specs, input_scales + output_scales, shapes):
        if not isinstance(shape, list) or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError("Invalid public instance shape")
        length = math.prod(shape)
        spec.check_size(length)
        raw = values[offset:offset + length]
        if len(raw) != length:
            raise ValueError("Truncated public instance column")
        tensors.append(BoundaryTensor(spec, scale, [
            encode_felt(decode_felt(value)) for value in raw
        ]))
        offset += length
    if offset != len(values):
        raise ValueError("Unexpected trailing public instances")
    return tensors[:len(input_specs)], tensors[len(input_specs):]


def check_public_graph(partitions: dict[str, tuple[list[BoundaryTensor], list[BoundaryTensor]]]):
    """Validate every graph edge, including fan-out and residual connections."""
    specs = {
        name: (tuple(t.spec for t in inputs), tuple(t.spec for t in outputs))
        for name, (inputs, outputs) in partitions.items()
    }
    order, producers = dependency_order(specs)
    outputs_by_name = {}
    edges = 0
    rescaled = 0
    for name in order:
        inputs, outputs = partitions[name]
        for target in inputs:
            if target.spec.name not in producers:
                continue
            source = outputs_by_name[target.spec.name]
            check_compatible_shape(source.spec, target.spec)
            if len(source.values) != len(target.values):
                raise ValueError(f"Boundary length mismatch: {target.spec.name}")
            expected = rescale_felts(source.values, source.scale, target.scale)
            if expected != target.values:
                raise ValueError(f"Public boundary mismatch: {target.spec.name} -> {name}")
            edges += 1
            rescaled += source.scale != target.scale
        outputs_by_name.update({tensor.spec.name: tensor for tensor in outputs})
    return {"checked_edges": edges, "rescaled_edges": rescaled}
