"""Unit tests for exact integer boundaries; no EZKL wheel required."""
from types import SimpleNamespace

import pytest

from zkinfer.backends.ezkl.preparation import (
    FIELD_MODULUS, BoundaryTensor, TensorSpec, check_public_graph,
    decode_felt, dependency_order, encode_felt, public_tensors,
    rescale_felts, rescale_integer, prepare_partitions, read_json, write_json,
)


def tensor(name, scale, values):
    return BoundaryTensor(
        TensorSpec(name, (len(values),)), scale, [encode_felt(v) for v in values]
    )


@pytest.mark.parametrize("value", [0, 1, -1, 123456, -123456, (1 << 63) - 1])
def test_field_roundtrip(value):
    assert decode_felt(encode_felt(value)) == value


def test_reject_noncanonical_and_overflow():
    with pytest.raises(ValueError):
        decode_felt(FIELD_MODULUS.to_bytes(32, "little").hex())
    with pytest.raises(ValueError):
        encode_felt(1 << 63)


@pytest.mark.parametrize("value,expected", [(3, 2), (-3, -2), (2, 1), (-2, -1), (1, 1)])
def test_rebase_rounding(value, expected):
    assert rescale_integer(value, 1, 0) == expected


def test_dag_fanout_and_rescaling():
    a = tensor("a", 2, [4, -6])
    b = tensor("b", 1, [2, -3])
    c = tensor("c", 2, [7, 8])
    d = tensor("d", 2, [11, 12])
    external = tensor("x", 2, [4, 4])
    parts = {
        "third": ([a, c], [d]),
        "second": ([tensor("a", 1, [2, -3])], [c]),
        "first": ([external], [a]),
    }
    assert check_public_graph(parts) == {"checked_edges": 3, "rescaled_edges": 1}
    parts["second"] = ([tensor("a", 1, [2, -2])], [c])
    with pytest.raises(ValueError, match="Public boundary mismatch"):
        check_public_graph(parts)


def test_cycles_and_duplicate_producers_rejected():
    a, b = TensorSpec("a", (1,)), TensorSpec("b", (1,))
    with pytest.raises(ValueError, match="cycle"):
        dependency_order({"one": ([b], [a]), "two": ([a], [b])})
    with pytest.raises(ValueError, match="Multiple producers"):
        dependency_order({"one": ([b], [a]), "two": ([b], [a])})


def test_public_instance_layout():
    settings = {
        "run_args": {
            "input_visibility": "Public", "output_visibility": "Public",
            "param_visibility": "Fixed", "tolerance": {"val": 0},
        },
        "model_input_scales": [3],
        "model_output_scales": [6],
        "model_instance_shapes": [[1, 2], [1, 1]],
    }
    proof = {"instances": [[encode_felt(2), encode_felt(-3), encode_felt(4)]]}
    ins, outs = public_tensors(
        proof, settings, [TensorSpec("x", (1, 2))], [TensorSpec("y", (1, 1))]
    )
    assert ins[0].values == [encode_felt(2), encode_felt(-3)]
    assert outs[0].scale == 6
    proof["instances"][0].append(encode_felt(0))
    with pytest.raises(ValueError, match="trailing"):
        public_tensors(
            proof, settings, [TensorSpec("x", (1, 2))], [TensorSpec("y", (1, 1))]
        )


class FakeValueInfo:
    """Minimal stand-in for ONNX value info; no ONNX or EZKL dependency."""
    def __init__(self, name):
        self.name = name
        dimension = SimpleNamespace(dim_value=1, HasField=lambda _: True)
        tensor_type = SimpleNamespace(
            shape=SimpleNamespace(dim=[dimension]), HasField=lambda _: True
        )
        self.type = SimpleNamespace(tensor_type=tensor_type)


def fake_partition(name, ins, outs, reference):
    graph = SimpleNamespace(
        input=[FakeValueInfo(v) for v in ins],
        output=[FakeValueInfo(v) for v in outs],
    )
    model = SimpleNamespace(
        graph=graph, SerializeToString=lambda: name.encode()
    )
    return SimpleNamespace(
        name=name, model=model, input_data={"input_data": reference}
    )


class FakeEZKL:
    """Deterministic mock backend for orchestration, not a crypto test."""
    __version__ = "test"
    def __init__(self):
        self.calibrations = []
        self.witness_calls = []

    def PyRunArgs(self):
        return SimpleNamespace(input_scale=7, param_scale=7)

    def gen_settings(self, model, settings, py_run_args):
        from pathlib import Path
        name = Path(model).read_text()
        scale = py_run_args.input_scale
        input_count = 2 if name == "join" else 1
        output_scale = scale + 2 if name == "first" else scale
        write_json(settings, {
            "run_args": {
                "input_visibility": py_run_args.input_visibility,
                "output_visibility": py_run_args.output_visibility,
                "param_visibility": py_run_args.param_visibility,
                "tolerance": {"val": 0}, "logrows": 8,
            },
            "model_input_scales": [scale] * input_count,
            "model_output_scales": [output_scale],
            "model_instance_shapes": [[1]] * (input_count + 1),
        })
        return True

    def calibrate_settings(self, data, model, settings, target, **kwargs):
        self.calibrations.append(kwargs.get("scales"))
        return True

    def compile_circuit(self, model, compiled, settings):
        from pathlib import Path
        Path(compiled).write_bytes(Path(settings).read_bytes())
        return True

    def gen_witness(self, data, compiled, witness):
        settings = read_json(compiled)
        raw = read_json(data)["input_data"]
        inputs = [
            [v if isinstance(v, str) else encode_felt(round(v * (1 << scale))) for v in values]
            for values, scale in zip(raw, settings["model_input_scales"])
        ]
        self.witness_calls.append(inputs)
        scale = settings["model_output_scales"][0]
        result = sum(
            rescale_integer(decode_felt(values[0]), inscale, scale)
            for values, inscale in zip(inputs, settings["model_input_scales"])
        )
        write_json(witness, {"inputs": inputs, "outputs": [[encode_felt(result)]]})
        return True


def test_prepare_dag_and_cache_without_duplicate_witnesses(tmp_path):
    # The deliberately wrong float reference at internal boundaries must not
    # become the actual proving input.
    partitions = [
        fake_partition("join", ["a", "b"], ["y"], [[999.0], [999.0]]),
        fake_partition("second", ["a"], ["b"], [[999.0]]),
        fake_partition("first", ["x"], ["a"], [[0.5]]),
    ]
    ezkl = FakeEZKL()
    cold = prepare_partitions(
        partitions, tmp_path / "cold", cache_root=tmp_path / "cache", ezkl_module=ezkl
    )
    assert list(cold) == ["first", "second", "join"]
    assert len(ezkl.witness_calls) == 3
    assert ezkl.calibrations == [None, [9], [9]]
    assert read_json(cold["first"].witness_path)["outputs"] == ezkl.witness_calls[1]
    assert ezkl.witness_calls[2] == [ezkl.witness_calls[1][0]] * 2

    warm = prepare_partitions(
        partitions, tmp_path / "warm", cache_root=tmp_path / "cache", ezkl_module=ezkl
    )
    assert len(ezkl.calibrations) == 3
    assert len(ezkl.witness_calls) == 6
    assert all(part.metrics["setup_cache_hit"] for part in warm.values())


def test_high_scale_fallback_is_explicit_and_exact(tmp_path):
    partitions = [
        fake_partition("first", ["x"], ["a"], [[0.5]]),
        fake_partition("second", ["a"], ["b"], [[999.0]]),
    ]
    backend = FakeEZKL()
    prepared = prepare_partitions(
        partitions, tmp_path / "allowed", max_propagated_scale=7, ezkl_module=backend
    )
    assert prepared["second"].metrics["rescaled_inputs"] == 1
    assert decode_felt(backend.witness_calls[1][0][0]) == 64
    with pytest.raises(ValueError, match="Cannot propagate"):
        prepare_partitions(
            partitions, tmp_path / "strict", max_propagated_scale=7,
            allow_rescale=False, ezkl_module=FakeEZKL()
        )
