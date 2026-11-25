"""Unit tests for exact integer boundaries; no EZKL wheel required."""
from types import SimpleNamespace

import pytest

from zkinfer.backends.ezkl.preparation import (
    FIELD_MODULUS, BoundaryTensor, TensorSpec, check_public_graph,
    decode_felt, dependency_order, encode_felt, public_tensors,
    rescale_felts, rescale_integer,
)


def tensor(name, scale, values):
    return BoundaryTensor(
        TensorSpec(name, (len(values),)), scale, [encode_felt(v) for v in values]
    )


@pytest.mark.parametrize("value", [0, 1, -1, 123456, -123456, (1 << 127) - 1])
def test_field_roundtrip(value):
    assert decode_felt(encode_felt(value)) == value


def test_reject_noncanonical_and_overflow():
    with pytest.raises(ValueError):
        decode_felt(FIELD_MODULUS.to_bytes(32, "little").hex())
    with pytest.raises(ValueError):
        encode_felt(1 << 127)


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
