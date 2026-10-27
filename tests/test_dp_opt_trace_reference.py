import importlib
from pathlib import Path

import pytest


@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("verify_trace")


@pytest.fixture
def paired_tokens():
    policies = ["kv_only_headroom", "kv_only_workaware"]
    config = {"policies": policies, "repeats": 2}
    maps = {(repeat, policy): {"a": [1, 2], "b": [3, 4]} for repeat in range(2) for policy in policies}
    references = [{"trace_request_id": "a", "token_ids": [1, 2]}, {"trace_request_id": "b", "token_ids": [3, 4]}]
    return config, maps, references


def test_two_policy_mechanism_does_not_require_no_move(verifier, paired_tokens):
    assert verifier.verify_paired_tokens(*paired_tokens) == "kv_only_headroom"


def test_no_move_remains_preferred_reference(verifier, paired_tokens):
    config, maps, references = paired_tokens
    config["policies"].append("no_move")
    maps.update({(repeat, "no_move"): maps[(repeat, "kv_only_headroom")] for repeat in range(2)})
    assert verifier.verify_paired_tokens(config, maps, references) == "no_move"


def test_explicit_reference_must_be_configured(verifier, paired_tokens):
    config, maps, references = paired_tokens
    config["reference_policy"] = "kv_only_workaware"
    assert verifier.verify_paired_tokens(config, maps, references) == "kv_only_workaware"
    config["reference_policy"] = "no_move"
    with pytest.raises(ValueError, match="configured"):
        verifier.verify_paired_tokens(config, maps, references)


def test_two_policy_tokens_still_must_match(verifier, paired_tokens):
    config, maps, references = paired_tokens
    maps[(1, "kv_only_workaware")] = {"a": [1, 9], "b": [3, 4]}
    with pytest.raises(ValueError, match="Paired policy outputs"):
        verifier.verify_paired_tokens(config, maps, references)


def test_serial_tokens_still_must_match(verifier, paired_tokens):
    config, maps, references = paired_tokens
    references[1]["token_ids"] = [3, 9]
    with pytest.raises(ValueError, match="Full serial reference"):
        verifier.verify_paired_tokens(config, maps, references)
