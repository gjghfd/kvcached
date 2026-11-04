import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
import audit_public_response_completion as completion


@pytest.mark.parametrize("response,status", [
    ("ordinary answer", "no_known_thought_marker"),
    ("<think>reasoning", "unclosed_thought_marker"),
    ("<think>reasoning</think>answer", "closed_thought_marker"),
    ("<think>one</think><think>two", "unclosed_thought_marker"),
    ("<|begin_of_thought|>reasoning", "unclosed_thought_marker"),
    ("<|begin_of_thought|>reasoning<|end_of_thought|>answer", "closed_thought_marker"),
    ("<think>first<|end_of_thought|>", "unclosed_thought_marker"),
    ("</think>answer", "no_known_thought_marker"),
])
def test_marker_balance_is_not_completion_claim(response, status):
    assert completion.marker_status(response) == status


def source_fixture(tmp_path):
    path = tmp_path / "source.jsonl"
    response = "<think>source reasoning"
    path.write_text(json.dumps({"messages": [
        {"role": "user", "content": "question"}, {"role": "assistant", "content": response}
    ]}) + "\n")
    source = {"label": "fixture", "dataset": "official/fixture", "revision": "fixed",
              "path": "data/source.jsonl", "destination": path.name,
              "size": path.stat().st_size, "sha256": completion.digest(path)}
    return source, response


def test_full_source_audit_keeps_original_response(tmp_path):
    source, _ = source_fixture(tmp_path)
    result = completion.audit(tmp_path, [source])
    assert result["responses"] == 1
    assert result["thought_marker_status"] == {"unclosed_thought_marker": 1}
    assert result["natural_eos_or_semantic_completion_verified"] is False
    assert result["sampling_changed"] is False


def test_raw_sha_mismatch_fails(tmp_path):
    source, _ = source_fixture(tmp_path)
    source["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Raw public source size/SHA"):
        completion.audit(tmp_path, [source])


def bank_fixture(tmp_path, monkeypatch, source, response):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    monkeypatch.setattr(completion, "validate", lambda _: {"sources": [source]})
    entry = {"dataset": source["dataset"], "revision": source["revision"],
             "source_file": source["path"], "source_row": 0, "assistant_index": 1,
             "original_response_sha256": hashlib.sha256(response.encode()).hexdigest()}
    (inputs / "prompt-bank.json").write_text(json.dumps({"prompts": {"source": entry}}))
    return inputs, entry


def test_bank_matches_source_path_not_local_destination(tmp_path, monkeypatch):
    source, response = source_fixture(tmp_path)
    inputs, _ = bank_fixture(tmp_path, monkeypatch, source, response)
    result = completion.audit(tmp_path, [source], inputs)
    assert result["bank_responses_sha_verified"] == 1
    assert result["bank_thought_marker_status"] == {"unclosed_thought_marker": 1}


def test_unreconstructed_bank_response_fails(tmp_path, monkeypatch):
    source, response = source_fixture(tmp_path)
    inputs, entry = bank_fixture(tmp_path, monkeypatch, source, response)
    entry["source_row"] = 10
    (inputs / "prompt-bank.json").write_text(json.dumps({"prompts": {"source": entry}}))
    with pytest.raises(ValueError, match="every frozen public response"):
        completion.audit(tmp_path, [source], inputs)


def test_changed_original_response_fails(tmp_path, monkeypatch):
    source, response = source_fixture(tmp_path)
    inputs, entry = bank_fixture(tmp_path, monkeypatch, source, response)
    entry["original_response_sha256"] = "0" * 64
    (inputs / "prompt-bank.json").write_text(json.dumps({"prompts": {"source": entry}}))
    with pytest.raises(ValueError, match="bank response SHA"):
        completion.audit(tmp_path, [source], inputs)
