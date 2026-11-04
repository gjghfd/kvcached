import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from batch_numerics_probe import analyze, compare_tokens, serialize_logprobs, stream_with_logprobs


def record(identifier="request", tokens=None):
    tokens = [10, 20, 30] if tokens is None else tokens
    return {"request_id": identifier, "model": 1, "prompt_tokens": 74,
            "token_ids": tokens, "token_logprobs": [{str(token): -.5, "99": -.51} for token in tokens]}


def test_shared_prefix_logprobs_not_a_correctness_tolerance():
    before, after = record(), record(tokens=[10, 99, 30])
    after["token_logprobs"][1] = {"20": -.501, "99": -.499}
    result = compare_tokens(after, before)
    assert result["first_difference_zero_based"] == 1
    assert result["shared_prefix_tokens"] == 1
    assert result["common_top_logprob_max_abs_delta"] == pytest.approx(.011)
    assert result["numerical_tolerance_acceptance_not_implemented"] is True


def test_reject_truncated_tokens_and_missing_logprobs():
    with pytest.raises(ValueError, match="truncate"):
        compare_tokens(record(tokens=[10]), record())
    with pytest.raises(RuntimeError, match="complete"):
        serialize_logprobs([None])
    with pytest.raises(RuntimeError, match="complete"):
        serialize_logprobs([{}])


def test_drain_original_lengths_and_preserve_serial_failure(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"policies": ["no_move"], "repeats": 2}))
    (tmp_path / "trace.json").write_text(json.dumps({"requests": [
        {"request_id": "request", "prompt_tokens": 74, "output_tokens": 3}]}))
    (tmp_path / "references.json").write_text(json.dumps([{**record(), "trace_request_id": "request"}]))
    error = {"type": "RuntimeError", "message": "Selected trace request differs from its full serial greedy reference"}
    (tmp_path / "error.json").write_text(json.dumps(error))
    for repeat in range(2):
        directory = tmp_path / f"r{repeat}-no_move"
        directory.mkdir()
        (directory / "requests.json").write_text(json.dumps([record(tokens=[10, 99, 30])]))
        (directory / "migrations.json").write_text("[]")
    result = analyze(tmp_path)
    assert result["baseline_serial_divergence_reproduced"]
    assert result["is_algorithm_benefit"] is False
    assert json.loads((tmp_path / "error.json").read_text()) == error
    (tmp_path / "r1-no_move/migrations.json").write_text("[{}]")
    with pytest.raises(ValueError, match="must not migrate"):
        analyze(tmp_path)
    (tmp_path / "r1-no_move/migrations.json").write_text("[]")
    (tmp_path / "error.json").write_text(json.dumps({"type": "RuntimeError", "message": "CUDA OOM"}))
    with pytest.raises(ValueError, match="unrelated"):
        analyze(tmp_path)


def test_stream_keeps_greedy_original_length_and_top_logprobs(monkeypatch):
    settings = {}

    def parameters(**kwargs):
        settings.update(kwargs)
        return kwargs

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=parameters))

    class Engine:
        async def generate(self, prompt, params, identifier, **kwargs):
            assert prompt == {"prompt_token_ids": [1, 2]}
            assert kwargs == {"data_parallel_rank": 1}
            yield SimpleNamespace(outputs=[SimpleNamespace(
                token_ids=[10, 20, 30], logprobs=[{token: SimpleNamespace(logprob=-.5)}
                                                 for token in (10, 20, 30)])])

    result = asyncio.run(stream_with_logprobs(Engine(), "request", [1, 2], 3, 1, 1, "probe", 0))
    assert settings == {"temperature": 0, "min_tokens": 3, "max_tokens": 3,
                        "ignore_eos": True, "logprobs": 2}
    assert result["token_ids"] == [10, 20, 30]
    assert result["token_logprobs"] == [{"10": -.5}, {"20": -.5}, {"30": -.5}]
