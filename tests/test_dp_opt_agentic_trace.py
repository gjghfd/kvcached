import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("agentic_trace"), importlib.import_module("live_trace")


def test_two_rates_and_distinct_marks_are_frozen(modules):
    generator, runner = modules
    code = [{"prompt_tokens": 100084, "output_tokens": 64, "prompt_key": "swe"}]
    conv = [{"prompt_tokens": 4096, "output_tokens": 512, "count": 5}]
    trace = generator.generate_mixed_trace(code, conv, .02, 1.8, 1200, 20261021)
    assert trace == generator.generate_mixed_trace(code, conv, .02, 1.8, 1200, 20261021)
    assert trace["model_rates_rps"] == [.02, 1.8]
    assert all(mark["prompt_tokens"] == (100084 if mark["model"] == 0 else 4096)
               for mark in trace["requests"])
    assert {mark["model"] for mark in trace["requests"]} == {0, 1}


def test_long_marks_cannot_use_old_32k_limit(modules):
    generator, runner = modules
    with pytest.raises(ValueError, match="full sequence"):
        generator.generate_mixed_trace([{"prompt_tokens": 100000, "output_tokens": 64}],
                                       [{"prompt_tokens": 2048, "output_tokens": 512}],
                                       .02, 1, 1200, 42, max_model_len=32768)


def test_agent_inputs_never_fallback_to_synthetic_length_only(modules):
    generator, runner = modules
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2])
    with pytest.raises(ValueError, match="verified prompt bank"):
        runner.prepare_prompts({"requests": [{"prompt_tokens": 100000, "prompt_key": "swe"}]},
                               tokenizer)


def test_prompt_keys_distinguish_same_length_agent_text(modules):
    generator, runner = modules
    prompts = {"swe-a": [1, 2], "swe-b": [3, 4], "2": [5, 6]}
    assert runner.request_prompt(prompts, {"prompt_tokens": 2, "prompt_key": "swe-a"}) == [1, 2]
    assert runner.request_prompt(prompts, {"prompt_tokens": 2, "prompt_key": "swe-b"}) == [3, 4]
    assert runner.request_prompt(prompts, {"prompt_tokens": 2}) == [5, 6]
