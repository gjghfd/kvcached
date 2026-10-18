import importlib.util
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture
def benchmark_module():
    path = Path(__file__).resolve().parents[1] / "experiments/dp_opt/benchmark.py"
    spec = importlib.util.spec_from_file_location("dp_opt_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_requests_include_explicit_dp_rank(benchmark_module):
    response = mock.MagicMock()
    response.__enter__.return_value.read.return_value = b'{"ok": true}'
    with mock.patch.object(benchmark_module.urllib.request, "urlopen", return_value=response) as send:
        assert benchmark_module.request_json("http://localhost", "/v1/models", rank=1) == {"ok": True}
    query = send.call_args.args[0]
    assert query.get_header("X-data-parallel-rank") == "1"


@pytest.mark.parametrize("token_ids,usage", [([], 0), ([1, 2], 1)])
def test_inconsistent_generation_is_rejected(benchmark_module, token_ids, usage):
    response = {"choices": [{"token_ids": token_ids, "text": "answer"}],
                "usage": {"completion_tokens": usage}}
    with mock.patch.object(benchmark_module, "request_json", return_value=response):
        with pytest.raises(RuntimeError, match="token counts"):
            benchmark_module.completion({"url": "http://localhost", "model": "model"},
                                        0, [1], 2, True)


def test_concurrent_batch_balances_both_models_and_ranks(benchmark_module):
    endpoints = [{"model": "first"}, {"model": "second"}]

    def complete(endpoint, rank, prompt, output_tokens, fixed_length, barrier):
        barrier.wait(timeout=5)
        return endpoint["model"], rank

    with mock.patch.object(benchmark_module, "completion", side_effect=complete):
        records = benchmark_module.concurrent_batch(endpoints, [1], 2, True, 16)
    for model in ("first", "second"):
        for rank in (0, 1):
            assert records.count((model, rank)) == 4
