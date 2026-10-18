import importlib.util
import io
import json
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture
def single_benchmark(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "experiments/dp_opt"
    monkeypatch.syspath_prepend(str(root))
    spec = importlib.util.spec_from_file_location("dp_opt_single_benchmark", root / "single_model_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_stream(token_ids=(10, 20), usage=None, finish_reason="length", done=True):
    chunks = [{"choices": [{"index": 0, "text": "answer", "token_ids": [token_id]}]}
              for token_id in token_ids]
    chunks.append({"choices": [{"index": 0, "text": "", "finish_reason": finish_reason}]})
    chunks.append({"choices": [], "usage": usage or {"prompt_tokens": 3, "completion_tokens": 2}})
    data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    if done:
        data += "data: [DONE]\n\n"
    return io.BytesIO(data.encode())


def test_streaming_uses_one_model_and_explicit_rank(single_benchmark):
    with mock.patch.object(single_benchmark.urllib.request, "urlopen", return_value=make_stream()) as send:
        record = single_benchmark.stream_completion({"url": "http://localhost", "model": "tenant-1"},
                                                    1, [1, 2, 3], 2)
    query = send.call_args.args[0]
    assert query.get_header("X-data-parallel-rank") == "1"
    assert json.loads(query.data)["model"] == "tenant-1"
    assert record["token_ids"] == [10, 20]
    assert record["usage"]["completion_tokens"] == 2
    assert record["ttft_s"] >= 0
    assert record["tpot_s"] >= 0


@pytest.mark.parametrize("stream_args", [
    {"token_ids": (10,)}, {"usage": {"prompt_tokens": 4, "completion_tokens": 2}},
    {"finish_reason": "stop"}, {"done": False},
])
def test_invalid_stream_is_rejected(single_benchmark, stream_args):
    with mock.patch.object(single_benchmark.urllib.request, "urlopen", return_value=make_stream(**stream_args)):
        with pytest.raises(RuntimeError):
            single_benchmark.stream_completion({"url": "http://localhost", "model": "tenant-1"},
                                               0, [1, 2, 3], 2)


def test_rank_schedule_preserves_unequal_request_counts(single_benchmark):
    assert single_benchmark.rank_schedule([3, 1]) == [0, 1, 0, 0]
    assert single_benchmark.rank_schedule([0, 2]) == [1, 1]
    with pytest.raises(ValueError):
        single_benchmark.rank_schedule([0, 0])


def test_batch_only_targets_its_own_model(single_benchmark):
    def complete(endpoint, rank, prompt, output_tokens, start_event):
        assert start_event.wait(timeout=5)
        return {"status": "passed", "model": endpoint["model"], "rank": rank}

    with mock.patch.object(single_benchmark, "stream_completion", side_effect=complete):
        records = single_benchmark.run_batch({"model": "tenant-2"}, [1], 2, [3, 1], 2)
    assert {record["model"] for record in records} == {"tenant-2"}
    assert [record["rank"] for record in records] == [0, 1, 0, 0]


def test_distribution_uses_interpolated_percentiles(single_benchmark):
    values = single_benchmark.distribution([1, 3, None])
    assert values["mean"] == 2
    assert values["p50"] == 2
    assert values["p95"] == pytest.approx(2.9)


def test_failed_requests_are_counted(single_benchmark):
    assert single_benchmark.summarize([{"status": "failed"}]) == {
        "successful_requests": 0, "failed_requests": 1,
    }
