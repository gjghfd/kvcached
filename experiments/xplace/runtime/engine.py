import json
import time
from pathlib import Path

SPARSE_CAPTURE_SIZES = [1, 8, 16, 32, 64, 128, 256]


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def engine_arguments(checkpoint, port, cache_bytes, max_model_len, num_ranks, gpu_memory_utilization=0.43):
    from vllm.engine.arg_utils import AsyncEngineArgs

    return AsyncEngineArgs(model=checkpoint, dtype="bfloat16", tensor_parallel_size=1,
                           data_parallel_size=num_ranks, data_parallel_size_local=num_ranks,
                           data_parallel_backend="mp", data_parallel_address="127.0.0.1",
                           data_parallel_rpc_port=port, enable_expert_parallel=True, enable_ep_weight_filter=True,
                           all2all_backend="allgather_reducescatter", moe_backend="triton",
                           worker_cls="worker.VerificationWorker", scheduler_cls="live_scheduler.PlacementScheduler",
                           max_model_len=max_model_len, max_num_seqs=256, max_num_batched_tokens=4096,
                           kv_cache_memory_bytes=cache_bytes, gpu_memory_utilization=gpu_memory_utilization,
                           enable_prefix_caching=False, async_scheduling=False, enable_dbo=False, enable_eplb=False,
                           compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                               "cudagraph_capture_sizes": list(SPARSE_CAPTURE_SIZES)})


async def stream(engine, request_id, prompt, output_tokens, rank, model, role, arrived_at, stride=32, progress=None):
    """Stream one greedy request; cadence is sampled every `stride` tokens to bound log size.

    `progress` (optional dict) is updated in place as tokens arrive, so a censored trial can record partial progress.
    """
    import hashlib

    from vllm import SamplingParams

    submitted_at = time.monotonic()
    cadence, latest, first_token_at, previous, maximum_gap, bucket = [], None, None, None, 0.0, -1
    if progress is not None:
        progress.update(submitted_at=submitted_at, count=0, first_token_at=None, last_at=None, max_gap=0.0,
                        cadence=cadence)
    async for output in engine.generate(
            {"prompt_token_ids": prompt}, SamplingParams(temperature=0, min_tokens=output_tokens,
                                                         max_tokens=output_tokens, ignore_eos=True),
            request_id, data_parallel_rank=rank):
        now = time.monotonic()
        latest = output
        count = len(output.outputs[0].token_ids)
        if count and first_token_at is None:
            first_token_at = now
        if previous is not None and first_token_at is not None:
            maximum_gap = max(maximum_gap, now - previous)
        previous = now
        if count // stride != bucket or count == output_tokens:
            cadence.append([now, count])
            bucket = count // stride
        if progress is not None:
            progress.update(count=count, first_token_at=first_token_at, last_at=now, max_gap=maximum_gap)
    if latest is None or len(latest.outputs[0].token_ids) != output_tokens or first_token_at is None:
        raise RuntimeError(f"Request {request_id} failed exact output-length/first-token check")
    tokens = list(latest.outputs[0].token_ids)
    return {"request_id": request_id, "model": model, "role": role, "rank_initial": rank,
            "prompt_tokens": len(prompt), "output_tokens": output_tokens, "arrived_at": arrived_at,
            "submitted_at": submitted_at, "first_token_at": first_token_at, "completed_at": time.monotonic(),
            "token_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
            "first_tokens": tokens[:16], "max_token_gap_s": maximum_gap, "cadence_stride": stride,
            "cadence": cadence}


def load_bank(path, expected_sha256=None):
    import hashlib

    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Prompt bank checksum {digest} differs from frozen trace {expected_sha256}")
    bank = json.loads(payload)
    return {key: entry["token_ids"] for key, entry in bank["prompts"].items()}, digest
