import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from benchmark import completion, request_json


def rank_schedule(rank_requests):
    if len(rank_requests) != 2 or min(rank_requests) < 0 or sum(rank_requests) < 1:
        raise ValueError("Two nonnegative rank counts with a positive total are required")
    return [rank for request_index in range(max(rank_requests))
            for rank, count in enumerate(rank_requests) if request_index < count]


def stream_completion(endpoint, rank, prompt, output_tokens, start_event=None, request_id=None):
    if start_event is not None and not start_event.wait(timeout=60):
        raise RuntimeError("Client request start gate timed out")
    body = {
        "model": endpoint["model"], "prompt": prompt, "temperature": 0,
        "seed": 42, "max_tokens": output_tokens, "min_tokens": output_tokens,
        "ignore_eos": True, "return_token_ids": True, "stream": True,
        "stream_options": {"include_usage": True},
    }
    query = urllib.request.Request(
        endpoint["url"] + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-data-parallel-rank": str(rank)},
    )
    if request_id is not None:
        query.add_header("X-Request-Id", request_id)
    started_ns = time.time_ns()
    started = time.perf_counter()
    token_ids = []
    token_events = []
    texts = []
    usage = None
    finish_reason = None
    done = False
    first_token = None
    last_token = None
    try:
        with urllib.request.urlopen(query, timeout=900) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(f"Streaming error: {chunk['error']}")
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if choice["index"] != 0:
                        raise RuntimeError("Unexpected streaming choice index")
                    new_tokens = choice.get("token_ids") or []
                    if new_tokens:
                        received = time.perf_counter()
                        if first_token is None:
                            first_token = received
                        last_token = received
                        token_ids.extend(new_tokens)
                        token_events.append({"time_ns": time.time_ns(), "tokens": len(new_tokens)})
                    texts.append(choice.get("text", ""))
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error
    ended = time.perf_counter()
    ended_ns = time.time_ns()
    if not done or first_token is None or finish_reason != "length":
        raise RuntimeError("Incomplete stream or unexpected finish reason")
    if (len(token_ids) != output_tokens or usage is None
            or usage.get("completion_tokens") != output_tokens
            or usage.get("prompt_tokens") != len(prompt)):
        raise RuntimeError("Streaming token counts do not match the requested lengths")
    return {
        "status": "passed", "model": endpoint["model"], "rank": rank,
        "request_id": request_id,
        "start_ns": started_ns, "end_ns": ended_ns,
        "latency_s": ended - started, "ttft_s": first_token - started,
        "tpot_s": (last_token - first_token) / (output_tokens - 1)
        if output_tokens > 1 else None,
        "usage": usage, "finish_reason": finish_reason,
        "token_ids": token_ids, "text": "".join(texts), "token_events": token_events,
        "max_tokens_per_chunk": max(event["tokens"] for event in token_events),
    }


def run_batch(endpoint, prompt, output_tokens, rank_requests, concurrency):
    schedule = rank_schedule(rank_requests)
    if concurrency < 1:
        raise ValueError("Concurrency must be positive")
    start_event = threading.Event()
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        pending = [(rank, executor.submit(
            stream_completion, endpoint, rank, prompt, output_tokens, start_event,
        )) for rank in schedule]
        start_event.set()
        for rank, future in pending:
            try:
                records.append(future.result())
            except Exception as error:
                records.append({"status": "failed", "model": endpoint["model"],
                                "rank": rank, "error": str(error)})
    return records


def distribution(values):
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"mean": sum(ordered) / len(ordered), "p50": percentile(0.5),
            "p95": percentile(0.95), "p99": percentile(0.99),
            "min": ordered[0], "max": ordered[-1]}


def summarize(records):
    successful = [record for record in records if record["status"] == "passed"]
    result = {"successful_requests": len(successful),
              "failed_requests": len(records) - len(successful)}
    if not successful:
        return result
    span_s = (max(record["end_ns"] for record in successful)
              - min(record["start_ns"] for record in successful)) / 1e9
    tokens = sum(record["usage"]["completion_tokens"] for record in successful)
    result.update({
        "completion_tokens": tokens, "request_span_s": span_s,
        "output_tokens_per_second": tokens / span_s,
        "requests_per_second": len(successful) / span_s,
        "latency_s": distribution(record["latency_s"] for record in successful),
        "ttft_s": distribution(record["ttft_s"] for record in successful),
        "tpot_s": distribution(record["tpot_s"] for record in successful),
        "rank_counts": {str(rank): sum(record["rank"] == rank for record in successful)
                        for rank in (0, 1)},
        "max_tokens_per_chunk": max(record["max_tokens_per_chunk"] for record in successful),
    })
    return result


def save_json(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--service", choices=("instance-1", "instance-2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sync-root", type=Path)
    parser.add_argument("--rank-requests", nargs=2, type=int, default=[8, 8])
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=256)
    args = parser.parse_args()
    try:
        rank_schedule(args.rank_requests)
    except ValueError as error:
        parser.error(str(error))
    if min(args.concurrency, args.rounds, args.input_tokens, args.output_tokens) < 1:
        parser.error("Concurrency, rounds and token lengths must be positive")
    endpoint = {"url": args.url, "model": args.model}
    models = request_json(args.url, "/v1/models")
    if args.model not in {model["id"] for model in models["data"]}:
        raise RuntimeError("Unexpected served model")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    arithmetic = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with one short sentence: what is 2+2?"}],
        tokenize=True, add_generation_prompt=True,
    )
    if not isinstance(arithmetic, list):
        arithmetic = arithmetic["input_ids"]
    serial = [completion(endpoint, rank, arithmetic, 128, False) for rank in (0, 1)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        pending = [executor.submit(completion, endpoint, rank, arithmetic, 128, False)
                   for rank in (0, 1, 0, 1)]
        concurrent_arithmetic = [future.result() for future in pending]
    for record in serial + concurrent_arithmetic:
        if (not re.search(r"\b(?:4|four)\b", record["text"], re.IGNORECASE)
                or record["token_ids"] != serial[record["rank"]]["token_ids"]):
            raise RuntimeError("Arithmetic correctness or same-rank determinism failed")
    prompt = tokenizer.encode("Summarize this example briefly. " * args.input_tokens,
                              add_special_tokens=False)[:args.input_tokens]
    if len(prompt) != args.input_tokens:
        raise RuntimeError("Could not construct exact-length token prompt")
    warmup = run_batch(endpoint, prompt, 32, [2, 2], 4)
    if any(record["status"] != "passed" for record in warmup):
        save_json(args.output, {"status": "failed", "warmup": warmup})
        raise RuntimeError("Streaming warmup failed")
    report = {
        "status": "running", "client_pid": os.getpid(), "service": args.service,
        "endpoint": endpoint, "rank_requests": args.rank_requests,
        "concurrency": args.concurrency, "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens, "serial_arithmetic": serial,
        "concurrent_arithmetic": concurrent_arithmetic, "warmup": warmup, "rounds": [],
        "scope": "Independent per-model closed-loop bursts; client-observed TTFT/TPOT",
    }
    for round_index in range(args.rounds):
        if args.sync_root is not None:
            args.sync_root.mkdir(parents=True, exist_ok=True)
            (args.sync_root / f"{args.service}-round-{round_index}.ready").write_text(str(os.getpid()))
            gate = args.sync_root / f"round-{round_index}.go"
            deadline = time.monotonic() + 1200
            while not gate.exists():
                if time.monotonic() > deadline:
                    raise RuntimeError("Cross-client synchronization timed out")
                time.sleep(0.05)
        records = run_batch(endpoint, prompt, args.output_tokens,
                            args.rank_requests, args.concurrency)
        report["rounds"].append({"round": round_index, "requests": records,
                                 "summary": summarize(records)})
        save_json(args.output, report)
        print(json.dumps({"service": args.service, "round": round_index,
                          "summary": report["rounds"][-1]["summary"]}), flush=True)
        if any(record["status"] != "passed" for record in records):
            report["status"] = "failed"
            save_json(args.output, report)
            raise RuntimeError("Measured requests failed; see client report")
    all_records = [record for round_result in report["rounds"]
                   for record in round_result["requests"]]
    active_s = sum(round_result["summary"]["request_span_s"] for round_result in report["rounds"])
    report["summary"] = summarize(all_records)
    report["summary"]["active_round_span_s"] = active_s
    report["summary"]["active_output_tokens_per_second"] = (
        report["summary"]["completion_tokens"] / active_s
    )
    report["status"] = "passed"
    save_json(args.output, report)


if __name__ == "__main__":
    main()
