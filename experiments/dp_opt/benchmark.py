import argparse
import concurrent.futures
import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def request_json(base_url, path, body=None, rank=None):
    headers = {"Content-Type": "application/json"}
    if rank is not None:
        headers["X-data-parallel-rank"] = str(rank)
    query = urllib.request.Request(
        base_url + path, headers=headers,
        data=None if body is None else json.dumps(body).encode(),
    )
    try:
        with urllib.request.urlopen(query, timeout=900) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error


def completion(endpoint, rank, prompt, output_tokens, fixed_length, barrier=None):
    if barrier is not None:
        barrier.wait(timeout=60)
    started_ns = time.time_ns()
    response = request_json(endpoint["url"], "/v1/completions", {
        "model": endpoint["model"], "prompt": prompt, "temperature": 0,
        "seed": 42, "max_tokens": output_tokens, "return_token_ids": True,
        "min_tokens": output_tokens if fixed_length else 0,
        "ignore_eos": fixed_length,
    }, rank=rank)
    ended_ns = time.time_ns()
    choice = response["choices"][0]
    token_ids = choice.get("token_ids")
    usage = response.get("usage", {})
    if not token_ids or len(token_ids) != usage.get("completion_tokens"):
        raise RuntimeError(f"Missing/inconsistent token counts: {response}")
    if fixed_length and len(token_ids) != output_tokens:
        raise RuntimeError(f"Incorrect fixed output length: {response}")
    return {
        "model": endpoint["model"], "rank": rank,
        "start_ns": started_ns, "end_ns": ended_ns,
        "latency_s": (ended_ns - started_ns) / 1e9,
        "usage": usage, "token_ids": token_ids, "text": choice["text"],
    }


def concurrent_batch(endpoints, prompt, output_tokens, fixed_length, count):
    barrier = threading.Barrier(count)
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as executor:
        futures = [executor.submit(
            completion, endpoints[request_index % 2], (request_index // 2) % 2,
            prompt, output_tokens, fixed_length, barrier,
        ) for request_index in range(count)]
        return [future.result() for future in futures]


def read_memory(names):
    from kvcached.cli.utils import get_kv_cache_limit

    result = {}
    for name in names:
        memory = get_kv_cache_limit(name)
        if memory is None:
            raise RuntimeError(f"KV control segment disappeared: {name}")
        result[name] = {
            "capacity_bytes": memory.total_size,
            "mapped_live_bytes": memory.used_size,
            "reserved_backing_bytes": memory.prealloc_size,
        }
    return result


def verify_workers(root):
    workers = [json.loads(path.read_text()) for path in sorted(root.glob("worker-*.json"))]
    if len(workers) != 4 or len({worker["pid"] for worker in workers}) != 4:
        raise RuntimeError("Expected four independent workers")
    if len({worker["ipc_name"] for worker in workers}) != 4:
        raise RuntimeError("Model/DP-rank KV namespaces are not isolated")
    for service in ("instance-1", "instance-2"):
        ranks = sorted([worker for worker in workers if worker["service"] == service],
                       key=lambda worker: worker["dp_rank"])
        if [worker["dp_rank"] for worker in ranks] != [0, 1]:
            raise RuntimeError("Missing DP rank")
        if len({worker["gpu_uuid"] for worker in ranks}) != 2:
            raise RuntimeError("DP workers must use distinct GPUs")
        for worker in ranks:
            for group, size in (("tp", 1), ("dp", 2), ("ep", 2)):
                if worker["groups"][group]["world_size"] != size:
                    raise RuntimeError(f"Incorrect {group} group")
            if len(worker["expert_ids"]) != 48 or worker["kv_num_blocks"] <= 0:
                raise RuntimeError("Incomplete Qwen model/cache initialization")
            if not 28 * 1024**3 < worker["parameter_bytes"] < 33 * 1024**3:
                raise RuntimeError("Unexpected EP-sharded weight size")
        for first_ids, second_ids in zip(ranks[0]["expert_ids"], ranks[1]["expert_ids"]):
            if (len(first_ids) != 64 or len(second_ids) != 64
                    or set(first_ids) & set(second_ids)
                    or set(first_ids) | set(second_ids) != set(range(128))):
                raise RuntimeError("Incorrect local expert coverage")
    first_gpus = {worker["gpu_uuid"] for worker in workers if worker["service"] == "instance-1"}
    second_gpus = {worker["gpu_uuid"] for worker in workers if worker["service"] == "instance-2"}
    if first_gpus != second_gpus:
        raise RuntimeError("The model tenants do not share the same GPU pair")
    return workers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", nargs=2, type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.requests < 4 or args.requests % 4:
        parser.error("--requests must be a positive multiple of four")
    endpoints = [{"url": f"http://127.0.0.1:{port}",
                  "model": f"Qwen/Qwen3-30B-A3B-Instruct-2507-instance-{service_index}"}
                 for service_index, port in enumerate(args.ports, start=1)]
    workers = verify_workers(args.evidence_root)
    if any(worker["checkpoint"] != args.checkpoint for worker in workers):
        raise RuntimeError("Workers do not use the requested checkpoint")
    for endpoint in endpoints:
        models = request_json(endpoint["url"], "/v1/models")
        if endpoint["model"] not in {model["id"] for model in models["data"]}:
            raise RuntimeError("Unexpected served model")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    arithmetic = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with one short sentence: what is 2+2?"}],
        tokenize=True, add_generation_prompt=True,
    )
    if not isinstance(arithmetic, list):
        arithmetic = arithmetic["input_ids"]
    serial = [completion(endpoint, rank, arithmetic, 128, False)
              for endpoint in endpoints for rank in (0, 1)]
    mixed = concurrent_batch(endpoints, arithmetic, 128, False, 8)
    baselines = {(record["model"], record["rank"]): record for record in serial}
    for record in serial + mixed:
        if not re.search(r"\b(?:4|four)\b", record["text"], re.IGNORECASE):
            raise RuntimeError(f"Incorrect arithmetic: {record}")
        if record["token_ids"] != baselines[(record["model"], record["rank"])]["token_ids"]:
            raise RuntimeError("Concurrent output differs from same-rank serial baseline")
    prompt = tokenizer.encode("Summarize this example briefly. " * args.input_tokens,
                              add_special_tokens=False)[:args.input_tokens]
    if len(prompt) != args.input_tokens:
        raise RuntimeError("Could not construct exact-length token prompt")
    names = [worker["ipc_name"] for worker in workers]
    samples = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                samples.append({"time_ns": time.time_ns(), "pools": read_memory(names)})
            except Exception as error:
                samples.append({"error": str(error)})
            stop.wait(0.5)

    before = read_memory(names)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        measured = concurrent_batch(endpoints, prompt, args.output_tokens, True, args.requests)
    finally:
        stop.set()
        monitor_thread.join(timeout=10)
    if any("error" in sample for sample in samples):
        raise RuntimeError("KV memory monitoring failed")
    if not any(all(pool["mapped_live_bytes"] > before[name]["mapped_live_bytes"]
                   for name, pool in sample["pools"].items())
               for sample in samples):
        raise RuntimeError("Did not observe all four KV pools grow beyond idle backing")
    for record in measured:
        if record["usage"]["prompt_tokens"] != args.input_tokens:
            raise RuntimeError("Server prompt token count does not match")
    span_s = (max(record["end_ns"] for record in measured)
              - min(record["start_ns"] for record in measured)) / 1e9
    total_tokens = sum(record["usage"]["completion_tokens"] for record in measured)
    report = {
        "status": "passed", "backend": "native_vllm+kvcached", "workers": workers,
        "serial_arithmetic": serial, "concurrent_arithmetic": mixed,
        "input_tokens": args.input_tokens, "output_tokens": args.output_tokens,
        "requests": measured, "completion_tokens": total_tokens,
        "makespan_s": span_s, "aggregate_output_tokens_per_s": total_tokens / span_s,
        "memory_before": before, "memory_samples": samples, "memory_after": read_memory(names),
        "scope": "startup/co-residency/concurrent inference baseline; no migration or SLO claim",
    }
    output = args.evidence_root / "benchmark.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": report["status"], "completion_tokens": total_tokens,
                      "makespan_s": span_s, "tokens_per_s": total_tokens / span_s}), flush=True)


if __name__ == "__main__":
    main()
