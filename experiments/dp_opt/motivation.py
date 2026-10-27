import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import random
import re
import statistics
import threading
import time
import urllib.request
from pathlib import Path

from benchmark import completion, read_memory, request_json
from single_model_benchmark import distribution, rank_schedule, save_json, stream_completion, summarize


def layouts(total=16):
    if total < 8 or total % 8:
        raise ValueError("Request count must be a positive multiple of eight")
    result = [(total // 2, total // 2)]
    for hot in (total * 5 // 8, total * 6 // 8, total * 7 // 8, total):
        result.extend(((hot, total - hot), (total - hot, hot)))
    return result


def build_plan(repeats=3, seed=42, extremes_only=False, total=16):
    balanced = list(layouts(total)[0])
    generator = random.Random(seed)
    result = []
    for repeat in range(repeats):
        result.append({"repeat": repeat, "layout": balanced, "role": "baseline_start"})
        unbalanced = [(total, 0), (0, total)] if extremes_only else layouts(total)[1:]
        generator.shuffle(unbalanced)
        result.extend({"repeat": repeat, "layout": list(layout), "role": "measured"}
                      for layout in unbalanced)
        result.append({"repeat": repeat, "layout": balanced, "role": "baseline_end"})
    for index, trial in enumerate(result):
        trial["id"] = f"trial-{index:02d}-r{trial['repeat']}-{trial['layout'][0]}-{trial['layout'][1]}"
    return result


def validate_capacity(workers, input_tokens, output_tokens, block_size=16, total=16):
    layouts(total)
    if input_tokens < 1 or output_tokens < 256:
        raise ValueError("Require positive input length and at least 256 output tokens")
    required_blocks = total * math.ceil((input_tokens + output_tokens) / block_size)
    for worker in workers:
        if worker.get("kv_block_size", block_size) != block_size:
            raise ValueError("Unexpected KV block size")
        if worker.get("max_model_len", input_tokens + output_tokens) < input_tokens + output_tokens:
            raise ValueError("Requested sequence exceeds max_model_len")
        if worker["kv_num_blocks"] < required_blocks:
            raise ValueError("Insufficient hot-rank virtual KV capacity for all requests")
        if worker.get("max_num_seqs", total) < total:
            raise ValueError("Insufficient hottest-rank sequence slots")
    return {"block_size": block_size, "required_hot_rank_blocks": required_blocks,
            "available_blocks_per_rank": [worker["kv_num_blocks"] for worker in workers]}


def parse_metrics(text, model):
    names = {"vllm:num_requests_running", "vllm:num_requests_waiting",
             "vllm:num_preemptions_total", "vllm:kv_cache_usage_perc"}
    result = {str(rank): {} for rank in (0, 1)}
    pattern = re.compile(r'^([^\s{]+)\{([^}]+)\}\s+([-+0-9.eE]+)')
    for line in text.splitlines():
        match = pattern.match(line)
        if match is None or match.group(1) not in names:
            continue
        labels = dict(re.findall(r'(\w+)="([^"\\]*)"', match.group(2)))
        if labels.get("model_name") == model and labels.get("engine") in result:
            result[labels["engine"]][match.group(1)] = float(match.group(3))
    if any(set(metrics) != names for metrics in result.values()):
        raise RuntimeError("Missing per-rank running/waiting/preemption/KV metrics")
    return result


def sample_metrics(endpoint):
    with urllib.request.urlopen(endpoint["url"] + "/metrics", timeout=10) as response:
        return parse_metrics(response.read().decode(), endpoint["model"])


def wait_idle(endpoint, names, baseline=None):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        metrics = sample_metrics(endpoint)
        memory = read_memory(names)
        idle = all(rank["vllm:num_requests_running"] == 0
                   and rank["vllm:num_requests_waiting"] == 0 for rank in metrics.values())
        reclaimed = baseline is None or all(
            memory[name]["mapped_live_bytes"] <= baseline[name]["mapped_live_bytes"] for name in names
        )
        if idle and reclaimed:
            return memory, metrics
        time.sleep(0.5)
    raise RuntimeError("Service did not become idle or reclaim previous trial KV")


def valid_decode(event, expected_ids):
    if not expected_ids:
        return event["kind"] == "dummy"
    requests = event["requests"]
    return (event["kind"] == "execute" and event["new_requests"] == 0
            and event["resumed_requests"] == 0
            and {request["id"] for request in requests} == expected_ids
            and all(request["tokens"] == 1 and request["output_tokens"] is not None
                    and request["output_tokens"] > 0
                    and request["computed_tokens"] is not None
                    and request["prompt_tokens"] is not None
                    and request["computed_tokens"] >= request["prompt_tokens"]
                    for request in requests))


def graph_padding(counts, capture_sizes=None):
    maximum = max(counts)
    if capture_sizes is None:
        return maximum
    if not capture_sizes or sorted(set(capture_sizes)) != list(capture_sizes) or min(capture_sizes) < 1:
        raise ValueError("Capture sizes must be strictly increasing positive integers")
    return next((size for size in capture_sizes if size >= maximum), None)


def stable_window(traces, counts, trim=32, minimum=128, expected_cudagraph_mode="NONE",
                  capture_sizes=None):
    request_ids = {}
    valid = {}
    for rank, trace in enumerate(traces):
        ids = {request["id"] for event in trace["events"] for request in event["requests"]}
        if len(ids) != counts[rank]:
            raise RuntimeError(f"Rank {rank} executed {len(ids)} requests, expected {counts[rank]}")
        request_ids[rank] = ids
        valid[rank] = [event for event in trace["events"] if valid_decode(event, ids)]
        if not valid[rank]:
            raise RuntimeError(f"Rank {rank} has no stable decode/dummy steps")
    start = max(events[0]["start_ns"] for events in valid.values())
    end = min(events[-1]["end_ns"] for events in valid.values())
    candidates = {rank: [event for event in events if event["start_ns"] >= start
                         and event["end_ns"] <= end] for rank, events in valid.items()}
    if any(len(events) < minimum + 2 * trim for events in candidates.values()):
        raise RuntimeError("Insufficient common stable decode steps before trimming")
    start = max(events[trim]["start_ns"] for events in candidates.values())
    end = min(events[-trim - 1]["end_ns"] for events in candidates.values())
    selected = {rank: [event for event in trace["events"] if event["start_ns"] >= start
                       and event["end_ns"] <= end] for rank, trace in enumerate(traces)}
    for rank, events in selected.items():
        if len(events) < minimum or any(not valid_decode(event, request_ids[rank]) for event in events):
            raise RuntimeError("Stable window has changing populations or too few steps")
    cadence = {}
    tokens = 0
    for rank, events in selected.items():
        intervals = [(second["start_ns"] - first["start_ns"]) / 1e9
                     for first, second in zip(events, events[1:])]
        batches = [event["batch"] for event in events if event["batch"] is not None]
        if not batches or len(batches) != len(events):
            raise RuntimeError("Missing batch/padding evidence in stable window")
        if any(batch["should_ubatch"] or batch["cudagraph_mode"].split(".")[-1] != expected_cudagraph_mode
               for batch in batches):
            raise RuntimeError(f"Expected {expected_cudagraph_mode} execution without DBO")
        expected_tokens = counts[rank] or 1
        if any(batch["unpadded_tokens"] != expected_tokens for batch in batches):
            raise RuntimeError("Unexpected unpadded decode population")
        if expected_cudagraph_mode == "NONE" and any(
                batch["padded_tokens"] != expected_tokens for batch in batches):
            raise RuntimeError("Eager decode must not silently pad the batch")
        if expected_cudagraph_mode == "FULL" and any(
                batch["padded_tokens"] != graph_padding(counts, capture_sizes) for batch in batches):
            raise RuntimeError("Full-graph decode must match its declared max-rank padding")
        cadence[str(rank)] = {
            "steps": len(events), "dummy_steps": sum(event["kind"] == "dummy" for event in events),
            "interval_s": distribution(intervals),
            "worker_host_execute_s": distribution((event["end_ns"] - event["start_ns"]) / 1e9
                                                   for event in events),
            "padded_tokens": sorted({batch["padded_tokens"] for batch in batches}),
            "unpadded_tokens": sorted({batch["unpadded_tokens"] for batch in batches}),
        }
        tokens += counts[rank] * len(events)
    durations = [cadence[str(rank)]["interval_s"]["p50"] for rank in (0, 1) if counts[rank]]
    return {"start_ns": start, "end_ns": end, "duration_s": (end - start) / 1e9,
            "tokens": tokens, "output_tokens_per_second": tokens / ((end - start) / 1e9),
            "critical_cadence_median_s": max(durations), "ranks": cadence}


def client_window(records, start, end):
    intervals = []
    for record in records:
        events = record["token_events"]
        intervals.extend((second["time_ns"] - first["time_ns"]) / 1e9
                         for first, second in zip(events, events[1:])
                         if start <= first["time_ns"] <= second["time_ns"] <= end)
    if len(intervals) < len(records) * 100:
        raise RuntimeError("Too few streamed token intervals in stable window")
    return distribution(intervals)


def run_trial(endpoint, prompt_set, trial, root, names, idle_memory, output_tokens,
              case_name="", profile_phases=False, expected_cudagraph_mode="NONE", capture_sizes=None):
    trial_id = trial["id"]
    before, metrics_before = wait_idle(endpoint, names, idle_memory)
    request_json(endpoint["url"], "/collective_rpc", {
        "method": "motivation_control",
        "args": ["start_profile" if profile_phases else "start", trial_id, case_name], "timeout": 60,
    })
    samples = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                samples.append({"time_ns": time.time_ns(), "pools": read_memory(names),
                                "metrics": sample_metrics(endpoint)})
            except Exception as error:
                samples.append({"error": str(error)})
            stop.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    total = len(prompt_set)
    barrier = threading.Barrier(total)
    schedule = rank_schedule(trial["layout"])
    shift = trial["repeat"] * 5 % total

    def complete(index, rank):
        barrier.wait(timeout=60)
        prompt_index = (index + shift) % total
        record = stream_completion(endpoint, rank, prompt_set[prompt_index], output_tokens,
                                   request_id=f"mot-{trial_id}-{index}")
        record["prompt_index"] = prompt_index
        return record

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=total) as executor:
            pending = [executor.submit(complete, index, rank) for index, rank in enumerate(schedule)]
            records = [future.result() for future in pending]
    finally:
        stop.set()
        thread.join(timeout=20)
        request_json(endpoint["url"], "/collective_rpc", {
            "method": "motivation_control", "args": ["stop", trial_id, case_name], "timeout": 60,
        })
    after, metrics_after = wait_idle(endpoint, names, idle_memory)
    save_json(root / "trials" / f"{trial_id}.json", {
        **trial, "requests": records, "samples": samples,
        "memory_before": before, "memory_after": after,
        "metrics_before": metrics_before, "metrics_after": metrics_after,
    })
    traces = [json.loads((root / "traces" / f"{trial_id}-dp{rank}.json").read_text()) for rank in (0, 1)]
    window = stable_window(traces, trial["layout"], expected_cudagraph_mode=expected_cudagraph_mode,
                           capture_sizes=capture_sizes)
    phase_profile = {}
    if profile_phases:
        for trace in traces:
            events = [event for event in trace["events"] if event.get("phase_cuda_ms")
                      and window["start_ns"] <= event["start_ns"] <= window["end_ns"]]
            if len(events) < 4:
                raise RuntimeError("Insufficient stable CUDA phase samples")
            if any(set(event["phase_cuda_ms"]) != {
                "attention", "attention_core", "moe", "dispatch", "combine", "worker_execute"}
                   for event in events):
                raise RuntimeError("Missing CUDA module phase coverage")
            phase_profile[str(trace["rank"])] = {
                "samples": len(events),
                "ranges_ms": {phase: distribution(event["phase_cuda_ms"][phase] for event in events)
                              for phase in events[0]["phase_cuda_ms"]},
                "collective_sizes": sorted({tuple(call["sizes"] or []) for event in events
                                             for call in event["collective_calls"]}),
            }
    if any("error" in sample for sample in samples):
        raise RuntimeError("Monitoring failed")
    stable_samples = [sample for sample in samples
                      if window["start_ns"] <= sample["time_ns"] <= window["end_ns"]]
    if not stable_samples:
        raise RuntimeError("No metrics in stable decode window")
    for rank in (0, 1):
        initial = metrics_before[str(rank)]["vllm:num_preemptions_total"]
        if metrics_after[str(rank)]["vllm:num_preemptions_total"] != initial:
            raise RuntimeError("Preemption invalidates compute-only experiment")
        for sample in stable_samples:
            metrics = sample["metrics"][str(rank)]
            if (metrics["vllm:num_requests_running"] != trial["layout"][rank]
                    or metrics["vllm:num_requests_waiting"] != 0
                    or metrics["vllm:kv_cache_usage_perc"] >= 0.9):
                raise RuntimeError("Stable metrics show incorrect population, waiting or KV pressure")
    if max(record["max_tokens_per_chunk"] for record in records) != 1:
        raise RuntimeError("Stable client ITL requires one token per stream chunk")
    summary = summarize(records)
    result = {**trial, "imbalance": abs(trial["layout"][0] - trial["layout"][1]) / total,
              "requests": records, "summary": summary, "stable_window": window,
              "phase_profile": phase_profile,
              "stable_client_itl_s": client_window(records, window["start_ns"], window["end_ns"]),
              "memory_before": before, "memory_after": after,
              "metrics_before": metrics_before, "metrics_after": metrics_after, "samples": samples}
    save_json(root / "trials" / f"{trial_id}.json", result)
    return result


def paired_summary(trials):
    output = []
    total = sum(trials[0]["layout"])
    balanced = list(layouts(total)[0])
    measured_layouts = {tuple(trial["layout"]) for trial in trials}
    for layout in (layout for layout in layouts(total) if layout in measured_layouts):
        ratios = []
        throughputs = []
        cadences = []
        tpot = []
        baseline_drift = []
        for repeat in sorted({trial["repeat"] for trial in trials}):
            cohort = [trial for trial in trials if trial["repeat"] == repeat]
            baselines = [trial for trial in cohort if trial["layout"] == balanced]
            reference = statistics.mean(trial["stable_window"]["critical_cadence_median_s"]
                                        for trial in baselines)
            baseline_drift.append(baselines[-1]["stable_window"]["critical_cadence_median_s"]
                                  / baselines[0]["stable_window"]["critical_cadence_median_s"] - 1)
            candidates = [trial for trial in cohort if trial["layout"] == list(layout)]
            cadence = statistics.mean(trial["stable_window"]["critical_cadence_median_s"]
                                      for trial in candidates)
            ratios.append(cadence / reference)
            cadences.append(cadence)
            throughputs.extend(trial["summary"]["output_tokens_per_second"] for trial in candidates)
            tpot.extend(trial["stable_client_itl_s"]["p95"] for trial in candidates)
        if len(ratios) < 2:
            interval = None
        else:
            margin = 4.303 * statistics.stdev(ratios) / math.sqrt(len(ratios))
            interval = [statistics.mean(ratios) - margin, statistics.mean(ratios) + margin]
        output.append({"layout": list(layout), "imbalance": abs(layout[0] - layout[1]) / total,
                       "paired_slowdown": statistics.mean(ratios), "paired_slowdown_t95_ci": interval,
                       "paired_ratios": ratios, "cadence_s": statistics.mean(cadences),
                       "output_tokens_per_second": statistics.mean(throughputs),
                       "stable_client_itl_p95_s": statistics.mean(tpot),
                       "baseline_drift": baseline_drift})
    return output


def phase_scaling(diagnostics):
    total = sum(diagnostics[0]["layout"])
    balanced_layout = list(layouts(total)[0])
    balanced = next(trial for trial in diagnostics if trial["layout"] == balanced_layout)
    output = []
    for trial in diagnostics:
        if trial["layout"] == balanced_layout:
            continue
        hot_rank = str(trial["layout"].index(total))
        reference = balanced["phase_profile"][hot_rank]["ranges_ms"]
        skewed = trial["phase_profile"][hot_rank]["ranges_ms"]
        output.append({
            "layout": trial["layout"], "hot_rank": int(hot_rank),
            "range_mean_ratio_skew_to_balanced": {
                phase: skewed[phase]["mean"] / reference[phase]["mean"]
                for phase in skewed if reference[phase]["mean"] > 0
            },
            "skew_attention_range_fraction": skewed["attention"]["mean"] / skewed["worker_execute"]["mean"],
            "linear_half_share_prediction": 0.5 * skewed["attention"]["mean"] / skewed["worker_execute"]["mean"],
            "measured_attention_delta_fraction": (
                skewed["attention"]["mean"] - reference["attention"]["mean"]
            ) / skewed["worker_execute"]["mean"],
        })
    return output


def write_summary(root, summary, repeats=3):
    columns = ["rank0_requests", "rank1_requests", "imbalance", "paired_slowdown",
               "ci95_lower", "ci95_upper", "cadence_ms", "burst_output_tokens_per_second",
               "stable_client_itl_p95_ms"]
    with (root / "summary.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        for result in summary:
            interval = result["paired_slowdown_t95_ci"] or [None, None]
            writer.writerow([*result["layout"], result["imbalance"], result["paired_slowdown"],
                             *interval, result["cadence_s"] * 1000,
                             result["output_tokens_per_second"], result["stable_client_itl_p95_s"] * 1000])
    maximum = max(1.1, max(max(result["paired_slowdown_t95_ci"] or [result["paired_slowdown"]])
                           for result in summary) * 1.08)
    minimum = min(0.9, min(min(result["paired_slowdown_t95_ci"] or [result["paired_slowdown"]])
                           for result in summary) * 0.95)

    def coordinate(imbalance, slowdown):
        return 85 + imbalance * 570, 390 - (slowdown - minimum) / (maximum - minimum) * 315

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="470" viewBox="0 0 760 470">',
        '<rect width="760" height="470" fill="white"/>',
        '<g font-family="sans-serif" font-size="14" fill="#222">',
        '<text x="85" y="28" font-size="18">Qwen single-model attention-DP imbalance</text>',
        f'<text x="85" y="52">Paired stable rank-host cadence / balanced cadence (n={repeats}; CI when n=3)</text>',
        '<path d="M85 75 V390 H655" stroke="#222" fill="none"/>',
        '<text x="270" y="447">Imbalance |n0-n1| / (n0+n1)</text>',
    ]
    for tick in range(5):
        imbalance = tick / 4
        horizontal, vertical = coordinate(imbalance, minimum)
        elements.append(f'<text x="{horizontal - 12}" y="414">{imbalance:.2f}</text>')
    for tick in range(6):
        slowdown = minimum + (maximum - minimum) * tick / 5
        horizontal, vertical = coordinate(0, slowdown)
        elements.append(f'<path d="M85 {vertical:.2f} H655" stroke="#eee"/>')
        elements.append(f'<text x="28" y="{vertical + 5:.2f}">{slowdown:.2f}x</text>')
    for hot_rank, color in ((0, "#1768ac"), (1, "#d97706")):
        results = sorted([result for result in summary if result["layout"][hot_rank]
                          >= result["layout"][1 - hot_rank]], key=lambda result: result["imbalance"])
        points = [coordinate(result["imbalance"], result["paired_slowdown"]) for result in results]
        coordinates = " ".join(f"{horizontal:.2f},{vertical:.2f}" for horizontal, vertical in points)
        elements.append(f'<polyline points="{coordinates}" stroke="{color}" fill="none" stroke-width="2"/>')
        for result, (horizontal, vertical) in zip(results, points):
            interval = result["paired_slowdown_t95_ci"]
            if interval is not None:
                top = coordinate(result["imbalance"], interval[1])[1]
                bottom = coordinate(result["imbalance"], interval[0])[1]
                elements.append(f'<path d="M{horizontal:.2f} {top:.2f} V{bottom:.2f}" stroke="{color}"/>')
            elements.append(f'<circle cx="{horizontal:.2f}" cy="{vertical:.2f}" r="4" fill="{color}"/>')
        elements.append(f'<text x="{430 + hot_rank * 115}" y="28" fill="{color}">hot rank {hot_rank}</text>')
    elements.append('</g></svg>')
    (root / "motivation.svg").write_text("\n".join(elements) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--case-name", default="")
    parser.add_argument("--extremes-only", action="store_true")
    parser.add_argument("--profile-phases", action="store_true")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--screening", action="store_true")
    parser.add_argument("--skip-overhead-controls", action="store_true")
    parser.add_argument("--expected-cudagraph-mode", choices=("NONE", "FULL", "PIECEWISE"), default="NONE")
    parser.add_argument("--capture-sizes", type=lambda value: [int(size) for size in value.split(",")])
    args = parser.parse_args()
    if args.repeats != (1 if args.screening else 3) or args.output_tokens < 256 or args.input_tokens < 1:
        parser.error("This fixture requires three repeats and at least 256 output tokens")
    try:
        balanced = list(layouts(args.requests)[0])
        if args.capture_sizes is not None:
            if args.expected_cudagraph_mode != "FULL" or graph_padding([args.requests, 0], args.capture_sizes) is None:
                raise ValueError("Declared capture sizes must cover the full-graph extreme")
    except ValueError as error:
        parser.error(str(error))
    if args.case_name and not re.fullmatch(r"[a-zA-Z0-9_-]+", args.case_name):
        parser.error("Case name must contain only letters, digits, underscores or hyphens")
    root = args.evidence_root / "motivation"
    if args.case_name:
        root = root / "cases" / args.case_name
    root.mkdir(parents=True, exist_ok=True)
    workers = [json.loads(path.read_text()) for path in sorted(args.evidence_root.glob("worker-*.json"))]
    if len(workers) != 2 or {worker["dp_rank"] for worker in workers} != {0, 1}:
        raise RuntimeError("Expected exactly one model with two DP workers")
    workers.sort(key=lambda worker: worker["dp_rank"])
    for worker in workers:
        if (worker["checkpoint"] != args.checkpoint or worker["service"] != "instance-1"
                or any(worker["groups"][group]["world_size"] != size
                       for group, size in (("tp", 1), ("dp", 2), ("ep", 2)))
                or not 28 * 1024**3 < worker["parameter_bytes"] < 33 * 1024**3):
            raise RuntimeError("Incorrect model, topology or expert-sharded weights")
    if len({worker["gpu_uuid"] for worker in workers}) != 2 or len({worker["ipc_name"] for worker in workers}) != 2:
        raise RuntimeError("DP workers must use distinct GPUs and KV namespaces")
    if any(len(worker["expert_ids"]) != 48 for worker in workers):
        raise RuntimeError("Expected expert coverage for all 48 layers")
    capacity = validate_capacity(workers, args.input_tokens, args.output_tokens, total=args.requests)
    for first, second in zip(workers[0]["expert_ids"], workers[1]["expert_ids"]):
        if len(first) != 64 or len(second) != 64 or set(first) & set(second) or set(first) | set(second) != set(range(128)):
            raise RuntimeError("Incorrect expert coverage")
    endpoint = {"url": args.url, "model": "Qwen/Qwen3-30B-A3B-Instruct-2507-instance-1"}
    if endpoint["model"] not in {model["id"] for model in request_json(args.url, "/v1/models")["data"]}:
        raise RuntimeError("Unexpected served model")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    prompts = []
    for index in range(args.requests):
        text = f"Example {index}: Explain this technical observation carefully. "
        tokens = tokenizer.encode(text * args.input_tokens, add_special_tokens=False)[:args.input_tokens]
        if len(tokens) != args.input_tokens:
            raise RuntimeError("Prompt fixture has incorrect token length")
        prompts.append(tokens)
    if len({tuple(prompt) for prompt in prompts}) != args.requests:
        raise RuntimeError("Motivation prompts must be distinct")
    plan = build_plan(args.repeats, extremes_only=args.extremes_only, total=args.requests)
    save_json(root / "fixture.json", {"prompts": prompts, "input_tokens": args.input_tokens,
                                      "output_tokens": args.output_tokens, "plan": plan,
                                      "sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest()})
    arithmetic = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with one short sentence: what is 2+2?"}],
        tokenize=True, add_generation_prompt=True,
    )
    if not isinstance(arithmetic, list):
        arithmetic = arithmetic["input_ids"]
    for rank in (0, 1):
        record = completion(endpoint, rank, arithmetic, 128, False)
        if not re.search(r"\b(?:4|four)\b", record["text"], re.IGNORECASE):
            raise RuntimeError("Arithmetic correctness check failed")
    names = [worker["ipc_name"] for worker in workers]
    warm_layouts = [tuple(balanced), (args.requests, 0), (0, args.requests)] if args.extremes_only else layouts(args.requests)
    for counts in warm_layouts:
        barrier = threading.Barrier(args.requests)

        def warm(index, rank):
            barrier.wait(timeout=60)
            return stream_completion(endpoint, rank, prompts[index], 32)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.requests) as executor:
            pending = [executor.submit(warm, index, rank)
                       for index, rank in enumerate(rank_schedule(counts))]
            for future in pending:
                future.result()
        wait_idle(endpoint, names)
    idle_memory, idle_metrics = wait_idle(endpoint, names)
    report = {"status": "running", "backend": "native_vllm+kvcached", "workers": workers,
              "scope": "single-model attention-DP compute motivation; no migration or placement",
              "protocol": {"requests": args.requests, "input_tokens": args.input_tokens, "output_tokens": args.output_tokens,
                           "repeats": args.repeats, "trim_steps_each_end": 32,
                           "minimum_stable_steps": 128, "hot_rank_swap": True,
                           "case_name": args.case_name, "capacity": capacity,
                           "extremes_only": args.extremes_only, "screening": args.screening,
                           "expected_cudagraph_mode": args.expected_cudagraph_mode,
                           "capture_sizes": args.capture_sizes,
                           "overhead_controls": not args.skip_overhead_controls}, "trials": []}
    overhead_controls = []
    for repeat in range(0 if args.skip_overhead_controls else 3):
        barrier = threading.Barrier(args.requests)

        def untraced(index, rank):
            barrier.wait(timeout=60)
            return stream_completion(endpoint, rank, prompts[index], args.output_tokens)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.requests) as executor:
            pending = [executor.submit(untraced, index, rank)
                       for index, rank in enumerate(rank_schedule(balanced))]
            records = [future.result() for future in pending]
        overhead_controls.append({"repeat": repeat, "summary": summarize(records), "requests": records})
        wait_idle(endpoint, names, idle_memory)
    save_json(root / "untraced-balanced.json", overhead_controls)
    report["untraced_balanced"] = [control["summary"] for control in overhead_controls]
    try:
        for trial in plan:
            result = run_trial(endpoint, prompts, trial, root, names, idle_memory, args.output_tokens,
                               args.case_name, expected_cudagraph_mode=args.expected_cudagraph_mode,
                               capture_sizes=args.capture_sizes)
            report["trials"].append({key: value for key, value in result.items()
                                      if key not in ("requests", "samples")})
            save_json(root / "results.json", report)
            print(json.dumps({"trial": trial, "throughput": result["summary"]["output_tokens_per_second"],
                              "stable_cadence_s": result["stable_window"]["critical_cadence_median_s"]}), flush=True)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save_json(root / "results.json", report)
        raise
    report["summary"] = paired_summary(report["trials"])
    traced_balanced = [trial["summary"]["tpot_s"]["mean"] for trial in report["trials"]
                       if trial["layout"] == balanced]
    untraced_balanced = [summary["tpot_s"]["mean"] for summary in report["untraced_balanced"]]
    report["instrumentation_control"] = None if args.skip_overhead_controls else {
        "untraced_balanced_mean_tpot_s": statistics.mean(untraced_balanced),
        "traced_balanced_mean_tpot_s": statistics.mean(traced_balanced),
        "observed_tpot_ratio": statistics.mean(traced_balanced) / statistics.mean(untraced_balanced),
        "caveat": "Sequential control, includes time drift; not an isolated causal overhead measurement",
    }
    write_summary(root, report["summary"], args.repeats)
    report["status"] = "passed"
    save_json(root / "results.json", report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    if args.profile_phases:
        diagnostics = []
        for index, counts in enumerate(warm_layouts[:1] + [(args.requests, 0), (0, args.requests)]):
            trial = {"id": f"profile-{index}-{counts[0]}-{counts[1]}", "repeat": 0,
                     "layout": list(counts), "role": "cuda_phase_diagnostic"}
            diagnostics.append(run_trial(endpoint, prompts, trial, root, names, idle_memory,
                                         args.output_tokens, args.case_name, True))
        save_json(root / "phase-diagnostics.json", {
            "status": "passed", "sampling_stride": 32,
            "caveat": "Independent instrumented trials, excluded from performance summary. "
                      "CUDA module ranges include projection, launch gaps and stream dependencies; "
                      "attention_core is nested inside attention (do not sum both). "
                      "Dispatch/combine ranges are nested inside MoE (do not sum twice). "
                      "Not isolated kernel timings; "
                      "phase delta fractions are diagnostic estimates, not causal predictions.",
            "scaling": phase_scaling(diagnostics),
            "trials": [{key: value for key, value in trial.items()
                        if key not in ("requests", "samples")} for trial in diagnostics],
        })


if __name__ == "__main__":
    main()
