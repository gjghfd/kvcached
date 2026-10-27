import argparse
import json
import math
from pathlib import Path

from motivation import build_plan, client_window, paired_summary, stable_window


def verify_case(root):
    report = json.loads((root / "results.json").read_text())
    if report["status"] != "passed":
        raise ValueError("Only passed cases can be independently verified")
    protocol = report["protocol"]
    plan = build_plan(repeats=protocol["repeats"], total=protocol["requests"],
                      extremes_only=protocol["extremes_only"])
    if len(report["trials"]) != len(plan):
        raise ValueError("Incomplete measured trial plan")
    verified = []
    request_count, output_tokens, minimum_steps, maximum_kv = 0, 0, math.inf, 0
    idle_backing = None
    graph_sizes = {}
    for expected, declared in zip(plan, report["trials"]):
        if any(declared[name] != expected[name] for name in ("id", "repeat", "role", "layout")):
            raise ValueError("Measured trial order/layout differs from frozen plan")
        trial = json.loads((root / "trials" / f'{declared["id"]}.json').read_text())
        if any(trial[name] != expected[name] for name in ("id", "repeat", "role", "layout")):
            raise ValueError("Raw trial metadata differs from the frozen plan")
        traces = [json.loads((root / "traces" / f'{declared["id"]}-dp{rank}.json').read_text()) for rank in (0, 1)]
        window = stable_window(traces, trial["layout"], trim=protocol["trim_steps_each_end"],
                               minimum=protocol["minimum_stable_steps"],
                               expected_cudagraph_mode=protocol["expected_cudagraph_mode"],
                               capture_sizes=protocol.get("capture_sizes"))
        if window != trial["stable_window"] or window != declared["stable_window"]:
            raise ValueError("Raw traces do not reproduce the declared stable window")
        if client_window(trial["requests"], window["start_ns"], window["end_ns"]) != trial["stable_client_itl_s"]:
            raise ValueError("Raw stream timestamps do not reproduce client ITL")
        requests = trial["requests"]
        if len(requests) != protocol["requests"]:
            raise ValueError("Incorrect completed request population")
        for rank in (0, 1):
            if sum(request["rank"] == rank for request in requests) != trial["layout"][rank]:
                raise ValueError("Incorrect client rank assignment")
        for request in requests:
            if (request["status"] != "passed" or request["finish_reason"] != "length"
                    or len(request["token_ids"]) != protocol["output_tokens"]
                    or request["usage"]["prompt_tokens"] != protocol["input_tokens"]
                    or request["usage"]["completion_tokens"] != protocol["output_tokens"]
                    or sum(event["tokens"] for event in request["token_events"]) != protocol["output_tokens"]
                    or request["max_tokens_per_chunk"] != 1):
                raise ValueError("Incorrect raw request lengths or stream token evidence")
        current_idle = {name: memory["mapped_live_bytes"] for name, memory in trial["memory_before"].items()}
        if idle_backing is None:
            idle_backing = current_idle
        if current_idle != idle_backing:
            raise ValueError("Idle mapped backing changed between trials")
        if any(memory["reserved_backing_bytes"] != 0 for memory in trial["memory_before"].values()):
            raise ValueError("Reserved backing is unexpectedly enabled")
        for name, memory in trial["memory_after"].items():
            if memory["mapped_live_bytes"] != idle_backing[name] or memory["reserved_backing_bytes"] != 0:
                raise ValueError("Trial did not restore the same idle backing")
        samples = [sample for sample in trial["samples"] if window["start_ns"] <= sample["time_ns"] <= window["end_ns"]]
        if not samples:
            raise ValueError("Stable monitoring samples are absent")
        for rank in (0, 1):
            initial_preemptions = trial["metrics_before"][str(rank)]["vllm:num_preemptions_total"]
            if trial["metrics_after"][str(rank)]["vllm:num_preemptions_total"] != initial_preemptions:
                raise ValueError("Trial contains a preemption")
            for sample in samples:
                metrics = sample["metrics"][str(rank)]
                if (metrics["vllm:num_requests_running"] != trial["layout"][rank]
                        or metrics["vllm:num_requests_waiting"] != 0
                        or metrics["vllm:num_preemptions_total"] != initial_preemptions
                        or metrics["vllm:kv_cache_usage_perc"] >= 0.9):
                    raise ValueError("Stable metrics violate population/waiting/preemption/KV limits")
                maximum_kv = max(maximum_kv, metrics["vllm:kv_cache_usage_perc"])
            minimum_steps = min(minimum_steps, window["ranks"][str(rank)]["steps"])
        graph_sizes[":".join(map(str, trial["layout"]))] = [window["ranks"][str(rank)]["padded_tokens"] for rank in (0, 1)]
        verified.append(trial)
        request_count += len(requests)
        output_tokens += sum(len(request["token_ids"]) for request in requests)
    if paired_summary(verified) != report["summary"]:
        raise ValueError("Raw trials do not reproduce paired summary")
    return {"status": "raw_verified", "trials": len(verified), "requests": request_count,
            "output_tokens": output_tokens, "minimum_stable_steps": minimum_steps,
            "maximum_stable_kv_usage": maximum_kv, "idle_mapped_bytes": idle_backing,
            "graph_sizes_by_layout": graph_sizes, "source_case": str(root)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verification = verify_case(args.case_root)
    args.output.write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
