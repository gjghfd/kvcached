import asyncio
import hashlib
import json
import math
import random
import statistics
import time

from live_benchmark import save
from live_benefit import capture_control, stream, wait_empty, wait_reserved


POPULATIONS = (8, 9, 10, 11, 12, 16)
LOOKUP_SOURCE_FILES = (
    "config.json", "summary.json", "layout-cases.json", "layout-mark.json", "layout-reference.json",
    "layout-trials.json", "layout-captures.json", "runtime-control.json", "exit-status",
    *(f"worker-instance-{model + 1}-dp{rank}.json" for model in (0, 1) for rank in (0, 1)),
)


def layout_cases():
    cases = []
    for population in POPULATIONS:
        for repeat in range(2):
            counts = list(range(population + 1))
            random.Random(20261023 + population * 100 + repeat).shuffle(counts)
            cases.extend({"population": population, "rank_zero": count, "repeat": repeat} for count in counts)
    return cases


def cadence(request):
    entries = request["cadence"]
    left = next((entry for entry in entries if entry["tokens"] >= 256), None)
    right = next((entry for entry in entries if entry["tokens"] >= 768), None)
    if left is None or right is None or right["tokens"] <= left["tokens"] or right["time"] <= left["time"]:
        raise ValueError("Layout calibration lacks complete stable decode cadence")
    return (right["time"] - left["time"]) / (right["tokens"] - left["tokens"])


def verify_shared_decode_window(requests):
    if any(not math.isfinite(request[field]) for request in requests
           for field in ("first_token_at", "completed_at")):
        raise ValueError("Layout decode window lacks finite first-token/completion evidence")
    all_prefill_finished = max(request["first_token_at"] for request in requests)
    first_request_completed = min(request["completed_at"] for request in requests)
    for request in requests:
        entries = request["cadence"]
        if (any(not math.isfinite(entry["time"]) for entry in entries)
                or any(later["time"] < earlier["time"] or later["tokens"] < earlier["tokens"]
                       for earlier, later in zip(entries, entries[1:]))):
            raise ValueError("Layout cadence timestamps or token counts are invalid")
        left = next(entry for entry in entries if entry["tokens"] >= 256)
        right = next(entry for entry in entries if entry["tokens"] >= 768)
        if left["time"] < all_prefill_finished or right["time"] > first_request_completed:
            raise ValueError("Predeclared 256..768 window contains population growth or drain")


def cost_tables(trials):
    if [trial["case"] for trial in trials] != layout_cases():
        raise ValueError("Layout calibration changed or omitted predeclared cases")
    tables = {}
    for population in POPULATIONS:
        values = [statistics.mean(cadence(request) for trial in trials
                   if trial["case"]["population"] == population and trial["case"]["rank_zero"] == count
                   for request in trial["requests"]) for count in range(population + 1)]
        baseline = statistics.mean(values[count] for count in {population // 2, population - population // 2})
        tables[str(population)] = {"cadence_s": values, "balanced_cadence_s": baseline,
                                  "all_counts_measured": True, "repeats_per_count": 2}
    return {"status": "raw_verified_small_population_lookup", "tables": tables,
            "scope": "one_active_Conv_tenant_other_resident_idle_not_Code_prefill_joint_interference"}


def measured_table(lookup, population, horizon_steps):
    if (lookup.get("status") != "raw_verified_small_population_lookup" or population <= 0
            or not math.isfinite(horizon_steps) or horizon_steps <= 0):
        raise ValueError("Require verified small-population lookup and positive horizon")
    row = lookup["tables"].get(str(population))
    if row is None:
        raise ValueError("Population not measured; lookup cannot silently extrapolate")
    values = row["cadence_s"]
    if (row.get("all_counts_measured") is not True or len(values) != population + 1
            or any(not math.isfinite(value) or value <= 0 for value in values)):
        raise ValueError("Lookup lacks valid direct measurements of every DP count")
    return [population * horizon_steps * (value - min(values)) for value in values]


def load_verified_lookup(path, expected_sha256, source_root):
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)):
        raise ValueError("Layout lookup requires an explicitly frozen SHA256")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Layout lookup differs from frozen SHA256")
    lookup = json.loads(payload)
    for name in LOOKUP_SOURCE_FILES:
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != lookup["source_files_sha256"].get(name):
            raise ValueError("Layout lookup differs from independently verified raw sources")
    if lookup != verify(source_root):
        raise ValueError("Layout lookup differs from independently verified raw sources")
    return lookup


async def run_layout_cost_pilot(engines, controller, args, tokenizer):
    marks = json.loads(args.admission_conv_marks.read_text())
    mark = next(mark for mark in marks if mark["prompt_tokens"] == 3584 and mark["output_tokens"] == 1024)
    prefix = tokenizer.encode("Calculate 2 + 2. The answer is", add_special_tokens=False)
    prompt = (prefix * math.ceil(3584 / len(prefix)))[:3584]
    reference = await stream(engines[1], "layout-reference", prompt, 1024, 0, 1, "reference", time.monotonic())
    save(args.output / "layout-reference.json", reference)
    save(args.output / "layout-mark.json", mark)
    save(args.output / "layout-cases.json", layout_cases())
    await capture_control(controller, "start")
    await stream(engines[0], "layout-idle-tenant-graph", prefix, 64, 0, 0, "graph_warmup", time.monotonic())
    trials = []
    for index, case in enumerate(layout_cases()):
        await wait_empty(controller)
        await controller.release_transfer_cache()
        tasks = []
        for position in range(case["population"]):
            rank = int(position >= case["rank_zero"])
            identifier = f"layout-{index}-{position}"
            task = asyncio.create_task(stream(engines[1], identifier, prompt, 1024, rank, 1,
                                              "layout_calibration", time.monotonic()))
            await wait_reserved(controller, task, identifier, 1, rank, timeout=120, target_only=True)
            tasks.append(task)
        samples = []
        for minimum in (256, 768):
            deadline = time.monotonic() + 120
            while True:
                observations = await controller.snapshots()
                active = [request for rank in observations[1] for request in rank["requests"]]
                if (len(active) == case["population"]
                        and all(request["movable"] and request["output_tokens"] >= minimum for request in active)):
                    samples.append(observations)
                    break
                if any(task.done() for task in tasks) or time.monotonic() >= deadline:
                    raise RuntimeError("Layout calibration population changed before stable decode window")
                await asyncio.sleep(.1)
        results = list(await asyncio.wait_for(asyncio.gather(*tasks), args.request_timeout_s))
        if any(request["token_ids"] != reference["token_ids"] for request in results):
            raise RuntimeError("Layout calibration differs from full serial greedy tokens")
        trials.append({"case": case, "requests": results, "window_observations": samples})
        await asyncio.to_thread(save, args.output / "layout-trials.json", trials)
        save(args.output / "progress.json", {"stage": "small_population_layout_calibration",
                                             "completed_cases": len(trials), "total_cases": len(layout_cases())})
    save(args.output / "layout-captures.json", await capture_control(controller, "stop"))
    save(args.output / "summary.json", {"status": "layout_cost_pilot_passed", "cases": len(trials)})


def verify(root, output=None):
    from verify_benefit import verify_ownership, verify_runtime_control
    from verify_trace import capture_entries

    read = lambda name: json.loads((root / name).read_text())
    config, summary, cases = read("config.json"), read("summary.json"), layout_cases()
    if (summary["status"] != "layout_cost_pilot_passed" or summary["cases"] != len(cases)
            or config["max_model_len"] != 131072 or config["cache_bytes"] != 24 * 1024**3
            or config["kv_budget_mib"] != 11328 or config["allocator_gpu_utilization"] != .98
            or read("layout-cases.json") != cases or (root / "exit-status").read_text().strip() != "0"):
        raise ValueError("Layout calibration did not complete its fixed serving settings/cases")
    workers = [[read(f"worker-instance-{model + 1}-dp{rank}.json") for rank in (0, 1)] for model in (0, 1)]
    if (verify_runtime_control(root, workers)["dp_pause_sync_interval"] != 32
            or len({worker["ipc_name"] for ranks in workers for worker in ranks}) != 4):
        raise ValueError("Layout calibration changed default consensus cadence")
    for model, ranks in enumerate(workers):
        for rank, worker in enumerate(ranks):
            if ("A800" not in worker["gpu_name"] or worker["gpu_uuid"] != workers[0][rank]["gpu_uuid"]
                    or ranks[0]["gpu_uuid"] == ranks[1]["gpu_uuid"]
                    or worker["checkpoint"] != workers[0][0]["checkpoint"]
                    or worker["max_model_len"] != 131072 or worker["kv_cache_memory_bytes"] != 24 * 1024**3
                    or any(worker["groups"][name]["world_size"] != size
                           for name, size in (("dp", 2), ("ep", 2), ("tp", 1)))):
                raise ValueError("Layout calibration topology changed")
    mark, reference, trials = read("layout-mark.json"), read("layout-reference.json"), read("layout-trials.json")
    if mark["prompt_tokens"] != 3584 or mark["output_tokens"] != 1024 or "source_internal_row" not in mark:
        raise ValueError("Layout calibration altered paired Azure lengths")
    if [trial["case"] for trial in trials] != cases:
        raise ValueError("Layout calibration omitted a case")
    for trial, case in zip(trials, cases):
        requests = trial["requests"]
        if (len(requests) != case["population"] or len({request["request_id"] for request in requests}) != len(requests)
                or sum(request["rank_initial"] == 0 for request in requests) != case["rank_zero"]
                or any(request["model"] != 1 or request["prompt_tokens"] != 3584
                       or len(request["token_ids"]) != 1024 or request["token_ids"] != reference["token_ids"]
                       for request in requests)):
            raise ValueError("Layout calibration lost or changed request lengths/tokens/placement")
        verify_shared_decode_window(requests)
        for minimum, observed in zip((256, 768), trial["window_observations"]):
            verify_ownership(observed, [11328 / 1024] * 2)
            if ([[len(rank["requests"]) for rank in ranks] for ranks in observed]
                    != [[0, 0], [case["rank_zero"], case["population"] - case["rank_zero"]]]
                    or any(not request["movable"] or request["output_tokens"] < minimum
                           for rank in observed[1] for request in rank["requests"])):
                raise ValueError("Layout calibration changed population during stable decode")
        if len(trial["window_observations"]) != 2:
            raise ValueError("Layout calibration lacks both stable window observations")
    captures = read("layout-captures.json")
    if len(captures) != 2 or any(len(ranks) != 2 for ranks in captures) or any(
            not any("FULL" in entry["mode"] and entry["steps"] > 0 for entry in capture_entries(capture))
            for ranks in captures for capture in ranks):
        raise ValueError("Layout calibration lacks real graph execution on all ranks")
    result = cost_tables(trials)
    result["source_trials_sha256"] = hashlib.sha256((root / "layout-trials.json").read_bytes()).hexdigest()
    result["cases"] = len(cases)
    result["requests"] = sum(case["population"] for case in cases)
    result["source_files_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in LOOKUP_SOURCE_FILES}
    if output is not None:
        output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    report = verify(options.root, options.output)
    print(json.dumps({key: value for key, value in report.items() if key != "tables"}, indent=2))
