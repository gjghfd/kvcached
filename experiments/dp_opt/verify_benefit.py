import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


POLICIES = ("no_move", "own_model", "kv_only", "ilp")


def percentile(values, quantile):
    samples = sorted(values)
    position = (len(samples) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return samples[lower] + (samples[upper] - samples[lower]) * (position - lower)


def verify_ownership(observations, budgets=None):
    identifiers = set()
    for ranks in observations:
        for observation in ranks:
            blocks = [block for request in observation["requests"] for block in request["blocks"]]
            if len(blocks) != len(set(blocks)) or any(block <= 0 for block in blocks):
                raise ValueError("Duplicate or null-owned request KV blocks")
            if observation["mapped_bytes"] != len(observation["pages"]) * observation["page_bytes"]:
                raise ValueError("Physical mapped bytes differ from page inventory")
            for request in observation["requests"]:
                if request["id"] in identifiers:
                    raise ValueError("Request owned by multiple tenant/ranks")
                identifiers.add(request["id"])
            for page in observation["pages"]:
                owners = sorted(request["id"] for request in observation["requests"]
                                if any(block // observation["page_block_capacity"] == page["id"]
                                       for block in request["blocks"]))
                if owners != page["owners"]:
                    raise ValueError("Page owner inventory differs from blocks")
    if budgets is not None:
        for rank in (0, 1):
            if sum(ranks[rank]["mapped_bytes"] for ranks in observations) / 1024**3 > budgets[rank] + 1e-6:
                raise ValueError("Physical mapped KV exceeded the controlled budget")


def verify_runtime_control(root, workers):
    artifact = root / "runtime-control.json"
    if not artifact.exists():
        return None
    control = json.loads(artifact.read_text())
    interval = control["dp_pause_sync_interval"]
    if (type(interval) is not int or interval not in (1, 32)
            or control.get("same_interval_for_all_policies") is not True
            or control.get("idle_rank_expert_execution_not_disabled") is not True):
        raise ValueError("Invalid or unequal DP pause control")
    if any(worker.get("dp_pause_sync_interval") != interval for ranks in workers for worker in ranks):
        raise ValueError("Worker DP pause intervals differ from runtime control")
    return control


def verify_decode_boundary(before, minimum):
    if any(not request["movable"] or request["output_tokens"] < minimum
           for ranks in before for rank in ranks for request in rank["requests"]):
        raise ValueError("Bootstrap did not satisfy the predeclared decode boundary")


def verify_capacity_witness(witness, before, workload):
    budgets = workload["capacity_gib"]
    for observations in (witness["before"], witness["after_rollback"], before):
        verify_ownership(observations, budgets)
    inventory = lambda observations: [[{"mapped": rank["mapped_bytes"], "pages": sorted(rank["pages"], key=lambda page: page["id"]),
                                       "requests": sorted((request["id"], request["blocks"]) for request in rank["requests"])}
                                      for rank in ranks] for ranks in observations]
    if inventory(witness["before"]) != inventory(before) or inventory(witness["after_rollback"]) != inventory(before):
        raise ValueError("Capacity diagnostic changed request/page ownership")
    initial = [sum(ranks[rank]["mapped_bytes"] for ranks in before) for rank in (0, 1)]
    reserved = sum(len(request["blocks"]) * rank["block_bytes"]
                   for ranks in before for rank in ranks for request in rank["requests"])
    incoming = 256 * before[0][0]["block_bytes"]
    all_reserved = reserved + incoming + 96 * before[1][0]["block_bytes"]
    total_budget = round(sum(budgets) * 1024**3)
    if (witness["total_budget_bytes"] != total_budget or witness["initial_mapped_bytes"] != sum(initial)
            or witness["global_free_bytes"] != total_budget - sum(initial)
            or witness["arrival_a_reserved_bytes"] != incoming or witness["all_request_reserved_bytes"] != all_reserved
            or total_budget - sum(initial) < incoming or all_reserved > total_budget):
        raise ValueError("Global KV capacity is not sufficient or evidence arithmetic differs")
    if len(witness["probes"]) != 2 or {probe["rank"] for probe in witness["probes"]} != {0, 1}:
        raise ValueError("Both actual attention-DP allocation probes are required")
    for probe in witness["probes"]:
        rank = probe["rank"]
        allocation = probe["allocator"]
        during = [[allocation["during"] if model == 0 and index == rank else observation
                   for index, observation in enumerate(ranks)] for model, ranks in enumerate(before)]
        verify_ownership(during)
        extra = allocation["during"]["mapped_bytes"] - before[0][rank]["mapped_bytes"]
        peak = [sum(ranks[index]["mapped_bytes"] for ranks in during) / 1024**3 for index in (0, 1)]
        if (probe["model"] != 0 or probe["allocated"] is not True or probe["fits_budget"] is not False
                or probe["rollback_verified"] is not True or allocation["allocated"] is not True
                or len(allocation["blocks"]) != 256 or probe["additional_mapped_bytes"] != extra
                or probe["allocation_peak_gib"] != peak or peak[rank] <= budgets[rank]
                or extra != 384 * 1024**2):
            raise ValueError("Actual no-move allocation should exceed the per-GPU KV budget")


def verify(root, manifest=None):
    read = lambda name: json.loads((root / name).read_text())
    if manifest is not None:
        for line in manifest.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            artifact = root / name
            if not artifact.resolve().is_relative_to(root.resolve()):
                raise ValueError("Manifest escapes evidence root")
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Manifest checksum mismatch: {name}")
    config, summary = read("config.json"), read("summary.json")
    if summary["status"] != "passed" or (root / "exit-status").read_text().strip() != "0":
        raise ValueError("Incomplete benefit allocation")
    if len(summary["trials"]) != 4 * config["repeats"]:
        raise ValueError("Incomplete paired trial population")
    references = read("references.json")
    scenario = config.get("scenario", "capacity")
    if scenario not in ("capacity", "tradeoff", "tradeoff_long", "long_kv"):
        raise ValueError("Unknown predeclared workload scenario")
    expected_output = 512 if scenario in ("tradeoff_long", "long_kv") else 256
    if config["output_tokens"] != expected_output:
        raise ValueError("Output tokens differ from the predeclared fixture")
    counts = {"capacity": ((3, 3), (4, 3)), "long_kv": ((3, 3), (3, 3))}.get(scenario, ((2, 8), (4, 2)))
    total_requests = sum(sum(ranks) for ranks in counts) + 2
    by_prompt = {request["prompt_tokens"]: request for request in references if "prompt_tokens" in request}
    workers = [[read(f"worker-instance-{model + 1}-dp{rank}.json") for rank in (0, 1)] for model in (0, 1)]
    runtime_control = verify_runtime_control(root, workers)
    if len({worker["ipc_name"] for ranks in workers for worker in ranks}) != 4:
        raise ValueError("Tenant/rank KV namespaces are not independent")
    for ranks in workers:
        if ranks[0]["gpu_uuid"] == ranks[1]["gpu_uuid"]:
            raise ValueError("Attention DP workers share one GPU")
        for worker in ranks:
            if any(worker["groups"][name]["world_size"] != size for name, size in (("tp", 1), ("dp", 2), ("ep", 2))):
                raise ValueError("Unexpected DP/EP/TP topology")
            if len(worker["expert_ids"]) != 48 or worker["parameter_bytes"] <= 0:
                raise ValueError("Missing resident MoE parameters")
        for left, right in zip(ranks[0]["expert_ids"], ranks[1]["expert_ids"]):
            if set(left) & set(right) or set(left) | set(right) != set(range(128)):
                raise ValueError("EP expert partitions are not complementary")
    if any(workers[0][rank]["gpu_uuid"] != workers[1][rank]["gpu_uuid"] for rank in (0, 1)):
        raise ValueError("Tenant DP ranks are not colocated")
    trials, layouts = [], set()
    capture_profiles = []
    for trial in summary["trials"]:
        repeat, policy = trial["repeat"], trial["policy"]
        if (repeat, policy) in layouts or policy not in POLICIES:
            raise ValueError("Duplicate or unknown paired policy")
        layouts.add((repeat, policy))
        prefix = f"r{repeat}-{policy}/"
        requests, workload, decisions = [read(prefix + name + ".json") for name in ("requests", "workload", "decisions")]
        before = read(prefix + "before.json")
        verify_ownership(before, workload["capacity_gib"])
        if "minimum_decode_tokens" in config:
            if [[len(rank["requests"]) for rank in ranks] for ranks in before] != [list(ranks) for ranks in counts]:
                raise ValueError("Missing bootstrap requests at the arrival boundary")
            verify_decode_boundary(before, config["minimum_decode_tokens"])
        expected_pages = {"capacity": [[1, 1], [2, 1]], "tradeoff": [[2, 3], [2, 1]],
                          "tradeoff_long": [[2, 5], [3, 2]], "long_kv": [[11, 11], [8, 8]]}[scenario]
        if [[len(rank["pages"]) for rank in ranks] for ranks in before] != expected_pages and by_prompt:
            raise ValueError("Starting allocator page geometry changed")
        diagnostic = None
        if scenario == "long_kv":
            diagnostic = read(prefix + "capacity-witness.json")
            verify_capacity_witness(diagnostic, before, workload)
        if len(requests) != total_requests or len({request["request_id"] for request in requests}) != total_requests:
            raise ValueError("Lost, dropped or duplicated workload requests")
        if sum(request["role"] == "arrival" for request in requests) != 2:
            raise ValueError("Both independent tenant arrivals must finish")
        expected_ids = {f"r{repeat}-{policy}-bootstrap-{model}-{rank}-{index}"
                        for model, ranks in enumerate(counts)
                        for rank, count in enumerate(ranks) for index in range(count)}
        expected_ids.update(f"r{repeat}-{policy}-arrival-{model}" for model in (0, 1))
        if {request["request_id"] for request in requests} != expected_ids:
            raise ValueError("Completed IDs differ from the predeclared workload")
        for request in requests:
            if scenario == "long_kv":
                expected_prompt = ((3584 if request["model"] == 0 else 1024) if request["role"] == "arrival"
                                   else 3072 if request["model"] == 0 else 2048)
                if request["prompt_tokens"] != expected_prompt:
                    raise ValueError("Long-KV request prompt differs from the predeclared long workload")
            reference = (by_prompt[request["prompt_tokens"]] if by_prompt
                         else references[int(request["model"] == 0 and request["role"] == "arrival")])
            if request["token_ids"] != reference["token_ids"] or len(request["token_ids"]) != config["output_tokens"]:
                raise ValueError("Request differs from matching serial reference")
            cadence = request["cadence"]
            if (any(right["time"] < left["time"] or right["tokens"] < left["tokens"]
                    for left, right in zip(cadence, cadence[1:]))
                    or cadence[-1]["tokens"] != config["output_tokens"]):
                raise ValueError("Nonmonotonic or incomplete client cadence")
            if not request["arrived_at"] <= request["submitted_at"] <= request["first_token_at"] <= request["completed_at"]:
                raise ValueError("Invalid queue/first-token/completion ordering")
        migrations = 0
        capture_path = root / prefix / "capture.json"
        if capture_path.exists():
            capture = read(prefix + "capture.json")
            if len(capture) != 2 or any(len(ranks) != 2 for ranks in capture):
                raise ValueError("Missing tenant/rank capture evidence")
            profile = {"repeat": repeat, "policy": policy, "models": []}
            for model, ranks in enumerate(capture):
                full = []
                for rank, workers in enumerate(ranks):
                    if len(workers) != 1 or workers[0]["rank"] != rank:
                        raise ValueError("Capture evidence rank differs from allocated topology")
                    for item in workers[0]["histogram"]:
                        if item["ubatch"] or item["steps"] <= 0:
                            raise ValueError("Unexpected DBO or invalid capture step count")
                        if item["mode"].split(".")[-1] == "FULL":
                            if item["padded_tokens"] not in (1, 8, 16, 32, 64, 128, 256):
                                raise ValueError("Decode capture differs from frozen sparse profile")
                            full.append(item)
                if not full:
                    raise ValueError("Missing FULL-decode replay evidence")
                profile["models"].append({"model": model,
                                          "replay_sizes": sorted({item["padded_tokens"] for item in full}),
                                          "steps_by_size": {str(size): sum(item["steps"] for item in full if item["padded_tokens"] == size)
                                                            for size in sorted({item["padded_tokens"] for item in full})}})
            capture_profiles.append(profile)
        for decision in decisions:
            migrations += len(decision["moves"])
            if policy == "no_move" and decision["moves"]:
                raise ValueError("No-move baseline migrated a request")
            if policy == "own_model" and any(move["model"] != decision["arriving_model"] for move in decision["moves"]):
                raise ValueError("Own-model baseline moved another tenant")
            if decision["end_to_end_control_s"] < decision.get("transaction_s_before_resume", 0):
                raise ValueError("Reported control time excludes resume")
            for name in ("observations", "before_admission", "after_admission"):
                if decision.get(name) is not None:
                    verify_ownership(decision[name], workload["capacity_gib"])
            for move in decision["moves"]:
                source = decision["observations"][move["model"]][move["source"]]
                destination = decision["before_admission"][move["model"]][move["destination"]]
                old = next(request for request in source["requests"] if request["id"] == move["request_id"])
                if move["kv_bytes"] != len(old["blocks"]) * source["block_bytes"] or len(move["sha256"]) != 64:
                    raise ValueError("Copied KV length or checksum metadata invalid")
                if not any(request["id"] == move["request_id"] for request in destination["requests"]):
                    raise ValueError("Migrated request missing from destination ownership")
                if any(value > cap + 1e-6 for value, cap in zip(move["copy_peak_gib"], workload["capacity_gib"])):
                    raise ValueError("Migration copy peak exceeded KV budget")
        latency = [request["completed_at"] - request["arrived_at"] for request in requests]
        ttft = [request["first_token_at"] - request["arrived_at"] for request in requests]
        tpot = [(request["completed_at"] - request["first_token_at"]) / (len(request["token_ids"]) - 1)
                for request in requests]
        makespan = max(request["completed_at"] for request in requests) - workload["origin"]
        met = sum(value <= config["deadline_s"] for value in latency)
        recomputed = {"makespan_s": makespan, "output_throughput_tokens_s": total_requests * config["output_tokens"] / makespan,
                      "request_latency_sum_s": sum(latency), "mean_request_latency_s": statistics.mean(latency),
                      "p99_request_latency_s": percentile(latency, .99), "mean_ttft_s": statistics.mean(ttft),
                      "p99_ttft_s": percentile(ttft, .99), "mean_tpot_s": statistics.mean(tpot),
                      "p99_tpot_s": percentile(tpot, .99), "deadline_met_requests": met,
                      "deadline_goodput_requests_s": met / makespan, "migration_count": migrations,
                      "control_s": sum(decision["end_to_end_control_s"] for decision in decisions),
                      "solve_s": sum(decision["solve_s"] for decision in decisions),
                      "arrival_a_ttft_s": next(request["first_token_at"] - request["arrived_at"]
                                                for request in requests if request["role"] == "arrival" and request["model"] == 0)}
        if diagnostic is not None:
            recomputed["diagnostic_control_s"] = diagnostic["diagnostic_control_s"]
            recomputed["arrival_a_queue_s"] = next(request["submitted_at"] - request["arrived_at"] for request in requests
                                                   if request["role"] == "arrival" and request["model"] == 0)
        for name, value in recomputed.items():
            if name == "arrival_a_queue_s":
                continue
            if not math.isclose(trial[name], value, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Raw metric mismatch: {prefix}{name}")
        trials.append(dict(recomputed, repeat=repeat, policy=policy))
    paired = {}
    for baseline in ("no_move", "own_model", "kv_only"):
        comparisons = []
        for repeat in range(config["repeats"]):
            left = next(trial for trial in trials if trial["repeat"] == repeat and trial["policy"] == baseline)
            right = next(trial for trial in trials if trial["repeat"] == repeat and trial["policy"] == "ilp")
            comparisons.append({"repeat": repeat,
                                "throughput_gain_fraction": right["output_throughput_tokens_s"] / left["output_throughput_tokens_s"] - 1,
                                "latency_sum_reduction_fraction": 1 - right["request_latency_sum_s"] / left["request_latency_sum_s"],
                                "a_ttft_reduction_fraction": 1 - right["arrival_a_ttft_s"] / left["arrival_a_ttft_s"],
                                "deadline_goodput_gain_fraction": (right["deadline_goodput_requests_s"] / left["deadline_goodput_requests_s"] - 1
                                                                    if left["deadline_goodput_requests_s"] else None)})
        paired[baseline] = {"per_repeat": comparisons,
                            "mean": {metric: statistics.mean(item[metric] for item in comparisons if item[metric] is not None)
                                     if any(item[metric] is not None for item in comparisons) else None
                                     for metric in comparisons[0] if metric != "repeat"}}
    return {"status": "raw_verified", "paired_repeats": config["repeats"], "trials": trials,
            "ilp_vs_baselines": paired, "scenario": scenario,
            "capture_profiles": capture_profiles,
            "runtime_control": runtime_control,
            "minimum_decode_tokens": config.get("minimum_decode_tokens"),
            "scope": "controlled_capacity_declared_decode_boundary_not_joint_calibrated_global_optimum"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.root, args.manifest)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
