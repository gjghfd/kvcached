import argparse
import json
import random
from pathlib import Path

from analyze_tail_slo import analyze, compare_group, metrics
from verify_benefit import percentile


def analyze_case(root):
    root = Path(root)
    trace = json.loads((root / "trace.json").read_text())
    config = json.loads((root / "config.json").read_text())
    start, stop = trace["evaluation_start_s"], trace["duration_s"]
    selected = {mark["request_id"] for mark in trace["requests"] if start <= mark["arrival_s"] < stop}
    comparisons = {}
    for baseline in ("no_move", "no_move_fair"):
        all_records = analyze(root, baseline, "kv_only")
        before, after = [{request["request_id"]: request for request in json.loads(
            (root / f"r0-{policy}" / "requests.json").read_text())} for policy in (baseline, "kv_only")]
        code = {identifier for identifier in selected if before[identifier]["model"] == 0}
        conv = selected - code
        old_conv = {identifier for identifier, request in before.items() if request["model"] == 1
                    and any(request["reserved_at"] <= before[code_id]["arrived_at"] < request["completed_at"]
                            for code_id in code)}
        groups = {"all_requests": selected, "Code": code, "Conv": conv, "old_Conv": old_conv}
        comparisons[baseline] = {"full_including_warmup": all_records,
                                "evaluation_window": {name: compare_group(before, after, identifiers, config["deadline_s"])
                                                      for name, identifiers in groups.items()},
                                "Code_paired_requests": [{"request_id": identifier,
                                    "baseline": metrics(before[identifier]), "kv_only": metrics(after[identifier])}
                                    for identifier in sorted(code)]}
    reductions = [comparison["evaluation_window"]["Code"]["changes"]["latency_s"]["p95_s"]["reduction_fraction"]
                  for comparison in comparisons.values()
                  if comparison["evaluation_window"]["Code"]["requests"] > 0]
    count = sum(mark["model"] == 0 and mark["request_id"] in selected for mark in trace["requests"])
    return {"status": "descriptive_paired_case_verified", "evaluation_start_s": start,
            "Code_samples": count, "minimum_Code_p95_reduction_fraction": min(reductions) if len(reductions) == 2 else None,
            "observed_50pct_both_baselines": len(reductions) == 2 and min(reductions) >= .5,
            "large_sample_or_heldout_confirmation": False, "comparisons": comparisons}


def select_best(cases):
    available = [case for case in cases if case["analysis"]["minimum_Code_p95_reduction_fraction"] is not None]
    if not available:
        return None
    return max(available, key=lambda case: case["analysis"]["minimum_Code_p95_reduction_fraction"])


def analyze_confirmations(roots, seeds, bootstrap_repeats=2000, bootstrap_seed=20261024):
    if len(roots) != len(seeds) or len(set(seeds)) != len(seeds) or not roots:
        raise ValueError("Confirmation must include every predeclared distinct seed exactly once")
    fixed = None
    comparisons = {}
    seed_results = []
    populations = {baseline: {"before": {}, "after": {}, "groups": {name: set() for name in
                   ("all_requests", "Code", "Conv", "old_Conv")}} for baseline in ("no_move", "no_move_fair")}
    clusters = []
    for root, seed in zip(map(Path, roots), seeds):
        setting = json.loads((root / "stress-setting.json").read_text())
        config = json.loads((root / "config.json").read_text())
        trace = json.loads((root / "trace.json").read_text())
        identity = {name: setting[name] for name in ("output_tokens", "budget_mib_per_gpu", "conv_rate_rps", "code_rate_rps")}
        identity.update({name: trace[name] for name in ("duration_s", "evaluation_start_s", "prompt_bank_sha256")})
        identity["deadline_s"] = config["deadline_s"]
        if (setting["seed"] != seed or trace["seed"] != seed
                or setting["fixed_validation_conv_rate_rps"] != setting["conv_rate_rps"]):
            raise ValueError("Heldout seed or fixed rate changed")
        if fixed is not None and identity != fixed:
            raise ValueError("Heldout workload, budget or SLO changed between seeds")
        fixed = identity
        analysis = analyze_case(root)
        seed_results.append({"seed": seed, "setting": setting, "analysis": analysis})
        cluster = {}
        for baseline, population in populations.items():
            records = {policy: {record["request_id"]: record for record in json.loads(
                       (root / f"r0-{policy}" / "requests.json").read_text())} for policy in (baseline, "kv_only")}
            groups = analysis["comparisons"][baseline]["evaluation_window"]
            for name, group in groups.items():
                population["groups"][name].update(f"{seed}:{identifier}" for identifier in group["request_ids"])
            for side, policy in (("before", baseline), ("after", "kv_only")):
                population[side].update({f"{seed}:{identifier}": record for identifier, record in records[policy].items()})
            cluster[baseline] = {policy: [metrics(records[policy][identifier])["latency_s"]
                                 for identifier in groups["Code"]["request_ids"]] for policy in (baseline, "kv_only")}
        clusters.append(cluster)
    for baseline, population in populations.items():
        comparisons[baseline] = {name: compare_group(population["before"], population["after"], identifiers, fixed["deadline_s"])
                                 for name, identifiers in population["groups"].items()}
    rng = random.Random(bootstrap_seed)
    draws = {baseline: [] for baseline in populations}
    minimums = []
    for unused in range(bootstrap_repeats):
        sample = [rng.choice(clusters) for index in clusters]
        reductions = []
        for baseline in populations:
            before = [value for cluster in sample for value in cluster[baseline][baseline]]
            after = [value for cluster in sample for value in cluster[baseline]["kv_only"]]
            if before and percentile(before, .95) > 0:
                reduction = 1 - percentile(after, .95) / percentile(before, .95)
                draws[baseline].append(reduction)
                reductions.append(reduction)
        if len(reductions) == 2:
            minimums.append(min(reductions))
    intervals = {baseline: [percentile(values, .025), percentile(values, .975)] if values else None
                 for baseline, values in draws.items()}
    minimum_interval = [percentile(minimums, .025), percentile(minimums, .975)] if minimums else None
    count = comparisons["no_move"]["Code"]["requests"]
    observed = (count > 0 and min(comparison["Code"]["changes"]["latency_s"]["p95_s"]["reduction_fraction"]
                                 for comparison in comparisons.values()) >= .5)
    confirmed = bool(observed and count >= 100 and len(seeds) >= 8 and minimum_interval
                     and minimum_interval[0] >= .5)
    return {"status": "all_predeclared_heldout_seeds_analyzed", "seeds": seeds, "fixed_setting": fixed,
            "Code_samples": count, "per_seed": seed_results, "pooled_comparisons": comparisons,
            "paired_seed_cluster_bootstrap": {"repeats": bootstrap_repeats, "seed": bootstrap_seed,
                "percentile_interval_95": intervals, "minimum_reduction_interval_95": minimum_interval,
                "valid_draws": len(minimums), "scope": "seed_cluster_percentile_bootstrap_small_seed_count_limitations"},
            "observed_50pct_Code_p95_both_baselines": observed,
            "confirmed_50pct_Code_p95_both_baselines": confirmed,
            "scope": "fixed_heldout_best_in_declared_grid_not_global_optimum_or_overall_throughput_gain"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.root / "stress-manifest.json").read_text())
    cases = [{**case, "analysis": analyze_case(args.root / case["directory"])} for case in manifest["completed_cases"]]
    result = {"cases": cases, "best_observed": select_best(cases),
              "scope": "best_in_declared_exploration_grid_not_global_optimum_or_large_sample_confirmation"}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed_cases": len(cases), "best": result["best_observed"]["setting"] if cases else None}))


if __name__ == "__main__":
    main()
