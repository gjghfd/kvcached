import argparse
import bisect
import collections
import hashlib
import itertools
import json
import math
import random
import statistics
from pathlib import Path


PAGE_MIB = 96
PAGE_BLOCKS = 64
BLOCK_MIB = 1.5
BLOCK_TOKENS = 16


def make_pool(requests):
    owners = collections.defaultdict(set)
    cursor = 1
    for request in requests:
        for page in range(cursor // PAGE_BLOCKS, (cursor + request["blocks"] - 1) // PAGE_BLOCKS + 1):
            owners[page].add(request["id"])
        cursor += request["blocks"]
    return {"requests": requests, "owners": dict(owners)}


def pool_at(pool, elapsed):
    remaining = [request for request in pool["requests"] if request["residual_s"] > elapsed]
    identifiers = {request["id"] for request in remaining}
    pages = {0} | {page for page, owners in pool["owners"].items() if owners & identifiers}
    blocks = sum(request["blocks"] for request in remaining)
    return {"requests": remaining, "pages": len(pages),
            "free_slots": len(pages) * PAGE_BLOCKS - 1 - blocks}


def packing_plan(pools, code_blocks, capacity_mib, elapsed=0):
    ranks = [pool_at(pool, elapsed) for pool in pools]
    incoming = math.ceil(max(0, code_blocks - PAGE_BLOCKS + 1) / PAGE_BLOCKS) * PAGE_MIB
    used = [(rank["pages"] + 1) * PAGE_MIB for rank in ranks]
    direct = [rank for rank in (0, 1) if used[rank] + incoming <= capacity_mib]
    if direct:
        return {"target": direct[0], "moves": 0, "copy_reserved_mib_upper": 0, "direct": True}
    if 2 * PAGE_MIB + incoming > capacity_mib:
        return None
    if any(request["movable_after_s"] > elapsed for rank in ranks for request in rank["requests"]):
        return None
    candidates = []
    for source in (0, 1):
        destination = 1 - source
        blocks = sum(request["blocks"] for request in ranks[source]["requests"])
        additional = math.ceil(max(0, blocks - ranks[destination]["free_slots"]) / PAGE_BLOCKS) * PAGE_MIB
        if used[destination] + additional <= capacity_mib:
            candidates.append({"target": source, "moves": len(ranks[source]["requests"]),
                               "copy_reserved_mib_upper": blocks * BLOCK_MIB, "direct": False})
    return min(candidates, key=lambda candidate: (candidate["moves"], candidate["copy_reserved_mib_upper"],
                                                 candidate["target"])) if candidates else None


def waiting_times(pools, code_blocks, capacity_mib):
    times = sorted({0} | {request[name] for pool in pools for request in pool["requests"]
                         for name in ("residual_s", "movable_after_s")})
    baseline = None
    treatment = None
    plan = None
    incoming = math.ceil(max(0, code_blocks - PAGE_BLOCKS + 1) / PAGE_BLOCKS) * PAGE_MIB
    for elapsed in times:
        ranks = [pool_at(pool, elapsed) for pool in pools]
        if baseline is None and any((rank["pages"] + 1) * PAGE_MIB + incoming <= capacity_mib for rank in ranks):
            baseline = elapsed
        candidate = packing_plan(pools, code_blocks, capacity_mib, elapsed)
        if treatment is None and candidate is not None:
            treatment, plan = elapsed, candidate
        if baseline is not None and treatment is not None:
            break
    return baseline, treatment, plan


def distribution(values):
    if not values:
        return None
    percentiles = statistics.quantiles(values, n=100, method="inclusive") if len(values) > 1 else [values[0]] * 99
    return {"samples": len(values), "mean": statistics.mean(values), "p50": percentiles[49],
            "p90": percentiles[89], "p95": percentiles[94], "p99": percentiles[98]}


def evaluate(conv_marks, code_marks, output_tokens, conv_rate, code_rate, capacity_mib,
             samples=10000, seed=20261024, decode_step_s=.02, prefill_s=.2,
             code_service_s=57, treatment_code_service_s=57, move_control_s=5):
    if (not conv_marks or not code_marks or samples <= 0
            or any(not math.isfinite(value) or value <= 0 for value in
                   (conv_rate, code_rate, capacity_mib, decode_step_s, prefill_s,
                    code_service_s, treatment_code_service_s, move_control_s))):
        raise ValueError("Require nonempty marks and positive finite proxy parameters")
    cumulative, marks, total, count = [], [], 0, 0
    memory_service, service_sum = 0, 0
    for mark in conv_marks:
        output = mark["output_tokens"] if output_tokens is None else output_tokens
        service = prefill_s + decode_step_s * output
        blocks = math.ceil((mark["prompt_tokens"] + output) / BLOCK_TOKENS)
        weight = mark.get("count", 1)
        total += service * weight
        service_sum += service * weight
        memory_service += blocks * BLOCK_MIB * service * weight
        count += weight
        cumulative.append(total)
        marks.append((blocks, service))
    mean_service = service_sum / count
    mean_population = conv_rate * mean_service
    rng = random.Random(seed)
    counters = collections.Counter()
    baseline_latencies, treatment_latencies, waits, moves = [], [], [], []
    for sample in range(samples):
        threshold, product, population = math.exp(-mean_population), 1, 0
        while product > threshold:
            product *= rng.random()
            population += 1
        population -= 1
        requests = []
        for identifier in range(population):
            blocks, service = marks[bisect.bisect_right(cumulative, rng.random() * total)]
            age = rng.random() * service
            requests.append({"id": identifier, "blocks": blocks, "residual_s": service - age,
                             "movable_after_s": max(0, prefill_s - age)})
        pools = [make_pool(requests[rank::2]) for rank in (0, 1)]
        code_mark = rng.choice(code_marks)
        code_blocks = math.ceil((code_mark["prompt_tokens"] + code_mark["output_tokens"]) / BLOCK_TOKENS)
        ranks = [pool_at(pool, 0) for pool in pools]
        used = [(rank["pages"] + 1) * PAGE_MIB for rank in ranks]
        if max(used) > capacity_mib:
            counters["offered_over_budget"] += 1
            continue
        baseline, treatment, plan = waiting_times(pools, code_blocks, capacity_mib)
        if baseline is None or treatment is None:
            counters["unsupported_code_even_empty"] += 1
            continue
        counters["safe_supported"] += 1
        if baseline > 0:
            counters["local_blocked"] += 1
            incoming = math.ceil(max(0, code_blocks - PAGE_BLOCKS + 1) / PAGE_BLOCKS) * PAGE_MIB
            if 2 * capacity_mib - sum(used) >= incoming:
                counters["global_enough_local_blocked"] += 1
        if treatment == 0 and not plan["direct"]:
            counters["immediate_constructive_migration"] += 1
        control = .1 if plan["direct"] else move_control_s
        before = baseline + code_service_s
        after = treatment + control + treatment_code_service_s
        baseline_latencies.append(before)
        treatment_latencies.append(after)
        waits.append(baseline)
        moves.append(plan["moves"])
        counters["per_request_latency_halved"] += after <= before * .5
    before, after = distribution(baseline_latencies), distribution(treatment_latencies)
    return {"status": "conditional_single_Code_CPU_constructive_envelope_not_GPU_result",
            "output_tokens": output_tokens, "conv_rate_rps": conv_rate, "code_rate_rps": code_rate,
            "capacity_mib_per_gpu": capacity_mib, "samples": samples, "seed": seed,
            "mean_Conv_proxy_service_s": mean_service, "mean_offered_Conv_count": mean_population,
            "mean_offered_Conv_reserved_mib": conv_rate * memory_service / count,
            "code_service_s": code_service_s, "treatment_code_service_s": treatment_code_service_s,
            "decode_step_proxy_s": decode_step_s, "prefill_proxy_s": prefill_s,
            "move_control_proxy_s": move_control_s,
            "other_Code_presence_unconstrained_Poisson_proxy": 1 - math.exp(-code_rate * treatment_code_service_s),
            "counters": dict(counters), "natural_wait_no_future_Conv_arrivals_s": distribution(waits),
            "baseline_Code_latency_proxy_s": before, "constructive_Code_latency_proxy_s": after,
            "migration_count": distribution(moves),
            "conditional_latency_reduction_fraction": {metric: 1 - after[metric] / before[metric]
                                                        for metric in ("mean", "p50", "p90", "p95", "p99")}
            if before and after else None,
            "limitations": ["Single incoming Code conditioned on no other resident Code; collision proxy is separate",
                            "Offered M/G/infinity population, balanced counts, uniform ages; no live queue feedback",
                            "Future Conv arrivals omitted; natural waiting is a no-refill comparator, not executed fair baseline",
                            "Whole-source packing is constructive, not an ILP-optimal plan or actual allocator probe",
                            "Long-output service extrapolation and fixed move cost require fresh joint GPU profiling",
                            "No old-Conv slowdown or compute contention modeled; do not claim global latency/throughput gains",
                            "Unsafe and unsupported offered states counted separately, not silently removed"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    conv_marks = [mark for mark in trace["requests"] if mark["model"] == 1]
    code_marks = [mark for mark in trace["requests"] if mark["model"] == 0]
    settings = [(None, .61, 11328, 57)]
    settings.extend((output, rate, capacity, service)
                    for (output, rate), capacity, service in itertools.product(
                        ((8192, .025), (12288, .015), (16384, .01)), (11328, 10240), (57, 75)))
    settings.extend((8192, .03, capacity, 75) for capacity in (11328, 10240))
    results = [evaluate(conv_marks, code_marks, output, rate, .0006, capacity, samples=args.samples,
                        seed=20261024, treatment_code_service_s=service)
               for output, rate, capacity, service in settings]
    sensitivity = []
    if args.sensitivity:
        sensitivity = [evaluate(conv_marks, code_marks, 8192, .025, .0006, 10240, samples=args.samples,
                                seed=20261024, decode_step_s=step, treatment_code_service_s=service,
                                move_control_s=control)
                       for step, service, control in itertools.product((.015, .02, .03), (57, 75, 90), (5, 10))]
    result = {"source_trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
              "scope": "local_design_only_new_synthetic_long_output_candidates_not_replacing_frozen_live_trace",
              "results": results, "sensitivity": sensitivity}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for entry in results:
        print(json.dumps({key: entry[key] for key in ("output_tokens", "conv_rate_rps", "capacity_mib_per_gpu",
                                                     "treatment_code_service_s", "counters",
                                                     "conditional_latency_reduction_fraction")}))


if __name__ == "__main__":
    main()
