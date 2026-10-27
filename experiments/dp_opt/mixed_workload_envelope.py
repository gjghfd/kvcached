import argparse
import bisect
import collections
import json
import math
import random
import statistics
from pathlib import Path


PAGE_MIB = 96
BLOCK_MIB = 1.5
BLOCK_TOKENS = 16
PAGE_BLOCKS = 64


class MarkSampler:
    def __init__(self, records, minimum_input=0, maximum_input=None, minimum_output=1, fixed_output=None,
                 decode_step_s=.04, prefill_s=.2):
        self.marks, self.arrivals, self.residents = [], [], []
        arrival_total, resident_total = 0, 0
        for record in records:
            context, output = record["prompt_tokens"], record["output_tokens"]
            if (context < minimum_input or maximum_input is not None and context >= maximum_input
                    or output < minimum_output):
                continue
            output = output if fixed_output is None else fixed_output
            service = prefill_s + decode_step_s * output
            frequency = record.get("count", 1)
            arrival_total += frequency
            resident_total += frequency * service
            self.marks.append((math.ceil((context + output) / BLOCK_TOKENS), service))
            self.arrivals.append(arrival_total)
            self.residents.append(resident_total)
        if not self.marks:
            raise ValueError("Empty workload condition")
        self.mean_service_s = resident_total / arrival_total
        self.prefill_s = prefill_s

    def sample(self, rng, resident=False):
        cumulative = self.residents if resident else self.arrivals
        return self.marks[bisect.bisect_right(cumulative, rng.random() * cumulative[-1])]


def poisson(rng, mean):
    threshold, product, count = math.exp(-mean), 1, 0
    while product > threshold:
        product *= rng.random()
        count += 1
    return count - 1


def make_pool(requests):
    cursor, resident = 1, []
    for identifier, blocks, residual, movable in requests:
        resident.append({"id": identifier, "blocks": tuple(range(cursor, cursor + blocks)),
                         "residual_s": residual, "movable": movable})
        cursor += blocks
    return {"requests": resident, "pages": set(range(math.ceil(cursor / PAGE_BLOCKS)))}


def mapped_mib(observations):
    return [sum(len(ranks[rank]["pages"]) * PAGE_MIB for ranks in observations) for rank in (0, 1)]


def free_slots(pool):
    return len(pool["pages"]) * PAGE_BLOCKS - 1 - sum(len(request["blocks"]) for request in pool["requests"])


def increment_mib(pool, blocks):
    return math.ceil(max(0, blocks - free_slots(pool)) / PAGE_BLOCKS) * PAGE_MIB


def direct_fit(observations, blocks, capacity_mib):
    used = mapped_mib(observations)
    return any(used[rank] + increment_mib(observations[0][rank], blocks) <= capacity_mib for rank in (0, 1))


def remove_request(pool, identifier):
    remaining = [request for request in pool["requests"] if request["id"] != identifier]
    owned = {block // PAGE_BLOCKS for request in remaining for block in request["blocks"]}
    return {"requests": remaining, "pages": {page for page in pool["pages"] if page == 0 or page in owned}}


def conv_one_move(observations, incoming_blocks, capacity_mib, all_decode_required):
    active = [request for ranks in observations for pool in ranks for request in pool["requests"]]
    if all_decode_required and any(not request["movable"] for request in active):
        return None
    used, candidates = mapped_mib(observations), []
    for source in (0, 1):
        destination = 1 - source
        source_pool, destination_pool = observations[1][source], observations[1][destination]
        for request in source_pool["requests"]:
            if not request["movable"]:
                continue
            additional = increment_mib(destination_pool, len(request["blocks"]))
            if used[destination] + additional > capacity_mib:
                continue
            after = remove_request(source_pool, request["id"])
            released = (len(source_pool["pages"]) - len(after["pages"])) * PAGE_MIB
            final_used = list(used)
            final_used[source] -= released
            final_used[destination] += additional
            for target in (0, 1):
                incoming = increment_mib(observations[0][target], incoming_blocks)
                if final_used[target] + incoming <= capacity_mib:
                    copy_peak = list(used)
                    copy_peak[destination] += additional
                    final_after_admission = list(final_used)
                    final_after_admission[target] += incoming
                    candidates.append({"id": request["id"], "source": source, "target": target,
                                       "copy_logical_mib": len(request["blocks"]) * BLOCK_MIB,
                                       "released_source_mib": released,
                                       "copy_peak_mib": copy_peak,
                                       "final_used_mib": final_used,
                                       "incoming_incremental_mib": incoming,
                                       "final_after_admission_mib": final_after_admission})
    return min(candidates, key=lambda item: (item["copy_logical_mib"], -item["released_source_mib"], item["id"])) if candidates else None


def natural_wait(observations, blocks, capacity_mib):
    events = sorted((request["residual_s"], model, rank, request["id"])
                    for model, ranks in enumerate(observations) for rank, pool in enumerate(ranks)
                    for request in pool["requests"])
    remaining = [list(ranks) for ranks in observations]
    for residual, model, rank, identifier in events:
        remaining[model][rank] = remove_request(remaining[model][rank], identifier)
        if direct_fit(remaining, blocks, capacity_mib):
            return residual
    return None


def describe(values):
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p90": sorted(values)[math.ceil(.9 * len(values)) - 1]} if values else {"count": 0}


def conservative_ilp_snapshot(observations, incoming_blocks, capacity_mib):
    from live_controller import select_candidates

    snapshot = {"counts": [[len(pool["requests"]) for pool in ranks] for ranks in observations],
                "initial_gib": [value / 1024 for value in mapped_mib(observations)],
                "capacity_gib": [capacity_mib / 1024] * 2,
                "new_gib": math.ceil(incoming_blocks / PAGE_BLOCKS) * PAGE_MIB / 1024,
                "source_pages": [], "movable": []}
    for model, ranks in enumerate(observations):
        for rank, pool in enumerate(ranks):
            for page in sorted(pool["pages"]):
                owners = [request["id"] for request in pool["requests"]
                          if any(block // PAGE_BLOCKS == page for block in request["blocks"])]
                snapshot["source_pages"].append({"id": f"{model}:{rank}:{page}", "rank": rank,
                                                 "pool": f"{model}:{rank}", "owners": owners,
                                                 "pinned": page == 0, "memory_gib": PAGE_MIB / 1024})
            if model == 1:
                for request in pool["requests"]:
                    if request["movable"]:
                        memory = math.ceil(len(request["blocks"]) / PAGE_BLOCKS) * PAGE_MIB / 1024
                        snapshot["movable"].append({"id": request["id"], "model": model, "source": rank,
                                                     "destination_pool": f"{model}:{1 - rank}", "memory_gib": memory,
                                                     "move_cost_s": 1 + memory})
    snapshot["movable"] = select_candidates(snapshot, limit=2)
    snapshot["performance_cost_tables_s"] = [[0] * (sum(snapshot["counts"][0]) + 2),
                                              [0] * (sum(snapshot["counts"][1]) + 1)]
    return snapshot


def evaluate(code, conv, code_rate, conv_rate, capacity_mib, samples, seed, require_all_decode=True,
             ilp_witness_limit=0):
    rng, counters, waits, populations, examples = random.Random(seed), collections.Counter(), [], [], []
    cost_gate_counts, witnesses = {str(cost): 0 for cost in (.5, 1, 2, 3, 5)}, []
    for sample_index in range(samples):
        observations, next_id = [], 0
        for model, sampler, rate in ((0, code, code_rate), (1, conv, conv_rate)):
            count = poisson(rng, rate * sampler.mean_service_s)
            counts = [count // 2, count - count // 2]
            rng.shuffle(counts)
            ranks = []
            for count in counts:
                requests = []
                for _ in range(count):
                    blocks, service = sampler.sample(rng, resident=True)
                    age = rng.uniform(0, service)
                    requests.append((f"{model}-{next_id}", blocks, service - age, age >= sampler.prefill_s))
                    next_id += 1
                ranks.append(make_pool(requests))
            observations.append(ranks)
        populations.append(next_id)
        incoming, _ = code.sample(rng)
        used = mapped_mib(observations)
        if max(used) > capacity_mib:
            counters["offered_state_over_budget"] += 1
            continue
        counters["safe"] += 1
        if not direct_fit(observations, incoming, capacity_mib) and incoming * BLOCK_MIB > capacity_mib - 2 * PAGE_MIB:
            counters["single_code_too_large"] += 1
        if direct_fit(observations, incoming, capacity_mib) or 2 * capacity_mib - sum(used) < incoming * BLOCK_MIB:
            continue
        counters["global_enough_local_blocked"] += 1
        move = conv_one_move(observations, incoming, capacity_mib, all_decode_required=False)
        if move is not None:
            counters["conv_move_recoverable_ignoring_other_prefill_gate"] += 1
        move = conv_one_move(observations, incoming, capacity_mib, all_decode_required=require_all_decode)
        if move is None:
            continue
        counters["conv_move_recoverable_current_gate"] += 1
        wait = natural_wait(observations, incoming, capacity_mib)
        if wait is None:
            raise ValueError("Migration fits but natural empty-pool admission never fits")
        waits.append(wait)
        if len(witnesses) < ilp_witness_limit:
            from ilp_reference import enumerate_targets

            snapshot = conservative_ilp_snapshot(observations, incoming, capacity_mib)
            plan = enumerate_targets(snapshot, max_moves=2, time_limit_s=1)
            own = dict(snapshot, movable=[])
            own_plan = enumerate_targets(own, max_moves=0, time_limit_s=1)
            witnesses.append({"sample": sample_index, "snapshot": snapshot, "conv_migration_plan": plan,
                              "own_model_plan": own_plan, "physical_geometry_one_move": move,
                              "scope": "offline_page_owner_feasibility_not_GPU_or_performance_calibrated"})
        for cost in (.5, 1, 2, 3, 5):
            cost_gate_counts[str(cost)] += wait > cost
        if len(examples) < 3:
            examples.append({"sample": sample_index, "initial_mapped_mib": used,
                             "global_free_mib": 2 * capacity_mib - sum(used),
                             "incoming_logical_mib": incoming * BLOCK_MIB, "move": move,
                             "natural_wait_s_proxy": wait,
                             "counts_code_conv": [[len(pool["requests"]) for pool in ranks] for ranks in observations]})
    denominator = counters["safe"]
    return {"code_rate_rps": code_rate, "conv_rate_rps": conv_rate, "capacity_mib_per_gpu": capacity_mib,
            "samples": samples, "seed": seed, "mean_total_offered_population": statistics.mean(populations),
            "counters": dict(counters), "offered_state_over_budget_fraction": counters["offered_state_over_budget"] / samples,
            "blocked_fraction_of_safe": counters["global_enough_local_blocked"] / denominator if denominator else None,
            "recoverable_fraction_of_safe": counters["conv_move_recoverable_current_gate"] / denominator if denominator else None,
            "recoverable_fraction_of_all_offered_states": counters["conv_move_recoverable_current_gate"] / samples,
            "natural_wait_for_recoverable_s": describe(waits),
            "recoverable_with_wait_gt_assumed_control_cost": cost_gate_counts, "examples": examples,
            "conservative_ilp_witnesses": witnesses}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--ilp-witness-limit", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior envelope")
    records = {dataset: json.loads((args.profile / f"azure-{dataset}-joint.json").read_text()) for dataset in ("code", "conv")}
    samplers = {
        "code_all": MarkSampler(records["code"]),
        "code_input_ge6000": MarkSampler(records["code"], minimum_input=6000),
        "conv_all": MarkSampler(records["conv"]),
        "conv_shortinput_longoutput": MarkSampler(records["conv"], maximum_input=2048, minimum_output=256)}
    cases = []
    for conv_rate in (1, 1.5, 2, 2.5):
        cases.append(("Azure_full_Code_and_Conversation", "code_all", "conv_all", .5, conv_rate, 2112))
    for conv_rate in (.5, .75, 1):
        cases.append(("Azure_declared_conditional_largeCode_shortinput_longdecodeConv", "code_input_ge6000",
                      "conv_shortinput_longoutput", .1, conv_rate, 2112))
    results = []
    for index, (name, code_name, conv_name, code_rate, conv_rate, capacity) in enumerate(cases):
        result = evaluate(samplers[code_name], samplers[conv_name], code_rate, conv_rate, capacity,
                          args.samples, 20261021 + index, ilp_witness_limit=args.ilp_witness_limit)
        result["workload"] = name
        results.append(result)
        print(name, code_rate, conv_rate, result["counters"], flush=True)
    report = {"scope": "CPU_stationary_MGinf_offered_state_countLB_page_owner_geometry_not_GPU_trace_or_benefit",
              "assumptions": {"decode_step_s": .04, "prefill_s": .2, "page_mib": 96, "block_tokens": 16,
                              "block_mib": 1.5, "length_biased_residents": True, "uniform_resident_age": True,
                              "all_decode_gate": True, "one_conv_move_copy_before_release": True},
              "limitations": ["safe_states_conditioned_offered_overbudget_states_also_reported", "no_real_queue_or_twoGPU_compute_contention",
                              "constant_prefill_not_longCode_calibrated", "wait_comparison_only_newCode_not_oldConv_SLO_cost"],
              "results": results}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
