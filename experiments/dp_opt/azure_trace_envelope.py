import csv
import json
import math
import random
import statistics
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3] / "evidence/azure-trace-theory-20261021"
SEED = 20261022
SNAPSHOTS = 6000
PAGE_MIB = 96
BLOCK_MIB = 1.5
PREFILL_S = .2
CADENCE_S = .04


rows = []
for artifact in sorted((ROOT / "ranges").glob("conv-*.csv.fragment")):
    text = artifact.read_bytes().decode().split("\n", 1)[1].rsplit("\n", 1)[0]
    for _, context, output in csv.reader(text.splitlines()):
        context, output = int(context), int(output)
        if context >= 2048 and 256 <= output <= 512 and context + output <= 4096:
            rows.append((math.ceil((context + output) / 16), PREFILL_S + CADENCE_S * output))
mean_service = statistics.mean(service for _, service in rows)


def poisson(rng, mean):
    threshold = math.exp(-mean)
    product, count = 1.0, 0
    while product > threshold:
        product *= rng.random()
        count += 1
    return count - 1


def pool(requests):
    cursor = 1
    entries = []
    for identifier, blocks, residual, movable in requests:
        entries.append({"id": identifier, "blocks": tuple(range(cursor, cursor + blocks)),
                        "residual_s": residual, "movable": movable})
        cursor += blocks
    return {"requests": entries, "pages": set(range(max(1, math.ceil(cursor / 64))))}


def remove_request(observation, identifier):
    remaining = [request for request in observation["requests"] if request["id"] != identifier]
    owned = {block // 64 for request in remaining for block in request["blocks"]}
    return {"requests": remaining, "pages": {page for page in observation["pages"] if page == 0 or page in owned}}


def free_slots(observation):
    return len(observation["pages"]) * 64 - 1 - sum(len(request["blocks"]) for request in observation["requests"])


def increment(observation, blocks):
    return math.ceil(max(0, blocks - free_slots(observation)) / 64) * PAGE_MIB


def used_memory(observations):
    return [sum(len(ranks[rank]["pages"]) * PAGE_MIB for ranks in observations) for rank in (0, 1)]


def direct_fit(observations, incoming, capacity):
    used = used_memory(observations)
    return any(used[rank] + increment(observations[0][rank], incoming) <= capacity for rank in (0, 1))


def one_move_fit(observations, incoming, capacity):
    if any(not request["movable"] for ranks in observations for rank in ranks for request in rank["requests"]):
        return False
    used = used_memory(observations)
    for model in (0, 1):
        for source in (0, 1):
            destination = 1 - source
            for request in observations[model][source]["requests"]:
                additional = increment(observations[model][destination], len(request["blocks"]))
                if used[destination] + additional > capacity:
                    continue
                after = remove_request(observations[model][source], request["id"])
                new_used = list(used)
                new_used[source] -= (len(observations[model][source]["pages"]) - len(after["pages"])) * PAGE_MIB
                new_used[destination] += additional
                for target in (0, 1):
                    observation = observations[0][target]
                    available = free_slots(observation)
                    if model == 0 and target == source:
                        available = free_slots(after)
                    elif model == 0 and target == destination:
                        available += additional / BLOCK_MIB - len(request["blocks"])
                    incoming_pages = math.ceil(max(0, incoming - available) / 64)
                    if new_used[target] + incoming_pages * PAGE_MIB <= capacity:
                        return True
    return False


def natural_wait(observations, incoming, capacity):
    events = sorted((request["residual_s"], model, rank, request["id"])
                    for model, ranks in enumerate(observations) for rank, observation in enumerate(ranks)
                    for request in observation["requests"])
    remaining = [list(ranks) for ranks in observations]
    for residual, model, rank, identifier in events:
        remaining[model][rank] = remove_request(remaining[model][rank], identifier)
        if direct_fit(remaining, incoming, capacity):
            return residual
    raise RuntimeError("Incoming long request should fit empty pools")


rng = random.Random(SEED)
weights = [service for _, service in rows]
results = []
for capacity in (1728, 2112):
    for rate in (.4, .5, .6, .65, .7, .8):
        safe = globally_sufficient_blocked = eligible_blocked = recoverable = 0
        waiting = []
        populations = []
        for _ in range(SNAPSHOTS):
            observations = []
            identifier = 0
            for model in (0, 1):
                count = poisson(rng, rate * mean_service / 2)
                counts = [count // 2, count - count // 2]
                rng.shuffle(counts)
                ranks = []
                for count in counts:
                    entries = []
                    for blocks, service in rng.choices(rows, weights=weights, k=count):
                        age = rng.uniform(0, service)
                        entries.append((identifier, blocks, service - age, age >= PREFILL_S))
                        identifier += 1
                    ranks.append(pool(entries))
                observations.append(ranks)
            incoming, _ = rng.choice(rows)
            used = used_memory(observations)
            populations.append(identifier)
            if max(used) > capacity:
                continue
            safe += 1
            if direct_fit(observations, incoming, capacity) or 2 * capacity - sum(used) < incoming * BLOCK_MIB:
                continue
            globally_sufficient_blocked += 1
            if all(request["movable"] for ranks in observations for rank in ranks for request in rank["requests"]):
                eligible_blocked += 1
            if one_move_fit(observations, incoming, capacity):
                recoverable += 1
                waiting.append(natural_wait(observations, incoming, capacity))
        result = {"capacity_mib_per_gpu": capacity, "total_rate_rps": rate, "snapshots": SNAPSHOTS,
                  "mean_population": statistics.mean(populations), "offered_state_over_budget_fraction": 1 - safe / SNAPSHOTS,
                  "global_enough_local_blocked_fraction_of_safe": globally_sufficient_blocked / safe,
                  "all_decode_fraction_of_blocked": eligible_blocked / globally_sufficient_blocked if globally_sufficient_blocked else None,
                  "one_move_recoverable_fraction_of_safe": recoverable / safe,
                  "one_move_recoverable_states": recoverable,
                  "natural_wait_mean_s_for_recoverable": statistics.mean(waiting) if waiting else None,
                  "natural_wait_median_s_for_recoverable": statistics.median(waiting) if waiting else None,
                  "natural_wait_above_pause_fraction": {str(pause): sum(wait > pause for wait in waiting) / len(waiting)
                                                        if waiting else None for pause in (1.3, 2.8, 5)}}
        results.append(result)
        print(json.dumps(result), flush=True)
report = {"created_at": datetime.now().astimezone().isoformat(), "seed": SEED,
          "scope": "Numerical stationary marked-Poisson memory-only thought model, not actual trace serving or an ILP performance measurement",
          "assumptions": {"population": "unconstrained M/G/infinity offered state with fixed service time; conditioning on fitting snapshots is not a finite-memory queue distribution",
                          "lengths": "Conversation conditional slice, context>=2048/output256..512/total<=4096; size-biased resident marks",
                          "placement": "each model count balanced in a snapshot, random hot rank; independent marks, contiguous initial block allocation",
                          "allocator": "64blocks per96MiB bundle, one pinned null block per pool, source page released only if all owner blocks removed",
                          "migrations": "one-move exact free-slot packing, every resident request considered; exceeds current two-candidate restriction and is an optimistic feasibility envelope",
                          "costs": "No measured context-dependent service, global pause duty, admission fastpath, copy contention, candidate cutoff or SLO simulation"},
          "mean_service_s": mean_service, "results": results}
(ROOT / "placement-envelope.json").write_text(json.dumps(report, indent=2) + "\n")
