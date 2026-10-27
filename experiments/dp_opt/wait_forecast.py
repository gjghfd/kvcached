import collections
import copy
import hashlib
import heapq
import json
import math
import random


PATHS = 32
HORIZON_S = 120
QUANTILE = .25
MIN_SAMPLES = 16
MAX_SAMPLES = 64


class Pool:
    def __init__(self, observation):
        self.page_bytes = observation["page_bytes"]
        self.capacity = observation["page_block_capacity"]
        self.pages = {page["id"]: {"pinned": page["pinned"], "owners": {}} for page in observation["pages"]}
        self.requests = {}
        self.virtual_free = observation.get("free_blocks", math.inf)
        for request in observation["requests"]:
            self.requests[request["id"]] = len(request["blocks"])
            for page, count in collections.Counter(block // self.capacity for block in request["blocks"]).items():
                self.pages[page]["owners"][request["id"]] = count

    def free(self, page):
        return self.capacity - sum(self.pages[page]["owners"].values()) - int(page == 0)

    def increment(self, blocks):
        if blocks > self.virtual_free:
            return math.inf
        return math.ceil(max(0, blocks - sum(self.free(page) for page in self.pages)) / self.capacity) * self.page_bytes

    def allocate(self, identifier, blocks):
        remaining = blocks
        for page in sorted(self.pages):
            amount = min(remaining, self.free(page))
            if amount:
                self.pages[page]["owners"][identifier] = amount
                remaining -= amount
            if remaining == 0:
                break
        while remaining:
            page = max(self.pages, default=0) + 1
            amount = min(remaining, self.capacity)
            self.pages[page] = {"pinned": False, "owners": {identifier: amount}}
            remaining -= amount
        self.requests[identifier] = blocks
        self.virtual_free -= blocks

    def remove(self, identifier):
        self.virtual_free += self.requests.pop(identifier)
        for page in list(self.pages):
            self.pages[page]["owners"].pop(identifier, None)
            if not self.pages[page]["owners"] and not self.pages[page]["pinned"]:
                del self.pages[page]


def forecast_context(request_id, model, records, rates, observations, now):
    other = 1 - model
    completed = sorted((record for record in records if record["model"] == other
                        and record["completed_at"] <= now), key=lambda record: record["completed_at"])[-MAX_SAMPLES:]
    if len(completed) < MIN_SAMPLES:
        return None
    geometry = observations[other][0]
    return {"incoming_request_id": request_id, "competing_model": other,
            "arrival_rate_rps": rates[other], "as_of": now,
            "samples": [{"request_id": record["request_id"], "completed_at": record["completed_at"],
                         "reserved_at": record["reserved_at"],
                         "blocks": math.ceil((record["prompt_tokens"] + len(record["token_ids"])) / geometry["block_size"]),
                         "service_s": record["completed_at"] - record["reserved_at"]} for record in completed]}


def forecast_wait(observations, model, tokens, budgets, context):
    if (context["competing_model"] != 1 - model or not MIN_SAMPLES <= len(context["samples"]) <= MAX_SAMPLES
            or not math.isfinite(context["arrival_rate_rps"]) or context["arrival_rate_rps"] <= 0
            or any(type(sample["blocks"]) is not int or sample["blocks"] <= 0
                   or not math.isfinite(sample["service_s"]) or sample["service_s"] <= 0
                   or sample["completed_at"] > context["as_of"] for sample in context["samples"])):
        raise ValueError("Wait forecast requires finite past-only paired samples and a positive competing rate")
    current = [(tenant, rank, request["id"], request.get("estimated_remaining_decode_s"))
               for tenant, ranks in enumerate(observations) for rank, observation in enumerate(ranks)
               for request in observation["requests"]]
    if any(remaining is None or not math.isfinite(remaining) or remaining < 0
           for tenant, rank, identifier, remaining in current):
        return {"status": "unavailable_current_decode_prediction", "scope": "legacy_wait_fallback"}
    initial = [[Pool(observation) for observation in ranks] for ranks in observations]
    incoming = [math.ceil(tokens / rank["block_size"]) for rank in observations[model]]
    capacities = [budget * 1024**3 for budget in budgets]
    seed = int.from_bytes(hashlib.sha256(context["incoming_request_id"].encode()).digest()[:8], "big")
    waits, censored = [], 0
    for trajectory in range(PATHS):
        generator = random.Random(seed + trajectory)
        pools = copy.deepcopy(initial)
        events, pending = [], collections.deque()
        sequence = 0

        def push(moment, kind, payload):
            nonlocal sequence
            sequence += 1
            heapq.heappush(events, (moment, sequence, kind, payload))

        def used():
            return [sum(len(ranks[rank].pages) * ranks[rank].page_bytes for ranks in pools) for rank in (0, 1)]

        def incoming_fits():
            mapped = used()
            return any(mapped[rank] + pools[model][rank].increment(incoming[rank]) <= capacities[rank]
                       for rank in (0, 1))

        def admit_pending(moment):
            other = 1 - model
            while pending:
                identifier, sample = pending[0]
                mapped = used()
                ranks = [rank for rank in (0, 1)
                         if mapped[rank] + pools[other][rank].increment(sample["blocks"]) <= capacities[rank]]
                if not ranks:
                    break
                rank = min(ranks, key=lambda index: (len(pools[other][index].requests), mapped[index], index))
                pending.popleft()
                pools[other][rank].allocate(identifier, sample["blocks"])
                push(moment + sample["service_s"], "complete", (other, rank, identifier))

        for tenant, rank, identifier, remaining in current:
            push(remaining, "complete", (tenant, rank, identifier))
        push(generator.expovariate(context["arrival_rate_rps"]), "arrival", None)
        wait, hit = HORIZON_S, incoming_fits()
        if hit:
            wait = 0
        while events and not hit:
            moment, serial, kind, payload = heapq.heappop(events)
            if moment > HORIZON_S:
                break
            if kind == "complete":
                tenant, rank, identifier = payload
                pools[tenant][rank].remove(identifier)
            if incoming_fits():
                wait, hit = moment, True
                break
            if kind == "arrival":
                pending.append((f"forecast-{serial}", generator.choice(context["samples"])))
                push(moment + generator.expovariate(context["arrival_rate_rps"]), "arrival", None)
            admit_pending(moment)
        waits.append(wait)
        censored += int(not hit)
    return {"status": "conditional_past_only_poisson_forecast", "estimated_wait_s": sorted(waits)[math.floor(QUANTILE * (PATHS - 1))],
            "waits_s": waits, "censored_paths": censored, "paths": PATHS, "horizon_s": HORIZON_S,
            "quantile": QUANTILE, "samples_sha256": hashlib.sha256(json.dumps(context["samples"], sort_keys=True).encode()).hexdigest(),
            "scope": "empirical_service_sorted_page_packing_optimistic_Code_priority_not_future_trace_or_proven_wait"}
