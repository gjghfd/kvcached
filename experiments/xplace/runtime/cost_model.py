import math

CAPTURE_SIZES = (1, 8, 16, 32, 64, 128, 256)


def capture_bucket(count):
    if count <= 0:
        return 0
    for size in CAPTURE_SIZES:
        if count <= size:
            return size
    raise ValueError(f"Per-rank batch {count} exceeds the largest CUDA graph capture size")


class StepModel:
    """Per-tenant decode step time for a per-rank layout.

    DP ranks are padded to the max per-rank batch (vLLM DP padding with CUDA graphs) and synchronized by EP
    collectives, so a tenant's decode step costs base + graph[bucket(max count)] + attention * max context.
    While the other tenant is active, cross-tenant GPU interference transforms the step: t * multiplier + additive.
    """

    def __init__(self, params):
        self.params = params
        self.tenants = {int(key): value for key, value in params["tenants"].items()}
        for tenant in self.tenants.values():
            tenant["graph_s"] = {int(key): float(value) for key, value in tenant["graph_s"].items()}

    def graph_cost(self, tenant, bucket):
        table = self.tenants[tenant]["graph_s"]
        if bucket in table:
            return table[bucket]
        known = sorted(table)
        if bucket < known[0]:
            return table[known[0]]
        if bucket > known[-1]:
            slope = self.tenants[tenant].get("graph_slope_s", 0.0)
            return table[known[-1]] + slope * (bucket - known[-1])
        lower = max(size for size in known if size < bucket)
        upper = min(size for size in known if size > bucket)
        weight = (bucket - lower) / (upper - lower)
        return table[lower] * (1 - weight) + table[upper] * weight

    def decode_step(self, tenant, counts, contexts, other_active=False, other_bucket=None):
        """other_active: False, True, or the other tenant's phase ("prefill" / "decode")."""
        if not any(counts):
            return 0.0
        params = self.tenants[tenant]
        bucket = capture_bucket(max(counts))
        step = params["base_s"] + self.graph_cost(tenant, bucket) + params["attention_s_per_token"] * max(contexts)
        bonus = params.get("full_bucket_bonus_s", {})
        if bonus and all(count == bucket for count in counts):
            step += float(bonus.get(str(bucket), bonus.get(bucket, 0.0)))
        if other_active:
            step = self.interfere(tenant, step, other_active, other_bucket if other_bucket is not None else bucket)
        return step

    def interfere(self, tenant, step, other_phase, bucket):
        interference = self.params.get("interference", {}).get(str(tenant), {})
        phase = interference.get(other_phase) if isinstance(other_phase, str) else None
        if phase is None:
            return step * interference.get("multiplier", 1.0) + interference.get("additive_s", 0.0)
        if isinstance(phase, dict):
            sizes = sorted(int(size) for size in phase)
            chosen = next((size for size in sizes if bucket <= size), sizes[-1])
            return step + float(phase[str(chosen)] if str(chosen) in phase else phase[chosen])
        return step + float(phase)

    def placement_terms(self, tenant, counts, contexts):
        """Layout-dependent part of the step (what placement can change)."""
        params = self.tenants[tenant]
        return self.graph_cost(tenant, capture_bucket(max(counts))) + params["attention_s_per_token"] * max(contexts)


def completion_segments(remaining, max_segments=16):
    """Split [0, max remaining] at completion thresholds (merged into at most max_segments)."""
    thresholds = sorted(set(value for value in remaining if value > 0))
    if len(thresholds) > max_segments:
        thresholds = [thresholds[math.ceil((index + 1) * len(thresholds) / max_segments) - 1]
                      for index in range(max_segments)]
        thresholds = sorted(set(thresholds))
    return thresholds


def integrated_cost(model, tenant, requests, num_ranks, horizon_steps=None, max_segments=16, other_active=False):
    """Request-seconds of decode time for (rank, context, remaining) requests until completion or horizon.

    Assumes no arrivals or moves; each alive request advances one token per step.
    """
    alive = [request for request in requests if request[2] > 0]
    if not alive:
        return 0.0
    thresholds = completion_segments([request[2] for request in alive], max_segments)
    if horizon_steps is not None:
        thresholds = [value for value in thresholds if value < horizon_steps] + [horizon_steps]
    total, previous = 0.0, 0
    for threshold in thresholds:
        steps = threshold - previous
        if steps <= 0:
            continue
        current = [request for request in alive if request[2] > previous]
        if not current:
            break
        middle = previous + steps / 2
        counts, contexts = [0] * num_ranks, [0.0] * num_ranks
        for rank, context, remaining in current:
            counts[rank] += 1
            contexts[rank] += context + min(middle, remaining)
        total += steps * len(current) * model.decode_step(tenant, counts, contexts, other_active)
        previous = threshold
    return total


def default_dp2_params():
    """Prior from the DP=2 Conv fit (public-performance-parallel-20261102/reused-context-fit.json)."""
    graph = {1: 0.779e-3, 8: 0.779e-3 * 8, 16: 0.779e-3 * 16}
    return {"source": "dp2_prior_reused_context_fit_linear_graph",
            "tenants": {"0": {"base_s": 9.803e-3, "graph_s": graph, "attention_s_per_token": 6.83e-8},
                        "1": {"base_s": 9.803e-3, "graph_s": graph, "attention_s_per_token": 6.83e-8}},
            "interference": {"0": {"multiplier": 1.0, "additive_s": 0.153},
                             "1": {"multiplier": 1.0, "additive_s": 0.157}},
            "code_prefill": {"solo_s_per_100k": 26.4, "with_other_s_per_100k": 54.0, "chunk_tokens": 4096}}
