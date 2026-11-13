import copy
import math

from cost_model import capture_bucket


def request_phase(computed_tokens, prompt_tokens, output_tokens, running):
    if not running:
        return "waiting"
    if computed_tokens < prompt_tokens or output_tokens == 0:
        return "prefill"
    return "decode"


def capture_scoring_observations(observed):
    keys = ("id", "computed_tokens", "prompt_tokens", "output_tokens", "in_flight_tokens", "phase")
    return [[{"rank": observation.get("rank"), "observed_at": observation.get("observed_at"),
              "phase_schema": observation.get("phase_schema"),
              "scheduler_paused": observation.get("scheduler_paused"),
              "requests": [{key: request.get(key) for key in keys} for request in observation["requests"]]}
             for observation in tenant] for tenant in observed]


def candidate_peer_state(state, tenant, as_of, max_age_s, moves=(), arrival=None, target=None):
    """Causal snapshot after proposed placement, not a prediction over the scoring horizon."""
    if (type(tenant) is not int or tenant not in (0, 1)
            or not isinstance(as_of, (int, float)) or isinstance(as_of, bool) or not math.isfinite(as_of)
            or not isinstance(max_age_s, (int, float)) or isinstance(max_age_s, bool)
            or not math.isfinite(max_age_s) or max_age_s < 0):
        raise ValueError("Invalid peer-state query or freshness bound")
    peer = 1 - tenant
    result = {"model": peer, "as_of": as_of, "max_age_s": max_age_s,
              "interpretation": "instantaneous_post_action_not_horizon_prediction"}

    def unavailable(reason):
        return dict(result, status="unavailable", reason=reason)

    observations = state.get("scoring_observations")
    if not observations or len(observations) != 2 or len(observations[peer]) != state["num_ranks"]:
        return unavailable("missing_rank_observations")
    records, timestamps, paused = [], [], []
    for rank, observation in enumerate(observations[peer]):
        if type(observation.get("rank")) is not int or observation["rank"] != rank:
            return unavailable("missing_or_invalid_rank")
        timestamp = observation.get("observed_at")
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or not math.isfinite(timestamp):
            return unavailable("missing_or_invalid_timestamp")
        if timestamp > as_of:
            return unavailable("future_timestamp")
        if as_of - timestamp > max_age_s:
            return unavailable("stale_timestamp")
        if (type(observation.get("phase_schema")) is not int or observation["phase_schema"] != 1
                or not isinstance(observation.get("scheduler_paused"), bool)):
            return unavailable("missing_phase_schema_or_pause_state")
        timestamps.append(timestamp)
        paused.append(observation["scheduler_paused"])
        for request in observation["requests"]:
            if request.get("phase") not in ("waiting", "prefill", "decode"):
                return unavailable("missing_or_invalid_phase")
            for key in ("computed_tokens", "prompt_tokens", "output_tokens", "in_flight_tokens"):
                value = request.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    return unavailable("missing_or_invalid_token_count")
            if not isinstance(request.get("id"), str) or not request["id"]:
                return unavailable("missing_request_id")
            if request["phase"] != request_phase(request["computed_tokens"], request["prompt_tokens"],
                                                  request["output_tokens"], request["phase"] != "waiting"):
                return unavailable("inconsistent_phase")
            records.append(dict(request, rank=rank, hypothetical=False))
    by_id = {request["id"]: request for request in records}
    if len(by_id) != len(records):
        return unavailable("duplicate_request_id")
    moved = set()
    for move in moves:
        if type(move["model"]) is not int or move["model"] not in (0, 1):
            raise ValueError("Invalid candidate move tenant")
        if move["model"] != peer:
            continue
        request = by_id.get(move["id"])
        destination = move["destination"]
        if (request is None or move["id"] in moved or move["source"] != request["rank"]
                or not isinstance(destination, int) or isinstance(destination, bool)
                or not 0 <= destination < state["num_ranks"] or request["phase"] != "decode"):
            raise ValueError("Invalid candidate peer move")
        request["rank"] = destination
        moved.add(move["id"])
    if arrival is not None and (type(arrival["model"]) is not int or arrival["model"] not in (0, 1)):
        raise ValueError("Invalid candidate arrival tenant")
    if arrival is not None and arrival["model"] == peer:
        prompt = arrival["prompt_tokens"]
        if (not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < state["num_ranks"]
                or not isinstance(prompt, int) or isinstance(prompt, bool) or prompt < 0):
            raise ValueError("Invalid candidate peer arrival")
        records.append({"id": None, "rank": target, "phase": "prefill", "computed_tokens": 0,
                        "prompt_tokens": prompt, "output_tokens": 0, "in_flight_tokens": 0,
                        "hypothetical": True})
    counts = {phase: [0] * state["num_ranks"] for phase in ("waiting", "prefill", "decode")}
    contexts = [0] * state["num_ranks"]
    prompts = [0] * state["num_ranks"]
    for request in records:
        rank = request["rank"]
        counts[request["phase"]][rank] += 1
        if request["phase"] != "waiting":
            contexts[rank] += request["computed_tokens"]
            prompts[rank] += request["prompt_tokens"]
    phases = [phase for phase in ("prefill", "decode") if any(counts[phase])]
    phase = "mixed" if len(phases) == 2 else phases[0] if phases else "idle"
    return dict(result, status="available", phase=phase, counts=counts, accounted_contexts=contexts,
                prompt_tokens=prompts, decode_bucket=capture_bucket(max(counts["decode"])),
                observed_at=timestamps, scheduler_paused=paused, requests=copy.deepcopy(records))
