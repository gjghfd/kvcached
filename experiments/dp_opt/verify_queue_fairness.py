import json
import math
from pathlib import Path


def verify(directory, threshold_s):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text())
    records = read("requests.json")
    requests = {record["request_id"]: record for record in records}
    trace_order = {mark["request_id"]: index for index, mark in enumerate(read("workload.json")["requests"])}
    events = read("fairness-events.json")
    decisions = read("decisions.json")
    summary = read("summary.json")
    journal = [json.loads(line) for line in (directory / "fairness-events.jsonl").read_text().splitlines()] if events else []
    if journal != events or len(requests) != len(records) or set(trace_order) != set(requests):
        raise ValueError("Fairness journal or request identity differs")
    if (summary["policy"] != "no_move_fair" or summary["moves"] != 0 or summary["ilp_attempts"] != 0
            or any(decision["moves"] or decision.get("pause_call_s", 0) != 0 for decision in decisions)):
        raise ValueError("Fair queue baseline must not migrate or pause models")
    fairness = summary["fairness"]
    if (fairness["threshold_s"] != threshold_s or fairness["owner_at_end"] is not None
            or not math.isfinite(threshold_s) or threshold_s <= 0):
        raise ValueError("Frozen threshold or final gate state differs")
    authorizations = {decision["request_id"]: decision["fairness_authorization"] for decision in decisions
                      if decision["admitted"] or decision.get("submitted", False)}
    if len(authorizations) != len(decisions) or set(authorizations) != set(requests):
        raise ValueError("Each fair baseline request requires one audited submission")
    for model in (0, 1):
        expected = sorted((record for record in records if record["model"] == model),
                          key=lambda record: (record["arrived_at"], trace_order[record["request_id"]]))
        actual = sorted(expected, key=lambda record: authorizations[record["request_id"]]["observed_at"])
        if [record["request_id"] for record in expected] != [record["request_id"] for record in actual]:
            raise ValueError("Fair baseline changed a tenant's FIFO submission order")
    timeline = [(event["observed_at"], event["event"], event) for event in events]
    timeline.extend((authorization["observed_at"], "authorization", authorization)
                    for authorization in authorizations.values())
    timeline.sort(key=lambda item: item[0])
    owner = None
    activations, releases, blocked = 0, 0, 0
    for now, kind, item in timeline:
        if not math.isfinite(now):
            raise ValueError("Fairness event clock is nonfinite")
        if kind == "latch":
            selected = item["owner"]
            record = requests[selected["request_id"]]
            if (owner is not None or selected["model"] != record["model"]
                    or selected["arrived_at"] != record["arrived_at"] or selected["activated_at"] != now
                    or item["threshold_s"] != threshold_s or now - record["arrived_at"] < threshold_s - 1e-8
                    or not math.isclose(item["queue_age_s"], now - record["arrived_at"], abs_tol=1e-8)):
                raise ValueError("Fair gate activates before threshold or with wrong owner")
            waiting = [record for identifier, record in requests.items()
                       if record["arrived_at"] + record["arrival_dispatch_lag_s"] <= now
                       and record["reservation_observed_at"] > now]
            heads = [min((record for record in waiting if record["model"] == model),
                         key=lambda record: (record["arrived_at"], trace_order[record["request_id"]]), default=None)
                     for model in (0, 1)]
            oldest = min((record for record in heads if record is not None),
                         key=lambda record: (record["arrived_at"], record["request_id"]), default=None)
            if oldest is None or oldest["request_id"] != selected["request_id"]:
                raise ValueError("Fair gate does not select the oldest observable waiting request")
            owner = selected
            activations += 1
        elif kind == "release":
            if (owner is None or item["owner"] != owner or item["reserved_at"] != requests[owner["request_id"]]["reserved_at"]
                    or requests[owner["request_id"]]["reservation_observed_at"] > now
                    or item["reserved_at"] > now
                    or not math.isclose(item["hold_s"], now - owner["activated_at"], abs_tol=1e-8)):
                raise ValueError("Fair gate releases without its owner's actual KV reservation")
            for record in records:
                if record["model"] != owner["model"] and owner["activated_at"] <= record["reserved_at"] < now:
                    raise ValueError("Foreign request obtained KV during an active fair gate")
            owner = None
            releases += 1
        elif kind in ("authorization", "blocked"):
            record = requests[item["request_id"]]
            allowed = owner is None or record["model"] == owner["model"]
            if (item["model"] != record["model"] or item["owner_request_id"] != (owner["request_id"] if owner else None)
                    or item["owner_model"] != (owner["model"] if owner else None) or item["allowed"] != allowed):
                raise ValueError("Admission authorization differs from replayed fair gate")
            if kind == "authorization":
                if not allowed or now > record["submitted_at"]:
                    raise ValueError("Foreign or future-authorized admission during fair gate")
            else:
                if allowed:
                    raise ValueError("Fair gate suppresses an eligible request")
                blocked += 1
        else:
            raise ValueError("Unknown fairness event")
    if (owner is not None or activations != fairness["activations"] or releases != fairness["releases"]
            or fairness["suppressed_attempts"] < blocked):
        raise ValueError("Fair gate summary differs from independently replayed events")
    return {"status": "raw_queue_fairness_verified", "threshold_s": threshold_s, "requests": len(records),
            "activations": activations, "releases": releases, "blocked_request_pairs": blocked,
            "strict_foreign_KV_reservations_during_gate": 0,
            "scope": "audited_external_FIFO_gate_and_actual_KV_reservations_not_latency_benefit"}
