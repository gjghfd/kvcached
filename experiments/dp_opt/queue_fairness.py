import math


DEFAULT_QUEUE_THRESHOLD_S = 30.0


class FairAdmissionGate:
    def __init__(self, threshold_s):
        if not math.isfinite(threshold_s) or threshold_s <= 0:
            raise ValueError("Fair queue threshold must be finite and positive")
        self.threshold_s = threshold_s
        self.owner = None
        self.events = []
        self.suppressed_attempts = 0
        self.recorded_blocks = set()

    def update(self, entries, now):
        if self.owner is not None:
            return None
        overdue = [entry for entry in entries if now - entry["arrived_at"] >= self.threshold_s]
        if not overdue:
            return None
        selected = min(overdue, key=lambda entry: (entry["arrived_at"], entry["mark"]["request_id"]))
        self.owner = {"request_id": selected["mark"]["request_id"], "model": selected["mark"]["model"],
                      "arrived_at": selected["arrived_at"], "activated_at": now}
        event = {"event": "latch", "observed_at": now, "owner": dict(self.owner),
                 "queue_age_s": now - selected["arrived_at"], "threshold_s": self.threshold_s}
        self.events.append(event)
        return event

    def authorization(self, model, request_id, now):
        return {"observed_at": now, "request_id": request_id, "model": model,
                "owner_request_id": self.owner["request_id"] if self.owner else None,
                "owner_model": self.owner["model"] if self.owner else None,
                "allowed": self.owner is None or self.owner["model"] == model}

    def blocked(self, model, request_id, now):
        self.suppressed_attempts += 1
        identity = (self.owner["request_id"], request_id)
        if identity in self.recorded_blocks:
            return None
        self.recorded_blocks.add(identity)
        event = {"event": "blocked", **self.authorization(model, request_id, now)}
        self.events.append(event)
        return event

    def reserved(self, request_id, reserved_at, now):
        if self.owner is None or self.owner["request_id"] != request_id:
            return None
        if not math.isfinite(reserved_at) or reserved_at > now or reserved_at < self.owner["arrived_at"]:
            raise ValueError("Fair gate release requires an actual completed KV reservation")
        event = {"event": "release", "observed_at": now, "reserved_at": reserved_at,
                 "owner": dict(self.owner), "hold_s": now - self.owner["activated_at"]}
        self.events.append(event)
        self.owner = None
        return event

    def summary(self):
        return {"enabled": True, "threshold_s": self.threshold_s,
                "activations": sum(event["event"] == "latch" for event in self.events),
                "releases": sum(event["event"] == "release" for event in self.events),
                "suppressed_attempts": self.suppressed_attempts,
                "owner_at_end": self.owner, "release_condition": "actual_full_KV_reservation_not_submission_or_completion"}
