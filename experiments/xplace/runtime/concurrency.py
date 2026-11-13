def validate_limits(limits):
    if (not isinstance(limits, (list, tuple)) or len(limits) != 2
            or any(type(value) is not int or value < 0 for value in limits)):
        raise ValueError("max_inflight must contain two nonnegative integers; zero means unlimited")
    return tuple(limits)


class InFlightLimiter:
    """Count submitted requests until completion, including engine queues, prefill and migration."""

    def __init__(self, limits=(0, 0)):
        self.limits = validate_limits(limits)
        self.active = [0, 0]
        self.peak = [0, 0]
        self.blocked_checks = [0, 0]

    def has_capacity(self, model):
        return not self.limits[model] or self.active[model] < self.limits[model]

    def allow(self, entry, now):
        model = entry["mark"]["model"]
        if not self.has_capacity(model):
            entry.setdefault("cap_blocked_since", now)
            self.blocked_checks[model] += 1
            return False
        started = entry.pop("cap_blocked_since", None)
        if started is not None:
            entry["concurrency_wait_s"] = entry.get("concurrency_wait_s", 0.0) + now - started
        return True

    def acquire(self, entry, now):
        if entry.get("inflight_slot") or not self.allow(entry, now):
            raise RuntimeError("Duplicate submission or exceeded shared inflight limit")
        model = entry["mark"]["model"]
        self.active[model] += 1
        self.peak[model] = max(self.peak[model], self.active[model])
        entry["inflight_slot"] = True

    def release(self, entry):
        if entry.pop("inflight_slot", False):
            self.active[entry["mark"]["model"]] -= 1

    def wait_s(self, entry, now):
        if entry is None:
            return 0.0
        started = entry.get("cap_blocked_since", now)
        return entry.get("concurrency_wait_s", 0.0) + now - started

    def snapshot(self):
        return {"max_inflight": list(self.limits), "peak_inflight": list(self.peak),
                "inflight_after_cleanup": list(self.active), "blocked_checks": list(self.blocked_checks)}
