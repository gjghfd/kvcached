import copy
import math

from prediction_state import finite_number
from reference_work import fingerprint, project_reference_layout, validate_reference


PHASES = ("prefill", "recompute", "decode", "mixed", "return", "residual", "drain", "transfer", "empty")
FEATURES = ("tokens", "context", "batch", "bytes", "peer_tokens", "peer_context", "peer_batch", "peer_bytes")


class DurationUnavailable(ValueError):
    pass


def positive(value, name, allow_zero=False):
    if not finite_number(value) or value < 0 or (value == 0 and not allow_zero):
        raise DurationUnavailable(f"invalid_{name}")
    return float(value)


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise DurationUnavailable(f"invalid_{name}")
    return value


class ResponseSurface:
    """Explicit directional service response; no fitted coefficients or fallback are supplied."""

    def __init__(self, parameters):
        self.parameters = copy.deepcopy(parameters)
        if not parameters.get("source_id") or parameters.get("kind") != "coupled_service_response_v1":
            raise ValueError("Response parameters require a source ID and explicit kind")
        self.identifier = fingerprint(self.parameters)

    def duration(self, model, own, peer):
        key = f"{model}:{own['phase']}:{peer['phase']}"
        row = self.parameters.get("rows", {}).get(key)
        if row is None:
            raise DurationUnavailable(f"uncovered_phase_pair:{key}")
        values = {name: own[name] for name in FEATURES[:4]}
        values.update({f"peer_{name}": peer[name] for name in FEATURES[:4]})
        duration = positive(row.get("intercept_s"), "intercept", allow_zero=True)
        for name, value in values.items():
            bounds = row.get("support", {}).get(name)
            if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                    or any(not finite_number(bound) for bound in bounds)
                    or not 0 <= bounds[0] <= value <= bounds[1]):
                raise DurationUnavailable(f"outside_response_support:{key}:{name}")
            coefficient = positive(row.get("coefficients", {}).get(name), name, allow_zero=True)
            duration += coefficient * value
        return positive(duration, "service_duration")


def receipt_time(completed_at, receipt):
    completed_at = positive(completed_at, "completion_time", allow_zero=True)
    period = positive(receipt.get("period_s"), "receipt_period")
    offset = positive(receipt.get("offset_s"), "receipt_offset", allow_zero=True)
    delay = positive(receipt.get("delay_s"), "receipt_delay", allow_zero=True)
    if offset >= period:
        raise DurationUnavailable("receipt_offset_outside_period")
    position = (completed_at - offset) / period
    if not math.isfinite(position):
        raise DurationUnavailable("nonfinite_receipt_position")
    nearest = round(position)
    if abs(position - nearest) <= 32 * math.ulp(position):
        position = nearest
    tick = max(0, math.ceil(position))
    return max(completed_at, offset + tick * period) + delay


class CoupledRollout:
    """Conditional single-batch, rank-barrier fluid model, not a vLLM scheduler emulator."""

    def __init__(self, reference, layout, initial, response, action, receipt, limits):
        validate_reference(reference)
        if reference.get("deployment_status") != "accounting_manifest_bound_not_runtime_acceptance":
            raise DurationUnavailable("unbound_accounting_reference")
        if fingerprint(response.parameters) != response.identifier:
            raise DurationUnavailable("response_parameters_changed")
        self.reference = reference
        self.response = response
        self.receipt = copy.deepcopy(receipt)
        receipt_time(0.0, self.receipt)
        self.time = 0.0
        self.events = []
        self.receipts = []
        self.requests = {}
        self.order = [[], []]
        self.active = [None, None]
        self.moves = [[], []]
        self.moved = [False, False]
        self.admissions = []
        self.limits = copy.deepcopy(limits)
        if len(limits) != 2:
            raise DurationUnavailable("incomplete_schedule_limits")
        for limit in self.limits:
            for name in ("max_tokens_per_rank", "max_requests_per_rank", "chunk_tokens"):
                integer(limit.get(name), name, 1)
        if (initial.get("reference_id") != reference["reference_id"]
                or initial.get("observation_sha256") != reference["observation_sha256"]
                or initial.get("as_of") != reference["as_of"] or not initial.get("source_id")
                or initial.get("model") != "single_batch_barrier_v1"):
            raise DurationUnavailable("missing_or_unbound_asof_residual_estimate")
        original = {(row["model"], row["id"]): row for row in reference["requests"]}
        for seed in initial.get("requests", []):
            identity = integer(seed.get("model"), "residual_model"), seed["id"]
            if identity not in original or identity in self.requests:
                raise DurationUnavailable("invalid_residual_cohort")
            request = original[identity]
            if request["phase"] == "waiting" and not request["external_arrival"]:
                raise DurationUnavailable("unsupported_resident_waiting_admission")
            ledger = request["ledger"]
            prefix = integer(seed.get("prefix_tokens"), "estimated_prefix")
            output = integer(seed.get("output_tokens"), "estimated_output")
            if (not ledger["retired_prefix_tokens"] <= prefix <= ledger["prompt_tokens"] + output
                    or output < ledger["observed_output_tokens"]
                    or (request["external_arrival"] and (prefix or output))):
                raise DurationUnavailable("inconsistent_asof_residual_estimate")
            self.requests[identity] = dict(model=identity[0], id=identity[1], rank=request["rank"], prefix=prefix,
                output=output, received=ledger["observed_output_tokens"], prompt=ledger["prompt_tokens"],
                target=ledger["target_output_tokens"], milestone=None, external=request["external_arrival"])
            self.order[identity[0]].append(identity)
        if set(self.requests) != set(original):
            raise DurationUnavailable("incomplete_residual_cohort")
        for sample in initial.get("pending_receipts", []):
            identity = sample["model"], sample["id"]
            if identity not in self.requests:
                raise DurationUnavailable("unknown_pending_receipt")
            output = integer(sample.get("output_tokens"), "pending_output")
            if not self.requests[identity]["received"] <= output <= self.requests[identity]["output"]:
                raise DurationUnavailable("invalid_pending_output")
            self.receipts.append((positive(sample.get("after_s"), "pending_receipt", True), identity, output))
        for identity, state in self.requests.items():
            covered = max([state["received"], *[output for _, owner, output in self.receipts if owner == identity]])
            if covered != state["output"]:
                raise DurationUnavailable("missing_sample_age_output_receipts")
        for batch in initial.get("batches", []):
            model = integer(batch.get("model"), "residual_model")
            if model not in (0, 1) or self.active[model] is not None:
                raise DurationUnavailable("unsupported_multiple_inflight_batches")
            allocations = []
            seen = set()
            for allocation in batch.get("allocations", []):
                identity = model, allocation["id"]
                if identity not in self.requests or identity in seen or self.requests[identity]["rank"] is None:
                    raise DurationUnavailable("invalid_residual_allocation")
                seen.add(identity)
                tokens = integer(allocation.get("tokens"), "residual_tokens")
                state = self.requests[identity]
                if state["prefix"] + tokens > state["prompt"] + state["output"]:
                    raise DurationUnavailable("residual_exceeds_known_frontier")
                allocations.append((identity, tokens))
            stage = batch.get("stage")
            if stage not in ("compute", "return", "residual") or (stage == "residual") != (not allocations):
                raise DurationUnavailable("unsupported_residual_stage")
            if stage == "compute" and any(tokens == 0 for _, tokens in allocations):
                raise DurationUnavailable("zero_compute_work")
            fraction = positive(batch.get("remaining_fraction"), "residual_fraction")
            if fraction > 1:
                raise DurationUnavailable("residual_fraction_above_one")
            self.active[model] = dict(stage=stage, allocations=allocations, remaining=fraction,
                                      descriptor=self.describe(model, stage, allocations))
        for identity, state in self.requests.items():
            active = self.active[identity[0]]
            tokens = dict(active["allocations"]).get(identity) if active else None
            if state["prefix"] < original[identity]["ledger"]["computed_tokens"]:
                if tokens is None or state["prefix"] + tokens < original[identity]["ledger"]["computed_tokens"]:
                    raise DurationUnavailable("missing_inflight_residual")
            if state["prefix"] == state["prompt"] + state["output"] and tokens is None:
                raise DurationUnavailable("missing_pending_sample_return")
        moves = action.get("moves", [])
        expected = project_reference_layout(reference, moves=[{name: move[name] for name in
            ("model", "id", "source", "destination")} for move in moves], arrival_target=action.get("arrival_target"))
        if layout != expected:
            raise DurationUnavailable("action_layout_mismatch")
        for move in moves:
            positive(move.get("bytes"), "transfer_bytes")
            self.moves[move["model"]].append(copy.deepcopy(move))
        admitted = set()
        for admission in action.get("admissions", []):
            identity = admission["model"], admission["id"]
            rank = integer(admission.get("rank"), "admission_rank")
            if (identity not in self.requests or not self.requests[identity]["external"] or identity in admitted
                    or rank >= reference["num_ranks"]
                    or (action.get("arrival_target") is not None and rank != action["arrival_target"])):
                raise DurationUnavailable("invalid_admission_forecast")
            admitted.add(identity)
            self.admissions.append((positive(admission.get("after_s"), "admission_time", True), identity, rank))

    def describe(self, model, stage, allocations):
        counts = [0] * self.reference["num_ranks"]
        tokens = counts.copy()
        contexts = counts.copy()
        phases = set()
        for identity, amount in allocations:
            state = self.requests[identity]
            rank = state["rank"]
            counts[rank] += 1
            tokens[rank] += amount
            contexts[rank] += state["prefix"]
            if state["prefix"] < state["prompt"]:
                phases.add("prefill")
                if state["prefix"] + amount > state["prompt"]:
                    phases.add("recompute")
            elif state["prefix"] < state["prompt"] + state["output"] - 1:
                phases.add("recompute")
            else:
                phases.add("decode")
        phase = (next(iter(phases)) if len(phases) == 1 else "mixed") if stage == "compute" else stage
        return dict(phase=phase, tokens=max(tokens), context=max(contexts), batch=max(counts), bytes=0,
                    rank_tokens=tokens, rank_contexts=contexts, rank_counts=counts)

    def new_batch(self, model):
        allocations = []
        token_budgets = [self.limits[model]["max_tokens_per_rank"]] * self.reference["num_ranks"]
        request_budgets = [self.limits[model]["max_requests_per_rank"]] * self.reference["num_ranks"]
        for identity in self.order[model]:
            state = self.requests[identity]
            rank = state["rank"]
            if rank is None or not token_budgets[rank] or not request_budgets[rank]:
                continue
            remaining = state["prompt"] + state["output"] - state["prefix"]
            if remaining <= 0:
                raise DurationUnavailable("unspecified_pending_output")
            amount = min(remaining, token_budgets[rank], self.limits[model]["chunk_tokens"])
            allocations.append((identity, amount))
            token_budgets[rank] -= amount
            request_budgets[rank] -= 1
        if not allocations:
            return None
        return dict(stage="compute", allocations=allocations, remaining=1.0,
                    descriptor=self.describe(model, "compute", allocations))

    def settle(self):
        due = [entry for entry in self.admissions if entry[0] <= self.time]
        self.admissions = [entry for entry in self.admissions if entry[0] > self.time]
        for _, identity, rank in due:
            self.requests[identity]["rank"] = rank
            self.events.append(dict(time_s=self.time, event="admit", model=identity[0], id=identity[1], rank=rank))
        due = [entry for entry in self.receipts if entry[0] <= self.time]
        self.receipts = [entry for entry in self.receipts if entry[0] > self.time]
        for _, identity, output in due:
            state = self.requests[identity]
            state["received"] = max(state["received"], output)
            if state["received"] >= state["target"] and state["milestone"] is None:
                state["milestone"] = self.time
                self.events.append(dict(time_s=self.time, event="milestone", model=identity[0], id=identity[1]))
        for model in (0, 1):
            if self.active[model] is not None:
                continue
            if self.moves[model] and not self.moved[model]:
                descriptor = self.describe(model, "drain", [])
                descriptor["bytes"] = sum(move["bytes"] for move in self.moves[model])
                self.active[model] = dict(stage="drain", allocations=[], remaining=1.0, descriptor=descriptor)
            else:
                self.active[model] = self.new_batch(model)

    def finish(self, model):
        active = self.active[model]
        stage = active["stage"]
        self.events.append(dict(time_s=self.time, event="stage_complete", model=model, stage=stage,
                                phase=active["descriptor"]["phase"]))
        if stage in ("compute", "drain"):
            active["stage"] = "return" if stage == "compute" else "transfer"
            active["descriptor"] = dict(active["descriptor"], phase=active["stage"])
            active["remaining"] = 1.0
            return
        if stage == "return":
            for identity, amount in active["allocations"]:
                state = self.requests[identity]
                state["prefix"] += amount
                if state["prefix"] == state["prompt"] + state["output"]:
                    state["output"] += 1
                    self.receipts.append((receipt_time(self.time, self.receipt), identity, state["output"]))
                self.events.append(dict(time_s=self.time, event="input_return", model=model, id=identity[1],
                                        prefix_tokens=state["prefix"], output_tokens=state["output"]))
        elif stage == "transfer":
            for move in self.moves[model]:
                self.requests[model, move["id"]]["rank"] = move["destination"]
            self.moved[model] = True
        self.active[model] = None

    def run(self, max_time_s, max_events):
        for _ in range(max_events):
            self.settle()
            if all(state["milestone"] is not None for state in self.requests.values()):
                return "complete_conditional_rollout"
            descriptors = [active["descriptor"] if active else self.describe(model, "empty", [])
                           for model, active in enumerate(self.active)]
            durations = [self.response.duration(model, descriptors[model], descriptors[1 - model]) if active else None
                         for model, active in enumerate(self.active)]
            completion = [self.time + active["remaining"] * durations[model] if active else math.inf
                          for model, active in enumerate(self.active)]
            next_time = min(*completion, *[entry[0] for entry in self.receipts],
                            *[entry[0] for entry in self.admissions])
            if math.isinf(next_time):
                return "stalled_unserved_cohort"
            if next_time > max_time_s:
                return "time_limit_censored"
            elapsed = next_time - self.time
            if elapsed <= 0 or not math.isfinite(elapsed):
                raise DurationUnavailable("nonpositive_or_nonfinite_event_interval")
            self.events.append(dict(time_s=self.time, event="service_interval", until_s=next_time,
                                    descriptors=copy.deepcopy(descriptors), full_service_s=durations))
            for model, active in enumerate(self.active):
                if active:
                    active["remaining"] = (completion[model] - next_time) / durations[model]
            self.time = next_time
            for model in (0, 1):
                if completion[model] == next_time:
                    self.finish(model)
        self.settle()
        if all(state["milestone"] is not None for state in self.requests.values()):
            return "complete_conditional_rollout"
        return "event_limit_censored"


def simulate_coupled_duration(reference, layout, initial, response, *, action, receipt, limits,
                              max_time_s, max_events):
    """Numerically evaluate conditional inputs without claiming independent predictive validity."""
    result = dict(status="unavailable", reference_id=reference.get("reference_id"), objective_s=None,
                  prediction_gate_passed=False, deployment_status="local_conditional_kernel_not_validated",
                  initialization_sha256=fingerprint(initial), layout_sha256=fingerprint(layout),
                  action_sha256=fingerprint(action), response_sha256=response.identifier,
                  protocol_sha256=fingerprint(dict(receipt=receipt, limits=limits, max_time_s=max_time_s,
                                                   max_events=max_events)), errors=[], outcomes=[], events=[])
    rollout = None
    try:
        positive(max_time_s, "max_time")
        integer(max_events, "max_events", 1)
        rollout = CoupledRollout(reference, layout, initial, response, action, receipt, limits)
        result["status"] = rollout.run(max_time_s, max_events)
    except (DurationUnavailable, ValueError, KeyError, TypeError) as error:
        result["errors"].append(str(error))
    if rollout is not None:
        result["events"] = rollout.events
        result["last_event_s"] = rollout.time
        result["outcomes"] = [dict(model=state["model"], id=state["id"], milestone_s=state["milestone"],
            output_tokens=state["output"], received_output_tokens=state["received"], prefix_tokens=state["prefix"],
            rank=state["rank"]) for state in rollout.requests.values()]
        if result["status"] == "complete_conditional_rollout":
            result["objective_s"] = sum(state["milestone"] for state in rollout.requests.values())
    if not result["outcomes"]:
        result["outcomes"] = [dict(model=row["model"], id=row["id"], milestone_s=None)
                              for row in reference.get("requests", [])]
    return result
