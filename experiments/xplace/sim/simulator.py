"""Discrete-event simulator of two DP/EP tenants sharing per-GPU KV budgets, driven by the live planner.

Each tenant steps all DP ranks together. A Conv decode step costs StepModel(tenant 1); Code prefill runs one
4096-token chunk per step. Cross-tenant interference applies while the other tenant has requests in flight.
Migrations pause both tenants at a 32-step boundary (idle tenant first) and cost fixed + copy time.
"""
import hashlib
import heapq
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from cost_model import StepModel, capture_bucket  # noqa: E402
from fairness import FairAdmissionGate  # noqa: E402
from planner import DEFAULT_PARAMS, build_state, plan_admission, plan_return  # noqa: E402

BLOCK_TOKENS, PAGE_BLOCKS = 16, 64
PAGE_BYTES, BLOCK_BYTES = 96 * 1024**2, int(1.5 * 1024**2)
PAGE_GIB, BLOCK_GIB = PAGE_BYTES / 1024**3, BLOCK_BYTES / 1024**3
CHUNK_TOKENS = 4096

POLICIES = {
    "no_move": {"admission": None, "return": False, "fair": False},
    "no_move_fair": {"admission": None, "return": False, "fair": True},
    "mig_random": {"admission": "mig_random", "return": False, "fair": False},
    "mig_bestfit": {"admission": "mig_bestfit", "return": False, "fair": False},
    "ca_admit": {"admission": "ca", "return": False, "fair": False},
    "ca_return": {"admission": "mig_random", "return": True, "fair": False},
    "ca_full": {"admission": "ca", "return": True, "fair": False},
}
# Exploratory (simulator only): pause Conv while any Code request is resident (tenant time-multiplexing).
for _name in ("no_move", "mig_random", "ca_full"):
    POLICIES[f"{_name}_tm"] = dict(POLICIES[_name], isolate=True)
# v2: control-plane changes. Fast Conv admission is not modeled (no RPC latency here); hold_first delays Code
# start until Conv reached its pause boundary.
POLICIES["no_move_v2"] = dict(POLICIES["no_move"])
POLICIES["mig_random_v2"] = dict(POLICIES["mig_random"])
POLICIES["ca_full_v2"] = dict(POLICIES["ca_full"])
POLICIES["no_move_tm_v2"] = dict(POLICIES["no_move_tm"], hold_first=True)
POLICIES["ca_full_tm_v2"] = dict(POLICIES["ca_full_tm"], hold_first=True)
# Exploratory: sliced TM. After `slice[0]` s of Code prefill, pause the Code for a Conv window of `slice[1]` s
# (plus the time Conv needs to reach its next pause boundary). Needs a per-step pausable Code tenant in live.
for _slice in ((3, 1.0), (5, 1.0), (10, 1.0), (5, 2.0)):
    POLICIES[f"ca_full_tm_s{_slice[0]}r{_slice[1]:g}"] = dict(POLICIES["ca_full_tm_v2"], slice=_slice)
POLICIES["ca_full_tm_s5"] = POLICIES["ca_full_tm_s5r1"]  # live policy name
POLICIES["ca_full_tm_s4"] = dict(POLICIES["ca_full_tm_v2"], slice=(4.0, 1.0))
POLICIES["ca_full_tm_slo"] = dict(POLICIES["ca_full_tm_v2"], slice=(2.0, 1.0, 15.0))

DEFAULT_SIM = {"num_ranks": 4, "budget_gib": 40.0, "pause_boundary": 32, "transaction_fixed_s": 0.3,
               "move_fixed_s": 0.05, "copy_gib_s": 10.0, "admission_overhead_s": 0.01,
               "fairness_threshold_s": 30.0, "return_period_s": 10.0, "return_cooldown_s": 10.0,
               "plan_retry_s": 2.0, "population_period_s": 5.0, "drain_timeout_s": 7200.0, "seed": 20261103,
               # live-measured per-move time (None: move_fixed_s + copy); peer copies contend with Code prefill
               "move_s": None, "move_s_during_code_prefill": None,
               # per-Code-episode noise: Conv-side interference factor ~ N(1, interference_cv) during that Code's
               # prefill, Code prefill speed factor ~ N(1, prefill_cv) (0: deterministic means)
               "interference_cv": 0.0, "prefill_cv": 0.0, "noise_seed": 0,
               # sliced TM: DP consensus after the Code keep-pause before Conv may run (live ~0.3 s)
               "slice_consensus_s": 0.3}


class Request:
    __slots__ = ("mark", "id", "model", "prompt", "output", "arrival", "rank", "rank_initial", "reserved_at",
                 "first_token_at", "completed_at", "start_step", "chunks_left", "pages", "blocks", "moves", "path",
                 "factor", "prefill_factor")

    def __init__(self, mark):
        self.mark, self.id, self.model = mark, mark["request_id"], mark["model"]
        self.prompt, self.output, self.arrival = mark["prompt_tokens"], mark["output_tokens"], mark["arrival_s"]
        self.rank = self.rank_initial = self.reserved_at = self.first_token_at = self.completed_at = None
        self.start_step = None
        self.chunks_left = 0
        self.blocks = math.ceil((self.prompt + self.output) / BLOCK_TOKENS)
        self.pages = math.ceil(self.blocks / PAGE_BLOCKS)
        self.moves, self.path = 0, None
        self.factor = self.prefill_factor = 1.0


class Tenant:
    def __init__(self, index, num_ranks):
        self.index, self.num_ranks = index, num_ranks
        self.active, self.joining = {}, {}
        self.count, self.context = [0] * num_ranks, [0.0] * num_ranks
        self.step_index, self.busy, self.paused, self.pause_pending, self.paused_at = 0, False, False, False, None
        self.held, self.hold_pending = False, False
        self.sliced = False
        self.completions, self.in_step = [], []

    def has_work(self):
        return bool(self.active or self.joining)

    def generated(self, request):
        return 0 if request.start_step is None else self.step_index - request.start_step + 1

    def current_context(self, request):
        return request.prompt + self.generated(request)


class Simulator:
    def __init__(self, trace, policy, sim_params=None, step_params=None, planner_step_params=None,
                 planner_params=None):
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy {policy}")
        self.trace, self.policy, self.spec = trace, policy, POLICIES[policy]
        self.params = dict(DEFAULT_SIM, **(sim_params or {}))
        self.num_ranks = self.params["num_ranks"]
        budget = self.params["budget_gib"]
        self.budgets = list(budget) if isinstance(budget, (list, tuple)) else [float(budget)] * self.num_ranks
        self.truth = StepModel(step_params)
        self.belief = StepModel(planner_step_params or step_params)
        self.prefill = step_params.get("code_prefill", {"solo_s_per_100k": 26.4, "with_other_s_per_100k": 54.0})
        self.planner_params = dict(DEFAULT_PARAMS, **(planner_params or {}))
        self.now, self.sequence, self.events = 0.0, 0, []
        self.tenants = [Tenant(0, self.num_ranks), Tenant(1, self.num_ranks)]
        self.queues = [[], []]
        self.used_pages = [2] * self.num_ranks
        self.transaction = None
        self.completed, self.decisions, self.migrations, self.population = [], [], [], []
        self.total = len(trace["requests"])
        self.fairness = FairAdmissionGate(self.params["fairness_threshold_s"]) if self.spec["fair"] else None
        self.plan_not_before = {}
        self.last_return = -math.inf
        self.code_run_since = None
        self.slice_armed = False
        self.hold_intervals = []
        self.code_arrivals = 0

    # ---------- event plumbing ----------
    def push(self, time, kind, payload=None):
        self.sequence += 1
        heapq.heappush(self.events, (time, self.sequence, kind, payload))

    def run(self):
        for mark in self.trace["requests"]:
            self.push(mark["arrival_s"], "arrival", Request(mark))
        self.push(0.0, "population")
        if self.spec["return"]:
            self.push(self.params["return_period_s"], "return_check")
        limit = self.trace["duration_s"] + self.params["drain_timeout_s"]
        while self.events and len(self.completed) < self.total:
            time, _, kind, payload = heapq.heappop(self.events)
            if time > limit:
                raise TimeoutError(f"{self.policy}: simulation exceeded drain timeout at t={time:.0f}")
            self.now = time
            getattr(self, "on_" + kind)(payload)
        return {"policy": self.policy, "requests": [self.record(request) for request in self.completed],
                "decisions": self.decisions, "migrations": self.migrations, "population": self.population,
                "budgets_gib": self.budgets, "hold_intervals": self.hold_intervals}

    def record(self, request):
        return {"request_id": request.id, "trace_request_id": request.id, "model": request.model,
                "prompt_tokens": request.prompt, "output_tokens": request.output, "arrived_at": request.arrival,
                "arrival_s": request.arrival,
                "reserved_at": request.reserved_at, "first_token_at": request.first_token_at,
                "completed_at": request.completed_at, "rank_initial": request.rank_initial,
                "moves": request.moves, "admission_path": request.path}

    # ---------- KV accounting ----------
    def used_gib(self):
        return [pages * PAGE_GIB for pages in self.used_pages]

    def fits(self, request, rank):
        return (self.used_pages[rank] + request.pages) * PAGE_GIB <= self.budgets[rank] + 1e-9

    def observed(self):
        observed = []
        for tenant in self.tenants:
            ranks = [{"requests": [], "pages": [{"owners": [], "pinned": True, "memory_gib": PAGE_GIB}],
                      "mapped_bytes": PAGE_BYTES, "page_bytes": PAGE_BYTES, "block_bytes": BLOCK_BYTES,
                      "page_block_capacity": PAGE_BLOCKS, "uniform_page_geometry": True, "block_size": BLOCK_TOKENS,
                      "mapped_free_block_slots": 0, "free_blocks": 10**7} for _ in range(self.num_ranks)]
            for request in list(tenant.active.values()) + list(tenant.joining.values()):
                rank = ranks[request.rank]
                decoding = request.start_step is not None
                rank["requests"].append({"id": request.id, "computed_tokens": tenant.current_context(request),
                                         "prompt_tokens": request.prompt, "output_tokens": tenant.generated(request),
                                         "max_tokens": request.output, "num_blocks": request.blocks,
                                         "movable": decoding})
                rank["pages"].append({"owners": [request.id], "pinned": False,
                                      "memory_gib": request.pages * PAGE_GIB})
                rank["mapped_bytes"] += request.pages * PAGE_BYTES
            observed.append(ranks)
        return observed

    def all_decoding(self, models=(0, 1)):
        return all(request.start_step is not None and request.chunks_left == 0
                   for model in models for tenant in (self.tenants[model],)
                   for request in list(tenant.active.values()) + list(tenant.joining.values()))

    # ---------- arrivals and admission ----------
    def on_arrival(self, request):
        self.queues[request.model].append(request)
        if request.model == 0:
            self.code_arrivals += 1
        self.dispatch()

    def admit(self, request, rank, path):
        request.rank = request.rank_initial = rank
        request.reserved_at, request.path = self.now, path
        self.used_pages[rank] += request.pages
        tenant = self.tenants[request.model]
        if request.model == 0:
            request.chunks_left = math.ceil(request.prompt / CHUNK_TOKENS)
            rng = random.Random(f"{self.params['noise_seed']}:{request.id}")
            interference, prefill = rng.gauss(1.0, 1.0), rng.gauss(1.0, 1.0)
            request.factor = max(0.2, 1.0 + self.params["interference_cv"] * (interference - 1.0))
            request.prefill_factor = max(0.5, 1.0 + self.params["prefill_cv"] * (prefill - 1.0))
        tenant.joining[request.id] = request
        if self.fairness is not None:
            self.fairness.reserved(request.id, self.now, self.now)
        if self.spec.get("isolate") and request.model == 0:
            self.hold_conv()
        if not tenant.busy and not tenant.paused:
            self.start_step(tenant)

    def hold_conv(self):
        conv = self.tenants[1]
        if conv.held:
            return
        if conv.busy:
            conv.hold_pending = True
        else:
            self.mark_held(conv)

    def mark_held(self, conv):
        conv.held, conv.hold_pending = True, False
        self.hold_intervals.append([self.now, None])

    def release_conv(self):
        conv = self.tenants[1]
        if conv.held and self.hold_intervals and self.hold_intervals[-1][1] is None:
            self.hold_intervals[-1][1] = self.now
        conv.held = conv.hold_pending = False
        if not conv.paused:
            self.start_step(conv)

    def code_prefilling(self):
        code = self.tenants[0]
        return any(request.chunks_left > 0 for request in list(code.active.values()) + list(code.joining.values()))

    def resume_code_slice(self):
        code = self.tenants[0]
        code.sliced = False
        self.code_run_since = self.now
        self.start_step(code)

    def on_slice_release(self, _):
        if not self.tenants[0].sliced:
            return
        self.release_conv()
        self.push(self.now + self.spec["slice"][1], "slice_end")

    def on_slice_end(self, _):
        if not self.tenants[0].sliced:
            return
        self.hold_conv()
        if self.tenants[1].held:
            self.resume_code_slice()

    def dispatch(self):
        if self.transaction is not None:
            return
        progressed = True
        while progressed:
            progressed = False
            heads = sorted((queue[0] for queue in self.queues if queue), key=lambda request: request.arrival)
            for request in heads:
                if self.fairness is not None:
                    waiting = [{"arrived_at": head.arrival, "mark": {"request_id": head.id, "model": head.model}}
                               for head in (queue[0] for queue in self.queues if queue)]
                    self.fairness.update(waiting, self.now)
                    if not self.fairness.authorization(request.model, request.id, self.now)["allowed"]:
                        continue
                candidates = [rank for rank in range(self.num_ranks) if self.fits(request, rank)]
                if candidates:
                    counts = self.tenants[request.model].count
                    joining = [0] * self.num_ranks
                    for item in self.tenants[request.model].joining.values():
                        joining[item.rank] += 1
                    rank = min(candidates, key=lambda r: (counts[r] + joining[r], self.used_pages[r], r))
                    self.queues[request.model].pop(0)
                    self.admit(request, rank, "direct")
                    progressed = True
                    continue
                if self.spec["admission"] is not None and self.try_migration(request):
                    return

    def try_migration(self, request):
        if self.now < self.plan_not_before.get(request.id, 0.0):
            return False
        self.plan_not_before[request.id] = self.now + self.params["plan_retry_s"]
        free = sum(self.budgets) - sum(self.used_gib())
        if request.pages * PAGE_GIB < self.planner_params["migrate_min_gib"] or free < request.pages * PAGE_GIB:
            return False
        arrival = {"model": request.model, "tokens": request.prompt + request.output,
                   "prompt_tokens": request.prompt, "output_tokens": request.output}
        seed = int.from_bytes(hashlib.sha256(f"{self.params['seed']}:{request.id}".encode()).digest()[:8], "big")

        def plan_fn():
            return plan_admission(build_state(self.observed(), self.budgets), arrival, self.spec["admission"],
                                  self.belief, self.planner_params, seed)

        preplan = plan_fn()
        if preplan.get("target") is None:
            return False
        pause_models = sorted({move["model"] for move in preplan["moves"]})
        if not self.all_decoding(pause_models):
            return False
        self.start_transaction("admission", plan_fn, request, pause_models)
        return True

    # ---------- stepping ----------
    def other_phase(self, tenant):
        other = self.tenants[1 - tenant.index]
        if not other.has_work() or (other.held and not other.busy):
            return False
        if other.index == 0 and not other.busy and other.sliced:
            return False  # the Code is paused for a Conv slice
        if other.index == 0 and not other.busy and self.spec.get("hold_first") and tenant.hold_pending:
            return False  # the Code is waiting for this tenant's pause boundary and is not running yet
        if any(request.chunks_left > 0 for request in list(other.active.values()) + list(other.joining.values())):
            return "prefill"
        return "decode"

    def prefill_factor(self, attribute="factor"):
        """Mean noise factor of Code requests currently in prefill (1.0 if none or noise disabled)."""
        factors = [getattr(request, attribute) for request in list(self.tenants[0].active.values())
                   + list(self.tenants[0].joining.values()) if request.chunks_left > 0]
        return sum(factors) / len(factors) if factors else 1.0

    def step_duration(self, tenant):
        phase = self.other_phase(tenant)
        if tenant.index == 0 and any(request.chunks_left > 0 for request in tenant.in_step):
            prompt = max(request.prompt for request in tenant.in_step if request.chunks_left > 0)
            per_100k = self.prefill["solo_s_per_100k"]
            if phase:
                value = self.prefill["with_other_s_per_100k"]
                if isinstance(value, dict):
                    bucket = capture_bucket(max(self.tenants[1].count) or 1)
                    sizes = sorted(int(size) for size in value)
                    chosen = next((size for size in sizes if bucket <= size), sizes[-1])
                    value = value[str(chosen)] if str(chosen) in value else value[chosen]
                per_100k = value * self.prefill_factor("prefill_factor")
            return per_100k * prompt / 100000 / math.ceil(prompt / CHUNK_TOKENS)
        if tenant.index == 1 and phase == "prefill":
            own = self.truth.decode_step(1, tenant.count, tenant.context, False)
            return own + self.prefill_factor() * (self.truth.decode_step(1, tenant.count, tenant.context, phase) - own)
        return self.truth.decode_step(tenant.index, tenant.count, tenant.context, phase)

    def slice_due(self, slice_cfg):
        """Periodic slices: (S, R). TTFT-triggered: (T_wait, R, S_max) -- slice when a held Conv request has not
        started and waited >= T_wait, or the hold lasted >= S_max."""
        held_for = self.now - self.code_run_since
        if len(slice_cfg) == 2:
            return held_for >= slice_cfg[0]
        waiting = [self.now - request.arrival for request in self.tenants[1].joining.values()]
        return held_for >= slice_cfg[2] or (waiting and max(waiting) >= slice_cfg[0])

    def start_step(self, tenant):
        if tenant.busy or tenant.paused or tenant.held or not tenant.has_work():
            return
        if tenant.index == 0 and self.spec.get("hold_first") and self.tenants[1].hold_pending:
            return  # Code waits until Conv reached its pause boundary
        if tenant.index == 0:
            if tenant.sliced:
                return
            slice_cfg = self.spec.get("slice")
            if slice_cfg and self.code_prefilling() and self.tenants[1].held and self.tenants[1].has_work():
                if self.code_run_since is None:
                    self.code_run_since = self.now
                elif self.slice_armed:
                    # live: the slicer's Code summary waits one step, then keep-pause waits for the next step
                    tenant.sliced, self.slice_armed = True, False
                    self.push(self.now + self.params["slice_consensus_s"], "slice_release")
                    return
                elif self.slice_due(slice_cfg):
                    self.slice_armed = True
            elif not self.code_prefilling():
                self.code_run_since, self.slice_armed = None, False
        for request in tenant.joining.values():
            tenant.active[request.id] = request
            tenant.count[request.rank] += 1
            tenant.context[request.rank] += request.prompt if request.model == 1 else 0
        tenant.joining.clear()
        tenant.in_step = list(tenant.active.values())
        tenant.busy = True
        self.push(self.now + self.step_duration(tenant), "step_end", tenant.index)

    def on_step_end(self, index):
        tenant = self.tenants[index]
        tenant.busy = False
        tenant.step_index += 1
        step = tenant.step_index
        prefilled_ranks = set()
        for request in tenant.in_step:
            if request.id not in tenant.active:
                continue
            if request.start_step is None:
                if request.chunks_left > 0:
                    if request.rank in prefilled_ranks:
                        continue
                    prefilled_ranks.add(request.rank)
                    request.chunks_left -= 1
                    if request.chunks_left:
                        continue
                    tenant.context[request.rank] += request.prompt
                request.start_step = step
                request.first_token_at = self.now
                heapq.heappush(tenant.completions, (step + request.output - 1, request.id))
            tenant.context[request.rank] += 1
        while tenant.completions and tenant.completions[0][0] <= step:
            _, identifier = heapq.heappop(tenant.completions)
            request = tenant.active.get(identifier)
            if request is not None and request.start_step is not None:
                self.complete(tenant, request)
        if tenant.pause_pending and (step % self.params["pause_boundary"] == 0 or not tenant.has_work()):
            self.pause_reached(tenant)
        elif tenant.hold_pending and (step % self.params["pause_boundary"] == 0 or not tenant.has_work()):
            self.mark_held(tenant)
            if self.tenants[0].sliced:
                self.resume_code_slice()
            else:
                self.start_step(self.tenants[0])
        elif not tenant.paused:
            self.start_step(tenant)

    def complete(self, tenant, request):
        del tenant.active[request.id]
        tenant.count[request.rank] -= 1
        tenant.context[request.rank] -= request.prompt + request.output
        self.used_pages[request.rank] -= request.pages
        request.completed_at = self.now
        self.completed.append(request)
        if self.spec.get("isolate") and request.model == 0 and not tenant.has_work():
            self.release_conv()
        if self.spec["return"] and request.model == 0:
            self.push(self.now + 0.5, "return_check", "release")
        self.dispatch()

    # ---------- transactions ----------
    def start_transaction(self, kind, plan_fn, arrival=None, pause_models=(0, 1)):
        order = sorted((self.tenants[model] for model in pause_models),
                       key=lambda tenant: (len(tenant.active) + len(tenant.joining), tenant.index))
        self.transaction = {"kind": kind, "plan_fn": plan_fn, "arrival": arrival, "requested_at": self.now,
                            "order": [tenant.index for tenant in order], "stage": 0}
        if order:
            self.request_pause(order[0])
        else:
            self.execute_transaction()

    def request_pause(self, tenant):
        if not tenant.busy:
            self.pause_reached(tenant)
        else:
            tenant.pause_pending = True

    def pause_reached(self, tenant):
        tenant.pause_pending, tenant.paused, tenant.paused_at = False, True, self.now
        transaction = self.transaction
        transaction["stage"] += 1
        if transaction["stage"] < len(transaction["order"]):
            self.request_pause(self.tenants[transaction["order"][transaction["stage"]]])
            return
        self.execute_transaction()

    def move_duration(self, move):
        if self.params["move_s"] is None:
            return self.params["move_fixed_s"] + move["blocks"] * BLOCK_GIB / self.params["copy_gib_s"]
        prefilling = any(request.chunks_left > 0 for request in self.tenants[0].active.values())
        slow = self.params["move_s_during_code_prefill"]
        return slow if prefilling and slow is not None else self.params["move_s"]

    def execute_transaction(self):
        transaction = self.transaction
        plan = transaction["plan_fn"]()
        plan["moves"] = [move for move in plan.get("moves", []) if move["model"] in transaction["order"]]
        duration = self.params["transaction_fixed_s"] + sum(self.move_duration(move) for move in plan["moves"])
        transaction["plan"] = plan
        self.push(self.now + duration, "transaction_end")

    def on_transaction_end(self, _):
        transaction, self.transaction = self.transaction, None
        plan = transaction["plan"]
        for move in plan.get("moves", []):
            tenant = self.tenants[move["model"]]
            request = tenant.active.get(move["id"]) or tenant.joining.get(move["id"])
            if request is None or request.rank != move["source"]:
                continue
            context = tenant.current_context(request)
            if request.id in tenant.active:
                tenant.count[move["source"]] -= 1
                tenant.context[move["source"]] -= context
                tenant.count[move["destination"]] += 1
                tenant.context[move["destination"]] += context
            self.used_pages[move["source"]] -= request.pages
            self.used_pages[move["destination"]] += request.pages
            request.rank = move["destination"]
            request.moves += 1
            self.migrations.append({"request_id": request.id, "model": move["model"], "source": move["source"],
                                    "destination": move["destination"], "at": self.now, "kind": transaction["kind"]})
        paused_request_s = sum((len(self.tenants[index].active) + len(self.tenants[index].joining))
                               * (self.now - self.tenants[index].paused_at) for index in transaction["order"])
        admitted = False
        arrival = transaction["arrival"]
        if arrival is not None and plan.get("target") is not None and self.fits(arrival, plan["target"]):
            self.queues[arrival.model].remove(arrival)
            self.admit(arrival, plan["target"], "migration")
            admitted = True
        self.decisions.append({"kind": transaction["kind"], "at": self.now, "moves": len(plan.get("moves", [])),
                               "control_s": self.now - transaction["requested_at"], "admitted": admitted,
                               "paused_request_s": paused_request_s, "target": plan.get("target"),
                               "request_id": arrival.id if arrival is not None else None})
        if transaction["kind"] == "return" and plan.get("moves"):
            self.last_return = self.now
        for tenant in self.tenants:
            tenant.paused = False
            self.start_step(tenant)
        self.dispatch()

    # ---------- periodic work ----------
    def on_return_check(self, reason):
        if reason != "release":
            self.push(self.now + self.params["return_period_s"], "return_check")
        if (self.transaction is not None or any(self.queues)
                or self.now - self.last_return < self.params["return_cooldown_s"] or not self.all_decoding((1,))):
            return
        counts = self.tenants[1].count
        if max(counts) - min(counts) < 2:
            return
        rate = (self.code_arrivals + 1) / (max(1.0, self.now) + 1 / self.planner_params["code_rate"])

        def plan_fn():
            return plan_return(build_state(self.observed(), self.budgets), self.belief,
                               dict(self.planner_params, code_rate=rate))

        if plan_fn()["moves"]:
            self.start_transaction("return", plan_fn, pause_models=(1,))

    def on_population(self, _):
        self.population.append({"t": self.now, "counts": [list(tenant.count) for tenant in self.tenants],
                                "used_gib": self.used_gib(), "queued": [len(queue) for queue in self.queues]})
        self.push(self.now + self.params["population_period_s"], "population")


def simulate(trace, policy, **kwargs):
    return Simulator(trace, policy, **kwargs).run()
