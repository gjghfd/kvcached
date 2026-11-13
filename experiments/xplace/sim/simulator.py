"""Discrete-event simulator of two DP/EP tenants sharing per-GPU KV budgets, driven by the live planner.

Each tenant steps all DP ranks together. A Conv decode step costs StepModel(tenant 1); Code prefill runs one
4096-token chunk per step. Cross-tenant interference applies while the other tenant has requests in flight.
Migrations pause both tenants at a 32-step boundary (idle tenant first) and cost fixed + copy time.

KV modes. "full" reserves prompt + output at admission (legacy: the output is the replayed true length, an oracle).
"prompt" reserves only the prompt (+ a fixed extra) and grows one block per 16 decoded tokens. "native" (policy
kvcached_native) is stock vLLM: count-based DP load balancing, a request starts once its current sequence fits on
its rank, and a Code's prefill KV is allocated chunk by chunk. Outside "full", when a rank runs out of pages the
tenant that needed the block retracts its newest request on that rank (vLLM FCFS preemption); the retracted request
waits in that rank's queue until its current length fits again and recomputes it before decoding (a Code retracted
mid-prefill restarts its prefill).
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
from planner import (DEFAULT_PARAMS, RELIEF_FLOOR_ON_TARGET, build_state, conv_growth_tokens, growth_tokens,  # noqa: E402
                     plan_admission, plan_return, run_seconds, solo_rate, tm_gain)

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
    "kvcached_native": {"admission": None, "return": False, "fair": False, "native": True},
}
# Exploratory (simulator only): pause Conv while any Code request is resident (tenant time-multiplexing).
for _name in ("no_move", "mig_random", "ca_full"):
    POLICIES[f"{_name}_tm"] = dict(POLICIES[_name], isolate=True)
# v2: control-plane changes. Fast Conv admission is not modeled (no RPC latency here); hold_first delays Code
# start until Conv reached its pause boundary.
POLICIES["no_move_v2"] = dict(POLICIES["no_move"])
POLICIES["no_move_fair_v2"] = dict(POLICIES["no_move_fair"])
POLICIES["mig_random_v2"] = dict(POLICIES["mig_random"])
POLICIES["ca_admit_v2"] = dict(POLICIES["ca_admit"])
POLICIES["ca_return_v2"] = dict(POLICIES["ca_return"])
POLICIES["ca_full_v2"] = dict(POLICIES["ca_full"])
POLICIES["no_move_tm_v2"] = dict(POLICIES["no_move_tm"], hold_first=True)
POLICIES["mig_random_tm_v2"] = dict(POLICIES["mig_random_tm"], hold_first=True)
POLICIES["ca_full_tm_v2"] = dict(POLICIES["ca_full_tm"], hold_first=True)
# Exploratory: sliced TM. After `slice[0]` s of Code prefill, pause the Code for a Conv window of `slice[1]` s
# (plus the time Conv needs to reach its next pause boundary). Needs a per-step pausable Code tenant in live.
for _slice in ((3, 1.0), (5, 1.0), (10, 1.0), (5, 2.0)):
    POLICIES[f"ca_full_tm_s{_slice[0]}r{_slice[1]:g}"] = dict(POLICIES["ca_full_tm_v2"], slice=_slice)
POLICIES["ca_full_tm_s5"] = POLICIES["ca_full_tm_s5r1"]  # live policy name
POLICIES["ca_full_tm_s4"] = dict(POLICIES["ca_full_tm_v2"], slice=(4.0, 1.0))
POLICIES["ca_full_tm_slo"] = dict(POLICIES["ca_full_tm_v2"], slice=(2.0, 1.0, 15.0))
# ablation (current-length setting): placement without growth headroom
POLICIES["ca_full_tm_slo_nh"] = dict(POLICIES["ca_full_tm_slo"], no_headroom=True)
POLICIES["ca_full_v2_nh"] = dict(POLICIES["ca_full_v2"], no_headroom=True)
# ablations of the final scheme ca_full_tm_slo (same names as live); fast Conv admission is not modeled, so the
# v1 variant differs from the final scheme only by admitting Code before Conv reaches its pause boundary
_SLO = (2.0, 1.0, 15.0)
POLICIES["no_move_tm_slo"] = dict(POLICIES["no_move_tm_v2"], slice=_SLO)
POLICIES["mig_random_tm_slo"] = dict(POLICIES["mig_random_tm_v2"], slice=_SLO)
POLICIES["ca_admit_tm_slo"] = dict(POLICIES["ca_admit"], isolate=True, hold_first=True, slice=_SLO)
POLICIES["ca_return_tm_slo"] = dict(POLICIES["ca_return"], isolate=True, hold_first=True, slice=_SLO)
POLICIES["ca_full_tm_slo_v1"] = dict(POLICIES["ca_full_tm"], slice=_SLO)
# baselines without growth headroom (stock current-length admission semantics, controller-side queueing)
POLICIES["no_move_v2_nh"] = dict(POLICIES["no_move_v2"], no_headroom=True)
POLICIES["no_move_tm_slo_nh"] = dict(POLICIES["no_move_tm_slo"], no_headroom=True)
# General (type-agnostic) algorithm (Paper/design.md §5, live policies *_g): per-request growth headroom on every
# admission, migration for any fragmentation-blocked request, episodes of time-multiplexing started by a prefill's
# gain (`isolate` enables them), returns over every tenant
for _name, _spec in list(POLICIES.items()):
    if not _spec.get("native"):
        POLICIES[f"{_name}_g"] = dict(_spec, general=True)

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
               "slice_consensus_s": 0.3,
               # admission of controlled policies ("full" legacy oracle reservation or "prompt" current length)
               "kv_mode": "full", "reserve_extra_tokens": 0,
               # a retracted request recomputes prompt + generated tokens in 4096-token chunks before decoding
               "recompute_s_per_token": 1.25e-4,
               # outside "full": Conv pages are shared by the rank's Conv and stay mapped as a high-water mark
               # (page_model "high_water", used by the forecasts up to V12)
               "page_sharing": True,
               # outside "full": "blocks" (per-page emulation of kvcached's allocator, PagePool) or "high_water"
               "page_model": "blocks",
               # an engine-internal queued request makes its DP group spin dummy batches that interfere
               "dp_dummy_interference": True}


class Request:
    __slots__ = ("mark", "id", "model", "prompt", "output", "arrival", "rank", "rank_initial", "reserved_at",
                 "first_token_at", "completed_at", "start_step", "chunks_left", "pages", "blocks", "moves", "path",
                 "factor", "prefill_factor", "generated", "held_blocks", "order", "recompute_left", "preemptions",
                 "ctx", "prefill_chunks", "holdings", "chunked")

    def __init__(self, mark):
        self.mark, self.id, self.model = mark, mark["request_id"], mark["model"]
        self.prompt, self.output, self.arrival = mark["prompt_tokens"], mark["output_tokens"], mark["arrival_s"]
        # a prompt longer than one 4096-token chunk prefills step by step (Code here); a short one joins decoding
        self.chunked = self.prompt > CHUNK_TOKENS
        self.rank = self.rank_initial = self.reserved_at = self.first_token_at = self.completed_at = None
        self.start_step = None
        self.chunks_left = self.prefill_chunks = 0
        self.blocks = math.ceil((self.prompt + self.output) / BLOCK_TOKENS)  # legacy full reservation
        self.pages = math.ceil(self.blocks / PAGE_BLOCKS)
        self.moves, self.path = 0, None
        self.factor = self.prefill_factor = 1.0
        self.generated = self.held_blocks = self.order = self.recompute_left = self.preemptions = 0
        self.ctx = 0.0
        self.holdings = {}  # page_model "blocks": page id -> blocks of this request on its rank


class PagePool:
    """kvcached's page allocator for one (tenant, rank) pool (`KVCacheManager._alloc_impl` / `_free`).

    A one-block decode allocation goes to the fullest mapped page with room (`alloc_packed`); a run takes the
    smallest page that holds all of it, else the emptiest page, and repeats. A new page is mapped only when no mapped
    page has a free block, and a page is unmapped once all its blocks are free. Requests growing one block at a time
    therefore interleave in the same pages, and moving some of a rank's requests frees only the pages they alone own.
    """

    def __init__(self):
        self.free = {}  # mapped page id -> free blocks
        self.owners = {}  # mapped page id -> {request id: blocks}
        self.by_free = [set() for _ in range(PAGE_BLOCKS + 1)]
        self.slots = 0  # free blocks in mapped pages
        self.next_id = 0

    def pages(self):
        return len(self.free)

    def _set_free(self, page, value):
        self.by_free[self.free[page]].discard(page)
        self.slots += value - self.free[page]
        self.free[page] = value
        self.by_free[value].add(page)

    def _pick(self, need):
        for free in range(min(need, PAGE_BLOCKS), PAGE_BLOCKS + 1):
            if self.by_free[free]:
                return min(self.by_free[free])
        for free in range(min(need, PAGE_BLOCKS + 1) - 1, 0, -1):
            if self.by_free[free]:
                return min(self.by_free[free])
        return None

    def alloc(self, request, blocks, packed=False):
        while blocks > 0:
            page = self._pick(1 if packed else blocks)
            if page is None:
                page, self.next_id = self.next_id, self.next_id + 1
                self.free[page], self.owners[page] = 0, {}
                self.by_free[0].add(page)
                self._set_free(page, PAGE_BLOCKS)
            take = min(self.free[page], blocks)
            self._set_free(page, self.free[page] - take)
            self.owners[page][request.id] = self.owners[page].get(request.id, 0) + take
            request.holdings[page] = request.holdings.get(page, 0) + take
            blocks -= take

    def release(self, request):
        for page, count in request.holdings.items():
            del self.owners[page][request.id]
            self._set_free(page, self.free[page] + count)
            if self.free[page] == PAGE_BLOCKS:
                self.by_free[PAGE_BLOCKS].discard(page)
                self.slots -= PAGE_BLOCKS
                del self.free[page], self.owners[page]
        request.holdings = {}


class Tenant:
    def __init__(self, index, num_ranks):
        self.index, self.num_ranks = index, num_ranks
        self.active, self.joining = {}, {}
        self.count, self.context = [0] * num_ranks, [0.0] * num_ranks
        self.step_index, self.busy, self.paused, self.pause_pending, self.paused_at = 0, False, False, False, None
        self.held, self.hold_pending = False, False
        self.sliced = False
        self.in_step = []
        self.waiting = [[] for _ in range(num_ranks)]  # vLLM-internal queues: native arrivals, retracted requests
        self.blocks = [0] * num_ranks  # held KV blocks per rank (shared page pool)
        # committed mapped pages per rank: Conv blocks grow one at a time into the fullest page with room, so the
        # rank's requests share pages and freed blocks stay mapped as holes (refilled by growth) until the rank holds
        # no Conv blocks at all (live: Conv mapped KV is a running high-water mark, V7/V8)
        self.high_water = [0] * num_ranks
        self.pools = [PagePool() for _ in range(num_ranks)]  # page_model "blocks"
        self.next_rank = 0  # native DP load balancing: rotating scan start

    def has_work(self):
        return bool(self.active or self.joining)

    def has_waiting(self):
        return any(self.waiting)


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
        self.native = bool(self.spec.get("native"))
        self.mode = "native" if self.native else self.params["kv_mode"]
        if self.mode not in ("full", "prompt", "native"):
            raise ValueError(f"Unknown kv_mode {self.mode}")
        if self.params["page_model"] not in ("high_water", "blocks"):
            raise ValueError(f"Unknown page_model {self.params['page_model']}")
        self.page_model = self.mode != "full" and self.params["page_model"] == "blocks"
        self.capacity_pages = [math.floor(value / PAGE_GIB + 1e-9) for value in self.budgets]
        if self.mode != "full":
            # live plans against the ballast-capped capacity: whole page bundles
            self.budgets = [pages * PAGE_GIB for pages in self.capacity_pages]
        self.truth = StepModel(step_params)
        self.belief = StepModel(planner_step_params or step_params)
        self.prefill = step_params.get("code_prefill", {"solo_s_per_100k": 26.4, "with_other_s_per_100k": 54.0})
        self.planner_params = dict(DEFAULT_PARAMS, **(planner_params or {}))
        self.target_floor = RELIEF_FLOOR_ON_TARGET[self.planner_params["relief_scope"]]
        self.now, self.sequence, self.events = 0.0, 0, []
        self.tenants = [Tenant(0, self.num_ranks), Tenant(1, self.num_ranks)]
        self.queues = [[], []]
        self.used_pages = [2] * self.num_ranks  # legacy full mode: per-request page sum
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
        self.order = 0
        self.preemptions, self.recompute_tokens = [0, 0], [0, 0]
        # general policies: the time-multiplexing episode in force ({"x", "triggers", "start"}) and arrival needs
        self.general = bool(self.spec.get("general"))
        self.episode = None
        self.episodes, self.arrival_needs = [], []

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
                "budgets_gib": self.budgets, "hold_intervals": self.hold_intervals, "kv_mode": self.mode,
                "preemptions": list(self.preemptions), "recompute_tokens": list(self.recompute_tokens),
                **({"episodes": self.episodes} if self.general else {})}

    def record(self, request):
        return {"request_id": request.id, "trace_request_id": request.id, "model": request.model,
                "prompt_tokens": request.prompt, "output_tokens": request.output, "arrived_at": request.arrival,
                "arrival_s": request.arrival,
                "reserved_at": request.reserved_at, "first_token_at": request.first_token_at,
                "completed_at": request.completed_at, "rank_initial": request.rank_initial,
                "moves": request.moves, "admission_path": request.path, "preemptions": request.preemptions}

    # ---------- KV accounting ----------
    def shared_pages(self, tenant):
        """Conv blocks interleave across pages (page sharing); the Code's big prefill runs own their pages."""
        return tenant.index == 1 and self.params["page_sharing"]

    def tenant_pages(self, tenant, rank, blocks=None):
        if self.page_model:
            return tenant.pools[rank].pages()
        blocks = tenant.blocks[rank] if blocks is None else blocks
        pages = math.ceil(blocks / PAGE_BLOCKS)
        if self.shared_pages(tenant) and blocks > 0:
            return max(tenant.high_water[rank], pages)
        return pages

    def commit_pages(self, tenant, rank):
        if self.page_model:
            return
        tenant.high_water[rank] = self.tenant_pages(tenant, rank) if tenant.blocks[rank] > 0 else 0

    def pages_on(self, rank):
        if self.mode == "full":
            return self.used_pages[rank]
        return 2 + sum(self.tenant_pages(tenant, rank) for tenant in self.tenants)

    def used_gib(self):
        return [self.pages_on(rank) * PAGE_GIB for rank in range(self.num_ranks)]

    def reservation_blocks(self, request):
        """Blocks a controlled admission books: the legacy oracle length, or the prompt plus the fixed extra."""
        if self.mode == "full":
            return request.blocks
        return math.ceil((request.prompt + self.params["reserve_extra_tokens"]) / BLOCK_TOKENS)

    def target_blocks(self, request):
        """Blocks the request needs now: current length (native Code prefill: computed chunks + the next one)."""
        tokens = request.prompt + request.generated
        if request.chunked and request.start_step is None and self.mode == "native":
            done = request.prefill_chunks - request.chunks_left
            tokens = min(request.prompt, (done + 1) * CHUNK_TOKENS)
        if self.mode == "prompt":
            tokens = max(tokens, request.prompt + self.params["reserve_extra_tokens"])
        return math.ceil(tokens / BLOCK_TOKENS)

    def set_blocks(self, tenant, request, blocks):
        if self.page_model:
            pool = tenant.pools[request.rank]
            if blocks < request.held_blocks:
                pool.release(request)
                pool.alloc(request, blocks)
            elif blocks > request.held_blocks:
                # decode growth of a running request is a one-block allocation (packed into the fullest page)
                pool.alloc(request, blocks - request.held_blocks,
                           packed=request.held_blocks > 0 and blocks - request.held_blocks == 1)
        tenant.blocks[request.rank] += blocks - request.held_blocks
        request.held_blocks = blocks
        if tenant.blocks[request.rank] == 0:
            tenant.high_water[request.rank] = 0

    def extra_pages(self, tenant, rank, blocks):
        if self.page_model:
            return math.ceil(max(0, blocks - tenant.pools[rank].slots) / PAGE_BLOCKS)
        return self.tenant_pages(tenant, rank, tenant.blocks[rank] + blocks) - self.tenant_pages(tenant, rank)

    def fits(self, request, rank, headroom=0.0):
        if self.mode == "full":
            return (self.used_pages[rank] + request.pages) * PAGE_GIB <= self.budgets[rank] + 1e-9
        pages = self.extra_pages(self.tenants[request.model], rank, self.reservation_blocks(request))
        return (self.pages_on(rank) + pages) * PAGE_GIB + headroom <= self.budgets[rank] + 1e-9

    def grow(self, tenant, request):
        """Allocate what the current length needs; retract on shortage. False if the request itself was retracted."""
        need = self.target_blocks(request)
        if need <= request.held_blocks:
            return True
        if self.page_model:
            # holes in mapped pages are used first; only a new page can hit the GPU limit, and vLLM then preempts
            # its own newest request on that rank, whose freed blocks become holes for the requester
            rank = request.rank
            while self.pages_on(rank) + self.extra_pages(tenant, rank, need - request.held_blocks) \
                    > self.capacity_pages[rank]:
                candidates = [item for item in list(tenant.active.values()) + list(tenant.joining.values())
                              if item.rank == rank]
                if not candidates:
                    break
                victim = max(candidates, key=lambda item: item.order)
                self.retract(tenant, victim)
                if victim is request:
                    return False
            self.set_blocks(tenant, request, need)
            return True
        self.set_blocks(tenant, request, need)
        ok = self.make_room(tenant, request.rank, request)
        self.commit_pages(tenant, request.rank)
        return ok

    def make_room(self, tenant, rank, requester):
        while self.pages_on(rank) > self.capacity_pages[rank]:
            candidates = [request for request in list(tenant.active.values()) + list(tenant.joining.values())
                          if request.rank == rank]
            if not candidates:
                return True
            victim = max(candidates, key=lambda request: request.order)
            self.retract(tenant, victim)
            if victim is requester:
                return False
        return True

    def retract(self, tenant, request):
        rank = request.rank
        self.set_blocks(tenant, request, 0)
        if request.id in tenant.active:
            del tenant.active[request.id]
            tenant.count[rank] -= 1
            tenant.context[rank] -= request.ctx
        else:
            tenant.joining.pop(request.id, None)
        request.ctx = 0.0
        request.preemptions += 1
        self.preemptions[tenant.index] += 1
        if request.chunked and request.start_step is None:
            done = request.prefill_chunks - request.chunks_left
            self.recompute_tokens[tenant.index] += min(request.prompt, done * CHUNK_TOKENS)
            request.chunks_left = request.prefill_chunks
        else:
            self.recompute_tokens[tenant.index] += request.prompt + request.generated
            request.recompute_left = request.prompt + request.generated
        tenant.waiting[rank].insert(0, request)

    def admit_waiting(self, tenant):
        """vLLM's own scheduling of its waiting queue: start the head once its current sequence fits (full ISL)."""
        for rank in range(self.num_ranks):
            queue = tenant.waiting[rank]
            while queue:
                request = queue[0]
                need = math.ceil((request.prompt + request.generated) / BLOCK_TOKENS)
                if self.pages_on(rank) + self.extra_pages(tenant, rank, need) > self.capacity_pages[rank]:
                    break
                queue.pop(0)
                self.order += 1
                request.order = self.order
                if request.recompute_left > 0:
                    self.set_blocks(tenant, request, need)
                else:
                    if request.chunked and request.start_step is None and not request.prefill_chunks:
                        self.init_code(request)
                    self.set_blocks(tenant, request, self.target_blocks(request))
                self.commit_pages(tenant, rank)
                tenant.joining[request.id] = request

    def kick(self):
        for tenant in self.tenants:
            if tenant.has_waiting() and not tenant.busy and not tenant.paused and not tenant.held:
                self.start_step(tenant)

    def observed(self):
        observed = []
        for tenant in self.tenants:
            ranks = [{"requests": [], "pages": [{"owners": [], "pinned": True, "memory_gib": PAGE_GIB}],
                      "mapped_bytes": PAGE_BYTES, "page_bytes": PAGE_BYTES, "block_bytes": BLOCK_BYTES,
                      "page_block_capacity": PAGE_BLOCKS, "uniform_page_geometry": True, "block_size": BLOCK_TOKENS,
                      "mapped_free_block_slots": 0, "free_blocks": 10**7} for _ in range(self.num_ranks)]
            waiting = [request for queue in tenant.waiting for request in queue]
            for request in list(tenant.active.values()) + list(tenant.joining.values()) + waiting:
                rank = ranks[request.rank]
                decoding = (request.start_step is not None and request.recompute_left == 0
                            and request.id in tenant.active)
                if self.mode == "full":
                    decoding = request.start_step is not None
                blocks = request.blocks if self.mode == "full" else request.held_blocks
                rank["requests"].append({"id": request.id, "computed_tokens": request.prompt + request.generated,
                                         "prompt_tokens": request.prompt, "output_tokens": request.generated,
                                         "num_blocks": blocks, "movable": decoding})
                if self.mode == "full":
                    rank["pages"].append({"owners": [request.id], "pinned": False,
                                          "memory_gib": request.pages * PAGE_GIB})
                    rank["mapped_bytes"] += request.pages * PAGE_BYTES
                elif blocks and not self.shared_pages(tenant) and not self.page_model:
                    rank["pages"].append({"owners": [request.id], "pinned": False,
                                          "memory_gib": blocks / PAGE_BLOCKS * PAGE_GIB})
            if self.mode != "full":
                for index, rank in enumerate(ranks):
                    pages = self.tenant_pages(tenant, index)
                    rank["mapped_bytes"] = (1 + pages) * PAGE_BYTES
                    rank["mapped_free_block_slots"] = pages * PAGE_BLOCKS - tenant.blocks[index]
                    # kvcached get_num_free_blocks() = available_size(): this pool's holes plus physically free pages
                    rank["free_blocks"] = (rank["mapped_free_block_slots"]
                                           + max(0, self.capacity_pages[index] - self.pages_on(index)) * PAGE_BLOCKS)
                    if self.page_model:
                        pool = tenant.pools[index]
                        rank["pages"].extend({"owners": sorted(pool.owners[page]), "pinned": False,
                                              "memory_gib": PAGE_GIB} for page in sorted(pool.free))
                    elif self.shared_pages(tenant) and pages:
                        owners = sorted(request["id"] for request in rank["requests"] if request["num_blocks"])
                        rank["pages"].extend({"owners": owners, "pinned": False, "memory_gib": PAGE_GIB}
                                             for _ in range(pages))
            observed.append(ranks)
        return observed

    def all_decoding(self, models=(0, 1)):
        return all(request.start_step is not None and request.chunks_left == 0
                   for model in models for tenant in (self.tenants[model],)
                   for request in list(tenant.active.values()) + list(tenant.joining.values()))

    # ---------- headroom (shared formula with the live runtime) ----------
    def growth_gib(self, arrival_prompt=None, floor=True):
        if self.mode == "full" or self.spec.get("no_headroom"):
            return 0.0
        conv = self.tenants[1]
        counts = [conv.count[rank] + sum(1 for request in conv.joining.values() if request.rank == rank)
                  for rank in range(self.num_ranks)]
        tokens = conv_growth_tokens(self.belief, self.planner_params, counts, list(conv.context), arrival_prompt,
                                    bool(self.spec.get("isolate")), self.spec.get("slice"), floor)
        return tokens * BLOCK_GIB / BLOCK_TOKENS

    def slack_gib(self):
        if self.mode == "full" or self.spec.get("no_headroom"):
            return 0.0
        return self.planner_params["slack_pages"] * PAGE_GIB

    def conv_on(self, rank):
        conv = self.tenants[1]
        return (conv.count[rank] + sum(1 for request in conv.joining.values() if request.rank == rank)
                + len(conv.waiting[rank]))

    def code_headroom(self, rank, arrival_prompt):
        if self.mode == "full" or self.spec.get("no_headroom"):
            return 0.0
        return self.conv_on(rank) * self.growth_gib(arrival_prompt, self.target_floor) + self.slack_gib()

    # ---------- general (type-agnostic) algorithm, Paper/design.md §5 ----------
    def exclusive(self):
        """Tenant that runs while the others are held: the episode's (general) or the Code tenant (legacy TM)."""
        if self.general:
            return self.episode["x"] if self.episode else None
        return 0 if self.spec.get("isolate") else None

    def residents(self, tenant):
        return (list(tenant.active.values()) + list(tenant.joining.values())
                + [request for queue in tenant.waiting for request in queue])

    def layout_of(self, tenant):
        counts = [0] * self.num_ranks
        for request in self.residents(tenant):
            counts[request.rank] += 1
        return counts, list(tenant.context)

    def computed_tokens(self, request):
        if request.start_step is not None:
            return request.prompt + request.generated
        if request.chunked:
            return min(request.prompt, (request.prefill_chunks - request.chunks_left) * CHUNK_TOKENS)
        return 0

    def trigger_left(self, request):
        """Seconds until a trigger no longer needs exclusivity: its prefill alone, then the grace after its first
        token."""
        if request.completed_at is not None:
            return 0.0
        grace = self.planner_params["tm_grace_s"]
        if request.first_token_at is not None:
            return max(0.0, grace - (self.now - request.first_token_at))
        remaining = request.prompt - self.computed_tokens(request)
        return remaining * self.belief.prefill_s_per_token(request.model) + grace

    def episode_scope(self, arrival=None, trigger=False):
        """(held tenants, seconds left) of the episode in force, or of the one admitting `arrival` starts."""
        x = arrival.model if trigger else self.exclusive() if self.general else None
        if x is None:
            return set(), 0.0
        left = ([self.trigger_left(request) for request in self.episode["triggers"].values()]
                if self.episode and self.episode["x"] == x else [])
        if trigger:
            left.append(self.trigger_left(arrival))
        return {tenant.index for tenant in self.tenants if tenant.index != x}, max([0.0] + left)

    def general_growth(self, arrival=None, trigger=False):
        """GROWTH (§5.1): GiB each request (and the arrival) may allocate beyond its reservation in the window."""
        if self.mode == "full" or self.spec.get("no_headroom"):
            return {}, 0.0
        held, left = self.episode_scope(arrival, trigger)
        rates = {tenant.index: solo_rate(self.belief, tenant.index, *self.layout_of(tenant)) for tenant in self.tenants}
        unit, extra = BLOCK_GIB / BLOCK_TOKENS, self.params["reserve_extra_tokens"]

        def gib(request, computed):
            return unit * growth_tokens(rates[request.model], request.prompt, computed, request.prompt + extra,
                                        self.belief.prefill_s_per_token(request.model),
                                        run_seconds(self.planner_params, request.model in held, left,
                                                    self.spec.get("slice")))

        result = {request.id: gib(request, self.computed_tokens(request))
                  for tenant in self.tenants for request in self.residents(tenant)}
        return result, gib(arrival, 0) if arrival is not None else 0.0

    def general_headroom(self, arrival=None, trigger=False):
        result, own = self.general_growth(arrival, trigger)
        totals = [self.slack_gib() + own] * self.num_ranks
        for tenant in self.tenants:
            for request in self.residents(tenant):
                totals[request.rank] += result.get(request.id, 0.0)
        return totals, own

    def episode_gain(self, request):
        others = [(tenant.index, *self.layout_of(tenant)) for tenant in self.tenants if tenant.index != request.model]
        remaining = request.prompt - self.computed_tokens(request)
        return tm_gain(self.belief, self.planner_params, request.model, remaining, others)

    def episode_safe(self, x, left):
        """The exclusive tenant's own growth during the episode fits each rank's free KV (held tenants free none)."""
        tenant = self.tenants[x]
        rate = solo_rate(self.belief, x, *self.layout_of(tenant))
        grow = [0.0] * self.num_ranks
        for request in self.residents(tenant):
            grow[request.rank] += growth_tokens(rate, request.prompt, self.computed_tokens(request),
                                                request.prompt + self.params["reserve_extra_tokens"],
                                                self.belief.prefill_s_per_token(x), left)
        return all(self.pages_on(rank) * PAGE_GIB + grow[rank] * BLOCK_GIB / BLOCK_TOKENS
                   <= self.budgets[rank] + 1e-9 for rank in range(self.num_ranks))

    def starts_episode(self, request):
        if not self.spec.get("isolate") or (self.episode and self.episode["x"] != request.model):
            return False
        return self.episode_gain(request) > 0 and self.episode_safe(request.model, self.trigger_left(request))

    def start_episode(self, request):
        if self.episode is None:
            self.episode = {"x": request.model, "triggers": {}, "start": self.now}
        self.episode["triggers"][request.id] = request
        for tenant in self.tenants:
            if tenant.index != request.model:
                self.hold(tenant)

    def on_episode_check(self, _):
        episode = self.episode
        if episode is None:
            return
        grace = self.planner_params["tm_grace_s"]
        if all(request.completed_at is not None
               or (request.first_token_at is not None and self.now - request.first_token_at >= grace - 1e-9)
               for request in episode["triggers"].values()):
            self.episodes.append({"model": episode["x"], "triggers": sorted(episode["triggers"]),
                                  "start": episode["start"], "end": self.now})
            self.episode, self.code_run_since, self.slice_armed = None, None, False
            self.tenants[episode["x"]].sliced = False
            for tenant in self.tenants:
                if tenant.index != episode["x"]:
                    self.release(tenant)
            self.start_step(self.tenants[episode["x"]])  # it may have been paused for a slice
            self.maybe_start_episode()
            self.dispatch()
            self.kick()

    def prefilling_requests(self):
        return [request for tenant in self.tenants for request in self.residents(tenant)
                if request.chunked and request.start_step is None]

    def maybe_start_episode(self):
        """A request still prefilling (e.g. admitted while another tenant was exclusive) starts an episode once
        holding the others gains and is safe."""
        if not (self.general and self.spec.get("isolate")) or self.episode is not None:
            return
        for request in sorted(self.prefilling_requests(), key=lambda item: item.arrival):
            if self.episode_gain(request) > 0 and self.episode_safe(request.model, self.trigger_left(request)):
                self.start_episode(request)
                return

    def long_prefill_in_flight(self):
        return any(self.episode_gain(request) > 0 for request in self.prefilling_requests())

    def risk_params(self):
        history = self.planner_params["history_s"]
        needs = [need for at, need in self.arrival_needs if at >= self.now - history]
        return {"arrival_needs": needs,
                "arrival_window_s": min(self.now, history) + self.planner_params["rate_prior_s"]}

    # ---------- arrivals and admission ----------
    def on_arrival(self, request):
        if request.model == 0:
            self.code_arrivals += 1
        if self.native:
            tenant = self.tenants[request.model]
            order = [(tenant.next_rank + offset) % self.num_ranks for offset in range(self.num_ranks)]
            joining = [sum(1 for item in tenant.joining.values() if item.rank == rank) for rank in range(self.num_ranks)]
            rank = min(order, key=lambda r: len(tenant.waiting[r]) * 4 + tenant.count[r] + joining[r])
            tenant.next_rank = (tenant.next_rank + 1) % self.num_ranks
            request.rank = request.rank_initial = rank
            request.reserved_at, request.path = self.now, "native"
            tenant.waiting[rank].append(request)
            self.kick()
            return
        if self.general:
            self.arrival_needs.append((self.now, self.reservation_blocks(request) * BLOCK_GIB))
        self.queues[request.model].append(request)
        self.dispatch()

    def init_code(self, request):
        request.chunks_left = request.prefill_chunks = math.ceil(request.prompt / CHUNK_TOKENS)
        rng = random.Random(f"{self.params['noise_seed']}:{request.id}")
        interference, prefill = rng.gauss(1.0, 1.0), rng.gauss(1.0, 1.0)
        request.factor = max(0.2, 1.0 + self.params["interference_cv"] * (interference - 1.0))
        request.prefill_factor = max(0.5, 1.0 + self.params["prefill_cv"] * (prefill - 1.0))

    def admit(self, request, rank, path, trigger=False):
        request.rank = request.rank_initial = rank
        request.reserved_at, request.path = self.now, path
        tenant = self.tenants[request.model]
        if self.mode == "full":
            self.used_pages[rank] += request.pages
        else:
            self.order += 1
            request.order = self.order
            self.set_blocks(tenant, request, self.reservation_blocks(request))
            self.commit_pages(tenant, rank)
        if request.chunked:
            self.init_code(request)
        tenant.joining[request.id] = request
        if self.fairness is not None:
            self.fairness.reserved(request.id, self.now, self.now)
        if self.general:
            if trigger:
                self.start_episode(request)
        elif self.spec.get("isolate") and request.model == 0:
            self.hold_conv()
        if not tenant.busy and not tenant.paused:
            self.start_step(tenant)

    def hold(self, tenant):
        if tenant.held:
            return
        if tenant.busy:
            tenant.hold_pending = True
        else:
            self.mark_held(tenant)

    def hold_conv(self):
        self.hold(self.tenants[1])

    def mark_held(self, conv):
        conv.held, conv.hold_pending = True, False
        self.hold_intervals.append([self.now, None])

    def release(self, tenant):
        if tenant.held and self.hold_intervals and self.hold_intervals[-1][1] is None:
            self.hold_intervals[-1][1] = self.now
        tenant.held = tenant.hold_pending = False
        if not tenant.paused:
            self.start_step(tenant)

    def release_conv(self):
        self.release(self.tenants[1])

    def prefilling(self, index):
        tenant = self.tenants[index]
        return any(request.chunks_left > 0 for request in list(tenant.active.values()) + list(tenant.joining.values()))

    def code_prefilling(self):
        return self.prefilling(0)

    def resume_slice(self, index):
        tenant = self.tenants[index]
        tenant.sliced = False
        self.code_run_since = self.now
        self.start_step(tenant)

    def resume_code_slice(self):
        self.resume_slice(0)

    def on_slice_release(self, _):
        x = self.exclusive()
        if x is None or not self.tenants[x].sliced:
            return
        for tenant in self.tenants:
            if tenant.index != x:
                self.release(tenant)
        self.push(self.now + self.spec["slice"][1], "slice_end")

    def on_slice_end(self, _):
        x = self.exclusive()
        if x is None or not self.tenants[x].sliced:
            return
        others = [tenant for tenant in self.tenants if tenant.index != x]
        for tenant in others:
            self.hold(tenant)
        if all(tenant.held for tenant in others):
            self.resume_slice(x)

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
                trigger = False
                if self.general:
                    trigger = self.starts_episode(request)
                    headroom, own = self.general_headroom(request, trigger)
                    candidates = [rank for rank in range(self.num_ranks) if self.fits(request, rank, headroom[rank])]
                else:
                    candidates = [rank for rank in range(self.num_ranks)
                                  if self.fits(request, rank, self.code_headroom(rank, request.prompt)
                                               if request.model == 0 else 0.0)]
                if candidates:
                    counts = self.tenants[request.model].count
                    joining = [0] * self.num_ranks
                    for item in self.tenants[request.model].joining.values():
                        joining[item.rank] += 1
                    rank = min(candidates, key=lambda r: (counts[r] + joining[r], self.pages_on(r), r))
                    self.queues[request.model].pop(0)
                    self.admit(request, rank, "direct", trigger)
                    progressed = True
                    continue
                if self.spec["admission"] is None:
                    continue
                if self.general:
                    if self.fragmented(request, headroom, own) and self.try_migration_general(request):
                        return
                elif self.try_migration(request):
                    return
        if self.general:
            self.maybe_start_episode()

    def fragmented(self, request, headroom, own):
        """Blocked by fragmentation, not capacity: the free KV beyond every rank's residents' headroom holds it."""
        need = self.reservation_blocks(request) * BLOCK_GIB
        spare = sum(max(0.0, budget - used - (room - own))
                    for budget, used, room in zip(self.budgets, self.used_gib(), headroom))
        return spare >= need

    def try_migration_general(self, request):
        """Algorithm 2 (§5.3) for any blocked request: any tenant's decoding requests may move."""
        if self.now < self.plan_not_before.get(request.id, 0.0):
            return False
        self.plan_not_before[request.id] = self.now + self.params["plan_retry_s"]
        arrival = {"model": request.model, "tokens": request.prompt + self.params["reserve_extra_tokens"],
                   "prompt_tokens": request.prompt}
        seed = int.from_bytes(hashlib.sha256(f"{self.params['seed']}:{request.id}".encode()).digest()[:8], "big")
        context = {}

        def plan_fn():
            context["trigger"] = self.starts_episode(request)
            growth, own = self.general_growth(request, context["trigger"])
            state = build_state(self.observed(), self.budgets, slack_gib=self.slack_gib(), request_growth=growth,
                                arrival_growth_gib=own)
            return plan_admission(state, arrival, self.spec["admission"], self.belief, self.planner_params, seed)

        preplan = plan_fn()
        if preplan.get("target") is None:
            return False
        self.start_transaction("admission", plan_fn, request, sorted({move["model"] for move in preplan["moves"]}))
        if self.transaction is not None:
            self.transaction["context"] = context
        return True

    def try_migration(self, request):
        if self.now < self.plan_not_before.get(request.id, 0.0):
            return False
        self.plan_not_before[request.id] = self.now + self.params["plan_retry_s"]
        need = self.reservation_blocks(request) * BLOCK_GIB if self.mode != "full" else request.pages * PAGE_GIB
        free = sum(self.budgets) - sum(self.used_gib())
        if need < self.planner_params["migrate_min_gib"] or free < need:
            return False
        tokens = (request.prompt + request.output if self.mode == "full"
                  else request.prompt + self.params["reserve_extra_tokens"])
        arrival = {"model": request.model, "tokens": tokens, "prompt_tokens": request.prompt}
        seed = int.from_bytes(hashlib.sha256(f"{self.params['seed']}:{request.id}".encode()).digest()[:8], "big")

        def plan_fn():
            state = build_state(self.observed(), self.budgets, (0.0, self.growth_gib(request.prompt)),
                                self.slack_gib(), (0.0, self.growth_gib(request.prompt, self.target_floor)))
            return plan_admission(state, arrival, self.spec["admission"], self.belief, self.planner_params, seed)

        preplan = plan_fn()
        if preplan.get("target") is None:
            return False
        pause_models = sorted({move["model"] for move in preplan["moves"]})
        if self.mode == "full" and not self.all_decoding(pause_models):
            return False
        self.start_transaction("admission", plan_fn, request, pause_models)
        return True

    # ---------- stepping ----------
    def other_phase(self, tenant):
        other = self.tenants[1 - tenant.index]
        if other.held and not other.busy:
            return False
        if not other.has_work():
            # vLLM DP: a request queued inside the engine (KV-blocked arrival or retracted request) keeps the whole DP
            # group stepping dummy batches (EP collectives every step), which interferes like a decode (V8 native)
            if other.has_waiting() and not other.paused and self.params["dp_dummy_interference"]:
                return "decode"
            return False
        if not other.busy and other.sliced:
            return False  # the exclusive tenant is paused for a slice of the held ones
        if (other.index == self.exclusive() and not other.busy and self.spec.get("hold_first")
                and tenant.hold_pending):
            return False  # the exclusive tenant waits for this tenant's pause boundary and is not running yet
        if any(request.chunks_left > 0 for request in list(other.active.values()) + list(other.joining.values())):
            return "prefill"
        return "decode"

    def prefill_factor(self, attribute="factor", index=0):
        """Mean noise factor of tenant `index`'s requests currently in prefill (1.0 if none or noise disabled)."""
        tenant = self.tenants[index]
        factors = [getattr(request, attribute) for request in list(tenant.active.values())
                   + list(tenant.joining.values()) if request.chunks_left > 0]
        return sum(factors) / len(factors) if factors else 1.0

    def recompute_s(self, tenant):
        """Extra step time for retracted requests re-prefilling one chunk per rank (DP ranks step together)."""
        chunks = [0] * self.num_ranks
        for request in tenant.in_step:
            if request.recompute_left > 0:
                chunks[request.rank] = max(chunks[request.rank], min(CHUNK_TOKENS, request.recompute_left))
        return max(chunks) * self.params["recompute_s_per_token"]

    def step_duration(self, tenant):
        phase = self.other_phase(tenant)
        other = self.tenants[1 - tenant.index]
        if any(request.chunks_left > 0 for request in tenant.in_step):
            prompt = max(request.prompt for request in tenant.in_step if request.chunks_left > 0)
            per_100k = self.prefill["solo_s_per_100k"]
            if phase:
                value = self.prefill["with_other_s_per_100k"]
                if isinstance(value, dict):
                    bucket = capture_bucket(max(other.count) or 1)
                    sizes = sorted(int(size) for size in value)
                    chosen = next((size for size in sizes if bucket <= size), sizes[-1])
                    value = value[str(chosen)] if str(chosen) in value else value[chosen]
                per_100k = value * self.prefill_factor("prefill_factor", tenant.index)
            return per_100k * prompt / 100000 / math.ceil(prompt / CHUNK_TOKENS)
        extra = self.recompute_s(tenant) if self.mode != "full" else 0.0
        if phase == "prefill":
            own = self.truth.decode_step(tenant.index, tenant.count, tenant.context, False)
            return own + self.prefill_factor("factor", other.index) * (
                self.truth.decode_step(tenant.index, tenant.count, tenant.context, phase) - own) + extra
        return self.truth.decode_step(tenant.index, tenant.count, tenant.context, phase) + extra

    def slice_due(self, slice_cfg):
        """Periodic slices: (S, R). TTFT-triggered: (T_wait, R, S_max) -- slice when a held request has not started
        and waited >= T_wait, or the hold lasted >= S_max."""
        held_for = self.now - self.code_run_since
        if len(slice_cfg) == 2:
            return held_for >= slice_cfg[0]
        x = self.exclusive()
        waiting = [self.now - request.arrival for tenant in self.tenants if tenant.index != x
                   for request in tenant.joining.values()]
        return held_for >= slice_cfg[2] or (waiting and max(waiting) >= slice_cfg[0])

    def start_step(self, tenant):
        if tenant.busy or tenant.paused or tenant.held:
            return
        if tenant.has_waiting():
            self.admit_waiting(tenant)
        if not tenant.has_work():
            return
        x = self.exclusive()
        if x is None or tenant.index != x:
            pass
        elif self.spec.get("hold_first") and any(other.hold_pending for other in self.tenants if other is not tenant):
            return  # the exclusive tenant waits until the held tenants reached their pause boundary
        else:
            if tenant.sliced:
                return
            slice_cfg = self.spec.get("slice")
            other = self.tenants[1 - x]
            if slice_cfg and self.prefilling(x) and other.held and other.has_work():
                if self.code_run_since is None:
                    self.code_run_since = self.now
                elif self.slice_armed:
                    # live: the slicer's Code summary waits one step, then keep-pause waits for the next step
                    tenant.sliced, self.slice_armed = True, False
                    self.push(self.now + self.params["slice_consensus_s"], "slice_release")
                    return
                elif self.slice_due(slice_cfg):
                    self.slice_armed = True
            elif not self.prefilling(x):
                self.code_run_since, self.slice_armed = None, False
        for request in tenant.joining.values():
            tenant.active[request.id] = request
            tenant.count[request.rank] += 1
            if not request.chunked or request.start_step is not None:
                request.ctx = request.prompt + request.generated
            else:
                request.ctx = 0.0
            tenant.context[request.rank] += request.ctx
        tenant.joining.clear()
        tenant.in_step = list(tenant.active.values())
        tenant.busy = True
        self.push(self.now + self.step_duration(tenant), "step_end", tenant.index)

    def on_step_end(self, index):
        tenant = self.tenants[index]
        tenant.busy = False
        tenant.step_index += 1
        step = tenant.step_index
        prefilled_ranks, recomputed_ranks, finished = set(), set(), []
        for request in tenant.in_step:
            if request.id not in tenant.active:
                continue
            if request.recompute_left > 0:
                if request.rank in recomputed_ranks:
                    continue
                recomputed_ranks.add(request.rank)
                request.recompute_left = max(0, request.recompute_left - CHUNK_TOKENS)
                continue
            if request.start_step is None:
                if request.chunks_left > 0:
                    if request.rank in prefilled_ranks:
                        continue
                    prefilled_ranks.add(request.rank)
                    request.chunks_left -= 1
                    if request.chunks_left:
                        if self.mode == "native":
                            self.grow(tenant, request)  # the next chunk's blocks
                        continue
                    tenant.context[request.rank] += request.prompt
                    request.ctx += request.prompt
                request.start_step = step
                if request.first_token_at is None:
                    request.first_token_at = self.now
                    if self.episode is not None and request.id in self.episode["triggers"]:
                        self.push(self.now + self.planner_params["tm_grace_s"], "episode_check")
            request.generated += 1
            tenant.context[request.rank] += 1
            request.ctx += 1
            if self.mode != "full" and not self.grow(tenant, request):
                continue
            if request.generated >= request.output:
                finished.append(request)
        for request in sorted(finished, key=lambda item: item.id):
            if request.id in tenant.active:
                self.complete(tenant, request)
        if tenant.pause_pending and (step % self.params["pause_boundary"] == 0 or not tenant.has_work()):
            self.pause_reached(tenant)
        elif tenant.hold_pending and (step % self.params["pause_boundary"] == 0 or not tenant.has_work()):
            self.mark_held(tenant)
            x = self.exclusive()
            if x is None:
                self.release(tenant)  # the episode ended before this tenant reached its boundary
            elif self.tenants[x].sliced:
                self.resume_slice(x)
            else:
                self.start_step(self.tenants[x])
        elif not tenant.paused:
            self.start_step(tenant)

    def complete(self, tenant, request):
        del tenant.active[request.id]
        tenant.count[request.rank] -= 1
        tenant.context[request.rank] -= request.ctx
        if self.mode == "full":
            self.used_pages[request.rank] -= request.pages
        else:
            self.set_blocks(tenant, request, 0)
        request.completed_at = self.now
        self.completed.append(request)
        if self.general:
            if self.episode is not None and request.id in self.episode["triggers"]:
                self.push(self.now, "episode_check")
            if self.spec["return"]:
                self.push(self.now + 0.5, "return_check", "release")
        else:
            if self.spec.get("isolate") and request.model == 0 and not tenant.has_work():
                self.release_conv()
            if self.spec["return"] and request.model == 0:
                self.push(self.now + 0.5, "return_check", "release")
        self.dispatch()
        self.kick()

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
        prefilling = any(request.chunks_left > 0 for tenant in self.tenants for request in tenant.active.values())
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
            if request.id in tenant.active:
                tenant.count[move["source"]] -= 1
                tenant.context[move["source"]] -= request.ctx
                tenant.count[move["destination"]] += 1
                tenant.context[move["destination"]] += request.ctx
            if self.mode == "full":
                self.used_pages[move["source"]] -= request.pages
                self.used_pages[move["destination"]] += request.pages
            else:
                if self.page_model:
                    # the destination prepares all blocks in one run; the source frees them after commit
                    tenant.pools[move["source"]].release(request)
                    tenant.pools[move["destination"]].alloc(request, request.held_blocks)
                tenant.blocks[move["source"]] -= request.held_blocks
                if tenant.blocks[move["source"]] == 0:
                    tenant.high_water[move["source"]] = 0
                tenant.blocks[move["destination"]] += request.held_blocks
                self.commit_pages(tenant, move["destination"])
            request.rank = move["destination"]
            request.moves += 1
            self.migrations.append({"request_id": request.id, "model": move["model"], "source": move["source"],
                                    "destination": move["destination"], "at": self.now, "kind": transaction["kind"]})
        paused_request_s = sum((len(self.tenants[index].active) + len(self.tenants[index].joining))
                               * (self.now - self.tenants[index].paused_at) for index in transaction["order"])
        admitted = False
        arrival = transaction["arrival"]
        trigger = bool(transaction.get("context", {}).get("trigger"))
        if arrival is not None and plan.get("target") is not None:
            if self.general:
                headroom = self.general_headroom(arrival, trigger)[0][plan["target"]]
            else:
                headroom = self.code_headroom(plan["target"], arrival.prompt) if arrival.model == 0 else 0.0
            if self.fits(arrival, plan["target"], headroom):
                self.queues[arrival.model].remove(arrival)
                self.admit(arrival, plan["target"], "migration", trigger)
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
        self.kick()

    # ---------- periodic work ----------
    def on_return_check(self, reason):
        if reason != "release":
            self.push(self.now + self.params["return_period_s"], "return_check")
        if self.general:
            self.general_return()
            return
        if (self.transaction is not None or any(self.queues)
                or self.now - self.last_return < self.params["return_cooldown_s"]
                or (self.mode == "full" and not self.all_decoding((1,)))):
            return
        if self.mode != "full" and (self.tenants[0].has_work() or self.tenants[0].has_waiting()):
            return  # no rebalancing while a Code is resident: its prefill slows peer copies and the gain is nil
        counts = self.tenants[1].count
        if max(counts) - min(counts) < 2:
            return
        rate = (self.code_arrivals + 1) / (max(1.0, self.now) + 1 / self.planner_params["code_rate"])

        def plan_fn():
            state = build_state(self.observed(), self.budgets, (0.0, self.growth_gib()), self.slack_gib())
            return plan_return(state, self.belief, dict(self.planner_params, code_rate=rate))

        if plan_fn()["moves"]:
            self.start_transaction("return", plan_fn, pause_models=(1,))

    def general_return(self):
        """Algorithm 3 (§5.4): level every tenant; never during an episode or a long prefill."""
        if (self.transaction is not None or any(self.queues)
                or self.now - self.last_return < self.params["return_cooldown_s"]
                or self.episode is not None or self.long_prefill_in_flight()):
            return
        if not any(max(tenant.count) - min(tenant.count) >= 2 for tenant in self.tenants):
            return
        return_params = dict(self.planner_params, return_tenants=(0, 1), **self.risk_params())

        def plan_fn():
            growth, _ = self.general_growth()
            state = build_state(self.observed(), self.budgets, slack_gib=self.slack_gib(), request_growth=growth)
            return plan_return(state, self.belief, return_params)

        moves = plan_fn()["moves"]
        if moves:
            self.start_transaction("return", plan_fn, pause_models=sorted({move["model"] for move in moves}))

    def on_population(self, _):
        self.population.append({"t": self.now, "counts": [list(tenant.count) for tenant in self.tenants],
                                "used_gib": self.used_gib(), "queued": [len(queue) for queue in self.queues],
                                "waiting": [[len(queue) for queue in tenant.waiting] for tenant in self.tenants],
                                "preemptions": list(self.preemptions)})
        self.push(self.now + self.params["population_period_s"], "population")


def simulate(trace, policy, **kwargs):
    return Simulator(trace, policy, **kwargs).run()
