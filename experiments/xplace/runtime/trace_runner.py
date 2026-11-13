import asyncio
import gc
import hashlib
import json
import math
import queue
import threading
import time
from pathlib import Path

from controller import GIB, used_gib
from concurrency import InFlightLimiter
from cost_model import StepModel
from engine import save, stream
from fairness import FairAdmissionGate
from planner import (DEFAULT_PARAMS, RELIEF_FLOOR_ON_TARGET, build_state, conv_growth_tokens, growth_tokens,
                     plan_admission, plan_return, run_seconds, solo_rate, tm_gain)

POLICIES = {
    "no_move": {"tag": "nm", "admission": None, "return": False, "fair": False},
    "no_move_fair": {"tag": "nf", "admission": None, "return": False, "fair": True},
    "mig_random": {"tag": "mr", "admission": "mig_random", "return": False, "fair": False},
    "mig_bestfit": {"tag": "mb", "admission": "mig_bestfit", "return": False, "fair": False},
    "ca_admit": {"tag": "ca", "admission": "ca", "return": False, "fair": False},
    "ca_return": {"tag": "cr", "admission": "mig_random", "return": True, "fair": False},
    "ca_full": {"tag": "cf", "admission": "ca", "return": True, "fair": False},
    # time-multiplexing: Conv is held paused while any Code request is resident
    "no_move_tm": {"tag": "nt", "admission": None, "return": False, "fair": False, "isolate": True},
    "mig_random_tm": {"tag": "mt", "admission": "mig_random", "return": False, "fair": False, "isolate": True},
    "ca_full_tm": {"tag": "ct", "admission": "ca", "return": True, "fair": False, "isolate": True},
    # v2: Conv admission without waiting on the Code tenant (cached Code view); TM holds Conv before admitting Code
    "no_move_v2": {"tag": "n2", "admission": None, "return": False, "fair": False, "fast": True},
    "no_move_fair_v2": {"tag": "f2", "admission": None, "return": False, "fair": True, "fast": True},
    "mig_random_v2": {"tag": "m2", "admission": "mig_random", "return": False, "fair": False, "fast": True},
    "ca_admit_v2": {"tag": "a2", "admission": "ca", "return": False, "fair": False, "fast": True},
    "ca_return_v2": {"tag": "r2", "admission": "mig_random", "return": True, "fair": False, "fast": True},
    "ca_full_v2": {"tag": "c2", "admission": "ca", "return": True, "fair": False, "fast": True},
    "ca_periodic_v2": {"tag": "p2", "admission": None, "return": True, "fair": False, "fast": True},
    "ca_no_imbalance_v2": {"tag": "i2", "admission": "ca", "return": True, "fair": False, "fast": True,
                            "imbalance_cost": False},
    "no_move_tm_v2": {"tag": "u2", "admission": None, "return": False, "fair": False, "isolate": True, "fast": True,
                      "hold_first": True},
    "mig_random_tm_v2": {"tag": "w2", "admission": "mig_random", "return": False, "fair": False, "isolate": True,
                         "fast": True, "hold_first": True},
    "ca_full_tm_v2": {"tag": "t2", "admission": "ca", "return": True, "fair": False, "isolate": True, "fast": True,
                      "hold_first": True},
    # v3: sliced TM. After `slice[0]` s of Conv hold while a Code prefills, pause the Code (vLLM keep-pause) and let
    # Conv run `slice[1]` s, so a Conv request never stalls much longer than one slice.
    "ca_full_tm_s5": {"tag": "s5", "admission": "ca", "return": True, "fair": False, "isolate": True, "fast": True,
                      "hold_first": True, "slice": (5.0, 1.0)},
    "ca_full_tm_s4": {"tag": "s4", "admission": "ca", "return": True, "fair": False, "isolate": True, "fast": True,
                      "hold_first": True, "slice": (4.0, 1.0)},
    # v3.1: SLO-aware slicing -- slice only when a held Conv request waits >= 2 s for its first token, or after 15 s
    "ca_full_tm_slo": {"tag": "sl", "admission": "ca", "return": True, "fair": False, "isolate": True, "fast": True,
                       "hold_first": True, "slice": (2.0, 1.0, 15.0)},
    # ablation (current-length setting): placement without growth headroom
    "ca_full_tm_slo_nh": {"tag": "sn", "admission": "ca", "return": True, "fair": False, "isolate": True, "fast": True,
                          "hold_first": True, "slice": (2.0, 1.0, 15.0), "no_headroom": True},
    # ablations of the final scheme ca_full_tm_slo: each removes or replaces one component, scheduling unchanged
    "no_move_tm_slo": {"tag": "us", "admission": None, "return": False, "fair": False, "isolate": True, "fast": True,
                       "hold_first": True, "slice": (2.0, 1.0, 15.0)},
    "mig_random_tm_slo": {"tag": "ws", "admission": "mig_random", "return": False, "fair": False, "isolate": True,
                          "fast": True, "hold_first": True, "slice": (2.0, 1.0, 15.0)},
    "ca_admit_tm_slo": {"tag": "as", "admission": "ca", "return": False, "fair": False, "isolate": True, "fast": True,
                        "hold_first": True, "slice": (2.0, 1.0, 15.0)},
    "ca_return_tm_slo": {"tag": "rs", "admission": "mig_random", "return": True, "fair": False, "isolate": True,
                         "fast": True, "hold_first": True, "slice": (2.0, 1.0, 15.0)},
    # v1 control plane: Conv admission waits on live Code-tenant RPCs, Code is admitted before Conv is held
    "ca_full_tm_slo_v1": {"tag": "v1", "admission": "ca", "return": True, "fair": False, "isolate": True,
                          "slice": (2.0, 1.0, 15.0)},
    "ca_full_v2_nh": {"tag": "cn", "admission": "ca", "return": True, "fair": False, "fast": True,
                      "no_headroom": True},
    # baselines without growth headroom: stock current-length admission semantics (admit when the request fits now,
    # recompute preemption on shortage) with controller-side queueing and least-loaded rank choice
    "no_move_v2_nh": {"tag": "nh", "admission": None, "return": False, "fair": False, "fast": True,
                      "no_headroom": True},
    "no_move_tm_slo_nh": {"tag": "uh", "admission": None, "return": False, "fair": False, "isolate": True,
                          "fast": True, "hold_first": True, "slice": (2.0, 1.0, 15.0), "no_headroom": True},
    # baseline: kvcached as deployed -- both vLLMs serve concurrently with stock admission (KV grows per step,
    # recompute preemption on shortage) and stock DP load balancing; no controller admission, migration or pausing.
    # main.py trims each GPU's free memory so the shared KV capacity equals the controlled budget.
    "kvcached_native": {"tag": "kn", "admission": None, "return": False, "fair": False, "native": True},
}
# General (type-agnostic) algorithm, Paper/design.md §5: the same switches, but no decision depends on a request's
# tenant -- every admission keeps per-request growth headroom, any request blocked by fragmentation may migrate any
# decoding request, an episode of time-multiplexing starts when a prefill gains from exclusivity, and returns level
# every tenant. `isolate` enables episodes; `slice` slices the held tenants.
for _name, _spec in list(POLICIES.items()):
    if not _spec.get("native"):
        POLICIES[f"{_name}_g"] = dict(_spec, tag=f"{_spec['tag']}g", general=True)
HEAVY_KEYS = ("observations", "plan", "pause_receipts", "transfer_memory")


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def light(report):
    record = {key: value for key, value in report.items() if key not in HEAVY_KEYS}
    plan = report.get("plan") or {}
    record["plan_summary"] = {key: plan.get(key) for key in ("target", "ordering", "objective_s", "performance_cost_s",
                                                            "migration_cost_s", "reason", "accepted", "chosen",
                                                            "candidates_evaluated", "horizon_steps", "diagnostics")}
    record["planned_moves"] = [{key: move[key] for key in ("id", "model", "source", "destination", "blocks")}
                               for move in plan.get("moves", [])]
    receipts = report.get("pause_receipts")
    if receipts:
        record["pause_reach_s"] = [[receipt.get("reached_at", math.nan) - receipt.get("armed_at", math.nan)
                                    for receipt in ranks] for ranks in receipts if ranks]
    if report.get("observations"):
        record["counts"] = [[len(rank["requests"]) for rank in ranks] for ranks in report["observations"]]
    return record


class Journal:
    """JSONL/snapshot writer on a background thread.

    The output directory lives on the cluster's shared filesystem, whose open/replace latency spikes (0.6-3.6 s
    observed) blocked the asyncio control loop when written inline.
    """

    def __init__(self, root):
        self.root, self.handles, self.errors = root, {}, []
        self.queue = queue.SimpleQueue()
        self.thread = threading.Thread(target=self._run, name="xplace-journal", daemon=True)
        self.thread.start()

    def write(self, name, value):
        self.queue.put(("line", name, json.dumps(value)))

    def save(self, path, value):
        self.queue.put(("save", path, value))

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            kind, target, payload = item
            try:
                if kind == "line":
                    handle = self.handles.get(target)
                    if handle is None:
                        handle = self.handles[target] = open(self.root / f"{target}.jsonl", "a")
                    handle.write(payload + "\n")
                    handle.flush()
                else:
                    save(target, payload)
            except Exception as error:  # keep the trial running; surfaced in summary.json
                self.errors.append(f"{type(error).__name__}: {error}")

    def close(self):
        self.queue.put(None)
        self.thread.join()
        for handle in self.handles.values():
            handle.close()


async def wait_empty(controller, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        summaries = await controller.summaries()
        if all(not rank["requests"] for ranks in summaries for rank in ranks):
            return summaries
        await asyncio.sleep(1)
    raise TimeoutError("Engines did not drain")


def conv_imbalanced(summaries, threshold=2, model=1):
    counts = [len(rank["requests"]) for rank in summaries[model]]
    return max(counts) - min(counts) >= threshold


async def run_trace(engines, controller, trace, prompts, policy, root, args, receipt_observer=None):
    """Replay a frozen two-tenant trace under one policy; raw records go to `root`."""
    if receipt_observer is not None and policy != "no_move":
        raise ValueError("Causal calibration currently supports only the shared-queue no_move path")
    spec = POLICIES[policy]
    limiter = InFlightLimiter(getattr(args, "max_inflight", (0, 0)))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    journal = Journal(root)
    idle = await wait_empty(controller)
    native = bool(spec.get("native"))
    kv_mode = "native" if native else getattr(args, "kv_mode", "full")
    reserve_extra = getattr(args, "reserve_extra_tokens", 0) if kv_mode == "prompt" else 0
    # Only the legacy full reservation lets the controller enforce the budget; otherwise KV grows per step under the
    # physical cap (ballast) and a shortage retracts requests inside vLLM.
    controller.enforce_budget = kv_mode == "full"
    await controller.set_mode(kv_mode, reserve_extra)
    preemptions_start = [sum(rank.get("preemptions_total", 0) for rank in ranks) for ranks in idle]
    recompute_start = [sum(rank.get("recompute_tokens_total", 0) for rank in ranks) for ranks in idle]
    observed = {"rank": {}, "preemptions": {}, "max_mapped_gib": [0.0] * controller.num_ranks,
                "capacity_min_gib": [math.inf] * controller.num_ranks}
    controller.events.clear()
    controller.transactions.clear()
    controller.holds.clear()
    controller.fast_admission = bool(spec.get("fast"))
    controller.code_view = None
    controller.view_stats = {"cached": 0, "fetched": 0}
    general = bool(spec.get("general"))
    controller.general = general
    controller.views.clear()
    params = dict(DEFAULT_PARAMS, **args.planner_params)
    params["score_imbalance_cost"] = spec.get("imbalance_cost", True)
    if not params["score_imbalance_cost"] and controller.num_ranks != 2:
        raise ValueError("Strict imbalance-cost ablation requires two attention groups")
    target_floor = RELIEF_FLOOR_ON_TARGET[params["relief_scope"]]
    step_model = StepModel(args.step_params)
    controller.max_decode_rate = max(1.0 / tenant["base_s"] for tenant in step_model.tenants.values())
    controller.min_prefill_s_per_token = min(step_model.prefill_s_per_token(model) for model in (0, 1))
    origin = time.monotonic() + 1.0
    total = len(trace["requests"])
    pending, results, decisions, tasks, errors = [[], []], [], [], [], []
    entries = {}  # trace request_id -> entry (incl. live progress), for censored partial records
    code_arrivals, version = [], {"value": 0}
    resident = {"code": 0}
    slices = []
    lags = {"max_s": 0.0, "stalls": 0}
    heartbeat, stall_stacks, gc_events, gc_started = {"t": time.monotonic()}, [], [], {}
    watchdog_stop = threading.Event()
    isolate = bool(spec.get("isolate"))
    done, failed, return_wakeup = asyncio.Event(), asyncio.Event(), asyncio.Event()
    fairness = FairAdmissionGate(args.fairness_threshold_s) if spec["fair"] else None
    # general policies: the time-multiplexing episode in force (exclusive tenant, trigger entries), the KV needs of
    # recent arrivals (rebalance risk), and the monitor's latest summaries
    episode = {"model": None, "triggers": {}, "started_at": None}
    episodes, arrival_needs, latest = [], [], {"summaries": None}
    slice_cfg = spec.get("slice")
    block_size, block_gib = idle[0][0]["block_size"], idle[0][0]["block_bytes"] / GIB
    save(root / "workload.json", {"policy": policy, "origin": origin, "budgets_gib": controller.budgets,
                                  "policy_switches": {"starvation_migration": spec["admission"] is not None,
                                                      "periodic_balance": bool(spec["return"]),
                                                      "imbalance_cost": params["score_imbalance_cost"]},
                                  "num_ranks": controller.num_ranks, "planner_params": params,
                                  "step_params": args.step_params, "trace_sha256": args.trace_sha256,
                                  "requests": total, "duration_s": trace["duration_s"], "native": native,
                                  "max_inflight": list(limiter.limits),
                                  "inflight_definition": "submitted until completion; central waiting queue excluded",
                                  "kv_mode": kv_mode, "reserve_extra_tokens": reserve_extra})

    def admission_tokens(mark):
        """KV the controller admits a request with: the legacy run uses the replayed output length (an oracle);
        current-length admission uses only the prompt plus the scheduler's fixed reservation extra."""
        if kv_mode == "full":
            return mark["prompt_tokens"] + mark["output_tokens"]
        return mark["prompt_tokens"] + reserve_extra

    def growth_gib(observations, arrival_prompt=None, floor=True):
        """Per-request growth allowance (Code, Conv) in GiB; zero under the legacy full reservation."""
        if kv_mode == "full" or spec.get("no_headroom"):
            return (0.0, 0.0)
        conv = observations[1]
        counts = [len(rank["requests"]) for rank in conv]
        contexts = [sum(request["computed_tokens"] for request in rank["requests"]) for rank in conv]
        tokens = conv_growth_tokens(step_model, params, counts, contexts, arrival_prompt, isolate, spec.get("slice"),
                                    floor)
        per_token = conv[0]["block_bytes"] / conv[0]["block_size"] / GIB
        return (0.0, tokens * per_token)

    def slack_gib(observations):
        if kv_mode == "full" or spec.get("no_headroom"):
            return 0.0
        return params["slack_pages"] * observations[1][0]["page_bytes"] / GIB

    def code_headroom(observations, arrival_prompt):
        """Per-rank headroom for placing a Code: growth of the Conv residents of that rank plus slack."""
        if kv_mode == "full" or spec.get("no_headroom"):
            return [0.0] * controller.num_ranks
        growth = growth_gib(observations, arrival_prompt, target_floor)[1]
        return [len(observations[1][rank]["requests"]) * growth + slack_gib(observations)
                for rank in range(controller.num_ranks)]

    # ---------- general (type-agnostic) algorithm: Paper/design.md §5 ----------
    def layouts(observations):
        return {model: ([len(rank["requests"]) for rank in observations[model]],
                        [sum(request["computed_tokens"] for request in rank["requests"])
                         for rank in observations[model]]) for model in (0, 1)}

    def remaining_prompt(entry, observations):
        prompt = entry["mark"]["prompt_tokens"]
        for rank in (observations[entry["mark"]["model"]] if observations else ()):
            for request in rank["requests"]:
                if request["id"].rsplit("-", 1)[0] == entry["id"]:
                    return max(0, prompt - request["computed_tokens"])
        return prompt

    def trigger_left(entry, observations):
        """Seconds until a trigger no longer needs exclusivity: its prefill alone, then the grace after its first
        token (the output length is unknown, so exclusivity ends a fixed grace after the first token)."""
        if entry.get("done"):
            return 0.0
        first = entry["live"].get("first_token_at")
        if first is not None:
            return max(0.0, params["tm_grace_s"] - (time.monotonic() - first))
        model = entry["mark"]["model"]
        return remaining_prompt(entry, observations) * step_model.prefill_s_per_token(model) + params["tm_grace_s"]

    def episode_scope(observations, entry=None, trigger=False):
        """(held tenants, seconds left) of the episode in force, or of the one admitting `entry` starts/extends."""
        model = entry["mark"]["model"] if trigger else episode["model"]
        if model is None:
            return set(), 0.0
        left = [trigger_left(item, observations) for item in episode["triggers"].values()] if \
            episode["model"] == model else []
        if trigger:
            left.append(trigger_left(entry, None))
        return {other for other in (0, 1) if other != model}, max([0.0] + left)

    def slack_value(observations):
        if spec.get("no_headroom") or kv_mode == "full":
            return 0.0
        return params["slack_pages"] * observations[0][0]["page_bytes"] / GIB

    def growth(observations, entry=None, trigger=False):
        """GROWTH (§5.1): GiB each observed request -- and the arriving entry -- may allocate beyond its reservation
        within the protection window, from decode rates and time only."""
        if spec.get("no_headroom") or kv_mode == "full":
            return {}, 0.0
        held, left = episode_scope(observations, entry, trigger)
        unit = observations[0][0]["block_bytes"] / observations[0][0]["block_size"] / GIB
        shapes = layouts(observations)
        rates = {model: solo_rate(step_model, model, *shapes[model]) for model in (0, 1)}

        def gib(model, prompt, computed, reservation):
            return unit * growth_tokens(rates[model], prompt, computed, reservation,
                                        step_model.prefill_s_per_token(model),
                                        run_seconds(params, model in held, left, slice_cfg))

        result = {request["id"]: gib(model, request["prompt_tokens"], request["computed_tokens"],
                                     request["reservation_tokens"])
                  for model in (0, 1) for rank in observations[model] for request in rank["requests"]}
        own = 0.0
        if entry is not None:
            own = gib(entry["mark"]["model"], entry["mark"]["prompt_tokens"], 0, admission_tokens(entry["mark"]))
        return result, own

    def headroom_ranks(observations, entry=None, trigger=False):
        """Per-rank headroom for placing `entry`: its residents' growth, its own growth and the slack page."""
        result, own = growth(observations, entry, trigger)
        totals = [slack_value(observations) + own] * controller.num_ranks
        for model in (0, 1):
            for rank, observation in enumerate(observations[model]):
                totals[rank] += sum(result.get(request["id"], 0.0) for request in observation["requests"])
        return totals, own

    def episode_safe(observations, model, left):
        """The exclusive tenant's own growth during the episode fits each rank's free KV (held tenants free none)."""
        unit = observations[0][0]["block_bytes"] / observations[0][0]["block_size"] / GIB
        rate = solo_rate(step_model, model, *layouts(observations)[model])
        used = used_gib(observations)
        for rank, observation in enumerate(observations[model]):
            grow = sum(growth_tokens(rate, request["prompt_tokens"], request["computed_tokens"],
                                     request["reservation_tokens"], step_model.prefill_s_per_token(model), left)
                       for request in observation["requests"])
            if used[rank] + unit * grow > controller.budgets[rank] + 1e-6:
                return False
        return True

    def gains_from_episode(entry, observations):
        """TM_GAIN (§2.2) of holding the other tenants while `entry` prefills what is left of its prompt."""
        model = entry["mark"]["model"]
        shapes = layouts(observations)
        others = [(other, *shapes[other]) for other in (0, 1) if other != model]
        return tm_gain(step_model, params, model, remaining_prompt(entry, observations), others) > 0

    def starts_episode(entry, observations):
        """True if admitting `entry` starts (or extends) an episode of its tenant: positive gain and safe."""
        model = entry["mark"]["model"]
        if not isolate or episode["model"] not in (None, model) or not gains_from_episode(entry, observations):
            return False
        return episode_safe(observations, model, trigger_left(entry, None))

    def start_episode(entry):
        if episode["model"] is None:
            episode.update(model=entry["mark"]["model"], triggers={}, started_at=time.monotonic())
        episode["triggers"][entry["id"]] = entry

    async def sync_episode():
        """Hold every tenant but the exclusive one while an episode is in force; release them otherwise."""
        try:
            for model in (0, 1):
                await controller.set_hold(model, lambda model=model: episode["model"] not in (None, model))
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                fail(error)
            raise

    def triggers_started():
        return all(item.get("done") or item["live"].get("first_token_at") is not None
                   for item in episode["triggers"].values())

    def episode_over():
        now = time.monotonic()
        return all(item.get("done") or (item["live"].get("first_token_at") is not None
                                        and now - item["live"]["first_token_at"] >= params["tm_grace_s"])
                   for item in episode["triggers"].values())

    def prefilling_entries():
        return [entry for entry in entries.values() if entry.get("admitted_at") is not None and not entry.get("done")
                and entry["live"].get("first_token_at") is None]

    def long_prefill_in_flight():
        observations = latest["summaries"]
        return observations is not None and any(gains_from_episode(entry, observations)
                                                for entry in prefilling_entries())

    def risk_params():
        now = time.monotonic()
        needs = [need for at, need in arrival_needs if at >= now - params["history_s"]]
        return {"arrival_needs": needs,
                "arrival_window_s": max(0.0, min(now - origin, params["history_s"])) + params["rate_prior_s"]}

    def fail(error):
        errors.append(f"{type(error).__name__}: {error}")
        failed.set()

    async def sync_hold():
        try:
            await controller.set_hold(1, lambda: resident["code"] > 0)
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                fail(error)
            raise

    def release_fairness(entry, reservation):
        if fairness is not None and reservation.get("reserved_at") is not None:
            event = fairness.reserved(entry["mark"]["request_id"], reservation["reserved_at"], time.monotonic())
            if event is not None:
                journal.write("fairness", event)

    async def submit(mark):
        arrived = origin + mark["arrival_s"]
        await asyncio.sleep(max(0.0, arrived - time.monotonic()))
        entry = {"mark": mark, "id": f"{spec['tag']}_{mark['request_id'].replace('-', '_')}", "arrived_at": arrived,
                 "dispatch_lag_s": time.monotonic() - arrived, "blocked_version": None, "last_attempt": 0.0,
                 "plan_not_before": 0.0, "live": {}}
        entries[mark["request_id"]] = entry
        if mark["model"] == 0:
            code_arrivals.append(arrived)
        if general:
            arrival_needs.append((arrived, math.ceil(admission_tokens(mark) / block_size) * block_gib))
        if receipt_observer is not None:
            receipt_observer.register_arrival(entry["id"], mark["model"], mark["prompt_tokens"], arrived)
        pending[mark["model"]].append(entry)

    async def complete(entry, task):
        try:
            result = await task
            reservation = await entry["reservation"]
            mark = entry["mark"]
            result.update({"trace_request_id": mark["request_id"], "arrival_s": mark["arrival_s"],
                           "prompt_key": mark.get("prompt_key"), "dispatch_lag_s": entry["dispatch_lag_s"],
                           "reserved_at": reservation.get("reserved_at"), "admission_path": entry.get("path"),
                           "concurrency_head_wait_s": limiter.wait_s(entry, time.monotonic()),
                           "preemptions": observed["preemptions"].get(entry["id"], 0)})
            if native:
                # No controller admission: the request is resident from the moment vLLM receives it, so queueing,
                # chunk waits and preemption inside vLLM all count in the prefill window and TTFT.
                result["reserved_at"] = result["submitted_at"]
                result["rank_initial"] = observed["rank"].get(entry["id"])
            results.append(result)
            journal.write("requests", result)
            version["value"] += 1
            entry["done"] = True
            if general:
                controller.drop_views((mark["model"],))  # its KV is freed: refetch the tenant's view
                if spec["return"]:
                    return_wakeup.set()
            else:
                if mark["model"] == 0:
                    controller.code_view = None  # Code KV freed: refresh the cached view on the next Conv admission
                if mark["model"] == 0 and isolate:
                    resident["code"] -= 1
                    tasks.append(asyncio.create_task(sync_hold()))
                if spec["return"] and mark["model"] == 0:
                    return_wakeup.set()
            if len(results) == total:
                done.set()
        except asyncio.CancelledError:
            raise  # deliberate cancellation (trial end or censoring), not a failure
        except BaseException as error:
            fail(error)
            raise
        finally:
            limiter.release(entry)

    async def wait_capacity(entry, rank):
        mark = entry["mark"]
        try:
            while True:
                state = await controller.retry_reservation(mark["model"], rank, entry["id"])
                if not state["reservation_pending"]:
                    entry["reservation"].set_result(state)
                    release_fairness(entry, state)
                    return
                await asyncio.sleep(.2)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            fail(error)
            raise

    def make_admit(entry, path, decision=None):
        async def admit(rank):
            mark = entry["mark"]
            limiter.acquire(entry, time.monotonic())
            entry["reservation"] = asyncio.get_running_loop().create_future()
            entry["path"] = path
            task = asyncio.create_task(stream(engines[mark["model"]], entry["id"], prompts[mark["prompt_key"]],
                                              mark["output_tokens"], rank, mark["model"], "trace", entry["arrived_at"],
                                              progress=entry["live"], receipt_observer=receipt_observer))
            tasks.extend([task, asyncio.create_task(complete(entry, task))])
            if general:
                entry["admitted_at"] = time.monotonic()
                if decision is not None and decision.get("trigger"):
                    start_episode(entry)
                    tasks.append(asyncio.create_task(sync_episode()))  # runs once the caller's lock is released
            elif mark["model"] == 0 and isolate:
                # Runs once the controller lock (held by the caller) is released.
                resident["code"] += 1
                tasks.append(asyncio.create_task(sync_hold()))
            reservation = await controller.wait_reserved(task, entry["id"], mark["model"], rank)
            if reservation["reservation_pending"]:
                tasks.append(asyncio.create_task(wait_capacity(entry, rank)))
            else:
                entry["reservation"].set_result(reservation)
                release_fairness(entry, reservation)
            return reservation
        return admit

    def record(report, entry=None):
        item = light(report)
        item["elapsed_s"] = time.monotonic() - origin
        if entry is not None:
            item["request_id"] = entry["mark"]["request_id"]
            item["waited_s"] = time.monotonic() - entry["arrived_at"]
        decisions.append(item)
        journal.write("decisions", item)

    async def try_migration(entry, report):
        mark = entry["mark"]
        tokens = admission_tokens(mark)
        observations = report["observations"]
        geometry = observations[mark["model"]][0]
        need = math.ceil(tokens / geometry["block_size"]) * geometry["block_bytes"] / GIB
        if need < params["migrate_min_gib"] or sum(controller.budgets) - sum(report["used_gib"]) < need:
            return False
        arrival = {"model": mark["model"], "tokens": tokens, "prompt_tokens": mark["prompt_tokens"]}
        seed = stable_seed(args.seed, mark["request_id"])

        def plan_fn(observed):
            state = build_state(observed, controller.budgets, growth_gib(observed, mark["prompt_tokens"]),
                                slack_gib(observed), growth_gib(observed, mark["prompt_tokens"], target_floor))
            return plan_admission(state, arrival, spec["admission"], step_model, params, seed)

        preplan = plan_fn(await controller.snapshots())
        if preplan.get("target") is None:
            record({"path": "admission_preplan_infeasible", "plan": preplan, "status": "skipped"}, entry)
            return False
        pause_models = sorted({move["model"] for move in preplan["moves"]})
        result = await controller.run_transaction(plan_fn, "admission", admission=arrival,
                                                  admit=make_admit(entry, "migration"), pause_models=pause_models,
                                                  hold_after=(1,) if isolate else (),
                                                  admission_headroom=lambda current: code_headroom(
                                                      current, mark["prompt_tokens"]))
        record(result, entry)
        return bool(result.get("submitted"))

    async def dispatch():
        try:
            while not done.is_set():
                progressed = False
                heads = sorted((queue[0] for queue in pending if queue), key=lambda entry: entry["arrived_at"])
                for entry in heads:
                    mark, now = entry["mark"], time.monotonic()
                    if not limiter.allow(entry, now):
                        continue
                    if native:
                        limiter.acquire(entry, now)
                        entry["path"] = "native"
                        entry["reservation"] = asyncio.get_running_loop().create_future()
                        entry["reservation"].set_result({"reservation_pending": False, "reserved_at": None})
                        task = asyncio.create_task(stream(
                            engines[mark["model"]], entry["id"], prompts[mark["prompt_key"]], mark["output_tokens"],
                            None, mark["model"], "trace", entry["arrived_at"], progress=entry["live"]))
                        tasks.extend([task, asyncio.create_task(complete(entry, task))])
                        pending[mark["model"]].pop(0)
                        progressed = True
                        continue
                    if fairness is not None:
                        event = fairness.update([queue[0] for queue in pending if queue], now)
                        if event is not None:
                            journal.write("fairness", event)
                        if not fairness.authorization(mark["model"], mark["request_id"], now)["allowed"]:
                            event = fairness.blocked(mark["model"], mark["request_id"], now)
                            if event is not None:
                                journal.write("fairness", event)
                            continue
                    if (entry["blocked_version"] == version["value"]
                            and now - entry["last_attempt"] < args.blocked_recheck_s):
                        continue
                    entry["last_attempt"] = now
                    before_admit = None
                    if mark["model"] == 0 and isolate and spec.get("hold_first"):
                        async def before_admit():
                            await controller.hold_locked(1)
                    report = await controller.try_direct_admission(
                        mark["model"], admission_tokens(mark), make_admit(entry, "direct"),
                        before_admit=before_admit,
                        headroom=((lambda observations, prompt=mark["prompt_tokens"]: code_headroom(observations,
                                                                                                    prompt))
                                  if mark["model"] == 0 else None))
                    if report.get("submitted"):
                        pending[mark["model"]].pop(0)
                        progressed = True
                        if mark["model"] == 0:
                            record(report, entry)
                        continue
                    entry["blocked_version"] = version["value"]
                    if spec["admission"] is None or now < entry["plan_not_before"]:
                        continue
                    if await try_migration(entry, report):
                        pending[mark["model"]].pop(0)
                        progressed = True
                    else:
                        entry["plan_not_before"] = time.monotonic() + args.plan_retry_s
                if not progressed:
                    await asyncio.sleep(args.poll_s)
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                fail(error)
            raise

    async def returner():
        last = -math.inf
        try:
            while not done.is_set():
                try:
                    await asyncio.wait_for(return_wakeup.wait(), timeout=args.return_period_s)
                except asyncio.TimeoutError:
                    pass
                return_wakeup.clear()
                if done.is_set() or time.monotonic() - last < args.return_cooldown_s:
                    continue
                if any(queue and limiter.has_capacity(model) for model, queue in enumerate(pending)):
                    continue
                summaries = await controller.summaries()
                if kv_mode != "full" and any(rank["requests"] for rank in summaries[0]):
                    continue  # no rebalancing while a Code is resident: its prefill slows peer copies, gain is nil
                if not conv_imbalanced(summaries):
                    continue
                elapsed = max(1.0, time.monotonic() - origin)
                rate = (len(code_arrivals) + 1) / (elapsed + 1 / params["code_rate"])

                def plan_fn(observed):
                    state = build_state(observed, controller.budgets, growth_gib(observed), slack_gib(observed))
                    return plan_return(state, step_model, dict(params, code_rate=rate))

                preplan = plan_fn(await controller.snapshots())
                if not preplan["moves"]:
                    record({"path": "return_preplan_rejected", "plan": preplan, "status": "skipped"})
                    continue
                result = await controller.run_transaction(plan_fn, "return", pause_models=(1,))
                record(result)
                if result.get("moves"):
                    last = time.monotonic()
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                fail(error)
            raise

    async def slicer():
        """Sliced TM. Periodic slice=(S, R): every S s of Conv hold while a Code prefills, pause the Code and let Conv
        run R s. SLO-aware slice=(T_wait, R, S_max): slice when a held Conv request has not started for T_wait s, or the
        hold reached S_max s; the Code is keep-paused first so Conv waits at most one Code step for its window."""
        cfg = spec["slice"]
        release_s = cfg[1]
        threshold = cfg[0] if len(cfg) == 2 else min(cfg[0], cfg[2])
        try:
            while not done.is_set():
                await asyncio.sleep(args.poll_s)
                hold = controller.held.get(1)
                if hold is None or resident["code"] <= 0 or time.monotonic() - hold["paused_at"] < threshold:
                    continue
                async with controller.lock:
                    hold = controller.held.get(1)
                    if hold is None:
                        continue
                    now = time.monotonic()
                    age = now - hold["paused_at"]
                    conv = await controller.tenant_summaries(1)
                    if not any(rank["requests"] for rank in conv):
                        continue
                    if len(cfg) == 2:
                        if age < cfg[0]:
                            continue
                        reason = "period"
                    else:
                        waits = [now - request["reserved_at"] for rank in conv for request in rank["requests"]
                                 if request["output_tokens"] == 0 and request.get("reserved_at")]
                        if age >= cfg[2]:
                            reason = "max_hold"
                        elif waits and max(waits) >= cfg[0]:
                            reason = "ttft"
                        else:
                            continue
                    started = time.monotonic()
                    if len(cfg) == 2:
                        code = await controller.tenant_summaries(0)
                        if all(request["decoding"] for rank in code for request in rank["requests"]):
                            continue  # every resident Code already decodes: finish it under the hold
                        await controller.pause_native(0)
                    else:
                        await controller.pause_native(0)
                        code = await controller.tenant_summaries(0)
                        if all(request["decoding"] for rank in code for request in rank["requests"]):
                            await controller.resume((0,))
                            continue
                    paused_code = time.monotonic()
                    await controller.release_locked(1, slice=True, reason=reason)
                await asyncio.sleep(release_s)
                async with controller.lock:
                    rehold = time.monotonic()
                    await controller.hold_locked(1)
                    held_again = time.monotonic()
                    await controller.resume((0,))
                    item = {"t": started - origin, "reason": reason, "code_pause_s": paused_code - started,
                            "conv_window_s": held_again - paused_code, "conv_rehold_s": held_again - rehold,
                            "code_paused_s": time.monotonic() - started}
                slices.append(item)
                journal.write("slices", item)
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                fail(error)
            raise

    async def try_migration_general(entry):
        """Algorithm 2 (§5.3) through a transaction (Algorithm 5): any tenant's decoding requests may move."""
        mark = entry["mark"]
        arrival = {"model": mark["model"], "tokens": admission_tokens(mark), "prompt_tokens": mark["prompt_tokens"]}
        seed = stable_seed(args.seed, mark["request_id"])
        context = {}

        def plan_fn(observed):
            context["trigger"] = starts_episode(entry, observed)
            result, own = growth(observed, entry, context["trigger"])
            state = build_state(observed, controller.budgets, slack_gib=slack_value(observed), request_growth=result,
                                arrival_growth_gib=own)
            return plan_admission(state, arrival, spec["admission"], step_model, params, seed)

        preplan = plan_fn(await controller.snapshots())
        if preplan.get("target") is None:
            record({"path": "admission_preplan_infeasible", "plan": preplan, "status": "skipped"}, entry)
            return False
        others = tuple(other for other in (0, 1) if other != mark["model"])
        result = await controller.run_transaction(
            plan_fn, "admission", admission=arrival, admit=make_admit(entry, "migration", context),
            pause_models=sorted({move["model"] for move in preplan["moves"]}),
            hold_after=others if context.get("trigger") else (),
            admission_headroom=lambda current: headroom_ranks(current, entry, context.get("trigger"))[0])
        record(result, entry)
        if isolate:
            tasks.append(asyncio.create_task(sync_episode()))  # holds follow the episode the final plan decided
        return bool(result.get("submitted"))

    async def dispatch_general():
        """Algorithm 1 (§5.2): admit a queue head that fits with its growth headroom (holding the other tenants
        first if its prefill starts an episode); one blocked by fragmentation tries Algorithm 2."""
        try:
            while not done.is_set():
                progressed = False
                heads = sorted((queue[0] for queue in pending if queue), key=lambda entry: entry["arrived_at"])
                for entry in heads:
                    mark, now = entry["mark"], time.monotonic()
                    if not limiter.allow(entry, now):
                        continue
                    if fairness is not None:
                        event = fairness.update([queue[0] for queue in pending if queue], now)
                        if event is not None:
                            journal.write("fairness", event)
                        if not fairness.authorization(mark["model"], mark["request_id"], now)["allowed"]:
                            event = fairness.blocked(mark["model"], mark["request_id"], now)
                            if event is not None:
                                journal.write("fairness", event)
                            continue
                    if (entry["blocked_version"] == version["value"]
                            and now - entry["last_attempt"] < args.blocked_recheck_s):
                        continue
                    entry["last_attempt"] = now
                    decision = {}

                    def headroom(observations, entry=entry, decision=decision):
                        decision["trigger"] = starts_episode(entry, observations)
                        decision["headroom"], decision["own"] = headroom_ranks(observations, entry, decision["trigger"])
                        return decision["headroom"]

                    async def before_admit(entry=entry, decision=decision):
                        if decision.get("trigger") and spec.get("hold_first"):
                            start_episode(entry)
                            for other in (0, 1):
                                if other != entry["mark"]["model"]:
                                    await controller.hold_locked(other)

                    report = await controller.try_direct_admission(
                        mark["model"], admission_tokens(mark), make_admit(entry, "direct", decision),
                        before_admit=before_admit, headroom=headroom)
                    if report.get("submitted"):
                        pending[mark["model"]].pop(0)
                        progressed = True
                        record(dict(report, episode=bool(decision.get("trigger"))), entry)
                        continue
                    entry["blocked_version"] = version["value"]
                    if spec["admission"] is None or now < entry["plan_not_before"]:
                        continue
                    need = math.ceil(admission_tokens(mark) / block_size) * block_gib
                    residents = [room - decision.get("own", 0.0) for room in decision.get("headroom", ())]
                    spare = sum(max(0.0, budget - used - room) for budget, used, room
                                in zip(controller.budgets, report["used_gib"], residents))
                    if spare >= need and await try_migration_general(entry):  # fragmentation, not capacity
                        pending[mark["model"]].pop(0)
                        progressed = True
                    else:
                        entry["plan_not_before"] = time.monotonic() + args.plan_retry_s
                if not progressed:
                    await asyncio.sleep(args.poll_s)
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                fail(error)
            raise

    async def returner_general():
        """Algorithm 3 (§5.4): level every tenant when the gain beats the pause and the admission risk; checked when
        any request completes and every return period, never during an episode or a long prefill."""
        last = -math.inf
        try:
            while not done.is_set():
                try:
                    await asyncio.wait_for(return_wakeup.wait(), timeout=args.return_period_s)
                except asyncio.TimeoutError:
                    pass
                return_wakeup.clear()
                if done.is_set() or time.monotonic() - last < args.return_cooldown_s:
                    continue
                if (any(queue and limiter.has_capacity(model) for model, queue in enumerate(pending))
                        or episode["model"] is not None or long_prefill_in_flight()):
                    continue
                summaries = await controller.summaries()
                latest["summaries"] = summaries
                if not any(conv_imbalanced(summaries, model=model) for model in (0, 1)):
                    continue
                return_params = dict(params, return_tenants=(0, 1), **risk_params())

                def plan_fn(observed):
                    result, _ = growth(observed)
                    state = build_state(observed, controller.budgets, slack_gib=slack_value(observed),
                                        request_growth=result)
                    return plan_return(state, step_model, return_params)

                preplan = plan_fn(await controller.snapshots())
                if not preplan["moves"]:
                    record({"path": "return_preplan_rejected", "plan": preplan, "status": "skipped"})
                    continue
                result = await controller.run_transaction(
                    plan_fn, "return", pause_models=sorted({move["model"] for move in preplan["moves"]}))
                record(result)
                if result.get("moves"):
                    last = time.monotonic()
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                fail(error)
            raise

    async def end_episode():
        item = {"model": episode["model"], "triggers": sorted(episode["triggers"]),
                "start_s": episode["started_at"] - origin, "end_s": time.monotonic() - origin}
        episodes.append(item)
        journal.write("episodes", item)
        episode.update(model=None, triggers={}, started_at=None)
        await sync_episode()

    async def maybe_start_episode():
        """An admitted request still prefilling (e.g. admitted while another tenant was exclusive) starts an
        episode once holding the others gains and is safe."""
        observations = latest["summaries"]
        if observations is None:
            return
        for entry in sorted(prefilling_entries(), key=lambda item: item["arrived_at"]):
            model = entry["mark"]["model"]
            if gains_from_episode(entry, observations) and episode_safe(observations, model,
                                                                        trigger_left(entry, observations)):
                start_episode(entry)
                await sync_episode()
                return

    async def tm_loop():
        """Algorithm 4 (§5.5): end the episode a grace after its triggers' first tokens, start one for a prefill
        that gains from exclusivity, and slice the held tenants: periodic (S, R) or SLO-aware (T_wait, R, S_max)."""
        cfg = slice_cfg
        threshold = (cfg[0] if len(cfg) == 2 else min(cfg[0], cfg[2])) if cfg else None
        try:
            while not done.is_set():
                await asyncio.sleep(args.poll_s)
                if episode["model"] is None:
                    await maybe_start_episode()
                    continue
                if episode_over():
                    await end_episode()
                    continue
                if cfg is None:
                    continue
                holds = [controller.held[other] for other in (0, 1)
                         if other != episode["model"] and other in controller.held]
                if not holds or time.monotonic() - min(hold["paused_at"] for hold in holds) < threshold:
                    continue
                async with controller.lock:
                    model = episode["model"]
                    held = [other for other in (0, 1) if model is not None and other != model
                            and other in controller.held]
                    if not held:
                        continue
                    now = time.monotonic()
                    age = now - min(controller.held[other]["paused_at"] for other in held)
                    views = {other: await controller.tenant_summaries(other) for other in held}
                    if not any(rank["requests"] for other in held for rank in views[other]):
                        continue
                    started = time.monotonic()
                    if len(cfg) == 2:
                        if age < cfg[0] or triggers_started():
                            continue  # every trigger already decodes: finish it under the hold
                        reason = "period"
                        await controller.pause_native(model)
                    else:
                        waits = [now - request["reserved_at"] for other in held for rank in views[other]
                                 for request in rank["requests"]
                                 if request["output_tokens"] == 0 and request.get("reserved_at")]
                        if age >= cfg[2]:
                            reason = "max_hold"
                        elif waits and max(waits) >= cfg[0]:
                            reason = "ttft"
                        else:
                            continue
                        await controller.pause_native(model)
                        if triggers_started():
                            await controller.resume((model,))
                            continue
                    paused_exclusive = time.monotonic()
                    for other in held:
                        await controller.release_locked(other, slice=True, reason=reason)
                await asyncio.sleep(cfg[1])
                async with controller.lock:
                    rehold = time.monotonic()
                    for other in held:
                        await controller.hold_locked(other)
                    held_again = time.monotonic()
                    await controller.resume((model,))
                    item = {"t": started - origin, "reason": reason, "exclusive": model,
                            "code_pause_s": paused_exclusive - started, "conv_window_s": held_again - paused_exclusive,
                            "conv_rehold_s": held_again - rehold, "code_paused_s": time.monotonic() - started}
                slices.append(item)
                journal.write("slices", item)
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                fail(error)
            raise

    async def loop_lag():
        """Record asyncio event-loop stalls (control-plane latency diagnostics)."""
        try:
            while not done.is_set():
                before = time.monotonic()
                heartbeat["t"] = before
                await asyncio.sleep(0.1)
                lag = time.monotonic() - before - 0.1
                lags["max_s"] = max(lags["max_s"], lag)
                if lag > 0.5:
                    lags["stalls"] += 1
                    journal.write("loop_lag", {"t": before - origin, "lag_s": lag,
                                               "stacks": stall_stacks[-3:], "gc": gc_events[-5:]})
                    stall_stacks.clear()
        except asyncio.CancelledError:
            raise

    def watchdog():
        """Thread: when the loop heartbeat is stale > 0.5 s, capture what the main thread is executing."""
        import sys
        import traceback
        main = threading.main_thread().ident
        while not watchdog_stop.is_set():
            time.sleep(0.1)
            if time.monotonic() - heartbeat["t"] > 0.5 and len(stall_stacks) < 20:
                frame = sys._current_frames().get(main)
                if frame is not None:
                    stall_stacks.append({"t": time.monotonic() - origin,
                                         "stack": traceback.format_stack(frame, limit=12)[-8:]})

    def gc_callback(phase, info):
        if phase == "start":
            gc_started["t"] = time.monotonic()
        elif gc_started.get("t") is not None:
            duration = time.monotonic() - gc_started["t"]
            if duration > 0.05:
                gc_events.append({"t": time.monotonic() - origin, "gen": info.get("generation"),
                                  "s": duration, "collected": info.get("collected")})

    def observe(summaries, used):
        for ranks in summaries:
            for index, rank in enumerate(ranks):
                for request in rank["requests"]:
                    key = request["id"].rsplit("-", 1)[0]
                    observed["rank"].setdefault(key, index)
                    observed["preemptions"][key] = max(observed["preemptions"].get(key, 0),
                                                       request.get("preemptions", 0))
        observed["max_mapped_gib"] = [max(seen, value) for seen, value in zip(observed["max_mapped_gib"], used)]

    async def monitor():
        try:
            iteration = 0
            while not done.is_set():
                epoch, epochs, taken = controller.code_epoch, list(controller.view_epoch), time.monotonic()
                summaries = await controller.summaries()
                if general:
                    latest["summaries"] = summaries
                    if controller.fast_admission:
                        for model in (0, 1):
                            controller.offer_view(model, epochs[model], summaries[model], taken)
                else:
                    controller.offer_code_view(epoch, summaries[0])
                observe(summaries, controller.check_capacity(summaries))
                if kv_mode != "full" and iteration % 12 == 0:
                    capacity = controller.kv_capacity(await controller.memory(), summaries,
                                                      getattr(args, "kv_utilization", 0.95))
                    observed["capacity_min_gib"] = [min(seen, value / GIB) for seen, value
                                                    in zip(observed["capacity_min_gib"], capacity)]
                iteration += 1
                now = time.monotonic()
                journal.write("population", {
                    "t": now - origin, "completed": len(results), "pending": [len(queue) for queue in pending],
                    "moves": len(controller.events), "transactions": len(controller.transactions),
                    "held": sorted(controller.held),
                    "counts": [[len(rank["requests"]) for rank in ranks] for ranks in summaries],
                    "decoding": [[sum(request["decoding"] for request in rank["requests"]) for rank in ranks]
                                 for ranks in summaries],
                    "mapped_gib": [[rank["mapped_bytes"] / GIB for rank in ranks] for ranks in summaries],
                    "preemptions": [sum(rank.get("preemptions_total", 0) for rank in ranks) for ranks in summaries]})
                journal.save(root / "progress.json", {"policy": policy, "elapsed_s": now - origin, "completed": len(results),
                                               "total": total, "moves": len(controller.events),
                                               "pending": [len(queue) for queue in pending]})
                await asyncio.sleep(args.monitor_s)
        except BaseException as error:
            if not isinstance(error, asyncio.CancelledError):
                fail(error)
            raise

    def censored_records(at):
        """Partial records of requests not completed by the deadline (same schema as completed ones, flagged).

        A request with no first token gets first_token_at = deadline, so its TTFT is a lower bound; one with >= 2
        tokens reports its observed per-token time through (last token - first token) / (tokens - 1).
        """
        finished = {item["trace_request_id"] for item in results}
        records = []
        for mark in trace["requests"]:
            if mark["request_id"] in finished:
                continue
            entry = entries.get(mark["request_id"])
            live = entry["live"] if entry else {}
            count = live.get("count") or 0
            first = live.get("first_token_at")
            reservation = entry.get("reservation") if entry else None
            reserved_at = (live.get("submitted_at") if native else
                           reservation.result().get("reserved_at")
                           if reservation is not None and reservation.done() and not reservation.cancelled()
                           else None)
            first_at = first if first is not None else at
            records.append({
                "request_id": entry["id"] if entry else None, "trace_request_id": mark["request_id"],
                "model": mark["model"], "role": "trace", "censored": True, "censored_at": at,
                "prompt_tokens": mark["prompt_tokens"], "target_output_tokens": mark["output_tokens"],
                "output_tokens": count, "arrived_at": entry["arrived_at"] if entry else origin + mark["arrival_s"],
                "arrival_s": mark["arrival_s"], "submitted_at": live.get("submitted_at"),
                "first_token_at": first_at, "first_token_observed": first is not None,
                "completed_at": live.get("last_at") if count >= 2 else first_at, "reserved_at": reserved_at,
                "admission_path": entry.get("path") if entry else "not_arrived",
                "concurrency_head_wait_s": limiter.wait_s(entry, at),
                "max_token_gap_s": live.get("max_gap"), "cadence": list(live.get("cadence") or []),
                "preemptions": observed["preemptions"].get(entry["id"], 0) if entry else 0,
                "rank_initial": observed["rank"].get(entry["id"]) if entry else None})
        return records

    arrivals = [asyncio.create_task(submit(mark)) for mark in trace["requests"]]
    if general:
        loops = [asyncio.create_task(dispatch_general()), asyncio.create_task(monitor()),
                 asyncio.create_task(loop_lag())]
        if spec["return"]:
            loops.append(asyncio.create_task(returner_general()))
        if isolate:
            loops.append(asyncio.create_task(tm_loop()))
    else:
        loops = [asyncio.create_task(dispatch()), asyncio.create_task(monitor()), asyncio.create_task(loop_lag())]
        if spec["return"]:
            loops.append(asyncio.create_task(returner()))
        if spec.get("slice"):
            loops.append(asyncio.create_task(slicer()))
    started = time.monotonic()
    watcher = threading.Thread(target=watchdog, name="xplace-loop-watchdog", daemon=True)
    watcher.start()
    gc.callbacks.append(gc_callback)
    incomplete, censored_at = [], None
    finished = asyncio.create_task(done.wait())
    broken = asyncio.create_task(failed.wait())
    try:
        trace_deadline = origin + trace["duration_s"] + args.drain_timeout_s
        await asyncio.wait([finished, broken], timeout=max(0.0, trace_deadline - time.monotonic()),
                           return_when=asyncio.FIRST_COMPLETED)
        if errors:
            raise RuntimeError(f"Trial {policy} failed: {errors[0]}")
        if not done.is_set():
            # Deadline reached without an error: censor (record partial progress, abort in-flight requests below)
            # so a pathologically slow policy bounds its round instead of failing the whole bundled job.
            censored_at = time.monotonic()
            incomplete = censored_records(censored_at)
    finally:
        if receipt_observer is not None:
            receipt_observer.mark_trace_cutoff(time.monotonic(), censored_at)
            try:
                await receipt_observer.stop(include_report=False)
            except Exception as error:
                errors.append(f"observation stop: {type(error).__name__}: {error}")
        for waiter in (finished, broken):
            waiter.cancel()
        await asyncio.gather(finished, broken, return_exceptions=True)
        watchdog_stop.set()
        if gc_callback in gc.callbacks:
            gc.callbacks.remove(gc_callback)
        for task in loops + arrivals:
            task.cancel()
        await asyncio.gather(*loops, *arrivals, return_exceptions=True)
        if errors or not done.is_set():
            for task in tasks:
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if controller.held and not errors:
            for model in sorted(controller.held):
                await controller.set_hold(model, lambda: False)
        if censored_at is not None:
            try:
                await wait_empty(controller, timeout=300)  # aborted requests must leave vLLM before the next trial
            except Exception as error:
                errors.append(f"post-censor drain: {type(error).__name__}: {error}")
        controller.enforce_budget = True
        try:
            final = await controller.summaries()
            preemptions = [sum(rank.get("preemptions_total", 0) for rank in ranks) - start
                           for ranks, start in zip(final, preemptions_start)]
            recompute_tokens = [sum(rank.get("recompute_tokens_total", 0) for rank in ranks) - start
                                for ranks, start in zip(final, recompute_start)]
        except Exception as error:
            preemptions = recompute_tokens = f"{type(error).__name__}: {error}"
        journal.close()
        save(root / "requests.json", results)
        if censored_at is not None:
            save(root / "incomplete.json", incomplete)
        save(root / "decisions.json", decisions)
        save(root / "migrations.json", controller.events)
        save(root / "transactions.json", controller.transactions)
        save(root / "holds.json", controller.holds)
        save(root / "slices.json", slices)
        if general:
            if episode["model"] is not None:  # the trial ended inside an episode
                episodes.append({"model": episode["model"], "triggers": sorted(episode["triggers"]),
                                 "start_s": episode["started_at"] - origin, "end_s": None})
            save(root / "episodes.json", episodes)
        status = ("passed" if done.is_set() and not errors else
                  "censored" if censored_at is not None and not errors else "failed")
        summary = {"policy": policy, "status": status,
                   "errors": errors, "requests": len(results), "expected": total,
                   "censored": len(incomplete), "censored_at_s": None if censored_at is None else censored_at - origin,
                   "moves": len(controller.events), "transactions": len(controller.transactions),
                   "wall_s": time.monotonic() - started,
                   "control_s": sum(item.get("control_s", 0) for item in controller.transactions),
                   "paused_request_s": sum(item.get("paused_request_s", 0) for item in controller.transactions),
                   "holds": len(controller.holds), "held_s": sum(item.get("held_s", 0) for item in controller.holds),
                   "admission_view": dict(controller.view_stats), "slices": len(slices), "loop_lag": dict(lags),
                   "gc_slow": len(gc_events), "gc_max_s": max((item["s"] for item in gc_events), default=0.0),
                   "journal_errors": journal.errors, "native": native, "kv_mode": kv_mode,
                   "concurrency": limiter.snapshot(),
                   "preemptions": preemptions, "recompute_tokens": recompute_tokens,
                   "preempted_requests": sum(1 for item in results if item.get("preemptions")),
                   "max_mapped_gib": observed["max_mapped_gib"],
                   "capacity_min_gib": [None if math.isinf(value) else value
                                        for value in observed["capacity_min_gib"]],
                   "rank_unobserved": sum(1 for item in results if item.get("rank_initial") is None)}
        if general:
            summary["episodes"] = len(episodes)
        save(root / "summary.json", summary)
    return summary
