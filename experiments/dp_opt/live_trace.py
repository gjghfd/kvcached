import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import time
from pathlib import Path

from live_benchmark import DEFAULT_KV_CACHE_BYTES, DEFAULT_MAX_MODEL_LEN, engine_arguments, save
from live_benefit import capture_control, percentile, stream, wait_empty, wait_reserved
from live_controller import PlacementController, allocation_increment, migration_precheck, stable_ownership, wait_migration_precheck
from queue_fairness import DEFAULT_QUEUE_THRESHOLD_S, FairAdmissionGate
from live_rebalance import ReleaseRebalancer


POLICIES = ("no_move", "own_model", "kv_only", "ilp")
TRACE_POLICIES = (*POLICIES, "no_move_fair", "kv_only_rebalance", "kv_only_amortized", "kv_only_headroom", "kv_only_workaware")
AMORTIZED_RUNTIME_SOURCES = (
    "amortized_rebalance.py", "ilp_reference.py", "live_controller.py", "live_scheduler.py",
    "live_trace.py", "live_rebalance.py", "verify_trace.py", "verify_rebalance.py",
    "trace-benefit.sbatch", "rebalance-trace.sbatch")


def archive_amortized_runtime(root, headroom=False):
    archive = root / "runtime-source"
    archive.mkdir()
    sources = {}
    names = (*AMORTIZED_RUNTIME_SOURCES, "headroom_rebalance.py") if headroom else AMORTIZED_RUNTIME_SOURCES
    for name in names:
        source = Path(__file__).resolve().parent / name
        shutil.copy2(source, archive / name)
        sources[name] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    save(root / "runtime-source.json", {"policy": "kv_only_headroom" if headroom else "kv_only_amortized", "sources": sources})


async def append_record(path, value):
    def write():
        with path.open("a") as output:
            output.write(json.dumps(value, separators=(",", ":")) + "\n")

    await asyncio.to_thread(write)


async def run_capacity_pilot(engines, controller, args, trace, tokenizer):
    from agentic_capacity import calibration_rates, capacity_setting

    code_marks = json.loads(args.calibration_code_marks.read_text())
    conv_marks = json.loads(args.calibration_conv_marks.read_text())
    code = max(code_marks, key=lambda mark: mark["prompt_tokens"] + mark["output_tokens"])
    conv = min(conv_marks, key=lambda mark: abs(mark["prompt_tokens"] - 3949) + abs(mark["output_tokens"] - 659))
    args.prepared_prompts = prepare_prompts({"requests": [code, conv]}, tokenizer, args.prompt_bank_data)
    await wait_empty(controller)
    await controller.pause()
    samples = []
    transitions = []

    async def measure(release_unused=False):
        before = await controller.snapshots()
        memory = await asyncio.gather(*[controller.call(model, rank, "collective_rpc",
                                                        "placement_memory_snapshot", 120, (release_unused,), {})
                                       for model in (0, 1) for rank in (0, 1)])
        observations = await controller.snapshots()
        gpu_memory = []
        for rank in (0, 1):
            replicas = [memory[model * 2 + rank][0] for model in (0, 1)]
            if (len({entry["gpu_uuid"] for entry in replicas}) != 1
                    or any("A800" not in entry["gpu_name"] for entry in replicas)):
                raise RuntimeError("Capacity pilot requires colocated A800 tenants")
            gpu_memory.append(min(replicas, key=lambda entry: entry["free_mib"]))
        sample = {"sampled_at": time.monotonic(), "gpu_memory": gpu_memory,
                  "mapped_mib": [sum(observations[model][rank]["mapped_bytes"] for model in (0, 1)) / 1024**2
                                 for rank in (0, 1)]}
        if stable_ownership(before) != stable_ownership(observations):
            transitions.append({"sample": sample, "before": before, "after": observations,
                                "reason": "allocation_ownership_changed_across_free_memory_query_not_coherent_headroom"})
            save(args.output / "memory-transitions.json", transitions)
            return observations
        sample["ownership_stable_across_memory_query"] = True
        sample["ownership_before_memory_query"] = stable_ownership(before)
        sample["ownership_after_memory_query"] = stable_ownership(observations)
        samples.append(sample)
        save(args.output / "memory-samples.json", samples)
        return observations

    observations = await measure(True)
    page_mib = observations[0][0]["page_bytes"] / 1024**2
    setting = capacity_setting(samples, code_marks, page_mib, args.capacity_guard_mib,
                               allocator_gpu_utilization=args.allocator_gpu_utilization)
    save(args.output / "capacity-setting-initial.json", setting)
    transaction = "a800-longest-code-capacity"
    probes = []
    try:
        await controller.begin(transaction)
        for model in (0, 1):
            for rank in (0, 1):
                probe = await controller.probe(transaction, model, rank, request_prompt(args.prepared_prompts, code),
                                               code["output_tokens"], [setting["kv_budget_mib_per_gpu"] / 1024] * 2)
                probes.append(probe)
                save(args.output / "long-context-probes.json", probes)
                if not probe["fits_budget"]:
                    raise MemoryError("Longest 100K Code allocation cannot fit selected A800 budget")
        await controller.end(transaction)
    finally:
        await controller.resume()
    await capture_control(controller, "start")
    origin = time.monotonic()
    tasks = [asyncio.create_task(stream(engines[1], f"capacity-conv-{index}",
                                       request_prompt(args.prepared_prompts, conv), conv["output_tokens"], 1,
                                       1, "capacity", origin)) for index in range(4)]
    tasks.append(asyncio.create_task(stream(engines[0], "capacity-code", request_prompt(args.prepared_prompts, code),
                                            code["output_tokens"], 0, 0, "capacity", origin)))
    try:
        while not all(task.done() for task in tasks):
            await measure()
            await asyncio.sleep(1)
        results = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    save(args.output / "requests.json", results)
    await wait_empty(controller)
    graph = await capture_control(controller, "stop")
    await controller.pause()
    try:
        await measure()
    finally:
        await controller.resume()
    setting = capacity_setting(samples, code_marks, page_mib, args.capacity_guard_mib,
                               allocator_gpu_utilization=args.allocator_gpu_utilization)
    conv_results = [result for result in results if result["model"] == 1]
    service = statistics.mean(result["completed_at"] - result["submitted_at"] for result in conv_results)
    rates = calibration_rates(setting, conv_marks, service, conv["output_tokens"])
    setting["calibration"] = rates
    save(args.output / "capacity-setting.json", setting)
    save(args.output / "summary.json", {"status": "capacity_passed", "setting": setting, "graph": graph,
                                         "scope": "100K_actual_allocation_and_mixed_prefill_not_four_policy_benefit"})
    with (args.output / "capacity-config.env").open("w") as output:
        output.write(f'export DP_OPT_TRACE_BUDGET_MIB={setting["kv_budget_mib_per_gpu"]}\n')
        output.write(f'export DP_OPT_TARGET_CONV_MIB={setting["target_conv_reserved_mib"]}\n')
        output.write('export DP_OPT_CALIBRATION_CONV_RATES="' + ' '.join(map(str, rates["conv_rates_rps"])) + '"\n')


def prepare_prompts(trace, tokenizer, prompt_bank=None):
    prefix = tokenizer.encode("Calculate 2 + 2. The answer is", add_special_tokens=False)
    prompts = {}
    for mark in trace["requests"]:
        key = mark.get("prompt_key", str(mark["prompt_tokens"]))
        if key in prompts:
            if len(prompts[key]) != mark["prompt_tokens"]:
                raise ValueError("Prompt key refers to inconsistent lengths")
            continue
        if "prompt_key" in mark:
            if prompt_bank is None or key not in prompt_bank["prompts"]:
                raise ValueError("Real agent input requires the verified prompt bank")
            entry = prompt_bank["prompts"][key]
            rendered = tokenizer.apply_chat_template(entry["messages"], tokenize=False,
                                                      add_generation_prompt=True)
            if hashlib.sha256(rendered.encode()).hexdigest() != entry["prompt_sha256"]:
                raise ValueError("Runtime chat template differs from profiled Qwen input")
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            if ids != entry["token_ids"] or len(ids) != mark["prompt_tokens"]:
                raise ValueError("Runtime Qwen token IDs differ from agent input bank")
            prompts[key] = ids
        else:
            prompts[key] = (prefix * math.ceil(mark["prompt_tokens"] / len(prefix)))[:mark["prompt_tokens"]]
    return prompts


def request_prompt(prompts, mark):
    return prompts[mark.get("prompt_key", str(mark["prompt_tokens"]))]


async def run_rate_pilot(engines, controller, args, trace, tokenizer):
    import subprocess

    code = next(mark for mark in trace["requests"] if mark["model"] == 0)
    conv = sorted((mark for mark in trace["requests"] if mark["model"] == 1),
                  key=lambda mark: abs(mark["prompt_tokens"] - 3949) + abs(mark["output_tokens"] - 659))[0]
    budgets = [args.kv_budget_mib / 1024] * 2
    await wait_empty(controller)
    save(args.output / "gpu-memory-loaded.json", {
        "nvidia_smi": subprocess.check_output([
            "nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used",
            "--format=csv"], text=True)})
    await controller.pause()
    transaction = "long-context-pilot"
    probes = []
    try:
        await controller.begin(transaction)
        for model in (0, 1):
            for rank in (0, 1):
                probe = await controller.probe(transaction, model, rank,
                                               request_prompt(args.prepared_prompts, code),
                                               code["output_tokens"], budgets)
                probes.append(probe)
                if not probe["fits_budget"]:
                    raise MemoryError("100K reservation cannot fit the enlarged KV budget")
        await controller.end(transaction)
    finally:
        await controller.resume()
    save(args.output / "long-context-probes.json", probes)
    await capture_control(controller, "start")
    started = time.monotonic()
    serial = await stream(engines[0], "pilot-code-serial", request_prompt(args.prepared_prompts, code),
                          code["output_tokens"], 0, 0, "pilot", started)
    await wait_empty(controller)
    save(args.output / "pilot-code-serial.json", serial)
    experiments = []
    for population in args.pilot_populations:
        origin = time.monotonic()
        tasks = [asyncio.create_task(stream(
            engines[1], f"pilot-conv-{population}-{index}",
            request_prompt(args.prepared_prompts, conv), conv["output_tokens"], index % 2,
            1, "pilot", origin)) for index in range(population)]
        await asyncio.sleep(3)
        code_started = time.monotonic()
        tasks.append(asyncio.create_task(stream(
            engines[0], f"pilot-code-mixed-{population}", request_prompt(args.prepared_prompts, code),
            code["output_tokens"], 0, 0, "pilot", code_started)))
        requests = await asyncio.wait_for(asyncio.gather(*tasks), timeout=args.request_timeout_s)
        await wait_empty(controller)
        conv_results = [request for request in requests if request["model"] == 1]
        services = [request["completed_at"] - request["submitted_at"] for request in conv_results]
        conv_memory = math.ceil((conv["prompt_tokens"] + conv["output_tokens"]) / 16) * 1.5
        mean_service = statistics.mean(services)
        experiment = {"population": population, "requests": requests,
                      "conv_mean_service_s": mean_service, "conv_mark": conv,
                      "conv_memory_mib": conv_memory,
                      "conv_target_memory_mib": args.pilot_target_conv_mib,
                      "suggested_conv_rate_rps": args.pilot_target_conv_mib / (conv_memory * mean_service),
                      "code_service_s": requests[-1]["completed_at"] - requests[-1]["submitted_at"],
                      "scope": "finite_equal_mark_population_with_code_not_sustainable_rate_calibration"}
        experiments.append(experiment)
        save(args.output / "pilot-progress.json", experiment)
    graph = await capture_control(controller, "stop")
    code_s = statistics.mean(experiment["code_service_s"] for experiment in experiments)
    result = {"status": "passed", "probes": probes, "serial_code": serial,
              "experiments": experiments, "graph": graph,
              "suggested_conv_rate_rps": experiments[-1]["suggested_conv_rate_rps"],
              "suggested_code_rate_rps": min(.02, .25 / code_s),
              "scope": "pilot_only_not_trace_benefit_or_stable_throughput"}
    save(args.output / "pilot-summary.json", result)
    save(args.output / "summary.json", {"status": "pilot_passed", "scope": result["scope"]})


async def run_rate_calibration(engines, controller, args, trace, tokenizer):
    from agentic_trace import generate_mixed_trace

    conv_marks = json.loads(args.calibration_conv_marks.read_text())
    code_marks = json.loads(args.calibration_code_marks.read_text()) if args.calibration_code_marks else list(
        {mark["prompt_key"]: {key: value for key, value in mark.items()
                             if key not in ("request_id", "model", "arrival_s")}
         for mark in trace["requests"] if mark["model"] == 0}.values())
    trials = []
    for index, conv_rate in enumerate(args.calibration_conv_rates):
        calibration = generate_mixed_trace(code_marks, conv_marks, args.calibration_code_rate,
                                            conv_rate, args.calibration_duration_s, args.seed,
                                            args.max_model_len)
        calibration["prompt_bank_sha256"] = trace["prompt_bank_sha256"]
        args.prepared_prompts = prepare_prompts(calibration, tokenizer, args.prompt_bank_data)
        directory = args.output / f"calibration-{index}"
        directory.mkdir(parents=True)
        save(directory / "trace.json", calibration)
        result = await run_trace(engines, controller, args, calibration, tokenizer, "no_move", index, directory)
        population = json.loads((directory / "population.json").read_text())
        samples = [sample for sample in population if args.calibration_duration_s / 3 <= sample["elapsed_s"]
                   < args.calibration_duration_s]
        conv_mib = [sum(sample["reserved_mib"][1]) for sample in samples]
        queues = [max(0, sample["unfinished"] - sample["active"] + sample.get("capacity_pending", 0))
                  for sample in samples]
        code_heads = [head for sample in samples for head in sample["pending_heads"] if head["model"] == 0]
        trial = {"conv_rate_rps": conv_rate, "code_rate_rps": args.calibration_code_rate,
                 "duration_s": args.calibration_duration_s, "summary": result,
                 "mean_sampled_conv_reserved_mib": statistics.mean(conv_mib),
                 "mean_sampled_external_queue_count": statistics.mean(queues),
                 "code_head_samples": len(code_heads),
                 "global_enough_local_blocked_code_samples": sum(head["global_enough_local_blocked"] for head in code_heads),
                 "scope": "open_loop_rate_calibration_not_four_policy_benefit"}
        trials.append(trial)
        save(args.output / "calibration-summary.json", {"status": "running", "trials": trials})
    save(args.output / "calibration-summary.json", {"status": "passed", "trials": trials,
                                                   "target_conv_reserved_mib": args.pilot_target_conv_mib,
                                                   "scope": "choose_rate_before_policy_comparison_not_outcome_selection"})
    save(args.output / "summary.json", {"status": "calibration_passed"})


def summarize_trace(requests, origin, duration, deadline):
    if not requests or len({request["request_id"] for request in requests}) != len(requests):
        raise ValueError("Require unique completed trace requests")
    finish = max(request["completed_at"] for request in requests)
    queue = [request.get("reserved_at", request["submitted_at"]) - request["arrived_at"] for request in requests]
    ttft = [request["first_token_at"] - request["arrived_at"] for request in requests]
    latency = [request["completed_at"] - request["arrived_at"] for request in requests]
    tpot = [(request["completed_at"] - request["first_token_at"]) / (len(request["token_ids"]) - 1)
            for request in requests if len(request["token_ids"]) > 1]
    if any(value < 0 for value in queue + ttft + latency):
        raise ValueError("Trace timing precedes frozen arrival")
    return {"completed_requests": len(requests), "output_tokens": sum(len(request["token_ids"]) for request in requests),
            "makespan_s": finish - origin, "drain_s": max(0, finish - origin - duration),
            "arrival_window_s": duration, "mean_queue_s": statistics.mean(queue),
            "p95_queue_s": percentile(queue, .95), "p99_queue_s": percentile(queue, .99),
            "mean_ttft_s": statistics.mean(ttft), "p95_ttft_s": percentile(ttft, .95),
            "p99_ttft_s": percentile(ttft, .99), "mean_latency_s": statistics.mean(latency),
            "p99_latency_s": percentile(latency, .99), "mean_tpot_s": statistics.mean(tpot) if tpot else None,
            "deadline_s": deadline, "deadline_met_requests": sum(value <= deadline for value in latency),
            "deadline_goodput_req_s": sum(value <= deadline for value in latency) / duration,
            "completed_within_arrival_window": sum(request["completed_at"] <= origin + duration for request in requests),
            "output_tokens_per_complete_drain_s": sum(len(request["token_ids"]) for request in requests) / (finish - origin)}


async def run_trace(engines, controller, args, trace, tokenizer, policy, repeat, root):
    root.mkdir(parents=True, exist_ok=True)
    await wait_empty(controller)
    if getattr(controller, "rank_local_admission", False):
        controller.admission_observations = None
    if getattr(controller, "release_unused_cache_on_trace", False):
        memory = await controller.release_transfer_cache()
        save(root / "memory-start.json", memory)
    controller.events.clear()
    await capture_control(controller, "start")
    budgets = [args.kv_budget_mib / 1024] * 2
    admission_policy = "kv_only" if policy in ("kv_only_rebalance", "kv_only_amortized", "kv_only_headroom", "kv_only_workaware") else policy
    origin = time.monotonic() + 1
    arrivals, tasks, decisions, results, population = [], [], [], [], []
    pending = [[], []]
    capacity_waiting = [None, None]
    capacity_entries = [None, None]
    fairness = (FairAdmissionGate(getattr(args, "fairness_queue_threshold_s", DEFAULT_QUEUE_THRESHOLD_S))
                if policy == "no_move_fair" else None)
    attempted_states = set()
    planning_not_before = [0, 0]
    planning_prechecks = {"checked": 0, "rejected": 0}
    wait_prechecks = {"checked": 0, "rejected": 0}
    recorded_wait_states = set()
    rebalancer = None
    balance_ledger = None
    if policy in ("kv_only_amortized", "kv_only_headroom", "kv_only_workaware"):
        from amortized_rebalance import MigrationInferenceLedger

        fixed = getattr(args, "rebalance_pause_fixed_s", None)
        if policy == "kv_only_workaware":
            from workaware_rebalance import WorkawareInferenceLedger

            balance_ledger = WorkawareInferenceLedger(args.workaware_rate_priors, origin,
                args.move_fixed_s if fixed is None else fixed, getattr(args, "rebalance_headroom_quantile", .95),
                getattr(args, "rebalance_headroom_history_limit", 64))
        elif policy == "kv_only_headroom":
            from headroom_rebalance import HeadroomInferenceLedger

            balance_ledger = HeadroomInferenceLedger(getattr(args, "rebalance_overhead_fraction", .01),
                args.move_fixed_s if fixed is None else fixed, getattr(args, "rebalance_headroom_quantile", .95),
                getattr(args, "rebalance_headroom_history_limit", 64))
        else:
            balance_ledger = MigrationInferenceLedger(getattr(args, "rebalance_overhead_fraction", .01),
                                                     args.move_fixed_s if fixed is None else fixed)
    headroom_arrivals = []
    journal_locks = {name: asyncio.Lock() for name in ("requests", "decisions", "population", "monitor-transitions", "wait-prechecks", "fairness-events", "headroom-arrivals")}
    prompts = getattr(args, "prepared_prompts", None) or prepare_prompts(trace, tokenizer)
    save(root / "workload.json", {"origin": origin, "policy": policy, "repeat": repeat,
                                  "capacity_gib": budgets, "trace_sha256": hashlib.sha256(
                                      (json.dumps(trace, indent=2) + "\n").encode()).hexdigest(),
                                  "duration_s": trace["duration_s"], "requests": trace["requests"]})

    async def record(name, value):
        async with journal_locks[name]:
            if balance_ledger is not None:
                if name == "decisions" and value["path"] != "release_rebalance_ilp":
                    balance_ledger.admission(value)
                elif name == "requests":
                    balance_ledger.completion(value)
            await append_record(root / f"{name}.jsonl", value)

    async def release_fairness(mark, reservation):
        if fairness is not None:
            if not isinstance(reservation, dict) or "reserved_at" not in reservation:
                raise ValueError("Fair baseline requires real KV reservation timing")
            reservation["reservation_observed_at"] = time.monotonic()
            event = fairness.reserved(mark["request_id"], reservation["reserved_at"], reservation["reservation_observed_at"])
            if event is not None:
                await record("fairness-events", event)

    async def submit(mark):
        arrived = origin + mark["arrival_s"]
        await asyncio.sleep(max(0, arrived - time.monotonic()))
        future = asyncio.get_running_loop().create_future()
        pending[mark["model"]].append({"mark": mark, "arrived_at": arrived,
                                       "arrival_dispatch_lag_s": time.monotonic() - arrived, "future": future})
        if policy in ("kv_only_headroom", "kv_only_workaware"):
            sample = balance_ledger.arrival(mark, arrived, time.monotonic())
            headroom_arrivals.append(sample)
            await record("headroom-arrivals", sample)
        return await asyncio.wait_for(future, timeout=args.request_timeout_s)

    async def complete(entry, stream_task):
        mark = entry["mark"]
        try:
            result = await stream_task
            reservation = await entry["reservation_future"]
            if isinstance(reservation, dict):
                result.update({"reserved_at": reservation["reserved_at"],
                               "reservation_time_source": reservation["reservation_time_source"]})
                if "reservation_observed_at" in reservation:
                    result["reservation_observed_at"] = reservation["reservation_observed_at"]
            result.update({"source_fragment": mark["source_fragment"], "source_internal_row": mark["source_internal_row"],
                           "arrival_offset_s": mark["arrival_s"], "arrival_dispatch_lag_s": entry["arrival_dispatch_lag_s"]})
            if "prompt_key" in mark:
                result["prompt_key"] = mark["prompt_key"]
            results.append(result)
            await record("requests", result)
            if rebalancer is not None and (balance_ledger is None or balance_ledger.eligible()):
                rebalancer.notify({"request_id": mark["request_id"], "model": mark["model"],
                                   "completed_at": result["completed_at"]})
            if not entry["future"].done():
                entry["future"].set_result(result)
        except BaseException as error:
            if not entry["future"].done():
                entry["future"].set_exception(error)
            raise

    async def dispatch(tenant=None):
        while len(results) < len(trace["requests"]):
            heads = sorted((queue[0] for model, queue in enumerate(pending)
                            if queue and (tenant is None or tenant == model)
                            and (capacity_waiting[model] is None or capacity_waiting[model].done())),
                           key=lambda entry: entry["arrived_at"])
            progressed = False
            for entry in heads:
                mark = entry["mark"]
                authorization = None
                if fairness is not None:
                    waiting = [queue[0] for queue in pending if queue]
                    waiting.extend(item for item in capacity_entries if item is not None)
                    event = fairness.update(waiting, time.monotonic())
                    if event is not None:
                        await record("fairness-events", event)
                    authorization = fairness.authorization(mark["model"], mark["request_id"], time.monotonic())
                    if not authorization["allowed"]:
                        event = fairness.blocked(mark["model"], mark["request_id"], time.monotonic())
                        if event is not None:
                            await record("fairness-events", event)
                        continue

                async def admit(rank, entry=entry, mark=mark):
                    entry["reservation_future"] = asyncio.get_running_loop().create_future()
                    task = asyncio.create_task(stream(engines[mark["model"]], mark["request_id"],
                                                      request_prompt(prompts, mark), mark["output_tokens"], rank,
                                                      mark["model"], "arrival", entry["arrived_at"]))
                    tasks.extend((task, asyncio.create_task(complete(entry, task))))
                    reservation = await wait_reserved(controller, task, mark["request_id"], mark["model"], rank,
                                                      target_only=True)
                    if isinstance(reservation, dict) and reservation["reservation_pending"]:
                        async def wait_capacity():
                            try:
                                remaining = max(.01, entry["arrived_at"] + args.request_timeout_s - time.monotonic())
                                final = await wait_reserved(controller, task, mark["request_id"], mark["model"], rank,
                                                            timeout=remaining, wait_for_capacity=True, budgets=budgets)
                                async with controller.lock:
                                    await controller.check_capacity(budgets)
                                capacity_entries[mark["model"]] = None
                                await release_fairness(mark, final)
                                entry["reservation_future"].set_result(final)
                            except BaseException as error:
                                if not entry["reservation_future"].done():
                                    entry["reservation_future"].set_exception(error)
                                raise

                        capacity_entries[mark["model"]] = entry
                        capacity_waiting[mark["model"]] = asyncio.create_task(wait_capacity())
                        tasks.append(capacity_waiting[mark["model"]])
                    else:
                        await release_fairness(mark, reservation)
                        entry["reservation_future"].set_result(reservation)
                    return reservation

                decision = await controller.try_direct_admission(mark["prompt_tokens"], mark["output_tokens"],
                                                                budgets, admit, arriving_model=mark["model"])
                if fairness is not None:
                    decision["fairness_authorization"] = authorization
                observations = decision["observations"]
                active = [request for ranks in observations for rank in ranks for request in rank["requests"]]
                state = (mark["request_id"], tuple(sorted(request["id"] for request in active)))
                free_bytes = sum(budgets) * 1024**3 - sum(rank["mapped_bytes"] for ranks in observations for rank in ranks)
                geometry = observations[mark["model"]][0]
                incoming_bytes = math.ceil((mark["prompt_tokens"] + mark["output_tokens"]) / geometry["block_size"]) * geometry["block_bytes"]
                if (not decision["admitted"] and not decision.get("submitted", False)
                        and admission_policy not in ("no_move", "no_move_fair") and free_bytes >= incoming_bytes
                        and all(request["movable"] for request in active) and state not in attempted_states
                        and time.monotonic() >= planning_not_before[mark["model"]]):
                    precheck = migration_precheck(observations, mark["model"],
                                                  mark["prompt_tokens"] + mark["output_tokens"], budgets,
                                                  admission_policy, args.max_moves, args.candidate_limit)
                    planning_prechecks["checked"] += 1
                    wait_gate = None
                    if precheck["may_fit"] and admission_policy == "ilp" and getattr(args, "wait_aware", False):
                        wait_gate = wait_migration_precheck(observations, mark["model"],
                            mark["prompt_tokens"] + mark["output_tokens"], budgets, args.max_moves,
                            args.candidate_limit, args.wait_pause_floor_s, args.wait_solve_floor_s,
                            args.wait_copy_seconds_per_gib, args.wait_safety_fraction,
                            boundary_cost=({"incoming_handshake_s": args.wait_incoming_handshake_s,
                                            "old_transaction_fixed_s": args.wait_old_transaction_fixed_s,
                                            "preplan_s": args.wait_solve_floor_s}
                                           if getattr(args, "preplan", False) else None))
                        wait_prechecks["checked"] += 1
                        if not wait_gate["may_reduce_latency_sum"]:
                            wait_prechecks["rejected"] += 1
                            wait_state = (state, wait_gate["estimated_wait_s"] is not None)
                            if wait_state not in recorded_wait_states:
                                await record("wait-prechecks", {"request_id": mark["request_id"], **wait_gate})
                                recorded_wait_states.add(wait_state)
                            precheck["may_fit"] = False
                    if precheck["may_fit"]:
                        attempted_states.add(state)
                        planning_started = time.monotonic()
                        prepared = None
                        if getattr(args, "preplan", False):
                            prepared = await controller.prepare_admission(
                                mark["prompt_tokens"], mark["output_tokens"], budgets, policy=admission_policy,
                                arriving_model=mark["model"], max_moves=args.max_moves,
                                candidate_limit=args.candidate_limit, solver_limit_s=args.solver_limit_s,
                                move_fixed_s=args.move_fixed_s,
                                move_bandwidth_gib_s=1 / args.wait_copy_seconds_per_gib if wait_gate is not None
                                else getattr(args, "move_bandwidth_gib_s", 1))
                        decision = await controller.schedule_admission(
                            mark["prompt_tokens"], mark["output_tokens"], budgets, admit, policy=admission_policy,
                            arriving_model=mark["model"], max_moves=args.max_moves,
                            candidate_limit=args.candidate_limit, solver_limit_s=args.solver_limit_s,
                            move_fixed_s=args.move_fixed_s,
                            **({"move_bandwidth_gib_s": getattr(args, "move_bandwidth_gib_s", 1)}
                               if wait_gate is None else {}),
                            **({"prepared": prepared} if prepared is not None else {}),
                            **({"wait_recheck_options": {"copy_seconds_per_gib": args.wait_copy_seconds_per_gib,
                                   "safety_fraction": args.wait_safety_fraction,
                                   "boundary_cost": {"incoming_handshake_s": args.wait_incoming_handshake_s,
                                                     "old_transaction_fixed_s": args.wait_old_transaction_fixed_s,
                                                     "preplan_s": args.wait_solve_floor_s}}}
                               if wait_gate is not None and getattr(args, "revalidate_prepared", False) else {}),
                            **({"wait_cost_s": wait_gate["estimated_wait_s"] * args.wait_safety_fraction,
                                "wait_fixed_cost_s": (0 if prepared is not None else
                                    wait_gate["active_plus_incoming"] * (args.wait_pause_floor_s + args.wait_solve_floor_s)),
                                "future_transaction_fixed_s": (args.wait_old_transaction_fixed_s
                                                               if prepared is not None else 0),
                                "move_bandwidth_gib_s": 1 / args.wait_copy_seconds_per_gib} if wait_gate is not None else {}))
                        planning_elapsed = time.monotonic() - planning_started
                        planning_not_before[mark["model"]] = time.monotonic() + max(
                            getattr(args, "planning_retry_min_s", 2),
                            planning_elapsed * getattr(args, "planning_retry_cost_factor", 4))
                        decision["path"] = "blocked_staged_ilp"
                        decision["no_pause_precheck"] = precheck
                        decision["wait_cost_precheck"] = wait_gate
                    else:
                        planning_prechecks["rejected"] += 1
                        decision["path"] = "wait_conservative_migration_bound"
                else:
                    decision["path"] = ("direct_fastpath" if decision["admitted"] else
                                        "direct_allocator_wait" if decision.get("submitted", False) else
                                        "wait_capacity_or_prefill")
                if decision["admitted"] or decision.get("submitted", False) or decision["path"] == "blocked_staged_ilp":
                    decision.update({"request_id": mark["request_id"], "elapsed_from_arrival_s": time.monotonic() - entry["arrived_at"],
                                     "policy": policy})
                    decisions.append(decision)
                    await record("decisions", decision)
                if decision["admitted"] or decision.get("submitted", False):
                    if pending[mark["model"]].pop(0) is not entry:
                        raise RuntimeError("Tenant FIFO admission head changed")
                    progressed = True
            if not progressed:
                await asyncio.sleep(args.queue_poll_s)

    async def monitor():
        while not arrivals or any(not task.done() for task in arrivals) or time.monotonic() < origin + trace["duration_s"]:
            monitoring = None
            if getattr(controller, "rank_local_admission", False):
                monitoring = await controller.monitor_snapshot()
                if not monitoring["admission_stable_across_observation"]:
                    await record("monitor-transitions", {
                        **monitoring, "elapsed_s": time.monotonic() - origin,
                        "reason": "admission_mutated_during_asynchronous_observation_not_atomic_population"})
                    await asyncio.sleep(2)
                    continue
                observations = monitoring["observations"]
            else:
                async with controller.lock:
                    observations = await controller.snapshots()
            now = time.monotonic()
            offered = sum(origin + request["arrival_s"] <= now for request in trace["requests"])
            completed = sum(request["completed_at"] <= now for request in results)
            active = sum(len(rank["requests"]) for ranks in observations for rank in ranks)
            capacity_pending = sum(request.get("reservation_pending", False)
                                   for ranks in observations for rank in ranks for request in rank["requests"])
            mapped = [sum(observations[model][rank]["mapped_bytes"] for model in (0, 1)) / 1024**3
                      for rank in (0, 1)]
            if any(value > budget + 1e-6 for value, budget in zip(mapped, budgets)):
                raise MemoryError("Trace mapped KV exceeded controlled budget")
            reserved = [[sum(len(request["blocks"]) * rank["block_bytes"] / 1024**2
                             for request in rank["requests"]) for rank in ranks] for ranks in observations]
            heads = []
            global_free = (sum(budgets) - sum(mapped)) * 1024**3
            for model, queue in enumerate(pending):
                if not queue:
                    continue
                mark = queue[0]["mark"]
                tokens = mark["prompt_tokens"] + mark["output_tokens"]
                increments = [allocation_increment(rank, tokens) for rank in observations[model]]
                geometry = observations[model][0]
                logical = math.ceil(tokens / geometry["block_size"]) * geometry["block_bytes"]
                blocked = all(mapped[rank] + increments[rank] > budgets[rank] + 1e-6 for rank in (0, 1))
                heads.append({"model": model, "request_id": mark["request_id"], "local_blocked": blocked,
                              "global_free_bytes": global_free, "incoming_logical_bytes": logical,
                              "global_enough_local_blocked": blocked and global_free >= logical,
                              "all_decode": all(request["movable"] for ranks in observations
                                                for rank in ranks for request in rank["requests"])})
            sample = {"elapsed_s": now - origin, "offered": offered, "completed": completed,
                      "active": active, "unfinished": offered - completed, "capacity_pending": capacity_pending,
                      "mapped_gib": mapped, "reserved_mib": reserved, "pending_heads": heads,
                      "counts": [[len(rank["requests"]) for rank in ranks] for ranks in observations]}
            if monitoring is not None:
                sample.update({key: value for key, value in monitoring.items() if key != "observations"})
            population.append(sample)
            await record("population", sample)
            save(args.output / "progress.json", {"policy": policy, "repeat": repeat,
                                                   "elapsed_s": now - origin, "offered": offered,
                                                   "completed": completed, "total": len(trace["requests"]),
                                                   "moves": len(controller.events)})
            await asyncio.sleep(2)

    if policy in ("kv_only_rebalance", "kv_only_amortized", "kv_only_headroom", "kv_only_workaware"):
        def admission_busy():
            return any(pending) or any(task is not None and not task.done() for task in capacity_waiting)

        async def rebalance(causes):
            pause_fixed = getattr(args, "rebalance_pause_fixed_s", None)
            return await controller.rebalance_after_release(
                budgets, max_moves=getattr(args, "rebalance_max_moves", 4),
                candidate_limit=getattr(args, "rebalance_candidate_limit", 8),
                move_bandwidth_gib_s=getattr(args, "move_bandwidth_gib_s", 1),
                solver_limit_s=args.solver_limit_s,
                pause_fixed_s=args.move_fixed_s if pause_fixed is None else pause_fixed,
                min_gain_s=getattr(args, "rebalance_min_gain_s", 1), admission_busy=admission_busy,
                workaware_fit=getattr(args, "workaware_fit_data", None),
                **({"balance_ledger": balance_ledger} if balance_ledger is not None else {}))

        async def record_rebalance(decision):
            decision["policy"] = policy
            if balance_ledger is not None:
                balance_ledger.settle(decision)
            decisions.append(decision)
            await record("decisions", decision)

        rebalancer = ReleaseRebalancer(rebalance, record_rebalance,
                                      retry_s=getattr(args, "rebalance_retry_s", 2),
                                      cooldown_s=getattr(args, "rebalance_cooldown_s", 10),
                                      retry_on_release_only=balance_ledger is not None)
    arrivals = [asyncio.create_task(submit(mark)) for mark in trace["requests"]]
    monitor_task = asyncio.create_task(monitor())
    dispatcher_tasks = [asyncio.create_task(dispatch(model)) for model in (0, 1)] if getattr(
        controller, "concurrent_admission", False) and fairness is None else [asyncio.create_task(dispatch())]
    async def drive_trace():
        await asyncio.gather(*arrivals, monitor_task, *dispatcher_tasks)
        if rebalancer is not None:
            await rebalancer.close()

    driver_task = asyncio.create_task(drive_trace())
    control_tasks = [driver_task] + ([rebalancer.task] if rebalancer is not None else [])
    try:
        await asyncio.wait_for(asyncio.gather(*control_tasks),
                               timeout=trace["duration_s"] + args.request_timeout_s * 2)
        await wait_empty(controller)
        captures = await capture_control(controller, "stop")
        summary = summarize_trace(results, origin, trace["duration_s"], args.deadline_s)
        summary.update({"policy": policy, "repeat": repeat, "moves": len(controller.events),
                        "migration_kv_bytes": sum(event["kv_bytes"] for event in controller.events),
                        "control_s": sum(decision["end_to_end_control_s"] for decision in decisions),
                        "control_s_including_preplan": sum(decision["end_to_end_control_s"]
                                                          + decision.get("preplan_s", 0) for decision in decisions),
                        "ilp_attempts": sum(decision["path"] == "blocked_staged_ilp" for decision in decisions),
                        "no_pause_planning_prechecks": planning_prechecks})
        summary["wait_cost_prechecks"] = wait_prechecks
        if rebalancer is not None:
            rebalances = [decision for decision in decisions if decision["path"] == "release_rebalance_ilp"]
            summary["release_rebalance"] = {
                "attempts": len(rebalances), "transactions": sum(bool(decision["moves"]) for decision in rebalances),
                "moves": sum(len(decision["moves"]) for decision in rebalances),
                "control_s_including_preplan": sum(decision["end_to_end_control_s"] + decision.get("preplan_s", 0)
                                                    for decision in rebalances)}
        if fairness is not None:
            if fairness.owner is not None:
                raise RuntimeError("Fairness owner remains after all trace requests completed")
            summary["fairness"] = fairness.summary()
        save(root / "captures.json", captures)
        save(root / "migrations.json", controller.events)
        save(root / "summary.json", summary)
        return summary
    finally:
        for task in arrivals + tasks + [monitor_task] + dispatcher_tasks + control_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*arrivals, *tasks, monitor_task, *dispatcher_tasks, *control_tasks, return_exceptions=True)
        for name, records in (("requests", results), ("decisions", decisions), ("population", population)):
            await asyncio.to_thread(save, root / f"{name}.json", records)
        if policy in ("kv_only_headroom", "kv_only_workaware"):
            await asyncio.to_thread(save, root / "headroom-arrivals.json", headroom_arrivals)
        if fairness is not None:
            await asyncio.to_thread(save, root / "fairness-events.json", fairness.events)
        if rebalancer is not None:
            await asyncio.to_thread(save, root / "migrations.json", controller.events)


async def run(args):
    from transformers import AutoTokenizer
    from vllm.v1.engine.async_llm import AsyncLLM

    args.allocator_gpu_utilization = float(os.environ.get("KVCACHED_GPU_UTILIZATION", "0.95"))
    if not 0 < args.allocator_gpu_utilization <= 1:
        raise ValueError("Require positive allocator GPU utilization no greater than one")
    trace = json.loads(args.trace.read_text())
    if trace["max_model_len"] > args.max_model_len:
        raise ValueError("Trace exceeds engine max length")
    args.output.mkdir(parents=True, exist_ok=True)
    if getattr(args, "workaware_fit", None) is not None:
        from workaware_pilot import SOURCE_FILES, load_verified_fit

        args.workaware_fit_data = load_verified_fit(args.workaware_fit, args.workaware_fit_sha256,
                                                     args.workaware_source_root)
        shutil.copy2(args.workaware_fit, args.output / "workaware-fit.json")
        source_archive = args.output / "workaware-cost-source"
        source_archive.mkdir()
        for name in SOURCE_FILES:
            destination = source_archive / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.workaware_source_root / name, destination)
    layout_lookup = None
    if getattr(args, "layout_cost_lookup", None) is not None:
        from layout_cost_pilot import LOOKUP_SOURCE_FILES, load_verified_lookup

        layout_lookup = load_verified_lookup(args.layout_cost_lookup, args.layout_cost_lookup_sha256,
                                             args.layout_cost_source_root)
        shutil.copy2(args.layout_cost_lookup, args.output / "layout-cost-lookup.json")
        source_archive = args.output / "layout-cost-source"
        source_archive.mkdir()
        for name in LOOKUP_SOURCE_FILES:
            shutil.copy2(args.layout_cost_source_root / name, source_archive / name)
        load_verified_lookup(args.output / "layout-cost-lookup.json", args.layout_cost_lookup_sha256,
                             source_archive)
    if getattr(args, "measured_control_config", None) is not None:
        from freeze_measured_control import load_control

        control_config = load_control(args.measured_control_config, args.measured_control_config_sha256)
        if (any(getattr(args, name) != value for name, value in control_config["parameters"].items())
                or not args.preplan or not args.wait_aware
                or os.environ.get("DP_OPT_KV_TRANSPORT") != "dp_echo" or layout_lookup is None
                or control_config["layout_lookup_sha256"] != args.layout_cost_lookup_sha256
                or control_config["layout_source_files_sha256"] != layout_lookup["source_files_sha256"]):
            raise ValueError("Serving arguments differ from independently frozen control configuration")
        shutil.copy2(args.measured_control_config, args.output / "control-config.json")
    save(args.output / "config.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    if getattr(args, "revalidate_prepared", False) or any(policy in args.policies for policy in ("kv_only_rebalance", "kv_only_amortized", "kv_only_headroom", "kv_only_workaware")):
        shutil.copy2(args.fit, args.output / "placement-fit.json")
    save(args.output / "trace.json", trace)
    if any(policy in args.policies for policy in ("kv_only_amortized", "kv_only_headroom")):
        archive_amortized_runtime(args.output, headroom="kv_only_headroom" in args.policies)
    if "kv_only_workaware" in args.policies or getattr(args, "workaware_cadence_pilot", False):
        from workaware_evidence import archive_runtime

        archive_runtime(args.output)
    save(args.output / "runtime-control.json", {"dp_pause_sync_interval": int(os.environ.get("DP_OPT_PAUSE_SYNC_INTERVAL", "32")),
                                               "same_interval_for_all_policies": True,
                                               "idle_rank_expert_execution_not_disabled": True,
                                               "admission": "centralized_per_tenant_FIFO_directfastpath_allocator_queue",
                                               "queue_metric": "full_KV_reservation_time_minus_frozen_arrival",
                                               "rank_local_admission": os.environ.get("DP_OPT_RANK_LOCAL_ADMISSION") == "1",
                                               "concurrent_admission": os.environ.get("DP_OPT_CONCURRENT_ADMISSION") == "1",
                                               "migration_gate": "global_free_all_decode_no_pause_bounded_source_release_then_frozen_recheck",
                                               "planning_retry_min_s": args.planning_retry_min_s,
                                               "planning_retry_cost_factor": args.planning_retry_cost_factor,
                                               "wait_aware_ilp": getattr(args, "wait_aware", False),
                                               "preplan_enabled": getattr(args, "preplan", False),
                                               "candidate_revalidation_enabled": getattr(args, "revalidate_prepared", False),
                                               "release_rebalance_policy": "kv_only_rebalance" if "kv_only_rebalance" in args.policies else None,
                                               "amortized_rebalance_policy": "kv_only_amortized" if "kv_only_amortized" in args.policies else None,
                                               "headroom_rebalance_policy": "kv_only_headroom" if "kv_only_headroom" in args.policies else None,
                                               "workaware_rebalance_policy": "kv_only_workaware" if "kv_only_workaware" in args.policies else None,
                                               "admission_cost_pilot": getattr(args, "admission_cost_pilot", False),
                                               "kv_transport": os.environ.get("DP_OPT_KV_TRANSPORT", "host_file"),
                                               "transfer_chunk_mib": int(os.environ.get("DP_OPT_TRANSFER_CHUNK_MIB", "32")),
                                               "wait_cost_scope": ("measured_244279_boundary_control_and_small_population_lookup_not_Code_prefill_counterfactual"
                                                                   if layout_lookup is not None else
                                                                   "online_remaining_decode_and_241580_host_staged_copy_floor_not_joint_calibrated_prediction")})
    os.environ["DP_OPT_EVIDENCE_ROOT"] = str(args.output)
    os.environ["DP_OPT_CONTROLLED_RESERVATION_RETRY"] = "1"
    os.environ["DP_OPT_CAPTURE_EVIDENCE"] = "1"
    engines = []
    try:
        for model in (0, 1):
            os.environ["DP_OPT_SERVICE"] = f"instance-{model + 1}"
            base = f"trace{os.environ['SLURM_JOB_ID']}_s{model}"
            os.environ["KVCACHED_IPC_NAME"] = base
            os.environ["KVCACHED_DP_IPC_BASE"] = base
            arguments = engine_arguments(args.checkpoint, args.port + model * 10, args.cache_bytes, args.max_model_len)
            if getattr(args, "workaware_cadence_pilot", False) or "kv_only_workaware" in args.policies:
                arguments.worker_cls = "workaware_worker.WorkawareVerificationWorker"
            engines.append(AsyncLLM.from_engine_args(arguments))
        controller = PlacementController(engines, json.loads(args.fit.read_text()), os.environ["DP_OPT_TRANSFER_ROOT"],
                                         allow_extrapolation=True)
        controller.rank_local_admission = os.environ.get("DP_OPT_RANK_LOCAL_ADMISSION") == "1"
        controller.concurrent_admission = os.environ.get("DP_OPT_CONCURRENT_ADMISSION") == "1"
        controller.layout_lookup = layout_lookup
        controller.revalidate_prepared = getattr(args, "revalidate_prepared", False)
        controller.release_unused_cache_on_trace = os.environ.get("DP_OPT_RELEASE_UNUSED_CACHE") == "1"
        if controller.concurrent_admission and not controller.rank_local_admission:
            raise ValueError("Concurrent admission requires controlled rank-local budget ledger")
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        bank = None
        if args.prompt_bank is not None:
            payload = args.prompt_bank.read_bytes()
            if hashlib.sha256(payload).hexdigest() != trace.get("prompt_bank_sha256"):
                raise ValueError("Prompt bank checksum differs from frozen trace")
            bank = json.loads(payload)
        args.prompt_bank_data = bank
        args.prepared_prompts = prepare_prompts(trace, tokenizer, bank)
        await controller.snapshots()
        workers = [[json.loads((args.output / f"worker-instance-{model + 1}-dp{rank}.json").read_text())
                    for rank in (0, 1)] for model in (0, 1)]
        if len({worker["ipc_name"] for ranks in workers for worker in ranks}) != 4:
            raise RuntimeError("Tenant/rank KV namespaces are not independent")
        for model in workers:
            for worker in model:
                if worker["max_model_len"] != args.max_model_len:
                    raise RuntimeError("Worker length differs from declared serving config")
                if any(worker["groups"][name]["world_size"] != size for name, size in (("tp", 1), ("dp", 2), ("ep", 2))):
                    raise RuntimeError("Worker topology differs from DP2/EP2/TP1")
            for left, right in zip(model[0]["expert_ids"], model[1]["expert_ids"]):
                if set(left) & set(right) or set(left) | set(right) != set(range(128)):
                    raise RuntimeError("Experts are not complementary EP shards")
        if any(workers[0][rank]["gpu_uuid"] != workers[1][rank]["gpu_uuid"] for rank in (0, 1)):
            raise RuntimeError("Tenant ranks not colocated")
        prefix = tokenizer.encode("Calculate 2 + 2. The answer is", add_special_tokens=False)
        for model in (0, 1):
            for rank in (0, 1):
                await stream(engines[model], f"warm-{model}-{rank}", prefix, 16, rank, model, "warmup", time.monotonic())
        from ilp_reference import enumerate_targets
        warm = {"counts": [[0, 0], [0, 0]], "initial_gib": [0, 0], "capacity_gib": [1, 1],
                "new_gib": .1, "source_pages": [], "movable": [], "performance_cost_tables_s": [[0, 0], [0]]}
        await asyncio.to_thread(enumerate_targets, warm, max_moves=0, time_limit_s=1)
        await controller.pause()
        await controller.resume()
        if os.environ.get("DP_OPT_RELEASE_UNUSED_CACHE") == "1":
            memory = await asyncio.gather(*[controller.call(model, rank, "collective_rpc",
                                                            "placement_memory_snapshot", 120, (True,), {})
                                           for model in (0, 1) for rank in (0, 1)])
            save(args.output / "memory-prepared.json", memory)
        if args.capacity_pilot:
            await run_capacity_pilot(engines, controller, args, trace, tokenizer)
            return
        if args.pilot:
            await run_rate_pilot(engines, controller, args, trace, tokenizer)
            return
        if getattr(args, "stress_sweep", False):
            from stress_sweep import run_stress_sweep

            await run_stress_sweep(engines, controller, args, trace, tokenizer)
            return
        if args.calibration_conv_marks is not None:
            await run_rate_calibration(engines, controller, args, trace, tokenizer)
            return
        if getattr(args, "transfer_pilot", False):
            from transfer_pilot import run_transfer_pilot

            await run_transfer_pilot(engines, controller, args, tokenizer)
            return
        if getattr(args, "admission_cost_pilot", False):
            from admission_cost_pilot import run_admission_cost_pilot

            await run_admission_cost_pilot(engines, controller, args, trace, tokenizer)
            return
        if getattr(args, "layout_cost_pilot", False):
            from layout_cost_pilot import run_layout_cost_pilot

            await run_layout_cost_pilot(engines, controller, args, tokenizer)
            return
        if getattr(args, "workaware_cadence_pilot", False):
            from workaware_pilot import run_workaware_pilot

            await run_workaware_pilot(engines, controller, args, tokenizer)
            return
        reference_marks = sorted(trace["requests"], key=lambda request: request["prompt_tokens"])
        reference_marks = [reference_marks[index] for index in (0, len(reference_marks) // 3,
                                                               2 * len(reference_marks) // 3, -1)]
        references = []
        for index, mark in enumerate(reference_marks):
            prompt = request_prompt(args.prepared_prompts, mark)
            reference = await stream(engines[mark["model"]], f"reference-{index}", prompt,
                                     mark["output_tokens"], 0, mark["model"], "reference", time.monotonic())
            reference["trace_request_id"] = mark["request_id"]
            references.append(reference)
        save(args.output / "references.json", references)
        await wait_empty(controller)
        order = []
        rng = random.Random(args.seed)
        for repeat in range(args.repeats):
            policies = list(args.policies)
            rng.shuffle(policies)
            order.extend({"repeat": repeat, "policy": policy} for policy in policies)
        save(args.output / "order.json", order)
        summaries = []
        for item in order:
            result = await run_trace(engines, controller, args, trace, tokenizer, item["policy"], item["repeat"],
                                     args.output / f"r{item['repeat']}-{item['policy']}")
            summaries.append(result)
            save(args.output / "summary.json", {"status": "running", "trials": summaries})
        for item in order:
            requests = json.loads((args.output / f"r{item['repeat']}-{item['policy']}" / "requests.json").read_text())
            by_id = {request["request_id"]: request for request in requests}
            for reference in references:
                if by_id[reference["trace_request_id"]]["token_ids"] != reference["token_ids"]:
                    raise RuntimeError("Selected trace request differs from its full serial greedy reference")
        if "no_move" in args.policies:
            for repeat in range(args.repeats):
                baseline = json.loads((args.output / f"r{repeat}-no_move" / "requests.json").read_text())
                baseline_tokens = {request["request_id"]: request["token_ids"] for request in baseline}
                for policy in args.policies:
                    requests = json.loads((args.output / f"r{repeat}-{policy}" / "requests.json").read_text())
                    if {request["request_id"]: request["token_ids"] for request in requests} != baseline_tokens:
                        raise RuntimeError("Paired policy trace token IDs differ from no-move baseline")
        save(args.output / "summary.json", {"status": "passed", "trials": summaries,
                                             "scope": "preliminary_one_trace_cached_sample_not_final_multiseed_benefit"})
    except BaseException as error:
        save(args.output / "error.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        for engine in reversed(engines):
            engine.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--prompt-bank", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--capacity-pilot", action="store_true")
    parser.add_argument("--transfer-pilot", action="store_true")
    parser.add_argument("--transfer-conv-marks", type=Path)
    parser.add_argument("--transfer-transports", nargs="+", choices=("host_file", "dp_allgatherv", "dp_echo"),
                        default=["host_file", "dp_allgatherv"])
    parser.add_argument("--transfer-pause-modes", nargs=2, choices=("native_keep", "default32_boundary"))
    parser.add_argument("--capacity-guard-mib", type=float, default=1024)
    parser.add_argument("--pilot-populations", nargs="+", type=int, default=[16, 32, 48])
    parser.add_argument("--pilot-target-conv-mib", type=float, default=23552)
    parser.add_argument("--calibration-conv-marks", type=Path)
    parser.add_argument("--calibration-code-marks", type=Path)
    parser.add_argument("--calibration-conv-rates", nargs="+", type=float, default=[1.65, 3.0, 4.0, 5.0])
    parser.add_argument("--calibration-code-rate", type=float, default=.013)
    parser.add_argument("--calibration-duration-s", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--cache-bytes", type=int, default=DEFAULT_KV_CACHE_BYTES)
    parser.add_argument("--kv-budget-mib", type=int, default=2112)
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--policies", nargs="+", choices=TRACE_POLICIES, default=list(POLICIES))
    parser.add_argument("--fairness-queue-threshold-s", type=float, default=DEFAULT_QUEUE_THRESHOLD_S)
    parser.add_argument("--request-timeout-s", type=float, default=300)
    parser.add_argument("--deadline-s", type=float, default=20)
    parser.add_argument("--queue-poll-s", type=float, default=.2)
    parser.add_argument("--max-moves", type=int, default=2)
    parser.add_argument("--candidate-limit", type=int, default=2)
    parser.add_argument("--solver-limit-s", type=float, default=1)
    parser.add_argument("--move-fixed-s", type=float, default=.475)
    parser.add_argument("--move-bandwidth-gib-s", type=float, default=1)
    parser.add_argument("--rebalance-max-moves", type=int, default=4)
    parser.add_argument("--rebalance-candidate-limit", type=int, default=8)
    parser.add_argument("--rebalance-pause-fixed-s", type=float)
    parser.add_argument("--rebalance-min-gain-s", type=float, default=1)
    parser.add_argument("--rebalance-retry-s", type=float, default=2)
    parser.add_argument("--rebalance-cooldown-s", type=float, default=10)
    parser.add_argument("--rebalance-overhead-fraction", type=float, default=.01)
    parser.add_argument("--rebalance-headroom-quantile", type=float, default=.95)
    parser.add_argument("--rebalance-headroom-history-limit", type=int, default=64)
    parser.add_argument("--planning-retry-min-s", type=float, default=2)
    parser.add_argument("--planning-retry-cost-factor", type=float, default=4)
    parser.add_argument("--wait-aware", action="store_true")
    parser.add_argument("--preplan", action="store_true")
    parser.add_argument("--revalidate-prepared", action="store_true")
    parser.add_argument("--admission-cost-pilot", action="store_true")
    parser.add_argument("--layout-cost-pilot", action="store_true")
    parser.add_argument("--workaware-cadence-pilot", action="store_true")
    parser.add_argument("--workaware-fit", type=Path)
    parser.add_argument("--workaware-fit-sha256")
    parser.add_argument("--workaware-source-root", type=Path)
    parser.add_argument("--workaware-rate-priors", nargs=2, type=float)
    parser.add_argument("--layout-cost-lookup", type=Path)
    parser.add_argument("--layout-cost-lookup-sha256")
    parser.add_argument("--layout-cost-source-root", type=Path)
    parser.add_argument("--measured-control-config", type=Path)
    parser.add_argument("--measured-control-config-sha256")
    parser.add_argument("--admission-conv-marks", type=Path)
    parser.add_argument("--stress-sweep", action="store_true")
    parser.add_argument("--stress-conv-marks", type=Path)
    parser.add_argument("--stress-code-marks", type=Path)
    parser.add_argument("--stress-outputs", nargs="+", type=int, default=[8192])
    parser.add_argument("--stress-budgets", nargs="+", type=int, default=[10240, 11328])
    parser.add_argument("--stress-rate-multipliers", nargs="+", type=float, default=[1])
    parser.add_argument("--stress-code-rate", type=float, default=.001)
    parser.add_argument("--stress-fixed-conv-rate", type=float)
    parser.add_argument("--stress-window-s", type=float, default=1800)
    parser.add_argument("--stress-warmup-s", type=float, default=600)
    parser.add_argument("--stress-target-conv-mib", type=float, default=4800)
    parser.add_argument("--wait-pause-floor-s", type=float, default=.35)
    parser.add_argument("--wait-solve-floor-s", type=float, default=.15)
    parser.add_argument("--wait-incoming-handshake-s", type=float, default=3.05)
    parser.add_argument("--wait-old-transaction-fixed-s", type=float, default=.1)
    parser.add_argument("--wait-copy-seconds-per-gib", type=float, default=5)
    parser.add_argument("--wait-safety-fraction", type=float, default=.8)
    args = parser.parse_args()
    if not math.isfinite(args.rebalance_overhead_fraction) or not 0 < args.rebalance_overhead_fraction < 1:
        parser.error("Rebalance overhead fraction must be strictly between zero and one")
    if (not math.isfinite(args.rebalance_headroom_quantile) or not 0 < args.rebalance_headroom_quantile <= 1
            or args.rebalance_headroom_history_limit < 1):
        parser.error("Headroom quantile must be in (0,1] with a positive history limit")
    if args.stress_sweep and (args.stress_conv_marks is None or args.stress_code_marks is None):
        parser.error("Stress sweep requires explicit source Conv and Code marks")
    if "no_move_fair" in args.policies and (not math.isfinite(args.fairness_queue_threshold_s)
                                           or args.fairness_queue_threshold_s <= 0):
        parser.error("Fair queue threshold must be finite and positive")
    if args.revalidate_prepared and not args.preplan:
        parser.error("Candidate revalidation requires pause-outside preplanning")
    if (args.measured_control_config is None) != (args.measured_control_config_sha256 is None):
        parser.error("Measured control configuration requires frozen SHA256")
    if any(value is not None for value in (args.workaware_fit, args.workaware_fit_sha256, args.workaware_source_root)) and not all(
            value is not None for value in (args.workaware_fit, args.workaware_fit_sha256, args.workaware_source_root)):
        parser.error("Workaware fit requires frozen SHA256 and independently verifiable raw source directory")
    if "kv_only_workaware" in args.policies and (args.workaware_fit is None or args.workaware_rate_priors is None):
        parser.error("Workaware recovery requires verified calibration and two frozen external rate priors")
    if args.workaware_rate_priors is not None and any(not math.isfinite(value) or value <= 0 for value in args.workaware_rate_priors):
        parser.error("Workaware rate priors must be finite positive RPS")
    if any(value is not None for value in (args.layout_cost_lookup, args.layout_cost_lookup_sha256,
                                          args.layout_cost_source_root)) and not all(
            value is not None for value in (args.layout_cost_lookup, args.layout_cost_lookup_sha256,
                                           args.layout_cost_source_root)):
        parser.error("Layout lookup requires frozen SHA256 and independent raw source directory")
    if (args.admission_cost_pilot or args.layout_cost_pilot) and args.admission_conv_marks is None:
        parser.error("Admission cost pilot requires paired Conv marks")
    if args.transfer_pilot and args.transfer_conv_marks is None:
        parser.error("Transfer pilot requires paired Conv marks")
    if args.transfer_pilot:
        from transfer_pilot import pilot_cases

        try:
            pilot_cases([{"prompt_tokens": 2048, "output_tokens": 512, "source_internal_row": 0}],
                        transports=args.transfer_transports, pause_modes=args.transfer_pause_modes)
        except ValueError as error:
            parser.error(str(error))
    if min(args.max_model_len, args.cache_bytes, args.kv_budget_mib, args.repeats,
           args.request_timeout_s, args.deadline_s, args.queue_poll_s,
           args.planning_retry_min_s, args.planning_retry_cost_factor) <= 0:
        parser.error("Lengths, capacities, repeat counts and timing limits must be positive")
    if (not 0 < args.wait_safety_fraction <= 1 or args.wait_copy_seconds_per_gib <= 0
            or any(not math.isfinite(value) or value < 0 for value in (
                args.wait_pause_floor_s, args.wait_solve_floor_s, args.wait_copy_seconds_per_gib,
                args.wait_incoming_handshake_s, args.wait_old_transaction_fixed_s))):
        parser.error("Wait costs must be finite and nonnegative with positive copy slope and unit safety fraction")
    if not math.isfinite(args.move_bandwidth_gib_s) or args.move_bandwidth_gib_s <= 0:
        parser.error("Move bandwidth must be positive and finite")
    if (args.rebalance_max_moves < 0 or args.rebalance_candidate_limit < 1
            or not math.isfinite(args.rebalance_min_gain_s) or args.rebalance_min_gain_s < 0
            or any(not math.isfinite(value) or value <= 0 for value in (args.rebalance_retry_s, args.rebalance_cooldown_s))
            or (args.rebalance_pause_fixed_s is not None and
                (not math.isfinite(args.rebalance_pause_fixed_s) or args.rebalance_pause_fixed_s < 0))):
        parser.error("Invalid release-rebalance limits or timing costs")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
