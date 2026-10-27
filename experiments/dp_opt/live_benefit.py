import argparse
import asyncio
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

from live_benchmark import DEFAULT_KV_CACHE_BYTES, DEFAULT_MAX_MODEL_LEN, engine_arguments, save
from live_controller import PlacementController, allocation_increment, stable_ownership


POLICIES = ("no_move", "own_model", "kv_only", "ilp")


def workload_fixture(scenario):
    if scenario == "capacity":
        inputs = [[[64] * 3, [64] * 3], [[64] * 4, [64] * 3]]
        pages, headroom = [[1, 1], [2, 1]], [2, 1]
    elif scenario == "tradeoff":
        inputs = [[[752, 64], [64] * 8], [[80, 80, 80, 512], [80, 80]]]
        pages, headroom = [[2, 3], [2, 1]], [2, 1]
    elif scenario == "tradeoff_long":
        inputs = [[[496, 64], [64] * 8], [[80, 80, 80, 512], [80, 80]]]
        pages, headroom = [[2, 5], [3, 2]], [2, 1]
    elif scenario == "long_kv":
        inputs = [[[3072] * 3, [3072] * 3], [[2048] * 3, [2048] * 3]]
        pages, headroom = [[11, 11], [8, 8]], [3, 3]
    else:
        raise ValueError("Unknown benefit workload scenario")
    return {"inputs": inputs, "counts": [[len(rank) for rank in ranks] for ranks in inputs],
            "pages": pages, "headroom": headroom,
            "output_tokens": 512 if scenario in ("tradeoff_long", "long_kv") else 256,
            "arrival_input_tokens": {"tradeoff_long": 2560, "long_kv": 3584}.get(scenario, 2816),
            "b_arrival_input_tokens": 1024 if scenario == "long_kv" else 64,
            "total_requests": sum(len(rank) for ranks in inputs for rank in ranks) + 2}


async def wait_reserved(controller, task, request_id, model, rank, timeout=30, wait_for_capacity=False, budgets=None,
                        target_only=False):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if target_only:
            target_observation = await controller.call(model, rank, "placement_snapshot")
            locations = [(rank, target_observation)]
        else:
            observations = await controller.snapshots()
            locations = [(location, observations[model][location]) for location in (rank, 1 - rank)]
        found = next(((location, observation, request) for location, observation in locations
                      for request in observation["requests"]
                      if request["id"] == request_id or request["id"].rsplit("-", 1)[0] == request_id), None)
        record = found[2] if found is not None else None
        if record is not None:
            pending = record.get("reservation_pending", False)
            if not pending or not wait_for_capacity:
                measured = record.get("reserved_at")
                return {"reservation_pending": pending,
                        "reserved_at": None if pending else measured if measured is not None else time.monotonic(),
                        "reservation_time_source": "pending" if pending else
                        "scheduler_monotonic" if measured is not None else "snapshot_upper_bound",
                        "rank_observation": found[1], "rank_observed": found[0]}
        if task.done():
            result = await task
            return {"reservation_pending": False, "reserved_at": result["first_token_at"],
                    "reservation_time_source": "first_token_upper_bound"}
        if record is not None and budgets is not None:
            location = found[0]
            await controller.retry_reservation(model, location, record["id"], budgets)
        await asyncio.sleep(.1 if wait_for_capacity else .05)
    raise TimeoutError("Request reservation was not observed")


async def capture_control(controller, command):
    return [await asyncio.gather(*[controller.call(model, rank, "collective_rpc", "benefit_capture",
                                                   120, (command,), {}) for rank in (0, 1)])
            for model in (0, 1)]


async def long_kv_preflight(controller, fixture, prompt, output_tokens, minimum_decode, budgets):
    async with controller.lock:
        transaction = "long-kv-probe-" + str(time.monotonic_ns())
        started = time.monotonic()
        await controller.pause()
        completed = False
        try:
            before = await controller.begin(transaction)
            if ([[len(rank["requests"]) for rank in ranks] for ranks in before] != fixture["counts"]
                    or any(not request["movable"] or request["output_tokens"] < minimum_decode
                           for ranks in before for rank in ranks for request in rank["requests"])):
                raise RuntimeError("Long-KV preflight requires the full declared decode population")
            probes = [await controller.probe(transaction, 0, rank, prompt, output_tokens, budgets) for rank in (0, 1)]
            after = await controller.snapshots()
            if stable_ownership(before) != stable_ownership(after):
                raise RuntimeError("Long-KV diagnostic changed serving ownership")
            incoming = math.ceil((len(prompt) + output_tokens) / before[0][0]["block_size"]) * before[0][0]["block_bytes"]
            total_budget = sum(budgets) * 1024**3
            mapped = sum(rank["mapped_bytes"] for ranks in before for rank in ranks)
            reserved = sum(len(request["blocks"]) * rank["block_bytes"]
                           for ranks in before for rank in ranks for request in rank["requests"])
            second = math.ceil((fixture["b_arrival_input_tokens"] + output_tokens) / before[1][0]["block_size"]) * before[1][0]["block_bytes"]
            if (any(probe["fits_budget"] or not probe["allocated"] for probe in probes)
                    or total_budget - mapped < incoming or reserved + incoming + second > total_budget):
                raise RuntimeError("Long-KV witness must have enough global capacity but no direct per-rank admission")
            result = {"before": before, "after_rollback": after, "probes": probes,
                      "capacity_gib": budgets, "total_budget_bytes": round(total_budget),
                      "initial_mapped_bytes": mapped, "global_free_bytes": round(total_budget - mapped),
                      "arrival_a_reserved_bytes": incoming, "all_request_reserved_bytes": reserved + incoming + second,
                      "scope": "diagnostic_temporary_allocations_may_exceed_controlled_budget_then_rollback_not_serving_oom"}
            await controller.end(transaction)
            completed = True
        finally:
            if completed:
                await controller.resume()
        result["diagnostic_control_s"] = time.monotonic() - started
        return result, after


def percentile(values, quantile):
    ordered = sorted(values)
    if not ordered or not 0 <= quantile <= 1:
        raise ValueError("Percentiles require samples and a unit-interval quantile")
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_requests(requests, origin, deadline_s):
    if not requests or len({request["request_id"] for request in requests}) != len(requests):
        raise ValueError("Require nonempty unique request results")
    end = max(request["completed_at"] for request in requests)
    makespan = end - origin
    if makespan <= 0:
        raise ValueError("Request completion must follow workload origin")
    arrivals = [request for request in requests if request["role"] == "arrival"]
    latency = [request["completed_at"] - request["arrived_at"] for request in requests]
    ttft = [request["first_token_at"] - request["arrived_at"] for request in requests]
    tpot = [(request["completed_at"] - request["first_token_at"]) / (len(request["token_ids"]) - 1)
            for request in requests if len(request["token_ids"]) > 1]
    if not math.isfinite(deadline_s) or deadline_s <= 0 or not tpot or any(value < 0 for value in latency + ttft + tpot):
        raise ValueError("Require positive SLO and valid request timing")
    return {"completed_requests": len(requests), "output_tokens": sum(len(request["token_ids"]) for request in requests),
            "makespan_s": makespan,
            "output_throughput_tokens_s": sum(len(request["token_ids"]) for request in requests) / makespan,
            "request_latency_sum_s": sum(request["completed_at"] - request["arrived_at"] for request in requests),
            "mean_request_latency_s": statistics.mean(request["completed_at"] - request["arrived_at"] for request in requests),
            "p99_request_latency_s": percentile(latency, 0.99), "mean_ttft_s": statistics.mean(ttft),
            "p99_ttft_s": percentile(ttft, 0.99), "mean_tpot_s": statistics.mean(tpot),
            "p99_tpot_s": percentile(tpot, 0.99),
            "mean_queue_s": statistics.mean(request["submitted_at"] - request["arrived_at"] for request in requests),
            "arrival_mean_ttft_s": statistics.mean(request["first_token_at"] - request["arrived_at"] for request in arrivals),
            "arrival_a_ttft_s": next(request["first_token_at"] - request["arrived_at"] for request in arrivals if request["model"] == 0),
            "slo_deadline_s": deadline_s,
            "deadline_met_requests": sum(request["completed_at"] - request["arrived_at"] <= deadline_s for request in requests),
            "deadline_goodput_requests_s": sum(request["completed_at"] - request["arrived_at"] <= deadline_s for request in requests) / makespan,
            "maximum_chunk_gap_s": max((right["time"] - left["time"] for request in requests
                                         for left, right in zip(request["cadence"], request["cadence"][1:])), default=0)}


async def stream(engine, request_id, prompt, output_tokens, rank, model, role, arrived_at):
    from vllm import SamplingParams

    submitted_at = time.monotonic()
    cadence, latest, first_token_at = [], None, None
    async for output in engine.generate(
            {"prompt_token_ids": prompt}, SamplingParams(temperature=0, min_tokens=output_tokens,
                                                         max_tokens=output_tokens, ignore_eos=True),
            request_id, data_parallel_rank=rank):
        now = time.monotonic()
        latest = output
        count = len(output.outputs[0].token_ids)
        if count and first_token_at is None:
            first_token_at = now
        cadence.append({"time": now, "tokens": count})
    if latest is None or len(latest.outputs[0].token_ids) != output_tokens or first_token_at is None:
        raise RuntimeError("Request failed exact output and first-token check")
    return {"request_id": request_id, "model": model, "role": role, "rank_initial": rank,
            "prompt_tokens": len(prompt), "arrived_at": arrived_at, "submitted_at": submitted_at,
            "first_token_at": first_token_at, "completed_at": time.monotonic(),
            "token_ids": list(latest.outputs[0].token_ids), "cadence": cadence}


async def wait_empty(controller, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshots = await controller.snapshots()
        if all(not rank["requests"] for ranks in snapshots for rank in ranks):
            return snapshots
        await asyncio.sleep(0.05)
    raise TimeoutError("Previous workload still owns live requests")


async def submit_arrival(controller, engines, tasks, decisions, args, model, request_id, prompt,
                         arrived_at, budgets, policy):
    await asyncio.sleep(max(0, arrived_at - time.monotonic()))
    first_attempt, task = True, None
    deadline = arrived_at + args.request_timeout_s
    while time.monotonic() < deadline:
        observations = await controller.snapshots()
        active = [request for ranks in observations for rank in ranks for request in rank["requests"]]
        eligible = all(request["movable"] for request in active)
        used = [sum(observations[tenant][rank]["mapped_bytes"] for tenant in (0, 1)) / 1024**3 for rank in (0, 1)]
        direct = any(used[rank] + allocation_increment(observations[model][rank], len(prompt) + args.output_tokens)
                     <= budgets[rank] + 1e-6 for rank in (0, 1))
        can_plan = first_attempt and policy != "no_move"
        if eligible and (direct or can_plan):

            async def admit(rank):
                nonlocal task
                task = asyncio.create_task(stream(engines[model], request_id, prompt, args.output_tokens,
                                                   rank, model, "arrival", arrived_at))
                tasks.append(task)
                return await wait_reserved(controller, task, request_id, model, rank)

            decision = await controller.schedule_admission(
                len(prompt), args.output_tokens, budgets, admit, policy=policy, arriving_model=model,
                max_moves=args.max_moves, candidate_limit=args.candidate_limit,
                move_fixed_s=args.move_fixed_s, solver_limit_s=args.solver_limit_s,
                allow_migrations=first_attempt)
            decision["request_id"] = request_id
            decision["elapsed_from_arrival_s"] = time.monotonic() - arrived_at
            decisions.append(decision)
            if not decision.get("deferred", False):
                first_attempt = False
            if decision["admitted"] or decision.get("submitted", False):
                return await task
        await asyncio.sleep(args.queue_poll_s)
    raise TimeoutError("Queued arrival did not complete admission; no requests are silently dropped")


async def run_trial(engines, controller, args, prompts, references, policy, repeat, root):
    root.mkdir(parents=True, exist_ok=True)
    fixture = workload_fixture(args.scenario)
    await capture_control(controller, "start")
    origin = time.monotonic()
    tasks, arrivals, decisions = [], [], []
    counts = fixture["counts"]
    try:
        for model in (0, 1):
            for rank in (0, 1):
                for index in range(counts[model][rank]):
                    request_id = f"r{repeat}-{policy}-bootstrap-{model}-{rank}-{index}"
                    prompt = prompts[fixture["inputs"][model][rank][index]]
                    task = asyncio.create_task(stream(engines[model], request_id, prompt, args.output_tokens,
                                                      rank, model, "bootstrap", origin))
                    tasks.append(task)
                    if args.scenario != "capacity":
                        await wait_reserved(controller, task, request_id, model, rank)
        until = origin + 60
        while time.monotonic() < until:
            observed = await controller.snapshots()
            requests = [request for ranks in observed for rank in ranks for request in rank["requests"]]
            if ([[len(rank["requests"]) for rank in ranks] for ranks in observed] == counts
                    and all(request["movable"] and request["output_tokens"] >= args.minimum_decode_tokens for request in requests)):
                break
            if any(task.done() for task in tasks):
                raise RuntimeError("Bootstrap finished before the measured arrival boundary")
            await asyncio.sleep(0.02)
        else:
            raise TimeoutError("Bootstrap did not reach the controlled decode boundary")
        boundary = time.monotonic()
        if [[rank["mapped_bytes"] // rank["page_bytes"] for rank in ranks] for ranks in observed] != fixture["pages"]:
            raise RuntimeError("Fresh paired trial did not restore the predeclared allocator geometry")
        if not any(not page["pinned"] and len(page["owners"]) == 1 for page in observed[1][0]["pages"]):
            raise RuntimeError("B rank0 lacks an exclusive releasable page")
        if args.scenario != "capacity" and not any(not page["pinned"] and len(page["owners"]) == 1
                                                   for page in observed[0][0]["pages"]):
            raise RuntimeError("Tradeoff A rank0 lacks its exclusive short-request page")
        budgets = [(sum(observed[model][rank]["mapped_bytes"] for model in (0, 1))
                    + fixture["headroom"][rank] * observed[0][rank]["page_bytes"]) / 1024**3 for rank in (0, 1)]
        diagnostic = None
        if args.scenario == "long_kv":
            diagnostic, observed = await long_kv_preflight(controller, fixture, prompts[fixture["arrival_input_tokens"]],
                                                         args.output_tokens, args.minimum_decode_tokens, budgets)
            save(root / "capacity-witness.json", diagnostic)
            boundary = time.monotonic()
        save(root / "before.json", observed)
        save(root / "workload.json", {"policy": policy, "repeat": repeat, "origin": origin, "arrival_boundary": boundary,
                                     "counts": counts, "capacity_gib": budgets, "scenario": args.scenario,
                                     "bootstrap_inputs": fixture["inputs"], "total_requests": fixture["total_requests"],
                                     "arrival_streams": [{"model": 0, "input_tokens": fixture["arrival_input_tokens"], "offset_s": 0},
                                                         {"model": 1, "input_tokens": fixture["b_arrival_input_tokens"], "offset_s": args.b_arrival_delay_s}]})
        for model, prompt, offset in ((0, prompts[fixture["arrival_input_tokens"]], 0),
                                     (1, prompts[fixture["b_arrival_input_tokens"]], args.b_arrival_delay_s)):
            request_id = f"r{repeat}-{policy}-arrival-{model}"
            arrivals.append(asyncio.create_task(submit_arrival(controller, engines, tasks, decisions, args, model,
                                                               request_id, prompt, boundary + offset, budgets, policy)))
        await asyncio.wait_for(asyncio.gather(*arrivals), timeout=args.request_timeout_s + 30)
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=args.request_timeout_s)
        if len(results) != fixture["total_requests"]:
            raise RuntimeError("Paired workload must complete every predeclared request")
        for request in results:
            reference = references[request["prompt_tokens"]]
            if request["token_ids"] != reference["token_ids"]:
                raise RuntimeError("Output token IDs differ from the matching prompt's serial reference")
        save(root / "requests.json", results)
        save(root / "decisions.json", decisions)
        await wait_empty(controller)
        save(root / "capture.json", await capture_control(controller, "stop"))
        report = summarize_requests(results, origin, args.deadline_s)
        report.update({"status": "passed", "policy": policy, "repeat": repeat, "scenario": args.scenario,
                       "token_matches": len(results),
                       "migration_count": len(controller.events), "moved_bytes": sum(event["kv_bytes"] for event in controller.events),
                       "diagnostic_control_s": diagnostic["diagnostic_control_s"] if diagnostic else 0,
                       "control_s": sum(decision["end_to_end_control_s"] for decision in decisions),
                       "solve_s": sum(decision["solve_s"] for decision in decisions),
                       "scope": "paired_two_stream_controlled_arrival_end_to_end_including_queue_and_host_staged_migration"})
        save(root / "summary.json", report)
        return report
    finally:
        for task in arrivals + tasks:
            if not task.done():
                task.cancel()
        if arrivals or tasks:
            await asyncio.gather(*(arrivals + tasks), return_exceptions=True)


async def run(args):
    from transformers import AutoTokenizer
    from vllm.v1.engine.async_llm import AsyncLLM

    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    save(root / "config.json", {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()})
    save(root / "runtime-control.json", {"dp_pause_sync_interval": int(os.environ.get("DP_OPT_PAUSE_SYNC_INTERVAL", "32")),
                                          "same_interval_for_all_policies": True,
                                          "idle_rank_expert_execution_not_disabled": True})
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    prefix = tokenizer.encode("Calculate 2 + 2. The answer is", add_special_tokens=False)
    fixture = workload_fixture(args.scenario)
    lengths = sorted({64, fixture["arrival_input_tokens"], fixture["b_arrival_input_tokens"]}
                     | {length for ranks in fixture["inputs"] for rank in ranks for length in rank})
    prompts = {length: (prefix * ((length + len(prefix) - 1) // len(prefix)))[:length] for length in lengths}
    fit = json.loads(args.fit.read_text())
    engines, reports = [], []
    os.environ["DP_OPT_EVIDENCE_ROOT"] = str(root)
    os.environ["DP_OPT_CAPTURE_EVIDENCE"] = "1"
    try:
        for model in (0, 1):
            os.environ["DP_OPT_SERVICE"] = f"instance-{model + 1}"
            base = f"benefit{os.environ['SLURM_JOB_ID']}_s{model}"
            if os.environ.get("DP_OPT_BENEFIT_CASE"):
                base += "_" + os.environ["DP_OPT_BENEFIT_CASE"]
            os.environ["KVCACHED_IPC_NAME"] = base
            os.environ["KVCACHED_DP_IPC_BASE"] = base
            engines.append(AsyncLLM.from_engine_args(engine_arguments(
                args.checkpoint, args.port + model * 10, args.cache_bytes, args.max_model_len)))
        controller = PlacementController(engines, fit, os.environ["DP_OPT_TRANSFER_ROOT"], allow_extrapolation=True)
        workers = [[json.loads((root / f"worker-instance-{model + 1}-dp{rank}.json").read_text())
                    for rank in (0, 1)] for model in (0, 1)]
        if len({worker["ipc_name"] for ranks in workers for worker in ranks}) != 4:
            raise RuntimeError("All tenant/rank KV namespaces must be independent")
        for ranks in workers:
            if ranks[0]["gpu_uuid"] == ranks[1]["gpu_uuid"]:
                raise RuntimeError("Attention DP must span two physical GPUs")
            for worker in ranks:
                if any(worker["groups"][name]["world_size"] != size for name, size in (("tp", 1), ("dp", 2), ("ep", 2))):
                    raise RuntimeError("Incorrect worker TP/DP/EP topology")
                if len(worker["expert_ids"]) != 48 or worker["parameter_bytes"] <= 0:
                    raise RuntimeError("Missing MoE expert layers or resident parameters")
            for left, right in zip(ranks[0]["expert_ids"], ranks[1]["expert_ids"]):
                if set(left) & set(right) or set(left) | set(right) != set(range(128)):
                    raise RuntimeError("EP workers do not hold complementary expert partitions")
        if any(workers[0][rank]["gpu_uuid"] != workers[1][rank]["gpu_uuid"] for rank in (0, 1)):
            raise RuntimeError("Tenant ranks must be colocated on matching physical GPUs")
        from ilp_reference import enumerate_targets
        warm_snapshot = {"counts": [[0, 0], [0, 0]], "initial_gib": [0, 0], "capacity_gib": [1, 1],
                         "new_gib": 0.1, "source_pages": [], "movable": [],
                         "performance_cost_tables_s": [[0, 0], [0]]}
        await asyncio.to_thread(enumerate_targets, warm_snapshot, max_moves=0, time_limit_s=args.solver_limit_s)
        for model in (0, 1):
            for rank in (0, 1):
                await stream(engines[model], f"warmup-{model}-{rank}", prompts[64], args.output_tokens,
                             rank, model, "warmup", time.monotonic())
        references = {}
        for length, prompt in prompts.items():
            references[length] = await stream(engines[0], f"reference-{length}", prompt, args.output_tokens, 0, 0,
                                              "reference", time.monotonic())
        save(root / "references.json", list(references.values()))
        await wait_empty(controller)
        if args.control_warmup:
            warmup_started = time.monotonic()
            await controller.pause()
            await controller.resume()
            warm = [asyncio.create_task(stream(engines[model], f"control-warmup-{model}-{rank}", prompts[64],
                                               args.output_tokens, rank, model, "warmup", time.monotonic()))
                    for model in (0, 1) for rank in (0, 1)]
            try:
                await asyncio.sleep(.5)
                paused = time.monotonic()
                await controller.pause()
                await controller.resume()
                save(root / "control-warmup.json", {"active_pause_resume_s": time.monotonic() - paused,
                                                     "started_at": warmup_started})
                await asyncio.gather(*warm)
            finally:
                for task in warm:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*warm, return_exceptions=True)
            await wait_empty(controller)
        order = []
        generator = random.Random(args.seed)
        for repeat in range(args.repeats):
            policies = list(POLICIES)
            generator.shuffle(policies)
            order.extend({"repeat": repeat, "policy": policy} for policy in policies)
        save(root / "order.json", order)
        for trial in order:
            controller.events.clear()
            await wait_empty(controller)
            report = await run_trial(engines, controller, args, prompts, references, trial["policy"], trial["repeat"],
                                     root / f"r{trial['repeat']}-{trial['policy']}")
            reports.append(report)
            save(root / "progress.json", {"completed": len(reports), "total": len(order), "last": report,
                                          "summaries": reports})
        save(root / "summary.json", {"status": "passed", "trials": reports,
                                     "cost_scope": "N128_single_model_surrogate_population_extrapolation_not_concurrent_calibration"})
    except BaseException as error:
        save(root / "error.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        for engine in reversed(engines):
            engine.shutdown()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenario", choices=("capacity", "tradeoff", "tradeoff_long", "long_kv"), default="capacity")
    parser.add_argument("--control-warmup", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--cache-bytes", type=int, default=DEFAULT_KV_CACHE_BYTES)
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--minimum-decode-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-moves", type=int, default=2)
    parser.add_argument("--candidate-limit", type=int, default=2)
    parser.add_argument("--move-fixed-s", type=float, default=0.475)
    parser.add_argument("--solver-limit-s", type=float, default=1)
    parser.add_argument("--queue-poll-s", type=float, default=0.2)
    parser.add_argument("--b-arrival-delay-s", type=float, default=1)
    parser.add_argument("--request-timeout-s", type=float, default=180)
    parser.add_argument("--deadline-s", type=float, default=20)
    args = parser.parse_args(argv)
    fixture = workload_fixture(args.scenario)
    prompt_lengths = [tokens for ranks in fixture["inputs"] for rank in ranks for tokens in rank]
    prompt_lengths.extend((fixture["arrival_input_tokens"], fixture["b_arrival_input_tokens"]))
    if (min(args.max_model_len, args.cache_bytes) <= 0
            or max(prompt_lengths) + fixture["output_tokens"] > args.max_model_len):
        parser.error("Max-model-len/cache must be positive and cover the declared fixture length")
    if (args.output_tokens != workload_fixture(args.scenario)["output_tokens"] or args.repeats < 1 or not 1 <= args.minimum_decode_tokens < args.output_tokens
            or args.candidate_limit < 1 or args.max_moves < 0 or args.b_arrival_delay_s < 0
            or min(args.solver_limit_s, args.queue_poll_s, args.request_timeout_s, args.deadline_s) <= 0):
        parser.error("Output tokens must match the declared fixture; repeats and timing limits must be positive")
    return args


def main():
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
