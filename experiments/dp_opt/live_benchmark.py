import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from live_controller import PlacementController


DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_KV_CACHE_BYTES = 4294967296


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def initial_counts(population, scenario):
    if scenario == "pressure":
        if population != 8:
            raise ValueError("Page-owner pressure witness requires population8")
        return [[3, 3], [4, 3]]
    return [[population // 2, population - population // 2 - 1],
            [population * 5 // 8, population - population * 5 // 8]]


def ownership_witness(observations):
    return [{"model": model, "rank": rank, "page": page["id"], "owners": page["owners"],
             "memory_gib": page["memory_gib"]}
            for model, ranks in enumerate(observations) for rank, observation in enumerate(ranks)
            for page in observation["pages"] if not page["pinned"] and page["owners"]]


def validate_probe_result(args, result):
    feasible = [probe["rank"] for probe in result["probes"] if probe["fits_budget"]]
    if args.scenario == "packing":
        if feasible != [0] or result["probes"][0]["additional_mapped_bytes"] != 192 * 1024**2:
            raise RuntimeError("Original packing witness differs from the frozen expected geometry")
    elif args.scenario == "pressure":
        if feasible:
            raise RuntimeError("Pressure setup allows zero-move admission under actual allocation")
        pages = result["releasable_pages"]
        if (len(pages) != 1 or pages[0]["model"] != 1 or pages[0]["rank"] != 0
                or len(pages[0]["owners"]) != 1):
            raise RuntimeError("Pressure witness must have only one releasable B/rank0 exclusive page")
        if any(value <= cap for value, cap in zip(result["no_b_migration_lower_bound_gib"], result["capacity_gib"])):
            raise RuntimeError("A-only reordering can satisfy the pressure budget lower bound")


async def allocation_preflight(controller, args, prompt, counts, admit):
    async with controller.lock:
        transaction = "allocator-" + str(time.monotonic_ns())
        started = time.monotonic()
        await controller.pause()
        completed = False
        try:
            observations = await controller.begin(transaction)
            if [[len(rank["requests"]) for rank in model] for model in observations] != counts:
                raise RuntimeError("Population changed before allocator witness")
            budgets = [(sum(observations[model][rank]["mapped_bytes"] for model in (0, 1))
                        + (2 if args.scenario == "packing" or rank == 0 else 1)
                        * observations[0][rank]["page_bytes"]) / 1024**3 for rank in (0, 1)]
            probes = [await controller.probe(transaction, 0, rank, prompt, args.output_tokens, budgets)
                      for rank in (0, 1)]
            result = {"observations": observations, "probes": probes, "capacity_gib": budgets,
                      "releasable_pages": ownership_witness(observations), "admitted_while_paused": False,
                      "scope": "temporary_actual_allocations_then_rollback_controlled_kv_caps_not_device_oom"}
            if args.scenario == "pressure":
                blocks = (len(prompt) + args.output_tokens + observations[0][0]["block_size"] - 1) // observations[0][0]["block_size"]
                minimum_a_pages = (blocks + 1 + observations[0][0]["page_block_capacity"] - 1) // observations[0][0]["page_block_capacity"]
                result["no_b_migration_lower_bound_gib"] = [
                    (minimum_a_pages * observations[0][rank]["page_bytes"]
                     + observations[1][rank]["mapped_bytes"]) / 1024**3 for rank in (0, 1)]
            save(args.output / "allocator-probes.json", result)
            validate_probe_result(args, result)
            if args.policy == "no_move":
                feasible = [probe for probe in probes if probe["fits_budget"]]
                if feasible:
                    target = min(feasible, key=lambda probe: (probe["allocation_peak_gib"][probe["rank"]],
                                                            probe["rank"]))["rank"]
                    result["before_admission"] = await controller.snapshots()
                    await admit(target)
                    await controller.check_capacity(budgets)
                    result.update({"target": target, "admitted_while_paused": True})
                else:
                    result["target"] = None
            await controller.end(transaction)
            completed = True
            result["pause_s"] = time.monotonic() - started
            return result
        finally:
            if completed:
                await controller.resume()


def validate_capacity_handoff(before, after_migrations, after, moves, arrival_id):
    source_releases = [{"model": event["model"], "rank": event["source"],
                       "released_bytes": before[event["model"]][event["source"]]["mapped_bytes"]
                       - event["source_mapped_after"]} for event in moves]
    if not any(item["model"] == 1 and item["released_bytes"] >= 96 * 1024**2 for item in source_releases):
        raise RuntimeError("Pressure admission did not release a B source page")
    arrival = next((model, rank, request) for model, ranks in enumerate(after)
                   for rank, observation in enumerate(ranks) for request in observation["requests"]
                   if request["id"].rsplit("-", 1)[0] == arrival_id)
    model, rank, request = arrival
    if model != 0 or rank != 0:
        raise RuntimeError("Cross-model capacity witness must admit A on B's source GPU")
    if after[0][rank]["mapped_bytes"] <= after_migrations[0][rank]["mapped_bytes"]:
        raise RuntimeError("Arrival did not map new A pages after B source release")
    return {"status": "cross_tenant_mapped_capacity_reclamation_verified",
            "source_releases": source_releases, "arrival_rank": rank,
            "arrival_blocks": request["blocks"],
            "arrival_new_a_mapped_bytes": after[0][rank]["mapped_bytes"] - after_migrations[0][rank]["mapped_bytes"],
            "scope": "controlled_aggregate_kv_budget_capacity_reuse_not_identical_cuda_physical_page_id_or_oom"}


async def generate(engine, request_id, prompt, output_tokens, rank):
    from vllm import SamplingParams

    started = time.monotonic()
    cadence = []
    latest = None
    async for output in engine.generate(
            {"prompt_token_ids": prompt}, SamplingParams(temperature=0, max_tokens=output_tokens,
                                                         min_tokens=output_tokens, ignore_eos=True),
            request_id, data_parallel_rank=rank):
        latest = output
        cadence.append({"time": time.monotonic(), "tokens": len(output.outputs[0].token_ids)})
    if latest is None or len(latest.outputs[0].token_ids) != output_tokens:
        raise RuntimeError("Request failed exact output-length check")
    return {"request_id": request_id, "rank_initial": rank, "latency_s": time.monotonic() - started,
            "token_ids": list(latest.outputs[0].token_ids), "cadence": cadence}


def engine_arguments(checkpoint, port, cache_bytes, max_model_len=DEFAULT_MAX_MODEL_LEN):
    from vllm.engine.arg_utils import AsyncEngineArgs

    return AsyncEngineArgs(model=checkpoint, dtype="bfloat16", tensor_parallel_size=1,
                           data_parallel_size=2, data_parallel_size_local=2, data_parallel_backend="mp",
                           data_parallel_address="127.0.0.1", data_parallel_rpc_port=port,
                           enable_expert_parallel=True, enable_ep_weight_filter=True,
                           all2all_backend="allgather_reducescatter", moe_backend="triton",
                           worker_cls="worker.VerificationWorker", scheduler_cls="live_scheduler.PlacementScheduler",
                           max_model_len=max_model_len, max_num_seqs=256, max_num_batched_tokens=4096,
                           kv_cache_memory_bytes=cache_bytes, gpu_memory_utilization=0.43,
                           enable_prefix_caching=False, async_scheduling=False, enable_dbo=False, enable_eplb=False,
                           compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                               "cudagraph_capture_sizes": [1, 8, 16, 32, 64, 128, 256]})


async def run(args):
    from vllm.v1.engine.async_llm import AsyncLLM
    from transformers import AutoTokenizer

    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    save(root / "config.json", {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()})
    os.environ["DP_OPT_EVIDENCE_ROOT"] = str(root)
    fit = json.loads(args.fit.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    prefix = tokenizer.encode("Calculate 2 + 2. The answer is", add_special_tokens=False)
    prompt = (prefix * ((args.input_tokens + len(prefix) - 1) // len(prefix)))[:args.input_tokens]
    arriving_prompt = (prefix * ((args.arrival_input_tokens + len(prefix) - 1) // len(prefix)))[:args.arrival_input_tokens]
    engines, tasks = [], []
    try:
        for model in (0, 1):
            os.environ["DP_OPT_SERVICE"] = f"instance-{model + 1}"
            base = f"live{os.environ['SLURM_JOB_ID']}_s{model}"
            os.environ["KVCACHED_IPC_NAME"] = base
            os.environ["KVCACHED_DP_IPC_BASE"] = base
            engines.append(AsyncLLM.from_engine_args(engine_arguments(
                args.checkpoint, args.port + model * 10, args.cache_bytes, args.max_model_len)))
        workers = [[json.loads((root / f"worker-instance-{model + 1}-dp{rank}.json").read_text())
                    for rank in (0, 1)] for model in (0, 1)]
        names = {worker["ipc_name"] for model in workers for worker in model}
        if len(names) != 4:
            raise RuntimeError("Each tenant/DP rank must have an independent KV namespace")
        for model in workers:
            if model[0]["gpu_uuid"] == model[1]["gpu_uuid"]:
                raise RuntimeError("The DP group does not span two physical GPUs")
            for worker in model:
                if any(worker["groups"][name]["world_size"] != size for name, size in (("tp", 1), ("dp", 2), ("ep", 2))):
                    raise RuntimeError("Worker TP/DP/EP topology differs from the requested experiment")
        if any(workers[0][rank]["gpu_uuid"] != workers[1][rank]["gpu_uuid"] for rank in (0, 1)):
            raise RuntimeError("Tenant DP ranks are not colocated on matching GPUs")
        controller = PlacementController(engines, fit, os.environ["DP_OPT_TRANSFER_ROOT"], args.allow_uncalibrated)
        save(root / "ready.json", {"services": 2, "dp": 2, "ep": 2, "attention_tp": 1,
                                   "capture_sizes": [1, 8, 16, 32, 64, 128, 256]})
        serial = await generate(engines[0], "serial-reference", prompt, args.output_tokens, 0)
        save(root / "serial-reference.json", serial)
        arrival_serial = None
        if args.mode != "migration":
            arrival_serial = await generate(engines[0], "arrival-serial-reference", arriving_prompt,
                                            args.output_tokens, 0)
            save(root / "arrival-serial-reference.json", arrival_serial)
        counts = initial_counts(args.population, args.scenario)
        for model in (0, 1):
            for rank in (0, 1):
                for request_index in range(counts[model][rank]):
                    tasks.append(asyncio.create_task(generate(
                        engines[model], f"live-{model}-{rank}-{request_index}", prompt, args.output_tokens, rank)))
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            observations = await controller.snapshots()
            observed_counts = [[len(rank["requests"]) for rank in model] for model in observations]
            if observed_counts == counts and all(request["movable"] for model in observations
                                                  for rank in model for request in rank["requests"]):
                break
            if any(task.done() for task in tasks):
                raise RuntimeError("Request finished or failed before placement snapshot")
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError("Live population did not reach all-decode placement window")
        save(root / "before.json", observations)
        if args.mode == "migration":
            await controller.pause()
            transaction = f"smoke-{os.environ['SLURM_JOB_ID']}"
            snapshots = await controller.begin(transaction)
            events = []
            for model in (0, 1):
                request = next(item for item in snapshots[model][0]["requests"] if item["movable"])
                events.append(await controller.migrate(transaction, {"id": request["id"], "model": model, "source": 0}))
            save(root / "migration.json", events)
            save(root / "after.json", await controller.snapshots())
            await controller.end(transaction)
            await controller.resume()
            placement = None
        else:
            async def admit(target):
                task = asyncio.create_task(generate(engines[0], "arriving-A", arriving_prompt,
                                                    args.output_tokens, target))
                tasks.append(task)
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    observed = await controller.snapshots()
                    if sum(len(rank["requests"]) for rank in observed[0]) == sum(counts[0]) + 1:
                        return
                    if task.done():
                        await task
                        raise RuntimeError("Arrival unexpectedly finished during paused admission")
                    await asyncio.sleep(0.01)
                raise TimeoutError("Arrival was not reserved by the paused scheduler")

            preflight = None
            if args.mode == "allocator":
                preflight = await allocation_preflight(controller, args, arriving_prompt, counts, admit)
                save(root / "allocator-probes.json", preflight)
            if args.mode == "allocator" and args.policy == "no_move":
                placement = {"plan": None, "moves": [], "preflight": preflight,
                             "admitted_while_paused": preflight["admitted_while_paused"]}
            else:
                placement = await controller.place(len(arriving_prompt) + args.output_tokens,
                                                    budgets_gib=preflight["capacity_gib"] if preflight else None,
                                                    policy=args.policy, required_counts=counts,
                                                    headroom_pages=args.headroom_pages, admit=admit)
            save(root / "placement.json", placement)
            after = await controller.snapshots()
            save(root / "after.json", after)
            if args.mode == "allocator" and args.scenario == "pressure":
                admitted = placement["admitted_while_paused"]
                if admitted != (args.policy in ("ilp", "kv_only")):
                    raise RuntimeError("Pressure policy admission differs from the intended capacity witness")
                if admitted:
                    save(root / "capacity-handoff.json", validate_capacity_handoff(
                        placement["observations"], placement["before_admission"], after,
                        placement["moves"], "arriving-A"))
        results = await asyncio.gather(*tasks)
        save(root / "requests.json", results)
        originals = [result for result in results if result["request_id"].startswith("live-")]
        matches = [result["token_ids"] == serial["token_ids"] for result in originals]
        arrivals = [result for result in results if result["request_id"] == "arriving-A"]
        arrival_matches = [result["token_ids"] == arrival_serial["token_ids"] for result in arrivals]
        no_plan = args.mode == "admission" and placement["plan"]["winner"] is None
        status = ("token_mismatch" if not all(matches + arrival_matches)
                  else "no_feasible_plan" if no_plan else "passed")
        report = {"status": status, "mode": args.mode,
                  "policy": args.policy, "initial_counts": counts, "input_tokens": args.input_tokens,
                  "output_tokens": args.output_tokens, "strict_token_matches": sum(matches),
                  "scenario": args.scenario, "arrival_strict_token_matches": sum(arrival_matches),
                  "arrival_requests": len(arrivals),
                  "admitted_while_paused": bool(placement and placement["admitted_while_paused"]),
                  "admission_outcome": ("not_applicable" if args.mode == "migration" else
                                        "admitted" if placement["admitted_while_paused"] else
                                        "rejected_controlled_capacity"),
                  "original_requests": len(originals), "migration_count": len(controller.events),
                  "moves": controller.events, "allow_uncalibrated": args.allow_uncalibrated,
                  "calibration_scope": "feasibility_only_single_model_coefficients_extrapolated",
                  "scope": "Live host-staged KV transfer; not optimized transport or validated concurrent costs"}
        save(root / "summary.json", report)
        if not all(matches + arrival_matches):
            raise RuntimeError("Concurrent/migrated token outputs differ from serial reference")
        if no_plan and args.policy == "ilp":
            raise RuntimeError("ILP admission found no feasible incumbent")
        print(json.dumps(report, indent=2))
    except BaseException as error:
        save(root / "error.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for engine in reversed(engines):
            engine.shutdown()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("migration", "admission", "allocator"), default="migration")
    parser.add_argument("--scenario", choices=("packing", "pressure"), default="packing")
    parser.add_argument("--policy", choices=("ilp", "own_model", "kv_only", "no_move"), default="ilp")
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--input-tokens", type=int, default=64)
    parser.add_argument("--arrival-input-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--headroom-pages", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--cache-bytes", type=int, default=DEFAULT_KV_CACHE_BYTES)
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--allow-uncalibrated", action="store_true")
    args = parser.parse_args(argv)
    if args.population < 8 or args.population % 8:
        parser.error("Population must be a positive multiple of eight")
    if (min(args.input_tokens, args.arrival_input_tokens, args.output_tokens,
            args.cache_bytes, args.max_model_len) <= 0
            or args.headroom_pages < 0 or args.population > 256
            or max(args.input_tokens, args.arrival_input_tokens) + args.output_tokens > args.max_model_len):
        parser.error("Lengths/cache must be positive, headroom nonnegative, population <=256, "
                     "total length <=max-model-len")
    if args.scenario == "pressure" and (args.population != 8 or args.input_tokens != 64
                                       or args.output_tokens != 256 or args.arrival_input_tokens != 2816
                                       or args.mode != "allocator"):
        parser.error("Pressure witness requires allocator mode, population8 and lengths64/2816/256")
    if args.mode == "allocator" and args.scenario == "packing" and args.policy != "no_move":
        parser.error("Packing allocator control directly admits without migration; select no_move")
    if args.mode == "allocator" and args.scenario == "packing" and (args.population != 8 or args.input_tokens != 64
                                                                   or args.output_tokens != 256 or args.arrival_input_tokens != 2048):
        parser.error("Packing witness requires population8 and lengths64/2048/256")
    if args.mode in ("admission", "allocator") and not args.allow_uncalibrated:
        parser.error("Concurrent/length/device costs are uncalibrated; explicitly enable feasibility-only override")
    return args


def main():
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
