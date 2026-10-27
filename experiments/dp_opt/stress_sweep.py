import asyncio
import hashlib
import itertools
import json
import math
import random
import shutil
import statistics
import time
from pathlib import Path

from live_benchmark import save
from live_benefit import capture_control, stream, wait_empty
from live_controller import stable_ownership
from live_trace import prepare_prompts, request_prompt, run_capacity_pilot, run_trace
from stress_trace import long_output_marks, make_stress_trace, rate_from_profile


def runtime_copy(source, destination, args, trace, capacity=False):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("runtime-control.json", "source.sha256", "placement-fit.json", "hardware.txt", "topology.txt"):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    for path in source.glob("worker-instance-*-dp*.json"):
        shutil.copy2(path, destination / path.name)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in ("prepared_prompts", "prompt_bank_data")}
    config["capacity_pilot"] = capacity
    save(destination / "config.json", config)
    save(destination / "trace.json", trace)
    with (destination / "source.sha256").open("a") as output:
        output.write(hashlib.sha256((destination / "trace.json").read_bytes()).hexdigest()
                     + "  " + str(destination / "trace.json") + "\n")
    control_path = destination / "runtime-control.json"
    if not capacity and control_path.exists():
        control = json.loads(control_path.read_text())
        control.update({"wait_cost_scope": "wait_aware_disabled_current_allocation_measured_migration_cost",
                        "move_cost_scope": "same_allocation_long_output_joint_profile",
                        "performance_cost_scope": "historical_single_model_surrogate_not_joint_long_output_fit",
                        "old_layout_lookup_reused": False})
        save(control_path, control)


async def collect_profile(tasks, controller, root, stage, timeout):
    deadline = time.monotonic() + timeout
    try:
        while not all(task.done() for task in tasks):
            if any(task.done() and task.exception() is not None for task in tasks):
                await asyncio.gather(*tasks)
            if time.monotonic() >= deadline:
                raise TimeoutError("Long-output profile exceeded declared request timeout")
            save(root / "progress.json", {"stage": stage, "completed": sum(task.done() for task in tasks),
                                          "requests": len(tasks), "observed_at": time.monotonic()})
            await asyncio.sleep(1)
        results = await asyncio.gather(*tasks)
        await wait_empty(controller)
        return results
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def wait_decode(controller, tasks, population, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        observed = await controller.snapshots()
        active = [request for rank in observed[1] for request in rank["requests"]]
        if len(active) == population and all(request["movable"] and request["output_tokens"] >= 64 for request in active):
            return observed
        if any(task.done() for task in tasks):
            await asyncio.gather(*tasks)
            raise RuntimeError("Long-output profile completed before migration boundary")
        await asyncio.sleep(.05)
    raise TimeoutError("Conv population never reached all-decode boundary")


async def profile_output(engines, controller, args, root, output_tokens, conv_marks, code_marks, tokenizer):
    directory = root / "profiles" / str(output_tokens)
    directory.mkdir(parents=True)
    representative = min(conv_marks, key=lambda mark: abs(mark["prompt_tokens"] - 3949))
    conv = long_output_marks([representative], output_tokens)[0]
    code = max(code_marks, key=lambda mark: mark["prompt_tokens"] + mark["output_tokens"])
    prompts = prepare_prompts({"requests": [conv, code]}, tokenizer, args.prompt_bank_data)
    conv_prompt, code_prompt = request_prompt(prompts, conv), request_prompt(prompts, code)
    budgets = [min(args.stress_budgets) / 1024] * 2
    save(directory / "setting.json", {"Conv": conv, "Code": code, "population": 4,
                                      "budgets_gib": budgets, "scope": "finite_population_joint_cost_pilot_not_trace_benefit"})
    await wait_empty(controller)
    await capture_control(controller, "start")
    references = []
    for model, prompt, output in ((1, conv_prompt, output_tokens), (0, code_prompt, code["output_tokens"])):
        references.append(await stream(engines[model], f"stress-ref-{output_tokens}-{model}", prompt,
                                       output, 0, model, "stress-profile-reference", time.monotonic()))
        await wait_empty(controller)
        save(directory / "references.json", references)
    trials = []
    for layout in ("balanced", "consolidated", "migrated"):
        await controller.release_transfer_cache()
        origin = time.monotonic()
        tasks = [asyncio.create_task(stream(engines[1], f"stress-profile-{output_tokens}-{layout}-{index}",
                                           conv_prompt, output_tokens, 1 if layout == "consolidated" else index % 2,
                                           1, "stress-profile", origin)) for index in range(4)]
        trial = {"layout": layout, "Conv": conv, "Code": code, "budgets_gib": budgets}
        try:
            observed = await wait_decode(controller, tasks, 4, args.request_timeout_s)
            await controller.check_capacity(budgets)
            trial["decode_ready"] = observed
            if layout == "migrated":
                started = time.monotonic()
                trial["pause_receipts"] = await controller.pause_boundary(observed)
                transaction = f"stress-profile-{output_tokens}"
                try:
                    before = await controller.begin(transaction)
                    trial["before"] = before
                    probes = [await controller.probe(transaction, 0, rank, code_prompt, code["output_tokens"], budgets)
                              for rank in (0, 1)]
                    trial["probes_before"] = probes
                    if stable_ownership(before) != stable_ownership(await controller.snapshots()):
                        raise RuntimeError("Code allocation/rollback changed profile ownership")
                    events = []
                    for request in before[1][0]["requests"]:
                        events.append(await controller.migrate(transaction, {"id": request["id"], "model": 1,
                                      "source": 0, "memory_gib": math.ceil(len(request["blocks"])
                                      * before[1][0]["block_bytes"] / before[1][0]["page_bytes"])
                                      * before[1][0]["page_bytes"] / 1024**3}, budgets))
                        await controller.release_transfer_cache()
                    after = await controller.snapshots(refresh_physical_capacity=True)
                    trial.update({"events": events, "after": after,
                                  "probe_after": await controller.probe(transaction, 0, 0, code_prompt,
                                                                       code["output_tokens"], budgets)})
                    if after[1][0]["requests"] or not trial["probe_after"]["fits_budget"]:
                        raise MemoryError("Consolidated Conv failed to free Code target rank")
                    await controller.end(transaction)
                finally:
                    await controller.resume()
                trial["control_s"] = time.monotonic() - started
            if layout != "balanced":
                tasks.append(asyncio.create_task(stream(engines[0], f"stress-profile-{output_tokens}-{layout}-code",
                                                       code_prompt, code["output_tokens"], 0, 0,
                                                       "stress-profile", time.monotonic())))
            save(directory / f"{layout}-inflight.json", trial)
            results = await collect_profile(tasks, controller, root, f"profile-{output_tokens}-{layout}", args.request_timeout_s)
            for result in results:
                reference = references[0 if result["model"] == 1 else 1]
                if result["token_ids"] != reference["token_ids"]:
                    raise RuntimeError("Long-output mixed/migrated tokens differ from serial reference")
            trial["requests"] = results
            trials.append(trial)
            save(directory / "trials.json", trials)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    graph = await capture_control(controller, "stop")
    service = statistics.mean(request["completed_at"] - request["submitted_at"] for request in trials[0]["requests"])
    rate = rate_from_profile(conv_marks, output_tokens, service, args.stress_target_conv_mib)
    result = {"status": "profile_passed", "rate": rate, "trials": trials, "graph": graph,
              "Code_serial_service_s": references[1]["completed_at"] - references[1]["submitted_at"],
              "Code_mixed_service_s": trials[1]["requests"][-1]["completed_at"] - trials[1]["requests"][-1]["submitted_at"],
              "scope": "finite_population_actual_migration_and_mixed_cost_not_open_loop_net_benefit"}
    save(directory / "summary.json", result)
    return result


async def run_case(engines, controller, args, tokenizer, root, trace, setting, index):
    directory = root / "cases" / f"case-{index}"
    args.output = directory
    args.kv_budget_mib = setting["budget_mib_per_gpu"]
    args.trace = directory / "trace.json"
    args.prepared_prompts = prepare_prompts(trace, tokenizer, args.prompt_bank_data)
    runtime_copy(root, directory, args, trace)
    save(directory / "stress-setting.json", setting)
    await wait_empty(controller)
    references = []
    marks = sorted(trace["requests"], key=lambda mark: mark["prompt_tokens"])
    selected = {0, len(marks) // 3, 2 * len(marks) // 3, len(marks) - 1}
    warmed_code = set()
    for position, mark in enumerate(marks):
        if mark["model"] == 0 and mark["prompt_key"] not in warmed_code:
            selected.add(position)
            warmed_code.add(mark["prompt_key"])
    rank_warmups = []
    for position in sorted(selected):
        mark = marks[position]
        request = await stream(engines[mark["model"]], f"stress-case-ref-{index}-{position}",
                               request_prompt(args.prepared_prompts, mark), mark["output_tokens"], 0,
                               mark["model"], "reference", time.monotonic())
        request.update({"trace_request_id": mark["request_id"]})
        references.append(request)
        save(directory / "references.json", references)
        await wait_empty(controller)
        if mark["model"] == 0:
            warmed = await stream(engines[0], f"stress-case-rank1-warm-{index}-{position}",
                                 request_prompt(args.prepared_prompts, mark), mark["output_tokens"], 1, 0,
                                 "uncached_Code_rank_warmup", time.monotonic())
            if warmed["token_ids"] != request["token_ids"]:
                raise RuntimeError("Code rank warmup differs from full serial reference")
            warmed["trace_request_id"] = mark["request_id"]
            rank_warmups.append(warmed)
            save(directory / "Code-rank1-warmups.json", rank_warmups)
            await wait_empty(controller)
    policies = list(args.policies)
    random.Random(args.seed + index).shuffle(policies)
    order = [{"repeat": 0, "policy": policy} for policy in policies]
    save(directory / "order.json", order)
    summaries = []
    for policy in policies:
        save(root / "phase.json", {"stage": "stress_trace_running", "case": index, "policy": policy,
                                   "setting": setting})
        result = await run_trace(engines, controller, args, trace, tokenizer, policy, 0, directory / f"r0-{policy}")
        summaries.append(result)
        save(directory / "summary.json", {"status": "running", "trials": summaries})
    baseline = {request["request_id"]: request["token_ids"] for request in
                json.loads((directory / "r0-no_move" / "requests.json").read_text())}
    for policy in policies:
        records = json.loads((directory / f"r0-{policy}" / "requests.json").read_text())
        if {request["request_id"]: request["token_ids"] for request in records} != baseline:
            raise RuntimeError("Paired stress policy output tokens differ")
    for reference in references:
        if baseline[reference["trace_request_id"]] != reference["token_ids"]:
            raise RuntimeError("Selected stress request differs from serial reference")
    save(directory / "summary.json", {"status": "passed", "trials": summaries,
                                        "scope": "one_frozen_stress_trace_all_policies_complete_not_large_sample_confirmation"})
    (directory / "exit-status").write_text("0\n")
    from verify_trace import verify
    from analyze_stress_sweep import analyze_case

    save(directory / "verification.json", verify(directory, args.prompt_bank))
    analysis = analyze_case(directory)
    save(directory / "analysis.json", analysis)
    return {"directory": str(directory.relative_to(root)), "setting": setting, "analysis": analysis}


async def run_stress_sweep(engines, controller, args, bootstrap, tokenizer):
    root, original_trace = args.output, args.trace
    conv_marks = json.loads(args.stress_conv_marks.read_text())
    code_marks = json.loads(args.stress_code_marks.read_text())
    if (set(args.policies) != {"no_move", "no_move_fair", "own_model", "kv_only", "ilp"}
            or any(output not in (8192, 12288, 16384) for output in args.stress_outputs)
            or any(budget not in (10240, 11328) for budget in args.stress_budgets)
            or any(not math.isfinite(value) or value <= 0 for value in args.stress_rate_multipliers)):
        raise ValueError("Stress sweep requires complete baseline policies and declared finite grid")
    source_configuration = json.loads((root / "config.json").read_text())
    args.output = root / "capacity"
    runtime_copy(root, args.output, args, bootstrap, capacity=True)
    save(root / "phase.json", {"stage": "stress_capacity_running"})
    await run_capacity_pilot(engines, controller, args, bootstrap, tokenizer)
    (args.output / "exit-status").write_text("0\n")
    capacity = json.loads((args.output / "capacity-setting.json").read_text())
    if capacity["kv_budget_mib_per_gpu"] < max(args.stress_budgets):
        raise MemoryError("A800 measured capacity is below declared stress budgets; do not truncate Code")
    args.output = root
    profiles = {}
    for output in args.stress_outputs:
        save(root / "phase.json", {"stage": "stress_joint_profile_running", "output_tokens": output})
        profiles[output] = await profile_output(engines, controller, args, root, output, conv_marks, code_marks, tokenizer)
    events = [event for profile in profiles.values() for trial in profile["trials"] for event in trial.get("events", [])]
    seconds = sum(event["elapsed_s"] for event in events)
    args.move_bandwidth_gib_s = sum(event["kv_bytes"] for event in events) / 1024**3 / seconds
    args.move_fixed_s = max(profile["trials"][-1]["control_s"]
                            - sum(event["elapsed_s"] for event in profile["trials"][-1]["events"])
                            for profile in profiles.values())
    frozen = []
    for output, budget, multiplier in itertools.product(args.stress_outputs, args.stress_budgets, args.stress_rate_multipliers):
        fixed = args.stress_fixed_conv_rate
        if fixed is not None and (len(args.stress_outputs) != 1 or not math.isfinite(fixed) or fixed <= 0):
            raise ValueError("Fixed validation rate requires one output length and positive finite rate")
        rate = (profiles[output]["rate"]["conv_rate_rps"] if fixed is None else fixed) * multiplier
        trace = make_stress_trace(code_marks, conv_marks, output, args.stress_code_rate, rate,
                                 args.stress_window_s, args.stress_warmup_s, args.seed, bootstrap["prompt_bank_sha256"])
        frozen.append({"trace": trace, "setting": {"output_tokens": output, "budget_mib_per_gpu": budget,
                       "rate_multiplier": multiplier, "conv_rate_rps": rate, "code_rate_rps": args.stress_code_rate,
                       "seed": args.seed, "profile_rate_scope": profiles[output]["rate"]["scope"],
                       "fixed_validation_conv_rate_rps": fixed,
                       "move_fixed_s": args.move_fixed_s, "move_bandwidth_gib_s": args.move_bandwidth_gib_s}})
    save(root / "stress-freeze.json", {"settings_and_traces": frozen, "frozen_before_policy_outcomes": True,
                                       "profile_files_sha256": {str((root / "profiles" / str(output) / "summary.json").relative_to(root)):
                                       hashlib.sha256((root / "profiles" / str(output) / "summary.json").read_bytes()).hexdigest()
                                       for output in profiles}})
    manifest = {"status": "running", "planned_cases": len(frozen), "completed_cases": [],
                "selection_metric": "minimum_Code_P95_latency_reduction_vs_both_baselines",
                "scope": "bounded_grid_exploration_all_settings_retained_not_global_optimum_or_heldout_confirmation"}
    save(root / "stress-manifest.json", manifest)
    for index, entry in enumerate(frozen):
        try:
            result = await run_case(engines, controller, args, tokenizer, root, entry["trace"], entry["setting"], index)
        finally:
            args.output = root
        manifest["completed_cases"].append(result)
        save(root / "stress-manifest.json", manifest)
        args.output = root
    manifest["status"] = "passed"
    save(root / "stress-manifest.json", manifest)
    args.output, args.trace = root, original_trace
    save(root / "initial-config.json", source_configuration)
    save(root / "summary.json", {"status": "stress_sweep_passed", "cases": len(frozen), "scope": manifest["scope"]})
    save(root / "phase.json", {"stage": "stress_sweep_complete", "cases": len(frozen)})
