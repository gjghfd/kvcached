import asyncio
import hashlib
import json
import math
import os
import statistics
import time

from calibrate import synthetic_prompt, wait_decoding, wait_idle, window_rate
from engine import save, stream
from measured_cost import interference_key, interference_samples, repeated_interference_plan
from timing_probe import EVENT_LIMIT, HEARTBEAT_S, FrontendTiming, nested_timing_enabled, validate_timing


BACKGROUND_SETTLE_S = 3


def observation_plan(nested=False):
    if type(nested) is not bool:
        raise ValueError("Invalid nested timing plan switch")
    cases = []
    for repeat in range(3):
        conditions = ("timing_off", "timing_on") if repeat % 2 == 0 else ("timing_on", "timing_off")
        for count in (0, 2, 8, 16):
            for condition in conditions:
                cases.append({"name": f"bal{count}", "repeat": repeat, "layout": [count, count],
                              "condition": condition, "case": f"{condition}-bal{count}-r{repeat}"})
    plan = {"schema": 1, "purpose": "mechanism_observation_not_predictive_validation", "cases": cases,
            "seed": 23, "code_output_tokens": 64, "conv_prompt_tokens": 4096, "conv_output_tokens": 3000,
            "cadence_stride": 1, "background_settle_s": BACKGROUND_SETTLE_S,
            "event_limit_per_process_per_case": EVENT_LIMIT, "heartbeat_s": HEARTBEAT_S,
            "soft_limit_from_sbatch_entry_s": 3300, "planned_requests": 336,
            "raw_contract": repeated_interference_plan(2, 23)}
    if nested:
        from nested_timing import INTERFACE_SHA256, SCOPES, source_sha256
        plan["nested_timing"] = {"schema": 1, "scopes": list(SCOPES), "source_sha256": source_sha256(),
                                 "interface_sha256": dict(INTERFACE_SHA256), "extra_cuda_sync": False}
    return plan


def allocation_deadline():
    try:
        started = float(os.environ["XPLACE_JOB_STARTED_AT"])
    except (KeyError, ValueError) as error:
        raise ValueError("Observability requires the sbatch entry timestamp") from error
    age = time.time() - started
    if not math.isfinite(age) or age < 0 or age >= 3300:
        raise ValueError("Invalid or exhausted observability job time budget")
    return time.monotonic() + 3300 - age


async def timing_control(controller, command, case, root):
    results = await asyncio.gather(*[controller.call(model, rank, "placement_timing", command, case)
                                    for model in range(2) for rank in range(controller.num_ranks)], return_exceptions=True)
    failures = [{"index": index, "type": type(result).__name__, "message": str(result)}
                for index, result in enumerate(results) if isinstance(result, BaseException)]
    save(root / f"timing-{command}.json", {"case": case, "command": command, "failures": failures,
                                          "results": [None if isinstance(result, BaseException) else result for result in results]})
    if failures:
        raise RuntimeError(f"Timing {command} failed: {failures}")
    return results


async def wait_timing_idle(controller, root, boundary):
    deadline = time.monotonic() + 30
    observations = []
    while time.monotonic() < deadline:
        replies = await asyncio.gather(*[controller.call(model, rank, "placement_timing", "ready", None)
                                         for model in range(2) for rank in range(controller.num_ranks)])
        observations.append({"at_ns": time.monotonic_ns(), "idle": [reply["idle"] for reply in replies]})
        if all(reply["idle"] for reply in replies):
            save(root / f"idle-{boundary}.json", observations)
            return
        await asyncio.sleep(0.05)
    save(root / f"idle-{boundary}.json", observations)
    raise TimeoutError("Engine timing boundary did not become idle")


async def measure_observation_case(engines, controller, case, code_prompt, root):
    initial = await controller.summaries()
    name, repeat, layout = case["name"], case["repeat"], case["layout"]
    progress, tasks = {}, []
    started_ns = time.monotonic_ns()

    def launch(request_id, prompt, output, rank, model):
        progress[request_id] = {"request_id": request_id, "model": model, "rank": rank, "prompt_tokens": len(prompt)}
        task = asyncio.create_task(stream(engines[model], request_id, prompt, output, rank, model,
                                          "interference", time.monotonic(), stride=1, progress=progress[request_id]))
        tasks.append(task)
        return task

    try:
        conv_tasks = [launch(f"if_{name}_{repeat}_{rank}_{index}",
                             synthetic_prompt(4096, 23 + 1009 * repeat + 101 * rank + index), 3000, rank, 1)
                      for rank, count in enumerate(layout) for index in range(count)]
        if conv_tasks:
            await wait_decoding(controller, 1, len(conv_tasks))
            await asyncio.sleep(BACKGROUND_SETTLE_S)
        code = await launch(f"if_code_{name}_{repeat}", code_prompt, 64, 0, 0)
        conv = list(await asyncio.gather(*conv_tasks))
    except BaseException as error:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        completed = [task.result() for task in tasks if not task.cancelled() and task.exception() is None]
        save(root / "failed-requests.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                                            "completed": completed, "partial_progress": progress})
        raise
    ended_ns = time.monotonic_ns()
    await wait_idle(controller)
    save(root / "requests.json", [code] + conv)
    final = await controller.summaries()
    row = {"name": name, "repeat": repeat, "layout": layout, "conv_per_rank": max(layout),
           "conv_on_code_gpu": layout[0], "code_prefill_s": code["first_token_at"] - code["submitted_at"],
           "code_tpot_s": (code["completed_at"] - code["first_token_at"]) / 63, "measurement_schema": 2,
           "preemptions": sum(after["preemptions_total"] - before["preemptions_total"]
                              for model in range(2) for before, after in zip(initial[model], final[model])),
           "condition": case["condition"], "case_elapsed_s": (ended_ns - started_ns) / 1e9,
           "measurement_started_ns": started_ns, "measurement_ended_ns": ended_ns}
    if conv:
        def median_rate(start, end):
            rates = [window_rate(request, start, end) for request in conv]
            return statistics.median(rates) if all(rate is not None for rate in rates) else None

        row.update(conv_before_s=median_rate(code["arrived_at"] - 3, code["arrived_at"]),
                   conv_during_prefill_s=median_rate(code["submitted_at"] + 1, code["first_token_at"] - 1),
                   conv_during_decode_s=median_rate(code["first_token_at"], code["completed_at"]),
                   conv_after_s=median_rate(code["completed_at"] + 2, min(request["completed_at"] for request in conv) - 1),
                   conv_whole_mean_tpot_s=statistics.mean((request["completed_at"] - request["first_token_at"]) /
                                                         (request["output_tokens"] - 1) for request in conv))
    save(root / "measurement.json", row)
    return row, [code] + conv


async def run_observability(engines, controller, root, code_prompt, deadline):
    if controller.num_ranks != 2 or controller.tensor_parallel_size != 2:
        raise ValueError("Observability requires both models DP2/TP2")
    if len(code_prompt) != 101167:
        raise ValueError("Observability requires the unchanged original Code prompt")
    nested = nested_timing_enabled()
    plan = observation_plan(nested=nested)
    root.mkdir(parents=True, exist_ok=False)
    save(root / "plan.json", plan)
    save(root / "inputs.json", {"code_prompt_tokens": len(code_prompt),
                               "code_prompt_sha256": hashlib.sha256(json.dumps(code_prompt).encode()).hexdigest(),
                               "sbatch_entry_wall_s": os.environ.get("XPLACE_JOB_STARTED_AT"),
                               "deadline_monotonic_s": deadline, "phase_entry_monotonic_s": time.monotonic()})
    records = {condition: [] for condition in ("timing_off", "timing_on")}
    raw = {condition: {} for condition in records}
    completed = []

    async def matrix():
        for case in plan["cases"]:
            await wait_idle(controller)
            case_root = root / case["condition"] / interference_key(case)
            case_root.mkdir(parents=True, exist_ok=False)
            save(case_root / "case.json", case)
            frontend = FrontendTiming(case["case"]) if case["condition"] == "timing_on" else None
            controls_requested = False
            stopped = False
            try:
                if frontend is not None:
                    await frontend.start()
                    await wait_timing_idle(controller, case_root, "start")
                    controls_requested = True
                    starts = await timing_control(controller, "start", case["case"], case_root)
                    if len(starts) != 4 or any(len(result["workers"]) != 2 for result in starts):
                        raise ValueError("Incomplete timing start acknowledgments")
                row, requests = await measure_observation_case(engines, controller, case, code_prompt, case_root)
                await wait_idle(controller)
                if frontend is not None:
                    await wait_timing_idle(controller, case_root, "stop")
                    reports = await timing_control(controller, "stop", case["case"], case_root)
                    stopped = True
                    front_report = await frontend.stop()
                    save(case_root / "timing-frontend.json", front_report)
                    validation = validate_timing(reports, front_report, case["case"], row["measurement_started_ns"],
                                                 row["measurement_ended_ns"], **({"nested": True} if nested else {}))
                    save(case_root / "timing-validation.json", validation)
                records[case["condition"]].append(row)
                raw[case["condition"]][interference_key(row)] = requests
                completed.append(case["case"])
                save(root / "progress.json", {"completed": completed, "planned": len(plan["cases"])})
            except BaseException as error:
                save(case_root / "error.json", {"type": type(error).__name__, "message": str(error)})
                if controls_requested and not stopped:
                    try:
                        await asyncio.wait_for(timing_control(controller, "abort", case["case"], case_root), timeout=30)
                    except Exception as cleanup:
                        save(case_root / "cleanup-error.json", {"type": type(cleanup).__name__, "message": str(cleanup)})
                if frontend is not None and frontend.buffer.active:
                    save(case_root / "timing-frontend.json", await frontend.stop())
                raise
        validations = {}
        for condition in records:
            save(root / f"{condition}-interference.json", records[condition])
        for condition in records:
            try:
                samples = interference_samples(records[condition], raw[condition], 2, plan["raw_contract"])
                validations[condition] = {"status": "passed", "cases": len(records[condition]), "samples": len(samples)}
            except (ValueError, KeyError, TypeError) as error:
                validations[condition] = {"status": "failed", "error": str(error), "cases": len(records[condition])}
        save(root / "raw-validation.json", validations)
        if any(value["status"] != "passed" for value in validations.values()):
            raise ValueError("At least one complete observation condition failed raw acceptance")
        pairs = []
        for off, on in zip(records["timing_off"], records["timing_on"]):
            if interference_key(off) != interference_key(on):
                raise ValueError("Observation pairing mismatch")
            metrics = ("case_elapsed_s", "code_prefill_s", "code_tpot_s", "conv_whole_mean_tpot_s")
            pairs.append({"case": interference_key(off), "timing_off": {key: off.get(key) for key in metrics},
                          "timing_on": {key: on.get(key) for key in metrics}})
        save(root / "paired-overhead.json", {"pairs": pairs, "performance_claim": False,
                                             "note": "All pairs retained; no overhead correction or automatic parameter promotion."})
        return validations

    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Observability soft deadline exhausted")
        validations = await asyncio.wait_for(matrix(), timeout=remaining)
    except BaseException as error:
        save(root / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                                    "completed": completed, "not_completed": [case["case"] for case in plan["cases"]
                                                                                if case["case"] not in completed]})
        raise
    result = {"status": "passed", "completed": completed, "conditions": validations, "independent_prediction_gate": False}
    save(root / "status.json", result)
    return result
