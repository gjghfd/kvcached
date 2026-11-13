"""Fixed-context decode calibration with dense, repeated placement sweeps."""
import asyncio
import copy
import math
import random
import time

from calibrate import steady_step, synthetic_prompt, wait_idle
from controller import fits, used_gib
from engine import SPARSE_CAPTURE_SIZES, save, stream


def capture_aligned_blocks():
    blocks = []
    for repeat in range(3):
        for batch in ((128, 256) if repeat % 2 == 0 else (256, 128)):
            balanced = batch // 2
            boundaries = {0, *(size for size in SPARSE_CAPTURE_SIZES if size <= balanced)}
            light_counts = {count + offset for count in boundaries for offset in (-1, 0, 1)
                            if 0 <= count + offset <= balanced}
            deltas = sorted(balanced - count for count in light_counts)
            blocks.append(dict(num_ranks=2, tensor_parallel_size=2, seed=20261110,
                               context_tokens=4096, output_tokens=1024, repeats=1, repeat_start=repeat,
                               case_timeout_s=150, max_scan_s=3000,
                               series=[dict(batch_size=batch, deltas=deltas)],
                               sampling_design="capture_boundaries_plus_minus_one_fixed_total",
                               capture_sizes=list(SPARSE_CAPTURE_SIZES)))
    return blocks


def scan_cases(config, num_ranks, tensor_parallel_size=None):
    configured_ranks = config.get("num_ranks", 4)
    configured_tp = config.get("tensor_parallel_size", 1)
    if (type(configured_ranks) is not int or type(configured_tp) is not int
            or (configured_ranks, configured_tp) not in ((4, 1), (2, 2))
            or configured_ranks != num_ranks
            or tensor_parallel_size is not None and configured_tp != tensor_parallel_size):
        raise ValueError("Calibration requires matching DP4/TP1 or DP2/TP2 on four GPUs")
    observe_graphs = config.get("observe_graph_execution", False)
    if type(observe_graphs) is not bool or observe_graphs and (configured_ranks, configured_tp) != (2, 2):
        raise ValueError("Graph execution observation requires an explicit DP2/TP2 boolean switch")
    if type(config["seed"]) is not int:
        raise ValueError("Calibration seed must be an integer")
    repeat_start = config.get("repeat_start", 0)
    if type(repeat_start) is not int or repeat_start < 0:
        raise ValueError("Calibration repeat_start must be a nonnegative integer")
    for key in ("context_tokens", "output_tokens", "repeats", "case_timeout_s", "max_scan_s"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"Expected a positive integer: {key}")
    if config["context_tokens"] + config["output_tokens"] > 131072:
        raise ValueError("Calibration request exceeds the model length limit")
    templates = []
    batches = set()
    for series in config["series"]:
        batch = series["batch_size"]
        if type(batch) is not int or batch <= 0 or batch % num_ranks or batch in batches:
            raise ValueError("Batch sizes must be unique positive multiples of the DP size")
        batches.add(batch)
        count = batch // num_ranks
        deltas = series["deltas"]
        if not deltas or deltas[0] != 0 or len(set(deltas)) != len(deltas):
            raise ValueError("Each series needs a balanced point and unique deltas")
        for delta in deltas:
            if type(delta) is not int or not 0 <= delta <= count or count + delta > 256:
                raise ValueError("Invalid imbalance or per-rank batch size")
            templates.append({"batch_size": batch, "delta": delta,
                              "layout": [count + delta] + [count] * (num_ranks - 2) + [count - delta]})
    if not templates:
        raise ValueError("Calibration series cannot be empty")
    cases = []
    for repeat in range(repeat_start, repeat_start + config["repeats"]):
        ordered = list(templates)
        random.Random(config["seed"] + repeat).shuffle(ordered)
        for template in ordered:
            rotation = repeat % num_ranks
            layout = template["layout"][rotation:] + template["layout"][:rotation]
            cases.append({**template, "name": f"bs{template['batch_size']}_d{template['delta']}_r{repeat}",
                          "repeat": repeat, "model": 1, "layout": layout,
                          "context_tokens": config["context_tokens"], "output_tokens": config["output_tokens"],
                          "imbalance": max(layout) - min(layout), "max_count": max(layout),
                          "capture_bucket": next(size for size in SPARSE_CAPTURE_SIZES if size >= max(layout))})
            if observe_graphs:
                cases[-1]["observe_graph_execution"] = True
    return cases


def fits_budget(controller, summaries, model, layout, contexts, output_tokens, margin_gib=0.5):
    used = used_gib(summaries)
    for rank, count in enumerate(layout):
        if count == 0:
            continue
        block_size = summaries[model][rank]["block_size"]
        tokens = sum(math.ceil((context + output_tokens) / block_size) * block_size
                     for context in contexts[rank][:count])
        if not fits(summaries, model, rank, tokens, controller.budgets, used, headroom=margin_gib):
            return False
    return True


async def measure_layout(engines, controller, case, prompts, root):
    prepared_at = time.monotonic()
    started, tasks, completed, progress, requests = None, [], {}, {}, []
    graph_attempted, graph_start, primary_error = False, None, None
    outcome = {**case, "status": "preparing", "prepared_at": prepared_at}
    prompt_index = 0
    for rank, count in enumerate(case["layout"]):
        for _ in range(count):
            identity = f"dense_{case['name']}_{prompt_index}"
            requests.append(dict(request_id=identity, rank_initial=rank, model=case["model"], prompt_index=prompt_index,
                                 prompt_tokens=case["context_tokens"], target_output_tokens=case["output_tokens"]))
            progress[identity] = {}
            prompt_index += 1
    save(root / f"{case['name']}-attempt.json", dict(case=case, prepared_at=prepared_at, requests=requests))
    save(root / f"{case['name']}-status.json", outcome)

    async def run_request(request):
        identity = request["request_id"]
        try:
            result = await stream(engines[case["model"]], identity, prompts[request["prompt_index"]],
                                  case["output_tokens"], request["rank_initial"], case["model"], "calibration", started,
                                  progress=progress[identity])
            completed[identity] = result
            return result
        except BaseException as error:
            progress[identity]["stream_error"] = dict(type=type(error).__name__, message=str(error))
            raise

    try:
        await wait_idle(controller)
        await controller.memory(release_unused=True)
        before = await controller.summaries()
        save(root / f"{case['name']}-before.json", before)
        contexts = [[case["context_tokens"]] * count for count in case["layout"]]
        if not fits_budget(controller, before, case["model"], case["layout"], contexts, case["output_tokens"]):
            outcome.update(status="skipped_budget", budget_gib=controller.budgets)
            return outcome
        if case.get("observe_graph_execution", False):
            from graph_execution import graph_control

            graph_attempted = True
            graph_start = await graph_control(controller, case["model"], "start", case["name"], root)
        started = time.monotonic()
        outcome.update(status="running", started_at=started)
        save(root / f"{case['name']}-status.json", outcome)
        tasks = [asyncio.create_task(run_request(request)) for request in requests]
        results = await asyncio.gather(*tasks)
        save(root / f"{case['name']}-requests.json", results)
        after = await controller.summaries()
        save(root / f"{case['name']}-after.json", after)
        preemptions = sum(finish["preemptions_total"] - initial["preemptions_total"]
                          for initial, finish in zip(before[case["model"]], after[case["model"]]))
        step, window = steady_step(results)
        valid = step is not None and preemptions == 0 and window["samples"] == case["batch_size"]
        outcome.update(status="measured" if valid else "invalid_window_or_preemption", step_s=step,
                       window=window, preemptions=preemptions, wall_s=time.monotonic() - started,
                       common_start=max(result["first_token_at"] for result in results) + 1.0,
                       common_end=min(result["completed_at"] for result in results) - 1.0,
                       first_token_spread_s=max(result["first_token_at"] for result in results)
                       - min(result["first_token_at"] for result in results), requests_file=f"{case['name']}-requests.json")
        return outcome
    except BaseException as error:
        primary_error = error
        censored_at = time.monotonic()
        cutoff_completed = copy.deepcopy(list(completed.values()))
        incomplete = [dict(request, censored_at=censored_at, partial=copy.deepcopy(progress[request["request_id"]]))
                      for request in requests if request["request_id"] not in completed]
        save(root / f"{case['name']}-requests.json", cutoff_completed)
        save(root / f"{case['name']}-incomplete.json", incomplete)
        outcome.update(status="cancelled" if isinstance(error, asyncio.CancelledError) else "error",
                       error=dict(type=type(error).__name__, message=str(error)), censored_at=censored_at,
                       completed_at_cutoff=len(cutoff_completed), incomplete_at_cutoff=len(incomplete),
                       requests_file=f"{case['name']}-requests.json", incomplete_file=f"{case['name']}-incomplete.json")
        save(root / f"{case['name']}-status.json", outcome)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            try:
                done, pending = await asyncio.wait(tasks, timeout=5)
            except BaseException as cleanup_error:
                outcome["cleanup_interrupted"] = dict(type=type(cleanup_error).__name__, message=str(cleanup_error))
                done = {task for task in tasks if task.done()}
                pending = set(tasks) - done
            outcome["cleanup_pending_requests"] = [request["request_id"] for request, task in zip(requests, tasks) if task in pending]
            for task in done:
                if not task.cancelled():
                    task.exception()
        save(root / f"{case['name']}-cleanup-results.json", list(completed.values()))
        raise
    finally:
        deferred_error = None
        if graph_attempted:
            try:
                graph_stop = await graph_control(controller, case["model"], "stop", case["name"], root)
                from graph_execution import validate_graph_pair

                validity = validate_graph_pair(graph_start, graph_stop)
                outcome["graph_evidence"] = validity
                if not validity["valid"] and outcome["status"] == "measured":
                    outcome["status"] = "invalid_graph_evidence"
            except BaseException as error:
                outcome["graph_evidence"] = dict(valid=False, error=dict(type=type(error).__name__, message=str(error)))
                if outcome["status"] == "measured":
                    outcome["status"] = "invalid_graph_evidence"
                if primary_error is None and not isinstance(error, Exception):
                    outcome["status"] = "cancelled" if isinstance(error, asyncio.CancelledError) else "error"
                    deferred_error = error
            outcome["graph_start_file"] = f"{case['name']}-graph-start.json"
            outcome["graph_stop_file"] = f"{case['name']}-graph-stop.json"
        outcome["finished_at"] = time.monotonic()
        save(root / f"{case['name']}-status.json", outcome)
        if deferred_error is not None:
            raise deferred_error


async def run_imbalance_scan(engines, controller, root, config, num_ranks):
    cases = scan_cases(config, num_ranks, getattr(controller, "tensor_parallel_size", 1))
    root.mkdir(parents=True, exist_ok=True)
    if (root / "plan.json").exists():
        raise FileExistsError("Refusing to overwrite an existing imbalance scan")
    save(root / "plan.json", {"config": config, "cases": cases})
    prompts = [synthetic_prompt(config["context_tokens"], config["seed"] + 104729 * index)
               for index in range(max(case["batch_size"] for case in cases))]
    largest = max(cases, key=lambda case: case["batch_size"])
    count = largest["batch_size"] // num_ranks
    warmup = {**largest, "name": "warmup", "layout": [count] * num_ranks,
              "output_tokens": 256, "delta": 0, "imbalance": 0, "max_count": count,
              "capture_bucket": next(size for size in SPARSE_CAPTURE_SIZES if size >= count)}
    started = time.monotonic()
    records, warmup_result, failure = [], None, None
    save(root / "results.json", records)

    async def attempt(case):
        try:
            return await asyncio.wait_for(measure_layout(engines, controller, case, prompts, root), timeout=config["case_timeout_s"])
        except BaseException as error:
            status = "timed_out" if isinstance(error, asyncio.TimeoutError) else "cancelled" if isinstance(error, asyncio.CancelledError) else "error"
            record = dict(case, status=status, error=dict(type=type(error).__name__, message=str(error)),
                          attempt_status_file=f"{case['name']}-status.json")
            if case["name"] == "warmup":
                save(root / "warmup.json", record)
            else:
                records.append(record)
                save(root / "results.json", records)
            raise

    try:
        warmup_result = await attempt(warmup)
        save(root / "warmup.json", warmup_result)
        if warmup_result["status"] == "measured":
            for case in cases:
                if time.monotonic() - started + config["case_timeout_s"] > config["max_scan_s"]:
                    break
                save(root / "progress.json", {"completed": len(records), "planned": len(cases), "current": case["name"]})
                result = await attempt(case)
                records.append(result)
                save(root / "results.json", records)
                if result.get("graph_evidence", {}).get("valid") is False:
                    break
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        summary = {"planned": len(cases), "completed": len(records),
                   "measured": sum(record["status"] == "measured" for record in records),
                   "wall_s": time.monotonic() - started, "failure": failure,
                   "max_scan_s": config["max_scan_s"], "case_timeout_s": config["case_timeout_s"],
                   "remaining_cases": [case["name"] for case in cases[len(records):]]}
        summary["status"] = "failed" if failure else "passed" if summary["measured"] == len(cases) else "incomplete"
        summary["stop_reason"] = ("exception" if failure else "warmup_not_measured" if warmup_result["status"] != "measured" else
                                  "graph_evidence_failed" if records and records[-1].get("graph_evidence", {}).get("valid") is False else
                                  "completed" if summary["status"] == "passed" else
                                  "scan_budget_exhausted" if len(records) < len(cases) else "invalid_measurements")
        statuses = records + [dict(case, status="not_run", reason=summary["stop_reason"]) for case in cases[len(records):]]
        save(root / "case-statuses.json", statuses)
        save(root / "summary.json", summary)
        save(root / "progress.json", summary)
    return summary
