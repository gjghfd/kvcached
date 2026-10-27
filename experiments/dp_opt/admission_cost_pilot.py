import asyncio
import json
import math
import time

from live_benchmark import save
from live_benefit import capture_control, stream, wait_empty, wait_reserved
from live_controller import admission_candidates, allocation_increment


def pilot_cases(marks):
    if not marks:
        raise ValueError("Admission cost pilot requires paired Azure marks")
    mark = min(marks, key=lambda item: abs(item["prompt_tokens"] - 3584)
               + abs(item["output_tokens"] - 1024))
    if (mark["prompt_tokens"] != 3584 or mark["output_tokens"] != 1024
            or "source_internal_row" not in mark):
        raise ValueError("Admission cost pilot requires intact paired Azure 3584/1024 mark")
    return [{"population": population, "mode": mode, "conv_mark": mark}
            for population in (8, 12, 16)
            for mode in (("native_inside", "preplan_boundary") if population != 12
                         else ("preplan_boundary", "native_inside"))]


async def run_admission_cost_pilot(engines, controller, args, trace, tokenizer):
    marks = json.loads(args.admission_conv_marks.read_text())
    cases = pilot_cases(marks)
    code = max((mark for mark in trace["requests"] if mark["model"] == 0),
               key=lambda mark: mark["prompt_tokens"])
    if code["prompt_tokens"] < 100000 or code["output_tokens"] != 64:
        raise ValueError("Admission cost pilot requires intact 100K Code input and output64")
    prefix = tokenizer.encode("Calculate 2 + 2. The answer is", add_special_tokens=False)
    mark = cases[0]["conv_mark"]
    conv_prompt = (prefix * math.ceil(mark["prompt_tokens"] / len(prefix)))[:mark["prompt_tokens"]]
    code_prompt = args.prepared_prompts[code.get("prompt_key", str(code["prompt_tokens"]))]
    budgets = [args.kv_budget_mib / 1024] * 2
    controller.kv_transport = "dp_echo"
    references = []
    for model, prompt, output in ((1, conv_prompt, mark["output_tokens"]),
                                   (0, code_prompt, code["output_tokens"])):
        references.append(await stream(engines[model], f"admission-reference-{model}", prompt,
                                       output, 0, model, "reference", time.monotonic()))
    save(args.output / "admission-references.json", references)
    save(args.output / "admission-cases.json", cases)
    save(args.output / "admission-code-mark.json", code)
    await wait_empty(controller)
    await capture_control(controller, "start")
    trials = []
    for index, case in enumerate(cases):
        await wait_empty(controller)
        await controller.release_transfer_cache()
        tasks = []
        for position in range(case["population"]):
            rank = position % 2
            identifier = f"admission-conv-{index}-{position}"
            task = asyncio.create_task(stream(engines[1], identifier, conv_prompt,
                                              mark["output_tokens"], rank, 1, "old_conv", time.monotonic()))
            await wait_reserved(controller, task, identifier, 1, rank, timeout=120, target_only=True)
            tasks.append(task)
        deadline = time.monotonic() + 120
        while True:
            before = await controller.snapshots()
            active = [request for rank in before[1] for request in rank["requests"]]
            if (len(active) == case["population"]
                    and all(request["movable"] and request["output_tokens"] >= 32 for request in active)):
                break
            if any(task.done() for task in tasks) or time.monotonic() >= deadline:
                raise RuntimeError("Admission pilot population did not reach mature simultaneous decode")
            await asyncio.sleep(.05)
        used, direct = admission_candidates(before, 0, code["prompt_tokens"] + code["output_tokens"], budgets)
        logical = math.ceil((code["prompt_tokens"] + code["output_tokens"]) / before[0][0]["block_size"]) * before[0][0]["block_bytes"]
        witness = {"initial_gib": used, "global_free_bytes": round((sum(budgets) - sum(used)) * 1024**3),
                   "incoming_logical_bytes": logical, "direct_ranks": direct,
                   "allocation_increment_gib": [allocation_increment(before[0][rank], code["prompt_tokens"]
                                                    + code["output_tokens"]) for rank in (0, 1)]}
        if direct or witness["global_free_bytes"] < logical:
            raise RuntimeError("Admission pilot lacks global-enough/local-blocked witness")
        arriving = time.monotonic()
        incoming_task = None

        async def admit(rank):
            nonlocal incoming_task
            incoming_task = asyncio.create_task(stream(engines[0], f"admission-code-{index}", code_prompt,
                code["output_tokens"], rank, 0, "code_arrival", arriving))
            reservation = await wait_reserved(controller, incoming_task, f"admission-code-{index}", 0,
                                              rank, timeout=120, target_only=True)
            if reservation.get("reservation_pending"):
                raise RuntimeError("Admission pilot migrated but failed actual Code reservation")
            return reservation

        options = {"max_moves": 8, "candidate_limit": 8, "move_fixed_s": .01,
                   "move_bandwidth_gib_s": 32, "solver_limit_s": 1}
        prepared = (await controller.prepare_admission(code["prompt_tokens"], code["output_tokens"],
                    budgets, **options) if case["mode"] == "preplan_boundary" else None)
        decision = await controller.schedule_admission(code["prompt_tokens"], code["output_tokens"],
                                                       budgets, admit, prepared=prepared, **options)
        decision["pilot_arrival_to_control_finish_s"] = time.monotonic() - arriving
        if not decision["admitted"] or not decision["moves"] or incoming_task is None:
            save(args.output / "admission-failed-decision.json", {"case": case, "decision": decision})
            raise RuntimeError("Admission cost pilot did not complete a migration and Code admission")
        results = list(await asyncio.wait_for(asyncio.gather(*tasks, incoming_task), args.request_timeout_s))
        if any(request["token_ids"] != references[0 if request["model"] == 1 else 1]["token_ids"] for request in results):
            raise RuntimeError("Admission cost pilot differs from full serial token reference")
        trial = {"case": case, "before": before, "witness": witness, "decision": decision,
                 "requests": results, "arriving_at": arriving,
                 "scope": "paired_joint_control_and_real_Code_admission_cost_not_Poisson_trace_benefit"}
        trials.append(trial)
        save(args.output / "admission-trials.json", trials)
        save(args.output / "progress.json", {"stage": "admission_cost_pilot", "completed_cases": len(trials),
                                            "total_cases": len(cases), "mode": case["mode"]})
    save(args.output / "admission-captures.json", await capture_control(controller, "stop"))
    save(args.output / "summary.json", {"status": "admission_cost_pilot_passed", "cases": len(cases),
                                         "scope": "cost_calibration_not_final_trace_benefit"})
