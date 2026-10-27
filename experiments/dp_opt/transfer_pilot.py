import asyncio
import json
import math
import os
import time
from pathlib import Path

from live_benchmark import save
from live_benefit import capture_control, stream, wait_empty
from live_controller import allocation_increment


def pilot_cases(marks, repeats=3, transports=("host_file", "dp_allgatherv"), pause_modes=None):
    if not marks or repeats < 1:
        raise ValueError("Transfer pilot requires paired Conv marks and positive repeats")
    if pause_modes is not None:
        if tuple(transports) != ("dp_echo",) or tuple(pause_modes) != ("native_keep", "default32_boundary"):
            raise ValueError("Pause pilot requires exact echo and native/default32 paired modes")
    elif (len(transports) != 2 or len(set(transports)) != 2
          or any(transport not in ("host_file", "dp_allgatherv", "dp_echo") for transport in transports)):
        raise ValueError("Transfer pilot needs two distinct supported transports")
    variants = [("dp_echo", mode) for mode in pause_modes] if pause_modes else [(name, None) for name in transports]
    cases = []
    for target in (2560, 4608, 8192):
        mark = min(marks, key=lambda item: abs(item["prompt_tokens"] + item["output_tokens"] - target)
                   + abs(item["output_tokens"] - 1024))
        if (mark["prompt_tokens"] < 2048 or not 512 <= mark["output_tokens"] <= 1024
                or "source_internal_row" not in mark):
            raise ValueError("Pilot marks must retain paired Azure lengths and source row")
        for repeat in range(repeats):
            order = variants if repeat % 2 == 0 else tuple(reversed(variants))
            for transport, pause_mode in order:
                for model in (0, 1):
                    case = {"target_tokens": target, "repeat": repeat, "transport": transport,
                            "model": model, "source": (repeat + model) % 2, "mark": mark}
                    if pause_mode is not None:
                        case["pause_mode"] = pause_mode
                    cases.append(case)
    return cases


async def run_transfer_pilot(engines, controller, args, tokenizer):
    marks = json.loads(args.transfer_conv_marks.read_text())
    transports = tuple(getattr(args, "transfer_transports", ("host_file", "dp_allgatherv")))
    pause_modes = getattr(args, "transfer_pause_modes", None)
    cases = pilot_cases(marks, transports=transports, pause_modes=pause_modes)
    save(args.output / "transfer-cases.json", cases)
    references, trials = {}, []
    budgets = [args.kv_budget_mib / 1024] * 2
    prefix = tokenizer.encode("Calculate 2 + 2. The answer is", add_special_tokens=False)
    await wait_empty(controller)
    await capture_control(controller, "start")
    for index, case in enumerate(cases):
        mark = case["mark"]
        prompt = (prefix * math.ceil(mark["prompt_tokens"] / len(prefix)))[:mark["prompt_tokens"]]
        key = (case["model"], mark["prompt_tokens"], mark["output_tokens"])
        if key not in references:
            reference = await stream(engines[case["model"]], f"transfer-reference-{len(references)}", prompt,
                                     mark["output_tokens"], case["source"], case["model"], "reference", time.monotonic())
            references[key] = reference
            save(args.output / "transfer-references.json", list(references.values()))
        await wait_empty(controller)
        await controller.release_transfer_cache()
        before = await controller.snapshots(refresh_physical_capacity=True)
        source, model = case["source"], case["model"]
        used = sum(before[tenant][source]["mapped_bytes"] for tenant in (0, 1)) / 1024**3
        if used + allocation_increment(before[model][source], len(prompt) + mark["output_tokens"]) > budgets[source] + 1e-6:
            raise MemoryError("Transfer pilot source exceeds common controlled KV budget")
        identifier = f"transfer-{index}"
        task = asyncio.create_task(stream(engines[model], identifier, prompt, mark["output_tokens"], source,
                                          model, "transfer-pilot", time.monotonic()))
        deadline = time.monotonic() + 120
        while True:
            observation = await controller.snapshots()
            record = next((request for request in observation[model][source]["requests"]
                           if request["id"] == identifier or request["id"].rsplit("-", 1)[0] == identifier), None)
            if record is not None and record["movable"] and record["output_tokens"] >= 32:
                break
            if task.done():
                await task
                raise RuntimeError("Pilot request completed before migration boundary")
            if time.monotonic() >= deadline:
                raise TimeoutError("Pilot request never reached mature decode boundary")
            await asyncio.sleep(.05)
        pause_started = time.monotonic()
        pause_receipts = (await controller.pause_boundary(observation)
                          if case.get("pause_mode") == "default32_boundary" else await controller.pause())
        pause_s = time.monotonic() - pause_started
        transaction = "transfer-pilot-" + str(index)
        frozen = await controller.begin(transaction)
        actual = next((request for request in frozen[model][source]["requests"] if request["id"] == record["id"]), None)
        if actual is None or not actual["movable"]:
            raise RuntimeError("Pilot request completed during default32 pause handshake")
        controller.kv_transport = case["transport"]
        event = await controller.migrate(transaction, {"id": actual["id"], "model": model, "source": source,
                                                       "memory_gib": math.ceil(len(actual["blocks"])
                                                        * frozen[model][source]["block_bytes"]
                                                        / frozen[model][source]["page_bytes"])
                                                       * frozen[model][source]["page_bytes"] / 1024**3}, budgets)
        await controller.release_transfer_cache()
        after = await controller.snapshots(refresh_physical_capacity=True)
        if (any(request["id"] == actual["id"] for request in after[model][source]["requests"])
                or not any(request["id"] == actual["id"] for request in after[model][1 - source]["requests"])):
            raise RuntimeError("Transfer pilot request did not change its unique KV owner")
        await controller.end(transaction)
        resume_started = time.monotonic()
        await controller.resume()
        resume_s = time.monotonic() - resume_started
        result = await asyncio.wait_for(task, args.request_timeout_s)
        if result["token_ids"] != references[key]["token_ids"]:
            raise RuntimeError("Migrated request differs from full serial greedy reference")
        trial = {"case": case, "before": frozen, "after": after, "event": event, "request": result,
                 "pause_s": pause_s, "resume_s": resume_s,
                 "pause_started_at": pause_started, "resume_started_at": resume_started,
                 "resume_completed_at": resume_started + resume_s, "pause_receipts": pause_receipts,
                 "paused_transaction_s": resume_started - pause_started - pause_s,
                 "scope": "single_request_real_KV_transport_cost_not_mixed_trace_net_benefit"}
        trials.append(trial)
        save(args.output / "transfer-trials.json", trials)
        save(args.output / "progress.json", {"stage": "transfer_pilot", "completed_cases": len(trials),
                                            "total_cases": len(cases), "transport": case["transport"]})
    captures = await capture_control(controller, "stop")
    save(args.output / "transfer-captures.json", captures)
    save(args.output / "transfer-summary.json", {"status": "transfer_pilot_passed", "cases": len(trials),
                                                "budget_mib_per_gpu": args.kv_budget_mib,
                                                "transports": list(transports),
                                                "pause_modes": pause_modes,
                                                "chunk_mib": int(os.environ.get("DP_OPT_TRANSFER_CHUNK_MIB", "32")),
                                                "scope": "correctness_and_cost_pilot_not_placement_benefit"})
