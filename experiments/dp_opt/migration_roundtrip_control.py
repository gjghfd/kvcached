import asyncio
import hashlib
import math
import time

from live_benchmark import save
from live_benefit import capture_control, stream, wait_empty


def locate_request(observed, model, rank, identifier):
    matches = [request for request in observed[model][rank]["requests"]
               if request["id"] == identifier or request["id"].rsplit("-", 1)[0] == identifier]
    if len(matches) != 1:
        raise RuntimeError("State control requires one uniquely owned request")
    return matches[0]


def exact_state_receipt(before, after):
    if (before["request_id"] != after["request_id"] or before["request"] != after["request"]
            or len(before["blocks"]) != len(after["blocks"])):
        raise RuntimeError("Paused roundtrip changed serialized request state or reservation length")
    if any(not blocks or len(blocks) != len(set(blocks))
           for blocks in (before["blocks"], after["blocks"])):
        raise RuntimeError("State control requires unique nonempty reservation blocks")
    return {"request_id": before["request_id"], "serialized_request_bytewise_equal": True,
            "serialized_request_sha256": hashlib.sha256(before["request"]).hexdigest(),
            "reservation_blocks": len(before["blocks"]),
            "physical_block_identifiers_expected_to_change": True}


def unique_owner(observed, model, identifier, expected_rank):
    owners = [rank for rank, snapshot in enumerate(observed[model])
              if any(request["id"] == identifier for request in snapshot["requests"])]
    if owners != [expected_rank]:
        raise RuntimeError("Roundtrip request lacks the expected unique KV owner")


async def paused_state_control(controller, observed, model, source, identifier, budgets, mode, directory):
    if mode not in ("boundary_noop", "paused_roundtrip"):
        raise ValueError("Require a same-shape noop or no-decode-between-hops roundtrip")
    if controller.kv_transport != "dp_echo":
        raise ValueError("Roundtrip requires exact device bytewise echo for both hops")
    receipt = await controller.pause_boundary(observed)
    transaction = "state-control-" + str(time.monotonic_ns())
    frozen = await controller.begin(transaction)
    save(directory / "inventory-before.json", frozen)
    record = locate_request(frozen, model, source, identifier)
    if not record["movable"]:
        raise RuntimeError("State-control request finished or became unsafe during pause")
    actual_id = record["id"]
    original = await controller.call(model, source, "placement_export", transaction, actual_id)
    (directory / "request-before.pkl").write_bytes(original["request"])
    hops, states = [], []
    if mode == "paused_roundtrip":
        for current_source in (source, 1 - source):
            inventory = await controller.snapshots()
            current = locate_request(inventory, model, current_source, actual_id)
            geometry = inventory[model][current_source]
            memory_gib = (math.ceil(len(current["blocks"]) * geometry["block_bytes"]
                                    / geometry["page_bytes"]) * geometry["page_bytes"] / 1024**3)
            event = await controller.migrate(transaction, {"id": actual_id, "model": model,
                "source": current_source, "memory_gib": memory_gib}, budgets)
            if (event.get("bitwise_verified") is not True
                    or event.get("integrity") != "device_echo_full_bytewise"):
                raise RuntimeError("Roundtrip hop lacks exact KV bytewise verification")
            hops.append(event)
            save(directory / "hops.json", hops)
            after = await controller.snapshots()
            save(directory / f"inventory-after-hop-{len(hops)}.json", after)
            unique_owner(after, model, actual_id, 1 - current_source)
            payload = await controller.call(model, 1 - current_source, "placement_export",
                                            transaction, actual_id)
            states.append(exact_state_receipt(original, payload))
            (directory / f"request-after-hop-{len(hops)}.pkl").write_bytes(payload["request"])
            save(directory / "state-receipts.json", states)
    final_inventory = await controller.snapshots()
    save(directory / "inventory-final.json", final_inventory)
    unique_owner(final_inventory, model, actual_id, source)
    final = await controller.call(model, source, "placement_export", transaction, actual_id)
    final_receipt = exact_state_receipt(original, final)
    (directory / "request-final.pkl").write_bytes(final["request"])
    await controller.release_transfer_cache()
    await controller.check_capacity(budgets)
    await controller.end(transaction)
    await controller.resume()
    result = {"mode": mode, "model": model, "source": source, "budgets_gib": budgets,
              "hops": hops, "states": states,
              "final_state": final_receipt, "pause_receipts": receipt,
              "inference_between_roundtrip_hops": False, "resumed_only_after_exact_checks": True}
    save(directory / "control.json", result)
    return result


async def wait_mature(controller, task, model, source, identifier, timeout_s=120):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        observed = await controller.snapshots()
        matches = [request for request in observed[model][source]["requests"]
                   if request["id"] == identifier or request["id"].rsplit("-", 1)[0] == identifier]
        if len(matches) == 1 and matches[0]["movable"] and matches[0]["output_tokens"] >= 1024:
            return observed
        if task.done():
            await task
            raise RuntimeError("Original response completed before the mature control boundary")
        await asyncio.sleep(.05)
    raise TimeoutError("Request did not reach the predeclared mature state boundary")


async def run_state_controls(engines, controller, args, trace):
    from live_trace import request_prompt

    marks = sorted(trace["requests"], key=lambda mark: (mark["model"], mark["request_id"]))
    if (len(marks) != 3 or sum(mark["model"] == 1 for mark in marks) != 2
            or any(mark["output_tokens"] < 2048 for mark in marks if mark["model"] == 1)):
        raise ValueError("Require the unchanged two-long-Conv/one-Code diagnostic trace")
    budgets = [args.kv_budget_mib / 1024] * 2
    results, references = [], {}
    await wait_empty(controller)
    await capture_control(controller, "start")
    for mark in marks:
        model = mark["model"]
        modes = ("serial", "boundary_noop", "paused_roundtrip") if model == 1 else ("serial",)
        for mode in modes:
            for source in (0, 1):
                directory = args.output / f"state-{mark['request_id']}-{mode}-rank{source}"
                directory.mkdir(parents=True, exist_ok=False)
                await wait_empty(controller)
                prompt = request_prompt(args.prepared_prompts, mark)
                identifier = f"state-control-{len(results)}"
                task = asyncio.create_task(stream(engines[model], identifier, prompt, mark["output_tokens"],
                                                  source, model, "state-control", time.monotonic()))
                if mode != "serial":
                    observed = await wait_mature(controller, task, model, source, identifier)
                    await paused_state_control(controller, observed, model, source, identifier,
                                               budgets, mode, directory)
                result = await asyncio.wait_for(task, args.request_timeout_s)
                save(directory / "request.json", result)
                key = (mark["request_id"], source)
                if mode == "serial":
                    references[key] = result["token_ids"]
                exact = result["token_ids"] == references[key]
                results.append({"request_id": mark["request_id"], "mode": mode, "source": source,
                                "original_output_tokens": mark["output_tokens"],
                                "same_source_full_serial_tokens_equal": exact})
                progress = {"trials": results}
                save(args.output / "state-control-progress.json", progress)
                save(args.output / "progress.json", progress)
                if not exact:
                    raise RuntimeError("Same-source state control differs from its full serial tokens")
    cross_rank = [{"request_id": mark["request_id"], "rank_serial_tokens_equal":
                   references[(mark["request_id"], 0)] == references[(mark["request_id"], 1)]}
                  for mark in marks]
    report = {"status": "passed_same_shape_roundtrip_control_not_benefit", "trials": results,
              "cross_rank_serial_comparisons": cross_rank, "is_algorithm_benefit": False,
              "mixed_batch_migration_correctness_proven": False,
              "lengths_extended_or_tokens_truncated": False, "greedy_tolerance_acceptance": False}
    save(args.output / "state-control-captures.json", await capture_control(controller, "stop"))
    save(args.output / "state-control-summary.json", report)
    return report
