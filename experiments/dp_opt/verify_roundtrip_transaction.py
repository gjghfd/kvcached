import argparse
import json
import math
from pathlib import Path

from migration_roundtrip_control import exact_state_receipt, locate_request, unique_owner
from verify_benefit import verify_ownership


def verify_inventory(observed, budgets):
    if len(observed) != 2 or any(len(ranks) != 2 for ranks in observed):
        raise ValueError("Control requires four tenant/rank inventories")
    verify_ownership(observed, budgets)
    for ranks in observed:
        for rank in ranks:
            if type(rank["page_block_capacity"]) is not int or rank["page_block_capacity"] <= 0:
                raise ValueError("Require measured positive page/block geometry")
            pages = {page["id"] for page in rank["pages"]}
            if len(pages) != len(rank["pages"]) or any(block // rank["page_block_capacity"] not in pages
                for request in rank["requests"] for block in request["blocks"]):
                raise ValueError("Control request blocks must have unique mapped source pages")
    if sum(len(rank["requests"]) for ranks in observed for rank in ranks) != 1:
        raise ValueError("Same-shape control must retain exactly one active request")


def verify(directory):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text())
    control = read("control.json")
    model, source, budgets = control["model"], control["source"], control["budgets_gib"]
    if (type(model) is not int or model not in (0, 1) or type(source) is not int or source not in (0, 1)
            or len(budgets) != 2 or any(isinstance(budget, bool) or not math.isfinite(budget) or budget <= 0
                                       for budget in budgets)):
        raise ValueError("Require two finite controlled budgets and valid tenant/rank identities")
    before = read("inventory-before.json")
    verify_inventory(before, budgets)
    identifier = control["final_state"]["request_id"]
    original_request = locate_request(before, model, source, identifier)
    if not original_request["movable"] or original_request["output_tokens"] < 1024:
        raise ValueError("State control did not reach the declared mature boundary")
    original_bytes = (directory / "request-before.pkl").read_bytes()
    original = {"request_id": identifier, "request": original_bytes, "blocks": original_request["blocks"]}
    expected_hops = 2 if control["mode"] == "paused_roundtrip" else 0
    if (control["mode"] not in ("paused_roundtrip", "boundary_noop")
            or len(control["hops"]) != expected_hops or len(control["states"]) != expected_hops):
        raise ValueError("Control changed its exact noop/roundtrip hop count")
    inventory = before
    for index, event in enumerate(control["hops"], 1):
        current_source = source if index == 1 else 1 - source
        after = read(f"inventory-after-hop-{index}.json")
        verify_inventory(after, budgets)
        unique_owner(after, model, identifier, 1 - current_source)
        old = locate_request(inventory, model, current_source, identifier)
        if (event["request_id"] != identifier or event["model"] != model
                or event["source"] != current_source or event["destination"] != 1 - current_source
                or event.get("bitwise_verified") is not True
                or event.get("integrity") != "device_echo_full_bytewise"
                or event["kv_bytes"] != len(old["blocks"]) * inventory[model][current_source]["block_bytes"]
                or len(event["copy_peak_gib"]) != 2
                or any(isinstance(peak, bool) or not math.isfinite(peak) or peak < 0 or peak > budget + 1e-6
                       for peak, budget in zip(event["copy_peak_gib"], budgets))):
            raise ValueError("Roundtrip hop lacks exact KV/owner/copy-peak evidence")
        moved = locate_request(after, model, 1 - current_source, identifier)
        state = {"request_id": identifier, "request": (directory / f"request-after-hop-{index}.pkl").read_bytes(),
                 "blocks": moved["blocks"]}
        if exact_state_receipt(original, state) != control["states"][index - 1]:
            raise ValueError("Independent serialized-state receipt differs")
        if any(moved[name] != original_request[name] for name in ("computed_tokens", "output_tokens")):
            raise ValueError("Inference progressed between paused roundtrip hops")
        inventory = after
    final = read("inventory-final.json")
    verify_inventory(final, budgets)
    unique_owner(final, model, identifier, source)
    record = locate_request(final, model, source, identifier)
    receipt = exact_state_receipt(original, {"request_id": identifier,
        "request": (directory / "request-final.pkl").read_bytes(), "blocks": record["blocks"]})
    if receipt != control["final_state"] or any(record[name] != original_request[name]
                                                for name in ("computed_tokens", "output_tokens")):
        raise ValueError("Roundtrip did not restore the exact same-source state")
    return {"status": "exact_paused_transaction_independently_verified", "model": model,
            "source": source, "request_id": identifier, "hops": expected_hops,
            "state_receipt": receipt, "mixed_batch_correctness_proven": False, "is_benefit": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))


if __name__ == "__main__":
    main()
