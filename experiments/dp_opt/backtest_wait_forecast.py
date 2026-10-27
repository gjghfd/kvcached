import argparse
import hashlib
import json
import time
from pathlib import Path

from wait_forecast import forecast_context, forecast_wait


def backtest(root):
    trace = json.loads((root / "trace.json").read_text())
    records = json.loads((root / "r0-ilp/requests.json").read_text())
    requests = {request["request_id"]: request for request in records}
    decisions = json.loads((root / "r0-ilp/decisions.json").read_text())
    cases = []
    for decision in decisions:
        if decision.get("wait_tradeoff") is None:
            continue
        check = decision["candidate_revalidations"][-1]
        observed = check["observations"]
        now = max(rank["observed_at"] for ranks in observed for rank in ranks)
        incoming = requests[decision["request_id"]]
        context = forecast_context(decision["request_id"], incoming["model"], records,
                                   trace["model_rates_rps"], observed, now)
        started = time.perf_counter()
        forecast = forecast_wait(observed, incoming["model"], incoming["prompt_tokens"] + len(incoming["token_ids"]),
                                 check["snapshot"]["capacity_gib"], context) if context else None
        elapsed = time.perf_counter() - started
        estimate = forecast.get("estimated_wait_s") if forecast else None
        actual = max(0, incoming["reserved_at"] - now)
        cost = decision["wait_tradeoff"]["migration_cost_s"] + decision["wait_tradeoff"]["fixed_cost_s"]
        accept = estimate is not None and cost < estimate * .8
        cases.append({"request_id": decision["request_id"], "as_of": now, "history_samples": len(context["samples"]) if context else 0,
                      "legacy_wait_s": check["wait_gate"]["estimated_wait_s"], "forecast": forecast,
                      "forecast_elapsed_s": elapsed, "candidate_cost_s": cost, "actual_later_reservation_wait_s": actual,
                      "would_accept": accept, "false_accept_against_same_policy_wait_label": accept and cost >= actual,
                      "context": context})
    return {"status": "past_only_forecast_backtest", "cases": cases,
            "accepted": sum(case["would_accept"] for case in cases),
            "false_accepts": sum(case["false_accept_against_same_policy_wait_label"] for case in cases),
            "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in ("trace.json", "r0-ilp/requests.json", "r0-ilp/decisions.json")},
            "scope": "labels_are_retrospective_same_policy_wait_not_counterfactual_migration_benefit_or_independent_samples"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = backtest(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                      "accepted": result["accepted"], "false_accepts": result["false_accepts"]}))


if __name__ == "__main__":
    main()
