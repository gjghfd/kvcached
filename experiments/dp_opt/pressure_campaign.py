import argparse
import json
import sys
from pathlib import Path

import amortized_campaign
from kv_pressure import read_pressure


EXTRA_SYNC_FILES = ["experiments/dp_opt/kv_pressure.py", "experiments/dp_opt/pressure_campaign.py",
                    "tests/test_dp_opt_kv_pressure.py", "experiments/dp_opt/docs/pressure-ratio-protocol.md",
                    "experiments/dp_opt/reports/pressure-ratio-status.md",
                    "experiments/dp_opt/stress_campaign.py", "tests/test_dp_opt_stress_campaign.py"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    amortized_campaign.SYNC_FILES = list(dict.fromkeys(amortized_campaign.SYNC_FILES + EXTRA_SYNC_FILES))
    sys.argv = [sys.argv[0], "--root", str(root), "--source", str(args.source.resolve()),
                "--inputs", str(args.inputs.resolve()), "--variant", "headroom"] + (["--resume"] if args.resume else [])
    amortized_campaign.main()
    state = json.loads((root / "campaign-state.json").read_text())
    pressures = read_pressure(root / f"live-trace-{state['job']}")
    setting = json.loads((args.source / "pressure-setting.json").read_text())
    result = {"job": state["job"], "setting": setting, "measured_pressure": pressures,
              "measured_no_move_ratio": pressures["no_move"]["local_physical_ratio"],
              "target_ratio": setting["setting"]["target_local_physical_ratio"],
              "GPU_experiment_complete_and_verified": state["GPU_experiment_complete"],
              "not_a_multi_seed_tail_confirmation": True}
    (root / "pressure-results.json").write_text(json.dumps(result, indent=2) + "\n")
    state.update({"phase": "lower_Conv_rate_trace_and_pressure_independently_verified",
                  "pressure_result": "pressure-results.json",
                  "next_action": "report_actual_ratio_and_all_tenant_tail_before_any_further_rate_adjustment"})
    amortized_campaign.save_state(root, state)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
