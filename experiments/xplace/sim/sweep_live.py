"""Regime sweep on non-final seeds with the live-calibrated simulator: A (Code TTFT) and B (Conv TPOT) per regime."""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))

from make_trace import DEFAULT_INPUTS, make_trace  # noqa: E402
from validate_live import sim_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--belief", type=Path, required=True)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--conv-rates", default="0.06,0.07,0.08")
    parser.add_argument("--code-rates", default="0.004,0.006,0.008")
    parser.add_argument("--budgets", default="16")
    parser.add_argument("--seeds", default="303,404")
    parser.add_argument("--duration-s", type=float, default=5300)
    parser.add_argument("--replicas", type=int, default=6)
    parser.add_argument("--policies", default="no_move,mig_random,ca_full")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth, belief, planner = (json.loads(path.read_text()) for path in (args.truth, args.belief, args.planner))
    policies = args.policies.split(",")
    rows = []
    for budget in map(float, args.budgets.split(",")):
        for conv in map(float, args.conv_rates.split(",")):
            for code in map(float, args.code_rates.split(",")):
                per = {policy: [] for policy in policies}
                for seed in map(int, args.seeds.split(",")):
                    trace = make_trace(DEFAULT_INPUTS, conv, code, args.duration_s, seed)
                    window = (300.0, args.duration_s)
                    for index, policy in enumerate(policies):
                        for replica in range(args.replicas):
                            per[policy].append(sim_run(trace, policy, truth, belief, planner, budget, window,
                                                       {"interference_cv": 0.37, "prefill_cv": 0.09,
                                                        "noise_seed": 1000 * (index + 1) + replica}))
                mean = lambda policy, key: statistics.mean(run[key] for run in per[policy])
                row = {"budget": budget, "conv_rate": conv, "code_rate": code,
                       **{f"{policy}_{key}": mean(policy, key) for policy in policies
                          for key in ("code_ttft_avg", "conv_tpot_avg", "conv_ttft_p99", "none_b16_share")}}
                if "no_move" in policies and "mig_random" in policies:
                    row["A_mig_vs_nm"] = row["mig_random_code_ttft_avg"] / row["no_move_code_ttft_avg"] - 1
                if "ca_full" in policies and "mig_random" in policies:
                    row["B_ca_vs_mig"] = row["ca_full_conv_tpot_avg"] / row["mig_random_conv_tpot_avg"] - 1
                    row["ca_vs_mig_code_ttft"] = row["ca_full_code_ttft_avg"] / row["mig_random_code_ttft_avg"] - 1
                if "ca_full" in policies and "no_move" in policies:
                    row["conv_ca_vs_nm"] = row["ca_full_conv_tpot_avg"] / row["no_move_conv_tpot_avg"] - 1
                rows.append(row)
                print(f"budget {budget:4.0f} conv {conv:.3f} code {code:.3f} | Code TTFT nm {row['no_move_code_ttft_avg']:6.1f}"
                      f" mig {row['mig_random_code_ttft_avg']:6.1f} ca {row['ca_full_code_ttft_avg']:6.1f}"
                      f" | Conv TPOT nm {1e3 * row['no_move_conv_tpot_avg']:5.1f} mig {1e3 * row['mig_random_conv_tpot_avg']:5.1f}"
                      f" ca {1e3 * row['ca_full_conv_tpot_avg']:5.1f} | B {100 * row['B_ca_vs_mig']:+5.1f}%"
                      f" | G16 mig {row['mig_random_none_b16_share']:.2f} ca {row['ca_full_none_b16_share']:.2f}"
                      f" | ConvTTFT p99 nm {row['no_move_conv_ttft_p99']:5.1f} ca {row['ca_full_conv_ttft_p99']:5.1f}",
                      flush=True)
    args.output.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
