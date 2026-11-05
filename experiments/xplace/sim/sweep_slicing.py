"""RQ4 (Fig. 10a): slicing parameters under time-multiplexing on the live traces (seeds 911, 914), simulator only.

Periodic slicing (S, R): every S s of Conv hold during a Code prefill, Conv runs R s. On-demand slicing
(T_wait, R, S_max): slice when a held Conv waits T_wait s for its first token, or the hold lasted S_max s.
Usage: python3 sim/sweep_slicing.py --replicas 3 --output X.json
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))

import simulator  # noqa: E402
from analyze import metrics  # noqa: E402

INPUTS = HERE.parents[3] / "evidence" / "xplace-inputs"
SLO = {"code": {"ttft_s": 120.0}, "conv": {"ttft_s": 5.0, "tpot_s": 0.050}}
VARIANTS = {"none": None}
VARIANTS.update({f"periodic_S{period}": (float(period), 1.0) for period in (3, 4, 5, 6, 8)})
VARIANTS.update({f"ondemand_T{wait}": (float(wait), 1.0, 15.0) for wait in (1, 2, 4, 8)})
VARIANTS.update({f"ondemand_T2_M{limit}": (2.0, 1.0, float(limit)) for limit in (8, 30)})
for _name, _slice in VARIANTS.items():
    _spec = dict(simulator.POLICIES["ca_full_tm_v2"])
    if _slice:
        _spec["slice"] = _slice
    simulator.POLICIES[f"sweep_{_name}"] = _spec


def job(item):
    seed, name, replica = item
    trace = json.loads((INPUTS / f"cur-b16-c012-p120-s{seed}-d900.json").read_text())
    truth = json.loads((HERE / "truth-pilot-nm-v2.json").read_text())
    belief = json.loads((INPUTS / "step-params-dp4-v1.json").read_text())
    planner = json.loads((INPUTS / "planner-params-v3.json").read_text())
    params = {"num_ranks": 4, "budget_gib": 16.0, **truth.get("sim", {}), "interference_cv": 0.37, "prefill_cv": 0.09,
              "noise_seed": 9000 + replica, "kv_mode": "prompt", "reserve_extra_tokens": 256}
    out = simulator.simulate(trace, f"sweep_{name}", sim_params=params, step_params=truth,
                             planner_step_params=belief, planner_params=planner)
    result = metrics(out["requests"], 0.0, 420.0, 900.0, slo=SLO)
    keys = ("code_ttft_avg", "conv_tpot_avg", "conv_ttft_avg", "conv_ttft_p99", "conv_slo_attainment")
    return item, {key: result[key] for key in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = [(seed, name, replica) for seed in (911, 914) for name in VARIANTS for replica in range(args.replicas)]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = [{"seed": item[0], "variant": item[1], "slice": VARIANTS[item[1]], "replica": item[2], **out}
                for item, out in pool.map(job, items)]
    args.output.write_text(json.dumps({"page_model": simulator.DEFAULT_SIM["page_model"], "runs": runs}) + "\n")
    print(f"{len(items)} runs in {time.time() - started:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
