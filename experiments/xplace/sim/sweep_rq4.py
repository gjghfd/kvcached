"""RQ4 sweep (Fig. 10): Conv load x per-GPU KV budget under current-length admission, simulator only.

Traces use the frozen trace generator on non-final seeds (801..), Code at 420/540/660/780 s, 900 s. Each point runs
every policy on each seed with `replicas` noise replicas; a run that exceeds the drain timeout is recorded as such.
Usage: python3 sim/sweep_rq4.py --rates 0.08,0.10,0.12,0.14 --budgets 16,20,24 --seeds 6 --replicas 2 --output X.json
"""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))

from make_trace import DEFAULT_INPUTS, make_trace  # noqa: E402
from validate_live import sim_run  # noqa: E402

INPUTS = HERE.parents[3] / "evidence" / "xplace-inputs"
POLICIES = ["kvcached_native", "no_move_v2", "mig_random_v2", "ca_full_v2", "ca_full_v2_nh", "no_move_tm_v2",
            "mig_random_tm_v2", "ca_full_tm_v2", "ca_full_tm_slo", "ca_full_tm_slo_nh"]
KEYS = ("code_ttft_avg", "conv_tpot_avg", "conv_ttft_avg", "conv_ttft_p99", "moves", "preemptions", "wall_s")
MODELS = {}


def models():
    if not MODELS:
        MODELS["truth"] = json.loads((HERE / "truth-pilot-nm-v2.json").read_text())
        MODELS["belief"] = json.loads((INPUTS / "step-params-dp4-v1.json").read_text())
        MODELS["planner"] = json.loads((INPUTS / "planner-params-v3.json").read_text())
    return MODELS


def job(item):
    rate, budget, seed, policy, replica, page_model = item
    trace = make_trace(DEFAULT_INPUTS, rate, 0.0, 900.0, seed, code_start_s=420.0, code_period_s=120.0,
                       code_end_s=900.0)
    loaded = models()
    try:
        result = sim_run(trace, policy, loaded["truth"], loaded["belief"], loaded["planner"], budget, (420.0, 900.0),
                         {"interference_cv": 0.37, "prefill_cv": 0.09,
                          "noise_seed": 1000 * (POLICIES.index(policy) + 1) + 100 * seed % 1000 + replica,
                          "kv_mode": "prompt", "reserve_extra_tokens": 256, "page_model": page_model})
    except TimeoutError:
        return item, {**{key: math.nan for key in KEYS}, "timeout": True}
    return item, {key: result.get(key, math.nan) for key in KEYS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", required=True)
    parser.add_argument("--budgets", required=True)
    parser.add_argument("--seeds", type=int, default=6, help="seeds 801 .. 800 + N")
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--page-model", choices=("blocks", "high_water"), default="blocks")
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = [(float(rate), float(budget), seed, policy, replica, args.page_model)
             for rate in args.rates.split(",") for budget in args.budgets.split(",")
             for seed in range(801, 801 + args.seeds) for policy in POLICIES for replica in range(args.replicas)]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = [{"rate": item[0], "budget": item[1], "seed": item[2], "policy": item[3], "replica": item[4], **out}
                   for item, out in pool.map(job, items, chunksize=2)]
    args.output.write_text(json.dumps({"page_model": args.page_model, "runs": results}) + "\n")
    print(f"{len(items)} runs in {time.time() - started:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
