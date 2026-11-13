"""Mechanism check (paper Fig. 7b): placement B vs the extra no-Code rank imbalance random migration leaves.

Non-final traces 801-812 (Conv 0.12 RPS, 16 GiB, blocks page model). Random migration is run with 5 seeds per trace
(the admission RNG seed and noise vary); cost-aware placement once. Writes all pairs to the output JSON.
Usage: python3 sim/b_vs_imbalance.py OUTPUT.json [--general]
--general pairs the general (type-agnostic) policies (mig_random_tm_v2_g / ca_full_tm_v2_g, mig_random_v2_g /
ca_full_v2_g) with planner-params-v5.json; default: the legacy policies with planner-params-v3.json.
"""
import argparse, json, sys, statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT + "/sim"); sys.path.insert(0, ROOT + "/tools")
E = str(Path(__file__).resolve().parents[4] / "evidence" / "xplace-inputs")
# (schedule, (random policy, cost-aware policy)) and planner file per mode (key: general?)
PAIRS = {False: (("tm", ("mig_random_tm_v2", "ca_full_tm_v2")), ("corun", ("mig_random_v2", "ca_full_v2"))),
         True: (("tm", ("mig_random_tm_v2_g", "ca_full_tm_v2_g")), ("corun", ("mig_random_v2_g", "ca_full_v2_g")))}
PLANNER = {False: "planner-params-v3.json", True: "planner-params-v5.json"}


def job(task):
    import simulator as S
    from analyze import metrics
    from make_trace import DEFAULT_INPUTS, make_trace
    trace_seed, policy, variant, sched, planner_file = task
    trace = make_trace(DEFAULT_INPUTS, 0.12, 0.0, 900.0, trace_seed, code_start_s=420.0, code_period_s=120.0,
                       code_end_s=900.0)
    truth = json.load(open(ROOT + "/sim/truth-pilot-nm-v2.json")); belief = json.load(open(E + "/step-params-dp4-v1.json"))
    planner = json.load(open(E + "/" + planner_file))
    p = {"num_ranks": 4, "budget_gib": 16.0, **truth.get("sim", {}), "kv_mode": "prompt", "reserve_extra_tokens": 256,
         "interference_cv": 0.37, "prefill_cv": 0.09, "noise_seed": 4242 + variant, "seed": 20261103 + 7919 * variant}
    out = S.simulate(trace, policy, sim_params=p, step_params=truth, planner_step_params=belief, planner_params=planner)
    samples = [x for x in out["population"] if x["t"] >= 420 and sum(x["counts"][0]) == 0 and sum(x["counts"][1]) > 0]
    spread = st.mean(max(x["counts"][1]) - min(x["counts"][1]) for x in samples) if samples else float("nan")
    m = metrics(out["requests"], 0.0, 420.0, 900.0)
    return task, {"spread": spread, "tpot": m["conv_tpot_avg"], "preemptions": sum(out["preemptions"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--general", action="store_true")
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    pairs, planner_file = PAIRS[args.general], PLANNER[args.general]
    tasks = []
    for trace_seed in range(801, 813):
        for sched, (rand, ca) in pairs:
            for variant in range(5):
                tasks.append((trace_seed, rand, variant, sched, planner_file))
            tasks.append((trace_seed, ca, 0, sched, planner_file))
    with ProcessPoolExecutor(args.workers) as pool:
        results = {task[:4]: out for task, out in pool.map(job, tasks)}
    summary = {"general": True, "planner": planner_file, "pairs": {}} if args.general else {"pairs": {}}
    for sched, (rand, ca) in pairs:
        xs, ys = [], []
        for trace_seed in range(801, 813):
            c = results[(trace_seed, ca, 0, sched)]
            for variant in range(5):
                r = results[(trace_seed, rand, variant, sched)]
                xs.append(r["spread"] - c["spread"]); ys.append(100 * (c["tpot"] / r["tpot"] - 1))
                summary["pairs"].setdefault(sched, []).append({"trace_seed": trace_seed, "variant": variant,
                    "extra_spread": xs[-1], "b_pct": ys[-1], "random_preemptions": r["preemptions"],
                    "cost_aware_preemptions": c["preemptions"]})
        n = len(xs); mx, my = st.mean(xs), st.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs); sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        syy = sum((y - my) ** 2 for y in ys)
        slope = sxy / sxx; r = sxy / (sxx * syy) ** 0.5
        through0 = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
        print(f"{sched:5s}: n={n} pairs | extra spread mean {mx:.2f} [{min(xs):.2f}, {max(xs):.2f}] | B mean {my:+.1f}% "
              f"[{min(ys):+.1f}, {max(ys):+.1f}] | slope {slope:+.2f}%/unit (through 0: {through0:+.2f}) | r={r:.2f}")
    args.output.write_text(json.dumps(summary) + "\n")
