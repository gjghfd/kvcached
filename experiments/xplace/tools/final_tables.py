"""Final tables for the short-job campaign: per-trial metrics, correctness, within-job and cross-job comparisons."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

from analyze import metrics  # noqa: E402
from report_main import SLO, bootstrap_ratio, mechanism, paired  # noqa: E402

PAIRS = [  # (claim, base policy, other policy)
    ("TM", "ca_full", "ca_full_tm"), ("TM", "no_move", "no_move_tm"), ("TM", "mig_random", "mig_random_tm"),
    ("A", "no_move", "ca_full"), ("A", "no_move", "mig_random"), ("A", "no_move_fair", "ca_full"),
    ("A+TM", "no_move_tm", "ca_full_tm"), ("A+TM", "no_move", "ca_full_tm"),
    ("B", "mig_random", "ca_full"), ("B", "no_move", "ca_full"),
    ("ablation", "mig_random", "ca_admit"), ("ablation", "mig_random", "ca_return"),
    ("ablation", "ca_admit", "ca_full"), ("ablation", "ca_return", "ca_full"),
    ("B+TM", "mig_random_tm", "ca_full_tm"),
]


def load_job(job):
    phases = json.loads((job / "out" / "phases.json").read_text())["phases"]
    probe = next((p for p in phases if p["phase"] == "probe"), {})
    trace = json.loads(next((job / "inputs").glob("short-*.json")).read_text())
    expected = {r["request_id"]: r["output_tokens"] for r in trace["requests"]}
    node = "?"
    for line in (job.parent / "logs").glob(f"slurm-{job.name.split('-')[-1]}.out"):
        node = line.read_text().split("node=")[-1].split()[0] if "node=" in line.read_text() else "?"
    trials = {}
    for root in sorted((job / "out" / "trace").iterdir()):
        if not (root / "summary.json").exists():
            continue
        summary = json.loads((root / "summary.json").read_text())
        if summary["status"] != "passed":
            continue
        requests = json.loads((root / "requests.json").read_text())
        origin = json.loads((root / "workload.json").read_text())["origin"]
        migrations = json.loads((root / "migrations.json").read_text())
        holds = json.loads((root / "holds.json").read_text()) if (root / "holds.json").exists() else []
        trials[root.name] = {"requests": requests, "origin": origin, "summary": summary, "holds": holds,
                             "metrics": metrics(requests, origin, 300.0, trace["duration_s"], slo=SLO),
                             "mechanism": mechanism(root, (300.0, trace["duration_s"])),
                             "exact": sum(expected.get(r["trace_request_id"]) == r["output_tokens"] for r in requests),
                             "expected": len(expected), "moves": len(migrations),
                             "verified": sum(bool(m.get("bitwise_verified")) for m in migrations),
                             "gaps": sorted(r["max_token_gap_s"] for r in requests if r["model"] == 1)}
    return {"job": job.name.split("-")[-1], "probe_ms": probe.get("step_ms", math.nan), "node": node,
            "trials": trials, "window": (300.0, trace["duration_s"])}


def fmt_change(item):
    if item is None:
        return "n/a"
    return f"{100 * item['change']:+.1f}% [{100 * item['ci95'][0]:+.1f}, {100 * item['ci95'][1]:+.1f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", type=Path, nargs="+")
    parser.add_argument("--nodes", nargs="*", default=[], help="job=node labels, e.g. 283933=d1n41a15g02")
    args = parser.parse_args()
    nodes = dict(item.split("=", 1) for item in args.nodes)
    jobs = [load_job(job) for job in args.jobs]
    print("| job | node | probe ms | policy | Code TTFT avg / p95 (s) | Conv TPOT avg / p95 / p99 (ms) | "
          "Conv TTFT avg / p99 (s) | Conv max gap p50 (s) | SLO Code / Conv | moves (verified) | holds | exact |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for job in jobs:
        for name, trial in job["trials"].items():
            m = trial["metrics"]
            print(f"| {job['job']} | {nodes.get(job['job'], job['node'])} | {job['probe_ms']:.1f} | {name} | "
                  f"{m['code_ttft_avg']:.1f} / {m['code_ttft_p95']:.1f} | {1e3 * m['conv_tpot_avg']:.2f} / "
                  f"{1e3 * m['conv_tpot_p95']:.1f} / {1e3 * m['conv_tpot_p99']:.1f} | {m['conv_ttft_avg']:.2f} / "
                  f"{m['conv_ttft_p99']:.1f} | {trial['gaps'][len(trial['gaps']) // 2]:.1f} | "
                  f"{m['code_slo_attainment']:.2f} / {m['conv_slo_attainment']:.2f} | {trial['moves']} "
                  f"({trial['verified']}) | {len(trial['holds'])} | {trial['exact']}/{trial['expected']} |")
    print("\n| job | policy | Conv TPOT = no Code + Code prefill + Code decode + held (ms) | no-Code step (ms) | "
          "no-Code G16 share | Code prefill mean (s) |")
    print("|---|---|---|---|---|---|")
    for job in jobs:
        for name, trial in job["trials"].items():
            parts = trial["mechanism"]["tpot_by_state_ms"]
            codes = [r for r in trial["requests"] if r["model"] == 0]
            prefill = sum(r["first_token_at"] - r["reserved_at"] for r in codes) / len(codes)
            print(f"| {job['job']} | {name} | {1e3 * trial['metrics']['conv_tpot_avg']:.2f} = "
                  f"{parts.get('none', 0):.2f} + {parts.get('prefill', 0):.2f} + {parts.get('decode', 0):.2f} + "
                  f"{parts.get('paused', 0):.2f} | {trial['mechanism']['none_step_ms']:.2f} | "
                  f"{trial['mechanism']['none_b16_share']:.2f} | {prefill:.1f} |")
    print("\n| claim | base → other | where | Code TTFT [95% CI] | Conv TPOT [95% CI] | Conv TTFT [95% CI] |")
    print("|---|---|---|---|---|---|")
    for claim, base, other in PAIRS:
        rows = []
        for job in jobs:
            if base in job["trials"] and other in job["trials"]:
                rows.append((f"within {job['job']}", job["trials"][base], job["trials"][other], job["window"]))
        if not rows:
            for job_a in jobs:
                for job_b in jobs:
                    if job_a is not job_b and base in job_a["trials"] and other in job_b["trials"]:
                        rows.append((f"{job_a['job']}({job_a['probe_ms']:.1f}ms) → {job_b['job']}"
                                     f"({job_b['probe_ms']:.1f}ms)", job_a["trials"][base], job_b["trials"][other],
                                     job_a["window"]))
        for where, a, b, window in rows:
            cells = [fmt_change(bootstrap_ratio(paired(a["requests"], a["origin"], b["requests"], b["origin"],
                                                       model, key, window)))
                     for model, key in ((0, "ttft_s"), (1, "tpot_s"), (1, "ttft_s"))]
            print(f"| {claim} | {base} → {other} | {where} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
