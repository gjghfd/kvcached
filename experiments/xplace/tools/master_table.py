"""Paper Table 2 (master strategy table): rows R1-R14 over several seeds, plus single-component pairs.

Each --job is SEED=LABEL=PATH/out/trace. A row's per-seed value is the mean over the runs of that seed; pairs use runs
of the same job (request-level paired bootstrap), and fall back to a cross-job pairing on the same trace, marked "x".
Usage: python3 tools/master_table.py --job 911=V10=.../job-289839/out/trace --job 914=V11=... --job 914=V12=...
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze import metrics  # noqa: E402
from report_main import SLO, bootstrap_ratio, load, paired  # noqa: E402

ROWS = {"R1": "kvcached_native", "R2": "no_move_v2", "R3": "no_move_fair_v2", "R4": "mig_random_v2",
        "R5": "ca_admit_v2", "R6": "ca_return_v2", "R7": "ca_full_v2", "R8": "no_move_tm_v2",
        "R9": "mig_random_tm_v2", "R10": "ca_full_tm_v2", "R11": "ca_full_tm_s4", "R12": "ca_full_tm_slo",
        "R13": "ca_full_v2_nh", "R14": "ca_full_tm_slo_nh"}
PAIRS = [("R1", "R2", "准入方式（控制器侧排队）"), ("R2", "R3", "公平门"), ("R2", "R4", "迁移（并行）"),
         ("R4", "R5", "代价感知准入"), ("R4", "R6", "代价感知回迁"), ("R6", "R7", "回迁之上加代价感知准入"),
         ("R4", "R7", "placement（并行）"), ("R2", "R8", "分时（不迁移）"), ("R8", "R9", "迁移（分时）"),
         ("R4", "R9", "分时（随机）"), ("R9", "R10", "placement（分时）"), ("R7", "R10", "分时（代价感知）"),
         ("R10", "R12", "按需切片"), ("R11", "R12", "按需对周期"), ("R13", "R7", "增长余量（并行）"),
         ("R14", "R12", "增长余量（分时）"), ("R2", "R12", "整体（控制器基线）"), ("R1", "R12", "整体（原生，截断）")]
WINDOW = (420.0, 900.0)


def summarize(requests, origin, transactions):
    result = metrics(requests, origin, *WINDOW, slo=SLO)
    result["moves"] = sum(len(item.get("moves", [])) for item in transactions)
    result["preemptions"] = sum(request.get("preemptions") or 0 for request in requests)
    result["censored"] = sum(1 for request in requests if request.get("censored"))
    return result


def change(run_a, run_b):
    (ra, oa), (rb, ob) = run_a, run_b
    return {"code_ttft": bootstrap_ratio(paired(ra, oa, rb, ob, 0, "ttft_s", WINDOW)),
            "conv_tpot": bootstrap_ratio(paired(ra, oa, rb, ob, 1, "tpot_s", WINDOW))}


def fmt(item, bold=False):
    if not item:
        return "—"
    text = f"{100 * item['change']:+.1f}% [{100 * item['ci95'][0]:+.1f}, {100 * item['ci95'][1]:+.1f}]"
    text = text.replace("-", "−")
    excludes_zero = item["ci95"][0] > 0 or item["ci95"][1] < 0
    return f"**{text}**" if bold and excludes_zero else text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", required=True, help="SEED=LABEL=PATH (repeatable)")
    args = parser.parse_args()
    jobs = []
    for spec in args.job:
        seed, label, path = spec.split("=", 2)
        jobs.append((seed, label, Path(path)))
    seeds = list(dict.fromkeys(seed for seed, _, _ in jobs))
    runs = {}  # (seed, row) -> [(label, requests, origin, summary)]
    for seed, label, root in jobs:
        for row, policy in ROWS.items():
            if (root / policy / "requests.json").exists():
                requests, origin, transactions, _ = load(root, policy)
                runs.setdefault((seed, row), []).append((label, requests, origin,
                                                         summarize(requests, origin, transactions)))
    print("| 行 | 策略 | " + " | ".join(f"seed {seed}: Code TTFT / Conv TPOT avg, p95, p99 / Conv TTFT p99 / Conv SLO / "
                                        f"抢占（轮次）" for seed in seeds) + " |")
    print("|---|---|" + "---|" * len(seeds))
    for row, policy in ROWS.items():
        cells = []
        for seed in seeds:
            items = runs.get((seed, row), [])
            if not items:
                cells.append("—")
                continue
            mean = lambda key: statistics.mean(item[3][key] for item in items)
            censored = any(item[3]["censored"] for item in items)
            cells.append(f"{'≥' if censored else ''}{mean('code_ttft_avg'):.1f}s / {1e3 * mean('conv_tpot_avg'):.2f}, "
                         f"{1e3 * mean('conv_tpot_p95'):.1f}, {1e3 * mean('conv_tpot_p99'):.1f}ms / "
                         f"{mean('conv_ttft_p99'):.2f}s / {mean('conv_slo_attainment'):.2f} / "
                         f"{'+'.join(str(item[3]['preemptions']) for item in items)}"
                         f"（{'+'.join(item[0] for item in items)}）")
        print(f"| {row} | `{policy}` | " + " | ".join(cells) + " |")
    print()
    print("| 对照 | 消融的组件 | " + " | ".join(f"seed {seed} Code TTFT | seed {seed} Conv TPOT" for seed in seeds) + " |")
    print("|---|---|" + "---|---|" * len(seeds))
    for base, other, name in PAIRS:
        cells = []
        for seed in seeds:
            a, b = runs.get((seed, base), []), runs.get((seed, other), [])
            same = [(x, y) for x in a for y in b if x[0] == y[0]]
            if not same and a and b:
                same, cross = [(a[0], b[0])], True
            else:
                cross = False
            if not same:
                cells += ["—", "—"]
                continue
            code, tpot = [], []
            for x, y in same:  # one entry per job that ran both policies
                item = change((x[1], x[2]), (y[1], y[2]))
                tag = f"（跨作业 {x[0]}→{y[0]}）" if cross else (f"（{x[0]}）" if len(same) > 1 else "")
                code.append(fmt(item["code_ttft"]) + tag)
                tpot.append(fmt(item["conv_tpot"], bold=True) + tag)
            cells += ["<br>".join(code), "<br>".join(tpot)]
        print(f"| {base}→{other} | {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
