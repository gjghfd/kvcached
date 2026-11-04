"""Compare live per-policy metrics with a pre-registered simulator forecast (sim/predict.py output)."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))

from analyze import metrics  # noqa: E402

KEYS = [("code_ttft_avg", "Code TTFT avg s", 1), ("code_ttft_p95", "Code TTFT p95 s", 1),
        ("conv_tpot_avg", "Conv TPOT avg ms", 1e3), ("conv_tpot_p99", "Conv TPOT p99 ms", 1e3),
        ("conv_ttft_avg", "Conv TTFT avg s", 1), ("conv_ttft_p99", "Conv TTFT p99 s", 1), ("moves", "moves", 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_roots", type=Path, nargs="+", help=".../out/trace directories (one or more jobs)")
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--warmup-s", type=float, default=300)
    parser.add_argument("--end-s", type=float, required=True)
    args = parser.parse_args()
    forecast = json.loads(args.forecast.read_text())["forecast"]
    print(f"{'job/policy':34s}" + "".join(f"{label:>30s}" for _, label, _ in KEYS))
    for trace_root in args.trace_roots:
        for root in sorted(path for path in trace_root.iterdir() if (path / "requests.json").exists()):
            workload = json.loads((root / "workload.json").read_text())
            live = metrics(json.loads((root / "requests.json").read_text()), workload["origin"], args.warmup_s,
                           args.end_s)
            live["moves"] = len(json.loads((root / "migrations.json").read_text()))
            expected = forecast.get(root.name, {})
            cells = []
            for key, _, scale in KEYS:
                value = scale * live.get(key, math.nan)
                band = expected.get(key)
                if band:
                    inside = band["p5"] <= live.get(key, math.nan) <= band["p95"]
                    cells.append(f"{value:8.1f} | {scale * band['mean']:6.1f} [{scale * band['p5']:5.1f},"
                                 f"{scale * band['p95']:5.1f}]{'' if inside else '*'}")
                else:
                    cells.append(f"{value:8.1f} | {'n/a':>15s}")
            print(f"{trace_root.parents[1].name + '/' + root.name:34s}" + "".join(f"{cell:>30s}" for cell in cells))
    print("cells: live | forecast mean [p5, p95]; * = live outside the forecast band")


if __name__ == "__main__":
    main()
