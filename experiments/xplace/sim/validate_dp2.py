"""Replay the DP=2 live cases in the simulator and compare Code TTFT / Conv TPOT per policy."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE.parent / "runtime"))

from analyze import metrics  # noqa: E402
from cost_model import default_dp2_params  # noqa: E402
from simulator import simulate  # noqa: E402

EVIDENCE = HERE.parents[3] / "evidence"
NATIVE = {
    "s1-N2": "public-performance-parallel-20261102/live-trace-278514/native-2026103101-N2",
    "s1-N3": "public-performance-followup-20261102-r5/stages/native-2026103101-N3/live-trace-278898/native-2026103101-N3",
    "s1-N4": "public-performance-followup-20261102-r5/stages/native-2026103101-N4/live-trace-279153/native-2026103101-N4",
    "s2-N2": "public-performance-followup-20261102-r5/stages/native-2026103102-N2/live-trace-279520/native-2026103102-N2",
    "s2-N3": "public-performance-followup-20261102-r5/stages/native-2026103102-N3/live-trace-280104/native-2026103102-N3",
    "s2-N4": "public-performance-followup-20261102-r5/stages/native-2026103102-N4/live-trace-280582/native-2026103102-N4",
}
POISSON = {
    "p-259091": "workaware-pressure-1p2-20261027/live-trace-259091/paired-trace",
    "p-260757": "workaware-random-baseline-20261028/live-trace-260757/paired-trace",
}
PAIRS = {"no_move": "no_move", "random_no_return": "mig_random", "kv_only_workaware": "ca_return"}


def live_metrics(root, policy):
    directory = root / f"r0-{policy}"
    requests = json.loads((directory / "requests.json").read_text())
    workload = json.loads((directory / "workload.json").read_text())
    for request in requests:
        request.setdefault("output_tokens", len(request.get("token_ids", [])) or request.get("output_tokens"))
    return metrics(requests, workload["origin"])


def load_trace(root):
    for candidate in (root / "trace.json", root.parent / "trace.json"):
        if candidate.exists():
            return json.loads(candidate.read_text())
    raise FileNotFoundError(root)


def main(cases=None):
    params = default_dp2_params()
    rows = []
    selected = {**NATIVE, **POISSON} if cases is None else {name: path for name, path in {**NATIVE, **POISSON}.items()
                                                             if name in cases}
    for name, relative in selected.items():
        root = EVIDENCE / relative
        trace = load_trace(root)
        for live_policy, sim_policy in PAIRS.items():
            if not (root / f"r0-{live_policy}").exists():
                continue
            live = live_metrics(root, live_policy)
            sim = metrics(simulate(trace, sim_policy, sim_params={"num_ranks": 2, "budget_gib": 10.0},
                                   step_params=params)["requests"])
            rows.append({"case": name, "policy": sim_policy,
                         "code_ttft_live": live.get("code_ttft_avg"), "code_ttft_sim": sim.get("code_ttft_avg"),
                         "conv_tpot_live_ms": 1e3 * live.get("conv_tpot_avg", float("nan")),
                         "conv_tpot_sim_ms": 1e3 * sim.get("conv_tpot_avg", float("nan"))})
            row = rows[-1]
            print(f"{name:9s} {sim_policy:10s} CodeTTFT live {row['code_ttft_live']:7.1f} sim {row['code_ttft_sim']:7.1f} | "
                  f"ConvTPOT live {row['conv_tpot_live_ms']:6.2f} sim {row['conv_tpot_sim_ms']:6.2f} ms", flush=True)
    return rows


if __name__ == "__main__":
    main(sys.argv[1:] or None)
