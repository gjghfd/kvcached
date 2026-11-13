"""Explicit evidence allowlist for the migration-only manuscript."""
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

XPLACE = Path(__file__).resolve().parents[1]
ROOT = XPLACE.parents[2]
EVIDENCE = ROOT / "evidence"
V16 = EVIDENCE / "xplace-v16-general-20261106/job-295488"
SEEDS = (911, 914)
WINDOW = (420.0, 900.0)
POLICIES = {"B": "no_move_v2_nh_g", "X": "ca_full_v2_g"}
NAMES = {"B": "不迁移基线", "X": "XPlace"}
CAPACITY_GIB = 170 * 96 / 1024
SLO = {"code": {"ttft_s": 120.0}, "conv": {"ttft_s": 5.0, "tpot_s": 0.050}}
J1B_GRID = EVIDENCE / "xplace-j1b-calib-20261103/job-282105/out/calibration/conv-grid.json"
DENSE_ROOT = EVIDENCE / "xplace-fig2-dense-20261107/job-300123/out/calibration/imbalance"
DENSE_ANALYSIS = ROOT / "Paper/draft/imbalance-dense-analysis.json"
TP2_ROOT = EVIDENCE / "xplace-fig2-ep4-tp2-20261108-r3/job-303594/out/calibration/imbalance"
TP2_ANALYSIS = ROOT / "Paper/draft/imbalance-ep4-tp2-analysis.json"
SWEEP = EVIDENCE / "xplace-sim/rq4-sweep-general-20261107.json"
IMBALANCE = EVIDENCE / "xplace-sim/b-vs-imbalance-general-20261107.json"
FIGURES = ("migration1_fragmentation", "migration2_padding", "migration3_design", "migration4_results",
           "migration5_mechanism", "migration6_factors", "imbalance_topologies")

sys.path.insert(0, str(XPLACE / "tools"))
from analyze import metrics
from report_main import bootstrap_ratio, paired


def read(path):
    return json.loads(path.read_text())


def trial_path(seed, sid):
    if seed not in SEEDS or sid not in POLICIES:
        raise ValueError(f"Trial outside the manuscript allowlist: {seed}, {sid}")
    segment = f"s{seed}-{'a' if sid == 'B' else 'c'}"
    return V16 / "out/trace" / segment / POLICIES[sid]


@lru_cache(maxsize=None)
def trial(seed, sid):
    root = trial_path(seed, sid)
    result = {name: read(root / f"{name}.json") for name in
              ("requests", "workload", "transactions", "migrations", "summary")}
    if result["workload"]["policy"] != POLICIES[sid]:
        raise ValueError("Archived policy does not match the allowlist")
    for filename in ("episodes.json", "slices.json", "holds.json"):
        if (root / filename).exists() and read(root / filename):
            raise ValueError(f"Unexpected execution mode in {root / filename}")
    return result


@lru_cache(maxsize=None)
def report(seed):
    return read(V16 / f"report-s{seed}.json")


def summary(seed, sid):
    data = trial(seed, sid)
    return metrics(data["requests"], data["workload"]["origin"], *WINDOW, slo=SLO)


def cohort(seed, sid, model):
    return [request for request in trial(seed, sid)["requests"]
            if request["model"] == model and WINDOW[0] <= request["arrival_s"] < WINDOW[1]]


def comparison(seed):
    baseline, ours = trial(seed, "B"), trial(seed, "X")
    result = {}
    for metric, model, field in (("code_ttft", 0, "ttft_s"), ("conv_tpot", 1, "tpot_s"),
                                 ("conv_ttft", 1, "ttft_s")):
        values = paired(baseline["requests"], baseline["workload"]["origin"], ours["requests"],
                        ours["workload"]["origin"], model, field, WINDOW)
        result[metric] = bootstrap_ratio(values)
    return result


@lru_cache(maxsize=None)
def population(seed, sid):
    return [json.loads(line) for line in (trial_path(seed, sid) / "population.jsonl").read_text().splitlines()
            if line.strip()]


def sample_before(samples, stamp):
    return max((sample for sample in samples if sample["t"] <= stamp), key=lambda sample: sample["t"])


def no_code_spread(seed, sid):
    samples = [sample for sample in population(seed, sid) if sample["t"] >= WINDOW[0]
               and not sum(sample["counts"][0]) and sum(sample["counts"][1])]
    return sum(max(sample["counts"][1]) - min(sample["counts"][1]) for sample in samples) / len(samples)


def free_at_arrivals(seed=914):
    samples = population(seed, "B")
    rows = []
    for request in sorted(cohort(seed, "B", 0), key=lambda item: item["arrival_s"]):
        sample = sample_before(samples, request["arrival_s"])
        free = [CAPACITY_GIB - sum(mapped) for mapped in zip(*sample["mapped_gib"])]
        need = math.ceil((request["prompt_tokens"] + 256) / 16) * 1.5 / 1024
        rows.append({"arrival_s": request["arrival_s"], "free_gib": free, "need_gib": need,
                     "queue_s": request["reserved_at"] - request["arrived_at"]})
    return rows


@lru_cache(maxsize=None)
def sweep_rows():
    return [row for row in read(SWEEP)["runs"] if row["policy"] in POLICIES.values()]


def imbalance_rows():
    return read(IMBALANCE)["pairs"]["corun"]
