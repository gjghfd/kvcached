"""Archived live runs used by the paper figures (read-only), plus small loaders."""
import json
import sys
from functools import lru_cache
from pathlib import Path

XPLACE = Path(__file__).resolve().parents[1]
EVIDENCE = XPLACE.parents[2] / "evidence"
sys.path.insert(0, str(XPLACE / "tools"))
sys.path.insert(0, str(XPLACE / "sim"))

from analyze import metrics  # noqa: E402
from report_main import bootstrap_ratio, load, paired  # noqa: E402

WINDOW = (420.0, 900.0)
SLO = {"code": {"ttft_s": 120.0}, "conv": {"ttft_s": 5.0, "tpot_s": 0.050}}
CAPACITY_GIB = 170 * 96 / 1024  # ballast-capped shared KV per GPU (16 GiB cap)

RUNS = {
    "V8": EVIDENCE / "xplace-v8-current-20261104/job-289203/out/trace",
    "V10": EVIDENCE / "xplace-v10-freeblocks-20261105/job-289839/out/trace",
    "V11": EVIDENCE / "xplace-v11-replicate-20261105/job-289851/out/trace",
    "V12": EVIDENCE / "xplace-v11-replicate-20261105/job-290422/out/trace",
    "V13-l14": EVIDENCE / "xplace-v13-anchors-20261105/job-291000/out/trace/s919-l14",
    "V13-l14c": EVIDENCE / "xplace-v13-anchors-20261105/job-291000/out/trace/s919-l14-corun",
    "V13-c24": EVIDENCE / "xplace-v13-anchors-20261105/job-291000/out/trace/s911-c24",
}
V14 = EVIDENCE / "xplace-v14-ablation-20261106/job-292505/out/trace"  # final-scheme ablations; absent until archived
RUNS.update({f"V14-{segment}": V14 / segment for segment in ("s911-a", "s911-b", "s914-a", "s914-b")})
# no-headroom baselines with the final scheme under planner v4 (relief floor only on migration destinations)
V15 = EVIDENCE / "xplace-v15-nh-baselines-20261106/job-294007/out/trace"  # job 293986 stopped at the node probe
RUNS.update({f"V15-{segment}": V15 / segment for segment in ("s911", "s914", "s919-l14")})
# V16: the general (type-agnostic) algorithm, policies `<old name>_g`, all in one allocation (job 295488). by-seed/<seed>
# holds symlinks to the trial directories of every segment of that seed, so one run key covers all policies of a seed
V16 = EVIDENCE / "xplace-v16-general-20261106/job-295488"
RUNS.update({f"V16-{seed}": V16 / "out/by-seed" / seed for seed in ("s911", "s914", "s919-l14")})
V16_REPORTS = {911: V16 / "report-s911.json", 914: V16 / "report-s914.json"}  # report_main.py output per seed
J1B_GRID = EVIDENCE / "xplace-j1b-calib-20261103/job-282105/out/calibration/conv-grid.json"
SIM = EVIDENCE / "xplace-sim"
# runs per seed at the main operating point (Conv 0.12 RPS, 16 GiB); later entries override earlier ones per policy
SEED_RUNS = {911: ["V10", "V14-s911-a", "V14-s911-b", "V15-s911", "V16-s911"],
             914: ["V11", "V12", "V14-s914-a", "V14-s914-b", "V15-s914", "V16-s914"]}
# same-allocation jobs per seed (a job may span several trace segments), oldest first; used for paired ablations
JOBS = {911: [["V10"], ["V14-s911-a", "V14-s911-b"], ["V15-s911"], ["V16-s911"]],
        914: [["V11"], ["V12"], ["V14-s914-a", "V14-s914-b"], ["V15-s914"], ["V16-s914"]]}
# kvcached native is not affected by the controller: the paper reuses V10 (seed 911) and V11 (seed 914), cut at 1900 s
NATIVE_RUNS = {911: "V10", 914: "V11"}
# trace names used in the paper (traces 1 and 2 at the main operating point, trace 3 at Conv 0.14 RPS)
TRACE = {911: "踪迹1", 914: "踪迹2", 919: "踪迹3"}

NAMES = {
    "kvcached_native": "kvcached 原生", "no_move_v2": "受控基线", "no_move_fair_v2": "公平门",
    "mig_random_v2": "随机迁移", "ca_admit_v2": "代价感知准入", "ca_return_v2": "代价感知回迁",
    "ca_full_v2": "代价感知放置", "ca_full_v2_nh": "无增长余量", "no_move_tm_v2": "分时",
    "mig_random_tm_v2": "分时+随机迁移", "ca_full_tm_v2": "分时+代价感知", "ca_full_tm_s4": "分时+周期切片",
    "ca_full_tm_slo": "XPlace（本文）", "ca_full_tm_slo_nh": "本文−增长余量",
    # general algorithm (V16), paper strategy table S1-S10
    "no_move_v2_nh_g": "受控基线", "ca_full_v2_g": "并行+代价感知放置",
    "no_move_tm_slo_nh_g": "分时+按需切片（不迁移）", "ca_full_tm_v2_g": "分时+代价感知（无切片）",
    "ca_full_tm_slo_g": "XPlace（本文）", "ca_admit_tm_slo_g": "XPlace−回迁", "ca_return_tm_slo_g": "XPlace−代价感知准入",
    "mig_random_tm_slo_g": "XPlace−代价感知放置", "ca_full_tm_slo_nh_g": "XPlace−增长余量",
}
# paper strategy IDs (general algorithm; S1 is the stock deployment, unaffected by the controller)
SID = {"kvcached_native": "S1", "no_move_v2_nh_g": "S2", "ca_full_v2_g": "S3", "no_move_tm_slo_nh_g": "S4",
       "ca_full_tm_v2_g": "S5", "ca_full_tm_slo_g": "S6", "ca_admit_tm_slo_g": "S7", "ca_return_tm_slo_g": "S8",
       "mig_random_tm_slo_g": "S9", "ca_full_tm_slo_nh_g": "S10"}
POLICY = {sid: policy for policy, sid in SID.items()}


def root(run):
    return RUNS[run]


@lru_cache(maxsize=None)
def run_data(run, policy):
    """(requests incl. censored records, origin, transactions, holds) of one archived trial."""
    return load(RUNS[run], policy)


def json_file(run, policy, name):
    path = RUNS[run] / policy / name
    return json.loads(path.read_text()) if path.exists() else []


@lru_cache(maxsize=None)
def v16_report(seed):
    """report_main.py output of one V16 seed (table, comparisons, mechanism), window [420, 900) s."""
    return json.loads(V16_REPORTS[seed].read_text())


def population(run, policy):
    path = RUNS[run] / policy / "population.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@lru_cache(maxsize=None)
def summary_metrics(run, policy):
    requests, origin, _, _ = run_data(run, policy)
    return metrics(requests, origin, *WINDOW, slo=SLO)


def pairs(run_a, policy_a, run_b, policy_b, model, key):
    """(a, b) values of the same trace requests in the window (model 0 Code, 1 Conv)."""
    ra, oa, _, _ = run_data(run_a, policy_a)
    rb, ob, _, _ = run_data(run_b, policy_b)
    return paired(ra, oa, rb, ob, model, key, WINDOW)


def pair(run_a, policy_a, run_b, policy_b, model, key):
    """Paired change b/a - 1 of the means with bootstrap CI over requests."""
    return bootstrap_ratio(pairs(run_a, policy_a, run_b, policy_b, model, key))


def runs_with(seed, policy):
    return [run for run in SEED_RUNS[seed] if (RUNS[run] / policy / "requests.json").exists()]


def window_conv(run, policy):
    requests, origin, _, _ = run_data(run, policy)
    return [r for r in requests if r["model"] == 1 and WINDOW[0] <= r["arrival_s"] < WINDOW[1]], origin


def no_code_spread(run, policy):
    """Mean (max - min) Conv count across ranks while no Code is resident and Conv is not held (t >= 420 s)."""
    samples = [x for x in population(run, policy) if x["t"] >= WINDOW[0] and sum(x["counts"][0]) == 0
               and sum(x["counts"][1]) > 0 and not x.get("held")]
    return sum(max(x["counts"][1]) - min(x["counts"][1]) for x in samples) / len(samples)
