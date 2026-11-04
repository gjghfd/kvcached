import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from amortized_campaign import check_frozen_sources
from analyze_return_recovery import analyze_recovery
from archive_trace_job import verify_raw
from public_response_inputs import digest, validate
from public_population_calibration import calibration_controls, calibration_plan, verify_calibration
from public_workload_campaign import SYNC_FILES as PREVIOUS_SYNC_FILES
from public_workload_search import make_poisson, write_new
from rebalance_campaign import checked_ssh, protected_sync
from stress_campaign import archive_job, check_pause, poll_job, save_state
from verify_trace import request_metrics, verify


POLICIES = ("no_move", "own_model", "kv_only", "no_move_fair", "random_no_return", "kv_only_headroom", "kv_only_workaware")
CODE_RATES = (.0005, .001)
EXPLORATION_SEED = 2026103104
HELDOUT_SEED = 2026103105
EXPLORATION_S = 3600
HELDOUT_S = 10800
SYNC_FILES = tuple(dict.fromkeys((*PREVIOUS_SYNC_FILES,
    "experiments/dp_opt/public_poisson_followup.py", "experiments/dp_opt/public-poisson-search.sbatch",
    "experiments/dp_opt/analyze_return_recovery.py", "tests/test_dp_opt_public_poisson_followup.py",
    "experiments/dp_opt/public_population_calibration.py", "tests/test_dp_opt_public_population_calibration.py",
    "tests/test_dp_opt_public_population_cli.py",
    "experiments/dp_opt/public_quality_followup.py", "tests/test_dp_opt_public_quality_followup.py",
    "experiments/dp_opt/public_quality_inputs.py", "tests/test_dp_opt_public_quality_inputs.py",
    "experiments/dp_opt/public_source_recovery.py", "tests/test_dp_opt_public_source_recovery.py",
    "tests/test_dp_opt_public_quality_recovery.py",
    "experiments/dp_opt/docs/public-quality-return-study.md",
    "experiments/dp_opt/reports/public-quality-response-profile.md",
    "experiments/dp_opt/docs/public-poisson-search.md")))


def search_rates(cases, candidate):
    if candidate is None or not cases[candidate]["eligible_for_separate_evaluation_rate_candidate"]:
        raise ValueError("Require a pressure/queue-qualified public calibration, not retuned synthetic rates")
    qualified = [name for name, case in cases.items() if case["eligible_for_separate_evaluation_rate_candidate"]]
    lower = min(qualified, key=lambda name: (cases[name]["physical_pressure"]["local_physical_ratio"], name))
    names = [candidate] if candidate == lower else [candidate, lower]
    return [{"calibration_case": name, "conv_rate": cases[name]["rate_rps"][1], "code_rate": code_rate}
            for name in names for code_rate in CODE_RATES]


def prepare(root, parent, state):
    previous = json.loads((parent / "campaign-state.json").read_text())
    if (previous.get("GPU_calibration_complete") is not True
            or previous.get("phase") != "public_source_mechanism_calibration_SHA_verified"
            or previous.get("state_last_observed") != "COMPLETED"):
        raise ValueError("Public native and rate calibration must independently close before evaluation")
    raw = parent / f"live-trace-{previous['job']}"
    verify_raw(raw, (parent / f"raw-{previous['job']}.sha256").read_text())
    result = json.loads((parent / "independent-verification.json").read_text())
    if result != json.loads((raw / "search-analysis.json").read_text()):
        raise ValueError("Parent raw/independent public calibration differs")
    grid = search_rates(result["calibration"], result["candidate_case"])
    frozen = root / "frozen"
    if frozen.exists():
        raise ValueError("Preserve existing exploration freeze, never reroll")
    shutil.copytree(raw / "frozen/inputs", frozen / "inputs")
    inputs = frozen / "inputs"
    validate(inputs)
    traces = {}
    for ordinal, setting in enumerate(grid):
        name = f"explore-{ordinal}.json"
        write_new(frozen / name, make_poisson(inputs, EXPLORATION_SEED, setting["conv_rate"],
                                             EXPLORATION_S, setting["code_rate"]))
        traces[name] = digest(frozen / name)
    shutil.copy2(frozen / "explore-0.json", frozen / "trace.json")
    plan = calibration_plan()
    plan_path = frozen / "population-calibration-plan.json"
    write_new(plan_path, plan)
    config = {"schema_version": 1, "grid": grid, "policies": list(POLICIES),
              "dataset_label": validate(inputs)["dataset_label"],
              "exploration_seed": EXPLORATION_SEED, "heldout_seed": HELDOUT_SEED,
              "exploration_s": EXPLORATION_S, "heldout_s": HELDOUT_S, "warmup_s": 600,
              "parent_job": previous["job"], "fit_sha256": digest(raw / "public-context-fit.json"),
              "population_calibration": {**calibration_controls(plan), "plan_sha256": digest(plan_path)},
              "inputs_sha256": digest(inputs / "input-manifest.json"), "trace_sha256": traces,
              "selection_rule": "equal_group_log_TTFT_gain_vs_no_move_plus_log_Conv_TPOT_gain_vs_random",
              "eligibility": "nonempty_measurement_tenants_and_positive_Conv_TPOT_gain_vs_random",
              "all_negative_empty_draws_retained": True, "research_goal_complete": False}
    write_new(frozen / "evaluation-config.json", config)
    state.update({"parent_job": previous["job"], "fit_sha256": config["fit_sha256"],
                  "protected_vllm_sha256": previous["protected_vllm_sha256"], "configuration": config})
    return config


def validate_frozen(frozen):
    inputs = frozen / "inputs"
    validate(inputs)
    config = json.loads((frozen / "evaluation-config.json").read_text())
    label = validate(inputs)["dataset_label"]
    if config.get("dataset_label", label) != label:
        raise ValueError("Public source label differs from the frozen original input bank")
    if (config["policies"] != list(POLICIES) or config["exploration_seed"] != EXPLORATION_SEED
            or config["heldout_seed"] != HELDOUT_SEED or config["exploration_s"] != EXPLORATION_S
            or config["heldout_s"] != HELDOUT_S or config["warmup_s"] != 600
            or config["inputs_sha256"] != digest(inputs / "input-manifest.json")
            or not 2 <= len(config["grid"]) <= 4):
        raise ValueError("Public finite-grid/held-out controls changed")
    plan_path = frozen / "population-calibration-plan.json"
    plan = json.loads(plan_path.read_text())
    if config.get("population_calibration") != {**calibration_controls(plan), "plan_sha256": digest(plan_path)}:
        raise ValueError("Frozen population calibration differs from the capacity-screened plan")
    for ordinal, setting in enumerate(config["grid"]):
        name = f"explore-{ordinal}.json"
        if (setting["code_rate"] not in CODE_RATES
                or digest(frozen / name) != config["trace_sha256"][name]
                or json.loads((frozen / name).read_text()) != make_poisson(
                    inputs, EXPLORATION_SEED, setting["conv_rate"], EXPLORATION_S, setting["code_rate"])):
            raise ValueError("Public search marks/arrivals cannot be independently rebuilt")
    if (frozen / "trace.json").read_bytes() != (frozen / "explore-0.json").read_bytes():
        raise ValueError("Frozen initial trace pointer changed")
    return config


def phase_metrics(phase):
    trace = json.loads((phase / "trace.json").read_text())
    result = {}
    for policy in POLICIES:
        case = phase / ("r0-" + policy)
        origin = json.loads((case / "workload.json").read_text())["origin"]
        requests = json.loads((case / "requests.json").read_text())
        if (len(requests) != len(trace["requests"])
                or {record["request_id"] for record in requests} != {mark["request_id"] for mark in trace["requests"]}):
            raise ValueError("Every baseline must complete exactly the same requests")
        result[policy] = {}
        for name, model in (("Code", 0), ("Conv", 1), ("combined", None)):
            selected = [record for record in requests if model is None or record["model"] == model]
            result[policy][name] = {
                "all_offered": request_metrics(selected, origin, trace["duration_s"], 120),
                "measurement": request_metrics([record for record in selected if
                    origin + 600 <= record["arrived_at"] < origin + trace["duration_s"]],
                    origin + 600, trace["duration_s"] - 600, 120)}
    return result


def score(metrics):
    before, after = metrics["no_move"], metrics["kv_only_workaware"]
    gains = []
    for group in ("Code", "Conv", "combined"):
        for field in ("mean_ttft_s", "p95_ttft_s", "p99_ttft_s"):
            baseline, treatment = before[group]["measurement"][field], after[group]["measurement"][field]
            if baseline is None or treatment is None or min(baseline, treatment) <= 0:
                return None
            gains.append(math.log(baseline / treatment))
    baseline = metrics["random_no_return"]["Conv"]["measurement"]["mean_tpot_s"]
    treatment = after["Conv"]["measurement"]["mean_tpot_s"]
    if baseline is None or treatment is None or not 0 < treatment < baseline:
        return None
    return sum(gains) / len(gains) + math.log(baseline / treatment)


def select(root):
    config = validate_frozen(root / "frozen")
    population_calibration = verify_calibration(root)
    cases, candidates = {}, []
    for ordinal, setting in enumerate(config["grid"]):
        name = f"explore-{ordinal}"
        phase = root / name
        verification = verify(phase, root / "frozen/inputs/prompt-bank.json")
        phase_config = json.loads((phase / "config.json").read_text())
        if phase_config.get("workaware_fit_sha256") != population_calibration["fit_sha256"]:
            raise ValueError("Public exploration did not use the same pre-benefit population profile")
        if digest(phase / "trace.json") != config["trace_sha256"][name + ".json"]:
            raise ValueError("Served exploration differs from its fixed public trace")
        metrics = phase_metrics(phase)
        value = score(metrics)
        cases[name] = {"setting": setting, "metrics": metrics, "score": value, "verification": verification}
        if value is not None:
            candidates.append((-value, ordinal, name))
    best = min(candidates)[2] if candidates else None
    return {"cases": cases, "best_case": best, "population_calibration": population_calibration,
            "dataset_label": config.get("dataset_label", "openthoughts3"),
            "best_setting": cases[best]["setting"] if best else None,
            "scope": "best_within_two_qualified_Conv_rates_two_Code_rates_not_global_optimum",
            "P99_statistically_significant": False, "research_goal_complete": False}


def freeze_heldout(root):
    result = select(root)
    write_new(root / "selection.json", result)
    if result["best_setting"] is not None:
        setting = result["best_setting"]
        trace = make_poisson(root / "frozen/inputs", HELDOUT_SEED, setting["conv_rate"], HELDOUT_S, setting["code_rate"])
        write_new(root / "heldout-trace.json", trace)
    return result


def analyze(root):
    result = select(root)
    if result != json.loads((root / "selection.json").read_text()):
        raise ValueError("Independent exploration selection changed")
    if result["best_setting"] is not None:
        setting = result["best_setting"]
        expected = make_poisson(root / "frozen/inputs", HELDOUT_SEED, setting["conv_rate"], HELDOUT_S, setting["code_rate"])
        if json.loads((root / "heldout/trace.json").read_text()) != expected:
            raise ValueError("Held-out trace changed or was rerolled after outcomes")
        heldout_config = json.loads((root / "heldout/config.json").read_text())
        if heldout_config.get("workaware_fit_sha256") != result["population_calibration"]["fit_sha256"]:
            raise ValueError("Held-out cannot change the exploration population profile")
        result["heldout"] = {"verification": verify(root / "heldout", root / "frozen/inputs/prompt-bank.json"),
                             "metrics": phase_metrics(root / "heldout"),
                             "return_recovery": analyze_recovery(root / "heldout"),
                             "score": score(phase_metrics(root / "heldout"))}
    result["all_heldout_negative_results_retained"] = True
    return result


def report(result, job, evidence):
    label = result.get("dataset_label", "openthoughts3")
    dataset_name = {"openthoughts3": "OpenThoughts3", "stratos": "Bespoke-Stratos",
                    "openthoughts3_closed": "OpenThoughts3 structurally-closed original-response cohort"}.get(label, label)
    lines = ["# Original public-response finite-grid results", "", f"Job{job}; evidence `{evidence}`.",
             dataset_name + " original-response-length replay and SWE-smith real100K/O64; not human chat or natural EOS.",
             "Best is restricted to the declared grid. All warmup/negative/empty draws are retained; P99 is descriptive.",
             f"Selected setting: `{json.dumps(result['best_setting'], sort_keys=True)}`.", ""]
    if label == "stratos":
        lines.append("Source is a SHA-pinned official dataset-viewer content snapshot; upstream Git/LFS revision is not verified.")
    cases = [(name, case["metrics"]) for name, case in result["cases"].items()]
    if "heldout" in result:
        cases.append(("fixed-heldout", result["heldout"]["metrics"]))
    for name, metrics in cases:
        for scope in ("all_offered", "measurement"):
            lines.extend([f"## {name}: {scope}", "", "TTFT seconds: avg/P95/P99; Conv TPOT milliseconds.", "",
                          "|Policy|Code TTFT|Conv TTFT|Combined TTFT|Conv TPOT|Code/Conv count|",
                          "|---|---|---|---|---:|---:|"])
            for policy in POLICIES:
                groups = {group: metrics[policy][group][scope] for group in ("Code", "Conv", "combined")}
                cells = ["/".join("N/A" if groups[group][field] is None else f"{groups[group][field]:.5f}"
                         for field in ("mean_ttft_s", "p95_ttft_s", "p99_ttft_s")) for group in groups]
                tpot = groups["Conv"]["mean_tpot_s"]
                lines.append("|" + "|".join([policy, *cells, "N/A" if tpot is None else f"{tpot * 1000:.5f}",
                    f"{groups['Code']['completed_requests']}/{groups['Conv']['completed_requests']}"]) + "|")
            lines.append("")
    lines.extend(["no_move is the same full-reservation custom admission, not unmodified vLLM dynamic growth.",
                  "Real pause/copy/control costs are included. Return recovery and all raw verification are in JSON.",
                  "Initial TTFT or a positive TPOT score alone is not causal return repayment proof.",
                  "Generalization/global optimum and P99 statistical significance are not established."])
    return "\n".join(lines) + "\n"


def submit_command(environment, ledger):
    exports = "; ".join(f"export {name}={shlex.quote(value)}" for name, value in environment.items())
    return (f'set -eu; test -z "$(squeue -h -u "$USER")"; test ! -e {shlex.quote(ledger)}; '
            'test "$(git -C "$HOME/kvcached/kvcached" branch --show-current)" = main; '
            'test "$(git -C "$HOME/kvcached/kvcached" remote get-url origin)" = git@github.com:gjghfd/kvcached.git; '
            + exports + '; sbatch --parsable --time=80:00:00 --export=ALL '
            '--output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
            '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
            '"$HOME/kvcached/kvcached/experiments/dp_opt/public-poisson-search.sbatch" | '
            f'tee {shlex.quote(ledger)}')


def run(root, parent):
    repo = Path(__file__).resolve().parents[2]
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "campaign-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "repo": str(repo), "parent_root": str(parent), "phase": "waiting_public_parent_independent_closure",
        "publication_authorized": False, "total_research_goal_completed": False,
        "frozen_runtime_sha256": {name: digest(repo / name) for name in SYNC_FILES if name.endswith((".py", ".sh", ".sbatch"))}}
    if state["repo"] != str(repo) or state["parent_root"] != str(parent) or "diagnosis_required" in state["phase"]:
        raise RuntimeError("Reconcile existing public follow-up identity/failure before resuming")
    if state.get("GPU_evaluation_complete"):
        return
    save_state(root, state)
    try:
        if "job" not in state:
            if state["phase"] != "waiting_public_parent_independent_closure":
                raise RuntimeError("Do not duplicate the recorded submission intent")
            while True:
                check_pause(root)
                check_pause(parent)
                check_frozen_sources(repo, state)
                previous = json.loads((parent / "campaign-state.json").read_text())
                if "diagnosis_required" in previous["phase"]:
                    raise RuntimeError("Public parent failed; diagnose complete raw archive before next allocation")
                state.update({"parent_job": previous.get("job"), "parent_phase": previous["phase"],
                              "parent_last_poll_at": previous.get("last_poll_at"), "SSH_queries_while_waiting": False})
                save_state(root, state)
                if previous.get("GPU_calibration_complete"):
                    break
                time.sleep(300)
            tests = sorted(name for name in SYNC_FILES if name.startswith("tests/"))
            with (root / "cpu-regression.log").open("w") as output:
                regression = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=repo,
                                            stdout=output, stderr=subprocess.STDOUT)
            (root / "cpu-regression.exit").write_text(str(regression.returncode) + "\n")
            if regression.returncode:
                raise RuntimeError("Public follow-up regression failed; do not deploy")
            check_frozen_sources(repo, state)
            from public_quality_followup import follow_quality

            if follow_quality(root, parent, repo, state):
                return
            config = prepare(root, parent, state)
            save_state(root, state)
            uploads = tuple(str(path.relative_to(root)) for path in sorted((root / "frozen").rglob("*"))
                            if path.is_file() and path.name != "trace.json")
            stage, home = protected_sync(repo, root, state, files=SYNC_FILES, prefix="public-poisson-search", upload_files=uploads)
            environment = {"DP_OPT_PUBLIC_POISSON_UPLOAD": stage + "/upload", "DP_OPT_PUBLIC_FIT_JOB": str(config["parent_job"]),
                           "DP_OPT_PUBLIC_FIT_SHA256": config["fit_sha256"],
                           "DP_OPT_MOVE_FIXED_S": "1.7743364535272121", "DP_OPT_MOVE_BANDWIDTH_GIB_S": "11.64086282917902"}
            ledger = stage + "/submitted-job.txt"
            state.update({"phase": "submission_intent_do_not_duplicate", "remote_sync_stage": stage,
                          "submission_ledger": ledger, "frozen_environment": environment})
            save_state(root, state)
            output = checked_ssh(submit_command(environment, ledger))
            state.update({"job": int(output.split(";")[0]), "phase": "submitted_public_grid_then_fixed_heldout"})
            save_state(root, state)
            remote = home + f"/run/kvcached-paracloud/state/live-trace-{state['job']}"
            state["remote_watcher_pid"] = int(checked_ssh(f'''set -eu
mkdir -p {shlex.quote(remote)}
DP_OPT_REMOTE_MAX_POLLS=2881 nohup bash "$HOME/kvcached/kvcached/experiments/dp_opt/watch-trace-remote.sh" {state['job']} \\
    >> {shlex.quote(remote + '/remote-query.log')} 2>&1 < /dev/null &
printf '%s\\n' "$!"'''))
            save_state(root, state)
        terminal = poll_job(root, state["job"], state)
        archive_job(root, state["job"], repo / "experiments/dp_opt", state)
        raw = root / f"live-trace-{state['job']}"
        check_frozen_sources(repo, state)
        if terminal != "COMPLETED" or (raw / "exit-status").read_text().strip() != "0":
            raise RuntimeError("Public grid/heldout failed; complete raw archive retained for diagnosis")
        result = analyze(raw)
        if result != json.loads((raw / "evaluation-analysis.json").read_text()):
            raise RuntimeError("Independent public grid/heldout differs from compute analysis")
        write_new(root / "independent-verification.json", result)
        (repo / "experiments/dp_opt/reports/public-poisson-results.md").write_text(report(result, state["job"], root))
        state.update({"phase": "public_grid_heldout_SHA_verified_requires_conclusion",
                      "GPU_evaluation_complete": True, "best_setting": result["best_setting"],
                      "next_action": "report_every_setting_and_heldout_then_generally_diagnose_any_unrepaid_return"})
        save_state(root, state)
    except Exception as error:
        state.update({"phase": "diagnosis_required_no_automatic_duplicate_submission", "error": repr(error)})
        save_state(root, state)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--action", choices=("run", "validate", "select", "analyze"), default="run")
    args = parser.parse_args()
    if args.action == "run":
        run(args.root.resolve(), args.parent_root.resolve())
    else:
        action = {"validate": lambda root: validate_frozen(root / "frozen"),
                  "select": freeze_heldout, "analyze": analyze}[args.action]
        print(json.dumps(action(args.root), indent=2))


if __name__ == "__main__":
    main()
