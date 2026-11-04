import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path, PurePosixPath

from amortized_campaign import check_frozen_sources, paired_window
from analyze_return_recovery import analyze_recovery
from analyze_return_repayment import analyze as repayment
from analyze_return_token_windows import analyze as token_windows
from archive_trace_job import verify_raw
from closed_phase_observer import OPTIONS, pull_phase
from stress_campaign import check_pause
from verify_trace import verify


PHASES = tuple(f"probe-{seed}" for seed in (20261030, 20261031, 20261032))


def latest_phase(path):
    if not path.is_file():
        return None
    with path.open("rb") as source:
        source.seek(0, 2)
        source.seek(max(0, source.tell() - 65536))
        text = source.read().decode("utf-8", errors="replace")
    matches = re.findall(r'^\{"phase":"(probe-2026103[012]|long-context-calibration)"\}$', text, re.MULTILINE)
    return matches[-1] if matches else None


def eligible_phases(state, phase):
    if state.get("GPU_experiment_complete") is True:
        if (state.get("state_last_observed") != "COMPLETED"
                or state.get("phase") != "long_Conv_probe_SHA_verified_analyzed"):
            raise ValueError("Require full verified GPU completion before using terminal raw")
        return PHASES
    if phase in PHASES:
        return PHASES[:PHASES.index(phase)]
    return ()


def pack_command(job, phase):
    if type(job) is not int or job < 1 or phase not in PHASES:
        raise ValueError("Unsafe job or closed probe phase")
    return f'''set -euo pipefail
root="$HOME/run/kvcached-paracloud/state/live-trace-{job}"
source="$root/{phase}"
test -d "$source" && test ! -L "$source"
test -f "$source/exit-status" && test ! -L "$source/exit-status"
test "$(cat "$source/exit-status")" = 0
test -z "$(find "$source" -type l -print -quit)"
stage="$(mktemp -d "$HOME/run/kvcached-paracloud/sync/closed-long-conv-{job}.XXXXXX")"
attempt="$(mktemp -d "$stage/phase-{phase}.XXXXXX")"
(cd "$source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$attempt/raw-before.sha256"
test -s "$attempt/raw-before.sha256"
nice -n 10 tar -C "$source" -czf "$attempt/raw.tar.gz.partial" .
gzip -t "$attempt/raw.tar.gz.partial"
test "$(cat "$source/exit-status")" = 0
test -z "$(find "$source" -type l -print -quit)"
(cd "$source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$attempt/raw-after.sha256"
cmp "$attempt/raw-before.sha256" "$attempt/raw-after.sha256"
mv "$attempt/raw.tar.gz.partial" "$attempt/raw.tar.gz"
printf 'stage=%s\\nattempt=%s\\nbundle_sha256=%s\\n' "$stage" "$(basename "$attempt")" \\
    "$(sha256sum "$attempt/raw.tar.gz" | awk '{{print $1}}')"
'''


def pack_metadata(text, job, phase):
    if type(job) is not int or job < 1 or phase not in PHASES:
        raise ValueError("Unsafe job or closed probe phase")
    entries = [line.split("=", 1) for line in text.splitlines()]
    if any(len(entry) != 2 for entry in entries):
        raise ValueError("Malformed archive metadata")
    metadata = dict(entries)
    if len(entries) != 3 or set(metadata) != {"stage", "attempt", "bundle_sha256"}:
        raise ValueError("Archive metadata must contain unique declared fields")
    stage = PurePosixPath(metadata["stage"])
    suffix = ("run", "kvcached-paracloud", "sync")
    if (not stage.is_absolute() or ".." in stage.parts or stage.parts[-4:-1] != suffix
            or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", component) for component in stage.parts[1:])
            or not re.fullmatch(r"closed-long-conv-" + str(job) + r"\.[A-Za-z0-9]+", stage.name)
            or not re.fullmatch(r"phase-" + phase + r"\.[A-Za-z0-9]+", metadata["attempt"])
            or not re.fullmatch(r"[0-9a-f]{64}", metadata["bundle_sha256"])):
        raise ValueError("Unsafe archive metadata")
    return metadata


def save_state(path, state):
    state["recorded_at"] = datetime.now().astimezone().isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def verify_and_analyze(raw, prompt_bank, expected_trace_sha):
    if hashlib.sha256((raw / "trace.json").read_bytes()).hexdigest() != expected_trace_sha:
        raise ValueError("Closed phase changed frozen trace")
    verification = verify(raw, prompt_bank)
    duration = json.loads((raw / "trace.json").read_text())["duration_s"]
    return {"verification": verification,
            "comparisons": {baseline: paired_window(raw, baseline, "kv_only_workaware", 0, duration)
                for baseline in ("no_move", "random_no_return", "kv_only_headroom")},
            "recovery": analyze_recovery(raw, start_s=0), "token_windows": token_windows(raw),
            "pause_inclusive_repayment": repayment(raw), "research_goal_complete": False}


def observe(root, manifest, once=False):
    directory = root / "pause-inclusive-phase-analysis"
    directory.mkdir(exist_ok=True)
    state_path = directory / "observer-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "phases": {}, "interval_s": 300, "Slurm_queries": False, "GPU_submissions": False,
        "runtime_changes": False, "research_goal_complete": False}
    expected_sources = manifest.read_text()
    if not expected_sources.strip():
        raise ValueError("Require a nonempty frozen phase-analysis source manifest")
    while True:
        try:
            check_pause(root)
            if manifest.read_text() != expected_sources:
                raise ValueError("Preserve phase-analysis expected source manifest")
            campaign = json.loads((root / "campaign-state.json").read_text())
            repo = Path(campaign["repo"])
            for line in expected_sources.splitlines():
                expected, name = line.split(maxsplit=1)
                path = repo / name
                if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(repo.resolve())
                        or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
                    raise ValueError("Frozen phase-analysis source changed")
            try:
                check_frozen_sources(repo, campaign)
            except OSError as error:
                raise ValueError("Frozen runtime source missing or unreadable") from error
            if state.get("job", campaign["job"]) != campaign["job"]:
                raise ValueError("Preserve phase-analysis job identity")
            state["job"] = campaign["job"]
            phase = latest_phase(root / f"poll-{campaign['job']}.log")
            state.update({"latest_remote_phase_from_main_poll": phase, "last_main_poll_at": campaign.get("last_poll_at"),
                          "main_GPU_experiment_complete": campaign.get("GPU_experiment_complete", False)})
            prompt_bank = Path(campaign["inputs"]) / "prompt-bank.json"
            for name in eligible_phases(campaign, phase):
                if name in state["phases"]:
                    continue
                check_pause(root)
                if campaign.get("GPU_experiment_complete"):
                    whole_raw = root / f"live-trace-{campaign['job']}"
                    verify_raw(whole_raw, (root / f"raw-{campaign['job']}.sha256").read_text())
                    raw, archive = whole_raw / name, {"source": "complete_terminal_SHA_archive"}
                else:
                    attempt = Path(tempfile.mkdtemp(prefix="pack-" + name + "-", dir=directory))
                    command = pack_command(campaign["job"], name)
                    (attempt / "remote-command.sh").write_text(command)
                    result = subprocess.run(["ssh", *OPTIONS, "paracloud", "bash -s"], input=command,
                                            text=True, capture_output=True, timeout=900)
                    (attempt / "remote.stdout").write_text(result.stdout)
                    (attempt / "remote.stderr").write_text(result.stderr)
                    result.check_returncode()
                    metadata = pack_metadata(result.stdout, campaign["job"], name)
                    raw, archive = pull_phase(metadata["stage"], name, metadata, directory / name, campaign["job"])
                state["verifying_raw"] = str(raw)
                save_state(state_path, state)
                expected_trace = campaign["probe_config"]["frozen_sha256"][name + ".json"]
                analysis = verify_and_analyze(raw, prompt_bank, expected_trace)
                output = directory / (name + "-independent-analysis.json")
                with output.open("x") as target:
                    target.write(json.dumps(analysis, indent=2) + "\n")
                state["phases"][name] = {"archive": archive, "raw": str(raw), "analysis": str(output),
                                         "status": "SHA_verified_closed_probe_not_final_trace_benefit"}
                state.pop("verifying_raw", None)
                save_state(state_path, state)
                print(json.dumps({"at": state["recorded_at"], "phase_verified": name,
                                  "repayment": analysis["pause_inclusive_repayment"]["observed_span_saved_request_s"]}), flush=True)
            if len(state["phases"]) == len(PHASES):
                state["status"] = "all_three_probe_analyses_verified_not_research_goal_complete"
                save_state(state_path, state)
                return
            if campaign.get("phase") == "diagnosis_required_no_automatic_duplicate_submission":
                state["status"] = "main_campaign_failed_preserve_closed_phase_evidence_no_GPU_submission"
                save_state(state_path, state)
                return
            state["status"] = "waiting_for_closed_phase_via_existing_main_poll_no_extra_Slurm_queries"
            save_state(state_path, state)
        except (subprocess.SubprocessError, OSError) as error:
            state["last_transient_archive_error"] = repr(error)
            save_state(state_path, state)
            print(json.dumps({"retry_after_s": 300, "error": repr(error)}), flush=True)
        except Exception:
            state["status"] = "diagnosis_required_no_GPU_submission"
            state["error"] = traceback.format_exc()
            save_state(state_path, state)
            raise
        if once:
            return
        print(json.dumps({"at": state["recorded_at"], "status": state["status"], "phases": list(state["phases"])}), flush=True)
        time.sleep(300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    observe(args.root.resolve(), args.source_manifest.resolve(), args.once)


if __name__ == "__main__":
    main()
