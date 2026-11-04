import argparse
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

from analyze_return_recovery import analyze_recovery
from archive_trace_job import extract_raw
from verify_trace import verify
from verify_workaware_profile import same_payload, verify_profile


PHASES = ("calibration-n567", "composite-profile", "cache-fixed-original", "lower-code-rate")
OPTIONS = ("-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3")


def ready_records(text):
    result = {}
    for line in text.splitlines():
        phase, attempt, digest = line.split("\t")
        if (phase not in PHASES or phase in result
                or not re.fullmatch("phase-" + phase + r"\.[A-Za-z0-9]+", attempt)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Unsafe or duplicate closed-phase archive metadata")
        result[phase] = {"attempt": attempt, "bundle_sha256": digest}
    return result


def registry_command(stage):
    return f'''set -eu
stage={shlex.quote(stage)}
for phase in {' '.join(PHASES)}; do
    marker="$stage/$phase.ready"
    if [ -f "$marker" ] && [ ! -L "$marker" ]; then
        attempt="$(sed -n 's/^attempt=//p' "$marker")"
        digest="$(sed -n 's/^bundle_sha256=//p' "$marker")"
        printf '%s\\t%s\\t%s\\n' "$phase" "$attempt" "$digest"
    fi
done
'''


def pull_phase(stage, phase, metadata, directory, job):
    directory.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="download-", dir=directory))
    remote = stage + "/" + metadata["attempt"]
    for name in ("raw.tar.gz", "raw-before.sha256", "raw-after.sha256"):
        subprocess.run(["scp", *OPTIONS, "paracloud:" + remote + "/" + name, str(attempt / name)],
                       check=True, timeout=900)
    bundle = attempt / "raw.tar.gz"
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != metadata["bundle_sha256"]:
        raise ValueError("Closed-phase archive bundle SHA mismatch")
    manifest = (attempt / "raw-before.sha256").read_text()
    if manifest != (attempt / "raw-after.sha256").read_text():
        raise ValueError("Closed phase changed while being archived")
    raw = attempt / "raw"
    raw.mkdir()
    if extract_raw(bundle, raw, manifest, job):
        raise ValueError("Closed phase must not contain the active job's progress pointer")
    return raw, {**metadata, "download": str(attempt), "raw_files_SHA_verified": len(manifest.splitlines())}


def verify_phase(phase, raw, prompt_bank):
    if phase in ("calibration-n567", "composite-profile"):
        fit = verify_profile(raw)
        if phase == "composite-profile" and not same_payload(fit, json.loads((raw / "fit.json").read_text())):
            raise ValueError("Archived composite fit does not reproduce from independent raw")
        return {"status": "independently_verified_calibration_not_algorithm_benefit",
                "fit": fit, "profiles": sorted(fit["population_profiles"]["profiles"])}
    return {"status": "complete_phase_independently_verified_not_research_goal_complete",
            "verification": verify(raw, prompt_bank), "recovery": analyze_recovery(raw)}


def save_state(path, state):
    state["recorded_at"] = datetime.now().astimezone().isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def observe(root, remote_stage):
    campaign = json.loads((root / "campaign-state.json").read_text())
    directory = root / "closed-phase-archives"
    directory.mkdir(exist_ok=True)
    state_path = directory / "observer-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "job": campaign["job"], "remote_stage": remote_stage, "phases": {},
        "research_goal_complete": False}
    if state["job"] != campaign["job"] or state["remote_stage"] != remote_stage:
        raise ValueError("Preserve closed-phase observer identity")
    prompt_bank = Path(campaign["inputs"]) / "prompt-bank.json"
    while True:
        try:
            query = subprocess.run(["ssh", *OPTIONS, "paracloud", registry_command(remote_stage)],
                                   text=True, capture_output=True, timeout=180)
            query.check_returncode()
            ready = ready_records(query.stdout)
            state["last_registry_query_exit"] = 0
            for phase in PHASES:
                if phase not in ready:
                    continue
                previous = state["phases"].get(phase)
                if previous:
                    if previous["bundle_sha256"] != ready[phase]["bundle_sha256"]:
                        raise ValueError("Preserve previously verified phase archive")
                    continue
                print(json.dumps({"at": datetime.now().astimezone().isoformat(), "phase_ready": phase}), flush=True)
                raw, archive = pull_phase(remote_stage, phase, ready[phase], directory / phase, campaign["job"])
                try:
                    result = verify_phase(phase, raw, prompt_bank)
                except Exception:
                    state["verification_failed_preserved_raw"] = str(raw)
                    save_state(state_path, state)
                    raise
                output = raw.parent / "independent-verification.json"
                with output.open("x") as target:
                    target.write(json.dumps(result, indent=2) + "\n")
                state["phases"][phase] = {**archive, "verification": str(output), "raw": str(raw), "status": result["status"]}
                save_state(state_path, state)
                print(json.dumps({"at": state["recorded_at"], "phase_verified": phase}), flush=True)
            save_state(state_path, state)
            if len(state["phases"]) == len(PHASES):
                print("all_closed_phases_verified_not_research_goal_complete", flush=True)
                return
            latest = json.loads((root / "campaign-state.json").read_text())
            if latest["phase"] == "diagnosis_required_no_automatic_duplicate_submission":
                print("campaign_failed_preserved_partial_archives", flush=True)
                return
        except (subprocess.SubprocessError, OSError) as error:
            state["last_registry_or_download_error"] = repr(error)
            save_state(state_path, state)
            print(json.dumps({"retry_after_s": 300, "error": repr(error)}), flush=True)
        except Exception:
            state["diagnosis_required_no_job_submission"] = traceback.format_exc()
            save_state(state_path, state)
            raise
        print(json.dumps({"at": datetime.now().astimezone().isoformat(), "verified_phases": list(state["phases"])}), flush=True)
        time.sleep(300)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--remote-stage", required=True)
    args = parser.parse_args()
    observe(args.campaign, args.remote_stage)


if __name__ == "__main__":
    main()
