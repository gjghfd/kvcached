import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def alias_command(job):
    if type(job) is not int or job <= 0:
        raise ValueError("Require one positive integer Poisson job ID")
    return f'''set -eu
date -u '+%Y-%m-%dT%H:%M:%SZ'
hostname
logs="$HOME/run/kvcached-paracloud/logs"
for extension in out err; do
    actual="$logs/long-poisson-{job}.$extension"
    alias="$logs/live-trace-{job}.$extension"
    test -f "$actual"
    if [ -L "$alias" ]; then
        test "$(readlink "$alias")" = "long-poisson-{job}.$extension"
    elif [ -e "$alias" ]; then
        printf 'Refusing existing non-symlink log alias: %s\\n' "$alias" >&2
        exit 2
    else
        ln -s "long-poisson-{job}.$extension" "$alias"
    fi
    test "$actual" -ef "$alias"
    printf 'verified_log_alias|%s|%s|%s\\n' "$extension" "$actual" "$alias"
done'''


def ready(state):
    return (type(state.get("job")) is int and state["job"] > 0
            and state.get("state_last_observed") in ("RUNNING", "COMPLETED"))


def save_status(root, status):
    temporary = root / "followup-state.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(root / "followup-state.json")


def run(root, parent):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "followup-state.json").open("x") as target:
        target.write('{}\n')
    status = {"parent_root": str(parent), "poll_interval_s": 300,
              "GPU_submissions": False, "Slurm_queries": False, "runtime_changes": False,
              "research_goal_complete": False}
    try:
        while True:
            status["observed_at"] = datetime.now().astimezone().isoformat()
            state = json.loads((parent / "campaign-state.json").read_text())
            pause_roots = (root, parent, Path(state["parent_root"])) if "parent_root" in state else (root, parent)
            if any((path / "PAUSE").exists() for path in pause_roots):
                status["status"] = "paused_without_log_alias_operation"
                save_status(root, status)
                return
            if "diagnosis_required" in state.get("phase", ""):
                raise RuntimeError("Poisson failure requires diagnosis, not automatic log repair")
            status.update({"job": state.get("job"), "parent_phase": state.get("phase")})
            if ready(state):
                command = alias_command(state["job"])
                (root / "alias-command.sh").write_text(command + "\n")
                result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                    "paracloud", command], text=True, capture_output=True, timeout=180)
                (root / "alias-receipt.log").write_text(result.stdout + result.stderr)
                status["SSH_exit"] = result.returncode
                if result.returncode or any(f"verified_log_alias|{extension}|" not in result.stdout
                                            for extension in ("out", "err")):
                    raise RuntimeError("Both actual Poisson logs must match guarded aliases")
                status["status"] = "log_aliases_verified_not_experiment_completion"
                save_status(root, status)
                return
            status["status"] = "waiting_local_state_no_additional_SSH_query"
            save_status(root, status)
            print(json.dumps(status), flush=True)
            time.sleep(300)
    except Exception as error:
        status.update({"status": "diagnosis_required_no_automatic_retry", "error": repr(error)})
        save_status(root, status)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.parent_root.resolve())


if __name__ == "__main__":
    main()
