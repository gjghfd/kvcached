import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import risk_certificate_portability
import stress_campaign


ENTRY_NAME = "experiments/dp_opt/host_archive/performance_path_recovery.py"
BRIDGE_NAME = "experiments/dp_opt/performance_archive_bridge.py"
ALIAS_NAME = "experiments/dp_opt/archive_progress_alias_recovery.py"
ARCHIVER_NAME = "experiments/dp_opt/archive_trace_job.py"
ORIGINAL_NAME = "experiments/dp_opt/performance_path_recovery.py"
RISK_NAME = "experiments/dp_opt/risk_certificate_portability.py"
ALIAS_ERROR = "ValueError: Unsafe raw progress pointer"
ORIGINAL_ARCHIVE_JOB = stress_campaign.archive_job


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SOURCE_SHA256_AT_IMPORT = digest(__file__)


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Require regular bridge evidence: " + str(path))
    return json.loads(path.read_text())


def receipt(path, value):
    with Path(path).open("x") as output:
        output.write(json.dumps(value, indent=2) + "\n")
    Path(path).chmod(0o444)


class SourceGuard:
    def __init__(self, root, manifest, host_manifest, entry, loaded_entry_sha256, original):
        self.root = Path(root).resolve()
        self.repo = Path(read_json(self.root / "campaign-state.json")["repo"]).resolve()
        self.manifest = Path(manifest).resolve()
        self.host_manifest = Path(host_manifest).resolve()
        self.original = original
        if not self.manifest.is_relative_to(self.root) or not self.host_manifest.is_relative_to(self.root):
            raise ValueError("Both independent host manifests must be bound to the campaign root")
        self.expected = read_json(self.manifest)
        if (set(self.expected) != {ENTRY_NAME, BRIDGE_NAME, ALIAS_NAME}
                or any(not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value)
                       for value in self.expected.values())):
            raise ValueError("Require the exact three-file archive-bridge manifest")
        if (Path(entry).resolve() != self.repo / ENTRY_NAME or Path(__file__).resolve() != self.repo / BRIDGE_NAME
                or Path(original.__file__).resolve() != self.repo / ORIGINAL_NAME
                or Path(risk_certificate_portability.__file__).resolve() != self.repo / RISK_NAME):
            raise ValueError("Loaded entry/bridge/original host source origin differs")
        self.loaded = {ENTRY_NAME: loaded_entry_sha256, BRIDGE_NAME: SOURCE_SHA256_AT_IMPORT}
        self.manifest_sha256 = digest(self.manifest)
        self.host_manifest_sha256 = original.verify_host_manifest(
            self.host_manifest, self.repo, risk_certificate_portability.SOURCE_AT_IMPORT_SHA256)
        source_path = self.repo / ALIAS_NAME
        source = source_path.read_bytes()
        self.loaded[ALIAS_NAME] = hashlib.sha256(source).hexdigest()
        if self.loaded != self.expected:
            raise ValueError("Bridge manifest differs from loaded host source SHA")
        specification = importlib.util.spec_from_file_location("_performance_archive_bridge_alias", source_path)
        self.alias = importlib.util.module_from_spec(specification)
        exec(compile(source, str(source_path), "exec"), self.alias.__dict__)
        self.check()

    def check(self):
        if digest(self.manifest) != self.manifest_sha256 or read_json(self.manifest) != self.expected:
            raise ValueError("Archive-bridge manifest changed after load")
        if digest(self.host_manifest) != self.host_manifest_sha256:
            raise ValueError("Original host-source manifest changed after load")
        if self.original.verify_host_manifest(self.host_manifest, self.repo,
                risk_certificate_portability.SOURCE_AT_IMPORT_SHA256) != self.host_manifest_sha256:
            raise ValueError("Original loaded host-source verification changed")
        state = read_json(self.root / "campaign-state.json")
        if Path(state["repo"]).resolve() != self.repo:
            raise ValueError("Master ledger repo changed")
        if any(name in state["frozen_runtime_sha256"] for name in self.expected):
            raise ValueError("Archive bridge is host-only, not part of the GPU manifest")
        for name, checksum in self.expected.items():
            if digest(self.repo / name) != checksum or self.loaded[name] != checksum:
                raise ValueError("Archive bridge loaded source changed: " + name)
        if (ARCHIVER_NAME not in state["frozen_runtime_sha256"]
                or digest(self.repo / ARCHIVER_NAME) != state["frozen_runtime_sha256"][ARCHIVER_NAME]):
            raise ValueError("Require the original frozen archiver SHA")
        return state


class ArchiveBridge:
    def __init__(self, guard, *, run=None, sleep=None, recover=None, save_state=None, check_pause=None):
        self.guard = guard
        self.run = subprocess.run if run is None else run
        self.sleep = time.sleep if sleep is None else sleep
        self.recover = guard.alias.recover if recover is None else recover
        self.save_state = stress_campaign.save_state if save_state is None else save_state
        self.check_pause = stress_campaign.check_pause if check_pause is None else check_pause
        self.pid = os.getpid()
        self.busy = threading.Lock()
        state = guard.check()
        self.evidence = Path(tempfile.mkdtemp(prefix="host-archive-bridge-", dir=guard.root))
        receipt(self.evidence / "loaded-source.json", {
            "root": str(guard.root), "repo": str(guard.repo), "coordinator_pid": self.pid,
            "argv_at_load": list(sys.argv), "archive_bridge_manifest": str(guard.manifest),
            "archive_bridge_manifest_sha256": guard.manifest_sha256, "loaded_source_sha256": guard.loaded,
            "host_source_manifest": str(guard.host_manifest), "host_source_manifest_sha256": guard.host_manifest_sha256,
            "master_ledger_sha256_at_load": digest(guard.root / "campaign-state.json"),
            "bound_stages_at_load": [{"stage_id": name, "job": stage["job"], "phase": stage.get("phase")}
                                     for name, stage in state["stages"].items() if "job" in stage],
            "GPU_manifest_files": len(state["frozen_runtime_sha256"]), "extra_sources_host_only": True,
            "GPU_manifest_modified": False, "independent_follow_started": False})

    def bind(self, archive, job, scripts, entry):
        state = self.guard.check()
        stage_id = entry.get("stage_id")
        if (type(job) is not int or job <= 0 or type(entry.get("job")) is not int or entry["job"] != job
                or not isinstance(stage_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", stage_id)):
            raise ValueError("Archive caller job/stage is not owned by the master ledger")
        master = state["stages"].get(stage_id)
        owners = [name for name, stage in state["stages"].items() if stage.get("job") == job]
        if (master is None or owners != [stage_id] or master.get("stage_id") != stage_id
                or type(master.get("job")) is not int or master["job"] != job or master.get("plan") != entry.get("plan")
                or any(master.get(field) != entry.get(field) for field in ("terminal", "accounting_exit_code"))):
            raise ValueError("Archive caller differs from the owned master stage")
        archive = Path(archive)
        if archive.is_symlink() or archive.resolve() != self.guard.root / "stages" / stage_id or not archive.is_dir():
            raise ValueError("Archive directory is not bound to the master root/stage")
        if Path(scripts).resolve() != self.guard.repo / "experiments/dp_opt":
            raise ValueError("Archive scripts differ from the frozen master repo")
        return archive.resolve(), stage_id

    def __call__(self, archive, job, scripts, entry):
        if os.getpid() != self.pid or not self.busy.acquire(blocking=False):
            raise RuntimeError("Archive bridge requires one synchronous owning coordinator")
        evidence = None
        try:
            archive, stage_id = self.bind(archive, job, scripts, entry)
            evidence = Path(tempfile.mkdtemp(prefix=f"job-{job}-{stage_id}-", dir=self.evidence))
            binding = {"root": str(self.guard.root), "archive": str(archive), "job": job, "stage_id": stage_id,
                       "coordinator_pid": self.pid, "plan": entry["plan"], "bridge_manifest_sha256": self.guard.manifest_sha256,
                       "host_source_manifest_sha256": self.guard.host_manifest_sha256,
                       "strict_downstream_verified": False}
            receipt(evidence / "caller.json", binding)
            entry["phase"] = "terminal_SHA_archive_and_independent_verification"
            self.save_state(archive, entry)
            for attempt in range(1, 13):
                self.check_pause(archive)
                self.bind(archive, job, scripts, entry)
                ordinals = [int(match.group(1)) for path in archive.glob(f"archive-{job}-attempt-*.log")
                            for match in [re.fullmatch(rf"archive-{job}-attempt-(\d+)\.log", path.name)] if match]
                ordinal = max(ordinals, default=0) + 1
                command = [sys.executable, str(Path(scripts) / "archive_trace_job.py"), str(job), "--archive", str(archive)]
                raw = archive / f"live-trace-{job}"
                if raw.exists():
                    command.append("--resume")
                result = self.run(command, text=True, capture_output=True)
                text = result.stdout + result.stderr
                log = archive / f"archive-{job}-attempt-{ordinal}.log"
                receipt(evidence / f"child-{attempt}.json", {**binding, "command": command, "attempt_in_call": attempt,
                        "log_ordinal": ordinal, "intended_closed_log": str(log), "child_returncode": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr})
                with log.open("x") as output:
                    output.write(text)
                    output.flush()
                    os.fsync(output.fileno())
                receipt(evidence / f"closed-{attempt}.json", {**binding, "command": command, "attempt_in_call": attempt,
                        "log_ordinal": ordinal, "closed_log": str(log), "closed_log_sha256": digest(log),
                        "child_returncode": result.returncode})
                self.bind(archive, job, scripts, entry)
                if result.returncode == 0:
                    receipt(evidence / "returned.json", {**binding, "action": "frozen_archiver_success"})
                    return
                if (text.rstrip().splitlines()[-1:] == [ALIAS_ERROR] and entry.get("terminal") == "COMPLETED"
                        and entry.get("accounting_exit_code") == "0:0"):
                    if "--resume" in command:
                        raise ValueError("Progress-pointer extraction error on existing raw is not a supported alias recovery")
                    recovered = self.recover(self.guard.root, stage_id)
                    self.bind(archive, job, scripts, entry)
                    if (type(recovered.get("job")) is not int or recovered["job"] != job
                            or recovered.get("stage_id") != stage_id):
                        raise ValueError("Alias recovery receipt differs from the owning coordinator stage")
                    receipt(evidence / "returned.json", {**binding, "action": "same_coordinator_alias_recovery",
                            "recovery": recovered, "closed_log": str(log), "closed_log_sha256": digest(log)})
                    return
                self.sleep(300)
            raise RuntimeError("Archive failed after bounded retries; preserve downloads and stop resubmission")
        except BaseException as error:
            if evidence is not None:
                receipt(evidence / "failed.json", {"error": repr(error), "root": str(self.guard.root), "job": job,
                        "stage_id": entry.get("stage_id"), "coordinator_pid": self.pid,
                        "attempt_evidence": str(evidence), "GPU_resubmission_authorized": False})
            raise
        finally:
            self.busy.release()


@contextmanager
def installed_archive_bridge(callback):
    if stress_campaign.archive_job is not ORIGINAL_ARCHIVE_JOB:
        raise RuntimeError("Refuse to replace another archive-job hook")
    stress_campaign.archive_job = callback
    try:
        yield
    finally:
        stress_campaign.archive_job = ORIGINAL_ARCHIVE_JOB


def run_entry(original, entry, loaded_entry_sha256, argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--archive-bridge-manifest", type=Path, required=True)
    bridge_args, forwarded = parser.parse_known_args(argv)
    controls = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    controls.add_argument("--root", type=Path, required=True)
    controls.add_argument("--host-source-manifest", type=Path, required=True)
    controls.add_argument("--risk-portability-audit", action="store_true", required=True)
    args = controls.parse_known_args(forwarded)[0]
    for option in ("--archive-bridge-manifest", "--root", "--host-source-manifest", "--risk-portability-audit"):
        if sum(argument.split("=", 1)[0] == option for argument in argv) != 1:
            raise ValueError("Require a unique explicit host option: " + option)
    guard = SourceGuard(args.root, bridge_args.archive_bridge_manifest, args.host_source_manifest,
                        entry, loaded_entry_sha256, original)
    callback = ArchiveBridge(guard)
    previous_argv = list(sys.argv)
    try:
        sys.argv = [str(entry), *forwarded]
        with installed_archive_bridge(callback):
            return original.main()
    finally:
        sys.argv = previous_argv
