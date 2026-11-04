import argparse
import copy
import ctypes
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath

import archive_trace_job


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def regular_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or nonregular evidence: " + str(path))
    return path


def write_new_json(path, value, immutable=False):
    with Path(path).open("x") as output:
        output.write(json.dumps(value, indent=2) + "\n")
    if immutable:
        Path(path).chmod(0o444)


@contextmanager
def local_install_lock(directory):
    path = directory / ".progress-alias-recovery.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another local alias installer owns this stage") from error
        yield
    finally:
        os.close(descriptor)


def pointer_binding(stage, owner, home, canonical_root, target):
    job, phase = stage["job"], stage["plan"]["phase_name"]
    if type(job) is not int or job <= 0 or not re.fullmatch(r"[a-zA-Z0-9_-]+", phase):
        raise ValueError("Invalid bound stage identity")
    if stage.get("terminal") != "COMPLETED" or stage.get("accounting_exit_code") != "0:0":
        raise ValueError("Require the completed original allocation")
    if home != stage["remote_home"] or not re.fullmatch(r"[a-zA-Z0-9_-]+", owner):
        raise ValueError("Actual owner/home binding differs")
    for path in (home, canonical_root, target):
        parsed = PurePosixPath(path)
        if not parsed.is_absolute() or ".." in parsed.parts or str(parsed) != path:
            raise ValueError("Invalid absolute alias path")
    suffix = f"/state/live-trace-{job}/{phase}/progress.json"
    expected = canonical_root + suffix
    alias = home + "/run/kvcached-paracloud" + suffix
    if target != expected or alias == target:
        raise ValueError("Progress pointer is not the bound canonical alias")
    return {"job": job, "phase": phase, "owner": owner, "remote_home": home,
            "canonical_root": canonical_root, "remote_target": target,
            "alias_target": alias, "archived_target": phase + "/progress.json"}


def extract_bound_alias(bundle, staging, manifest, job, binding, **kwargs):
    if job != binding["job"]:
        raise ValueError("Archive caller job differs from alias receipt")
    retained = Path(tempfile.mkdtemp(prefix="progress-alias-parser-", dir=Path(staging).parent))
    normalized = retained / "parser-only.tar"
    with tarfile.open(bundle) as source, tarfile.open(normalized, "w") as destination:
        matched = 0
        for member in source.getmembers():
            copied = copy.copy(member)
            if member.issym() and str(PurePosixPath(member.name)) == "progress.json":
                if member.linkname != binding["remote_target"]:
                    raise ValueError("Bundle progress pointer differs from actual receipt")
                copied.linkname = binding["alias_target"]
                copied.pax_headers = {name: value for name, value in copied.pax_headers.items() if name != "linkpath"}
                matched += 1
            content = source.extractfile(member) if member.isfile() else None
            destination.addfile(copied, content)
        if matched != 1:
            raise ValueError("Require exactly one original progress pointer")
    pointers = archive_trace_job.extract_raw(normalized, staging, manifest, job, **kwargs)
    if len(pointers) != 1 or pointers[0]["archived_target"] != binding["archived_target"]:
        raise ValueError("Extracted pointer differs from bound stage")
    pointers[0].update({"remote_target": binding["remote_target"],
                       "parser_alias_target": binding["alias_target"],
                       "actual_owner_home_samefile_verified": True})
    return pointers


def check_runtime(state):
    for name, checksum in state["frozen_runtime_sha256"].items():
        if digest(Path(state["repo"]) / name) != checksum:
            raise ValueError("Active performance source SHA changed")


def latest_failure_log(directory, job):
    logs = [regular_file(path) for path in directory.glob(f"archive-{job}-attempt-*.log")
            if re.fullmatch(rf"archive-{job}-attempt-\d+\.log", path.name)]
    if not logs:
        raise ValueError("No closed coordinator failure log; archive may still be in progress")
    logs.sort(key=lambda path: path.stat().st_mtime_ns)
    if len(logs) > 1 and logs[-1].stat().st_mtime_ns == logs[-2].stat().st_mtime_ns:
        raise ValueError("Ambiguous current coordinator failure log")
    log = logs[-1]
    text = log.read_text().rstrip()
    if not text:
        raise ValueError("Empty current coordinator log is not a healthy archive")
    if not text.endswith("ValueError: Unsafe raw progress pointer"):
        raise ValueError("Current coordinator failure is not the bound progress alias: " + text.splitlines()[-1])
    return log


def file_witness(path):
    path = regular_file(path)
    before = path.stat()
    checksum = digest(path)
    after = path.stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if identity(before) != identity(after):
        raise ValueError("Evidence changed while reading: " + str(path))
    return {"path": str(path), "sha256": checksum, "stat": list(identity(after))}


def check_witnesses(witnesses):
    for witness in witnesses:
        if file_witness(witness["path"]) != witness:
            raise ValueError("Closed coordinator evidence changed; preserve attempt")


def check_closed_current(directory, job, closed):
    check_witnesses(closed["witnesses"])
    attempts = sorted(path for path in directory.glob(f"archive-{job}-attempt-*") if path.is_dir())
    if (not attempts or str(attempts[-1]) != closed["attempt"]
            or str(latest_failure_log(directory, job)) != closed["failure_log"]):
        raise ValueError("New coordinator attempt appeared during alias recovery")


def bind_closed_attempt(directory, job, retained):
    log = latest_failure_log(directory, job)
    attempts = sorted(path for path in directory.glob(f"archive-{job}-attempt-*") if path.is_dir())
    if not attempts or any(path.is_symlink() for path in attempts):
        raise ValueError("Missing closed coordinator attempt")
    attempt = attempts[-1]
    if (attempt / "report.json").exists() or attempt.stat().st_mtime_ns > log.stat().st_mtime_ns:
        raise ValueError("Latest coordinator attempt is not the closed alias failure")
    suffix = attempt.name.removeprefix(f"archive-{job}-attempt-")
    bundle = directory / f"raw-{job}-{suffix}.tar.gz"
    files = [log, bundle, attempt / "raw.sha256", attempt / "query.stdout", attempt / "accounting.txt"]
    witnesses = [file_witness(path) for path in files]
    accounting = (attempt / "accounting.txt").read_text().strip().splitlines()[0].split("|")
    if accounting[:3] != [str(job), "COMPLETED", "0:0"]:
        raise ValueError("Closed coordinator accounting differs from completed allocation")
    if (attempt / "query.stdout").read_text().partition("RAW_MANIFEST\n")[2] != (attempt / "raw.sha256").read_text():
        raise ValueError("Closed query and expected manifest differ")
    if any(Path(witness["path"]).stat().st_mtime_ns > log.stat().st_mtime_ns for witness in witnesses[1:]):
        raise ValueError("Bundle/manifest not closed by current failure log")
    incoming = retained / "closed-input"
    incoming.mkdir()
    copies = {}
    for ordinal, witness in enumerate(witnesses):
        source = Path(witness["path"])
        copied = incoming / f"{ordinal}-{source.name}"
        shutil.copy2(source, copied)
        if digest(copied) != witness["sha256"]:
            raise ValueError("Closed input copy checksum mismatch")
        copied.chmod(0o444)
        copies[str(source)] = str(copied)
    check_witnesses(witnesses)
    binding = {"attempt": str(attempt), "failure_log": str(log), "bundle": str(bundle),
               "witnesses": witnesses, "retained_copies": copies}
    write_new_json(retained / "closed-input.json", binding, immutable=True)
    return binding


def alias_proof(stage, stage_id, target, bundle, result):
    result.check_returncode()
    records = {}
    for prefix in ("OWNER", "HOME", "DATA", "TARGET_SHA"):
        values = [line.split("|", 1)[1] for line in result.stdout.splitlines() if line.startswith(prefix + "|")]
        if len(values) != 1:
            raise ValueError("Missing actual alias proof: " + prefix)
        records[prefix] = values[0]
    job = stage["job"]
    rows = [line.rstrip("|").split("|") for line in result.stdout.splitlines() if line.startswith(str(job) + "|")]
    if rows != [[str(job), records["OWNER"], "qwen-public-perf-followup", "gpu_a800", "COMPLETED", "0:0"]]:
        raise ValueError("Actual accounting owner/name/A800/exit differs")
    binding = pointer_binding(stage, records["OWNER"], records["HOME"], records["DATA"], target)
    checksum_fields = records["TARGET_SHA"].split()
    if (len(checksum_fields) != 2 or not re.fullmatch(r"[a-f0-9]{64}", checksum_fields[0])
            or checksum_fields[1] != binding["alias_target"]):
        raise ValueError("Actual target checksum/path differs")
    return {**binding, "stage_id": stage_id, "observed_at": datetime.now().astimezone().isoformat(),
            "original_bundle": str(bundle), "original_bundle_sha256": digest(bundle),
            "target_sha256": checksum_fields[0], "SSH_exit": result.returncode, "SSH_stdout": result.stdout,
            "active_runtime_changed": False, "original_bundle_changed": False}


def query_alias(stage, stage_id, target, bundle, retained):
    job = stage["job"]
    alias = stage["remote_home"] + f"/run/kvcached-paracloud/state/live-trace-{job}/{stage['plan']['phase_name']}/progress.json"
    command = ("set -eu\nprintf 'OWNER|%s\\n' \"$(id -un)\"\nprintf 'HOME|%s\\n' \"$HOME\"\n"
               "printf 'DATA|%s\\n' \"$(readlink -f \"$HOME/run/kvcached-paracloud\")\"\n"
               f"sacct -n -X -P -j {job} --format=JobIDRaw,User,JobName%128,Partition,State,ExitCode\n"
               f"test {shlex.quote(target)} -ef {shlex.quote(alias)}\n"
               f"test \"$(readlink -f {shlex.quote(alias)})\" = {shlex.quote(target)}\n"
               f"printf 'TARGET_SHA|'; sha256sum {shlex.quote(alias)}\n")
    try:
        result = subprocess.run(["ssh", "paracloud", command], text=True, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired as error:
        for suffix, content in (("stdout", error.stdout), ("stderr", error.stderr)):
            (retained / f"samefile-query.{suffix}").write_bytes(
                content.encode() if isinstance(content, str) else content or b"")
        raise
    (retained / "samefile-query.stdout").write_text(result.stdout)
    (retained / "samefile-query.stderr").write_text(result.stderr)
    return alias_proof(stage, stage_id, target, bundle, result)


def run_archiver(recovered, bundle, proof):
    original_extract, original_argv = archive_trace_job.extract_raw, list(sys.argv)

    def extract(bundle_path, staging, manifest, actual_job, **kwargs):
        archive_trace_job.extract_raw = original_extract
        try:
            pointers = extract_bound_alias(bundle_path, staging, manifest, actual_job, proof, **kwargs)
            if digest(staging / proof["archived_target"]) != proof["target_sha256"]:
                raise ValueError("Archived regular progress differs from actual samefile proof")
            return pointers
        finally:
            archive_trace_job.extract_raw = extract

    archive_trace_job.extract_raw = extract
    sys.argv = ["archive_trace_job", str(proof["job"]), "--archive", str(recovered), "--bundle", str(bundle)]
    try:
        archive_trace_job.main()
    finally:
        archive_trace_job.extract_raw, sys.argv = original_extract, original_argv


def verify_archive(directory, stage, proof):
    job = stage["job"]
    raw = directory / f"live-trace-{job}"
    if raw.is_symlink() or not raw.is_dir() or any(path.is_symlink() for path in raw.rglob("*")):
        raise ValueError("Nonregular canonical raw tree")
    manifest_path = regular_file(directory / f"raw-{job}.sha256")
    manifest = manifest_path.read_text()
    for line in manifest.splitlines():
        checksum, filename = line.split("  ", 1)
        parsed = PurePosixPath(filename)
        if not re.fullmatch(r"[a-f0-9]{64}", checksum) or parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError("Unsafe raw manifest")
    archive_trace_job.verify_raw(raw, manifest)
    report_path = regular_file(directory / f"archive-{job}.json")
    report_witness = file_witness(report_path)
    report = json.loads(report_path.read_text())
    if (type(report.get("job")) is not int or report["job"] != job or report.get("state") != "COMPLETED"
            or report.get("missing_job_logs") != [] or report.get("raw") != str(raw)
            or type(report.get("raw_files_verified")) is not int
            or report["raw_files_verified"] != len(manifest.splitlines())
            or report.get("watcher_barrier") not in ("shared_markers", "no_watcher")):
        raise ValueError("Incomplete canonical archive report")
    for field in ("raw_manifest", "pointer_metadata"):
        path = regular_file(report[field])
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Report path outside permanent archive")
    attempt = Path(report["attempt"])
    if attempt.is_symlink() or not attempt.is_dir() or not attempt.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Report attempt outside permanent archive")
    regular_file(attempt / "report.json")
    if digest(regular_file(attempt / "raw.sha256")) != digest(manifest_path):
        raise ValueError("Report attempt manifest differs")
    accounting = regular_file(directory / f"accounting-{job}.txt").read_text().strip().splitlines()[0].split("|")
    if accounting[:3] != [str(job), "COMPLETED", "0:0"]:
        raise ValueError("Canonical accounting differs from completed allocation")
    if digest(report["raw_manifest"]) != digest(manifest_path):
        raise ValueError("Report manifest differs")
    log_sha256 = report.get("log_sha256", {})
    if set(log_sha256) != {"out", "err"}:
        raise ValueError("Both original job logs are mandatory")
    for suffix, checksum in log_sha256.items():
        if digest(regular_file(directory / f"live-trace-{job}.{suffix}")) != checksum:
            raise ValueError("Canonical job log checksum mismatch: " + suffix)
    pointers_path = regular_file(directory / f"raw-{job}-pointers.json")
    pointers = json.loads(pointers_path.read_text())
    if (digest(report["pointer_metadata"]) != digest(pointers_path) or len(pointers) != 1
            or pointers[0].get("archived_target") != proof["archived_target"]
            or pointers[0].get("remote_target") != proof["remote_target"]
            or pointers[0].get("parser_alias_target") != proof["alias_target"]
            or pointers[0].get("actual_owner_home_samefile_verified") is not True
            or pointers[0].get("action") != "recorded_not_extracted"):
        raise ValueError("Canonical pointer proof differs")
    if digest(raw / proof["archived_target"]) != proof["target_sha256"]:
        raise ValueError("Canonical progress proof checksum mismatch")
    if file_witness(report_path) != report_witness:
        raise ValueError("Coordinator report changed during verification")
    return report


def atomic_copy(source, destination, immutable=False):
    source = regular_file(source)
    if destination.exists() or destination.is_symlink():
        if digest(regular_file(destination)) != digest(source):
            raise ValueError("Preserve differing prior artifact: " + destination.name)
        return
    descriptor, temporary = tempfile.mkstemp(prefix="publish-", dir=source.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    shutil.copy2(source, temporary)
    temporary.chmod(0o444 if immutable else temporary.stat().st_mode | 0o200)
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if digest(regular_file(destination)) != digest(source):
            raise ValueError("Concurrent archive installed a differing artifact: " + destination.name)
    finally:
        temporary.unlink()


def exclusive_raw_rename(source, destination):
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        operation = library.renamex_np
        operation.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        arguments = [os.fsencode(source), os.fsencode(destination), 4]
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        operation = library.renameat2
        operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        arguments = [-100, os.fsencode(source), -100, os.fsencode(destination), 1]
    else:
        raise RuntimeError("No supported exclusive raw rename; preserve verified stage")
    operation.restype = ctypes.c_int
    if operation(*arguments) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def write_report_path_remap(original, replacements, output):
    original, output = regular_file(original), Path(output)
    report = json.loads(original.read_text())
    resolved, paths = dict(report), {}
    for field in ("raw", "attempt", "raw_manifest", "pointer_metadata", "bundle_preserved"):
        recorded = Path(report[field])
        permanent = Path(replacements.get(field, recorded))
        if permanent.is_symlink() or not permanent.exists():
            raise ValueError("No verified permanent report path: " + str(permanent))
        paths[field] = {"recorded": str(recorded), "permanent": str(permanent), "exists": True}
        if permanent.is_file():
            paths[field]["sha256"] = digest(permanent)
        resolved[field] = str(permanent)
    resolved_path = output.with_name(output.stem + "-resolved-report.json")
    write_new_json(resolved_path, resolved, immutable=True)
    write_new_json(output, {"schema": 1, "original_report": str(original), "original_report_sha256": digest(original),
                   "resolved_report": str(resolved_path), "resolved_report_sha256": digest(resolved_path),
                   "paths": paths, "path_mapping_only": True, "numerical_quality_verified": False}, immutable=True)
    return output


def retain_report_paths(retained, directory, job):
    originals = list(retained.rglob("report.json")) + list(retained.rglob(f"archive-{job}.json"))
    for original in sorted(originals):
        report = json.loads(original.read_text())
        replacements = {}
        if not Path(report["raw"]).exists() and (directory / f"live-trace-{job}").exists():
            replacements["raw"] = directory / f"live-trace-{job}"
        write_report_path_remap(original, replacements, original.with_name("report-relocation.json"))


def verify_installed(directory, stage, stage_id):
    receipt_path = regular_file(directory / f"progress-alias-{stage['job']}-installation.json")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != 1 or receipt.get("job") != stage["job"] or receipt.get("stage_id") != stage_id
            or receipt.get("installation_protocol") != "verified_metadata_then_raw_local_lock"):
        raise ValueError("Incomplete immutable installation receipt")
    artifact_paths = [artifact["path"] for artifact in receipt["artifacts"]]
    proof_path = Path(receipt["proof"])
    required = {str(proof_path), str(proof_path.parent / "installation-report.json"),
                str(proof_path.parent / "closed-input.json")}
    if (len(artifact_paths) != 4 or len(set(artifact_paths)) != 4 or not required.issubset(artifact_paths)
            or sum(Path(path).name == "report.json" for path in artifact_paths) != 1):
        raise ValueError("Incomplete immutable installation artifacts")
    for artifact in receipt["artifacts"]:
        path = regular_file(artifact["path"])
        if not path.resolve().is_relative_to(directory.resolve()) or digest(path) != artifact["sha256"]:
            raise ValueError("Immutable installation evidence checksum mismatch")
    proof = json.loads(regular_file(receipt["proof"]).read_text())
    if type(proof["SSH_exit"]) is not int or proof["SSH_exit"] != 0:
        raise ValueError("Immutable samefile proof did not pass")
    if digest(regular_file(directory / "progress-alias-recovery.json")) != digest(proof_path):
        raise ValueError("Canonical proof differs from immutable receipt")
    closed = json.loads(regular_file(proof_path.parent / "closed-input.json").read_text())
    for witness in closed["witnesses"]:
        copied = regular_file(closed["retained_copies"][witness["path"]])
        if not copied.resolve().is_relative_to(proof_path.parent.resolve()) or digest(copied) != witness["sha256"]:
            raise ValueError("Retained closed input checksum mismatch")
    original = regular_file(proof["original_bundle"])
    result = subprocess.CompletedProcess([], proof["SSH_exit"], stdout=proof["SSH_stdout"])
    reconstructed = alias_proof(stage, stage_id, proof["remote_target"], original, result)
    for field in ("job", "stage_id", "phase", "owner", "remote_home", "canonical_root", "remote_target",
                  "alias_target", "archived_target", "target_sha256", "original_bundle_sha256"):
        if proof[field] != reconstructed[field]:
            raise ValueError("Immutable samefile proof binding differs")
    report = verify_archive(directory, stage, proof)
    if (report["log_sha256"] != receipt["log_sha256"]
            or digest(directory / f"raw-{stage['job']}.sha256") != receipt["manifest_sha256"]
            or digest(regular_file(report["bundle_preserved"])) != proof["original_bundle_sha256"]):
        raise ValueError("Canonical archive differs from immutable installation receipt")
    return {"job": stage["job"], "stage_id": stage_id, "already_SHA_archived": True,
            "installation_receipt": str(receipt_path), "alias_recovery_proof": receipt["proof"]}


def install_verified(directory, recovered, retained, stage, stage_id, proof, closed, state):
    job = stage["job"]
    report = verify_archive(recovered, stage, proof)
    original_manifest = Path(closed["retained_copies"][str(Path(closed["attempt"]) / "raw.sha256")])
    attempt = Path(report["attempt"])
    for path in (recovered / f"raw-{job}.sha256", attempt / "raw.sha256", attempt / "raw-after.sha256"):
        if digest(regular_file(path)) != digest(original_manifest):
            raise ValueError("Closed attempt and before/bundle/after manifests differ")
    destination = directory / f"live-trace-{job}"
    proof_path = retained / "samefile-proof.json"
    normalized = {**report, "raw": str(destination), "raw_manifest": str(directory / f"raw-{job}.sha256"),
                  "pointer_metadata": str(directory / f"raw-{job}-pointers.json"),
                  "bundle_preserved": proof["original_bundle"], "alias_recovery_proof": str(proof_path)}
    install_report = retained / "installation-report.json"
    write_new_json(install_report, normalized, immutable=True)
    artifacts = [proof_path, install_report, retained / "closed-input.json", attempt / "report.json"]
    receipt = {"schema": 1, "job": job, "stage_id": stage_id, "proof": str(proof_path),
               "installation_protocol": "verified_metadata_then_raw_local_lock", "coordinator_shares_lock": False,
               "manifest_sha256": digest(original_manifest), "log_sha256": report["log_sha256"],
               "artifacts": [{"path": str(path), "sha256": digest(path)} for path in artifacts]}
    receipt_source = retained / "installation-receipt.json"
    write_new_json(receipt_source, receipt, immutable=True)
    check_closed_current(directory, job, closed)
    check_runtime(state)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Coordinator installed raw during recovery; preserve both attempts")
    for name in (f"raw-{job}.sha256", f"raw-{job}-pointers.json", f"accounting-{job}.txt",
                 f"live-trace-{job}.out", f"live-trace-{job}.err"):
        atomic_copy(recovered / name, directory / name)
    atomic_copy(install_report, directory / f"archive-{job}.json")
    atomic_copy(proof_path, directory / "progress-alias-recovery.json", immutable=True)
    atomic_copy(receipt_source, directory / f"progress-alias-{job}-installation.json", immutable=True)
    check_closed_current(directory, job, closed)
    check_runtime(state)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Coordinator installed raw before final handoff; preserve both attempts")
    exclusive_raw_rename(recovered / f"live-trace-{job}", destination)
    return verify_installed(directory, stage, stage_id)


def recover(root, stage_id):
    root = Path(root).resolve()
    state = json.loads((root / "campaign-state.json").read_text())
    stage = state["stages"][stage_id]
    if (stage["stage_id"] != stage_id or not re.fullmatch(r"[a-zA-Z0-9_-]+", stage_id)
            or type(stage["job"]) is not int or stage["job"] <= 0
            or stage.get("terminal") != "COMPLETED" or stage.get("accounting_exit_code") != "0:0"):
        raise ValueError("Caller stage differs from completed bound ledger")
    directory = root / "stages" / stage_id
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Missing permanent stage directory")
    job = stage["job"]
    with local_install_lock(directory):
        retained = Path(tempfile.mkdtemp(prefix=f"progress-alias-attempt-{job}-", dir=directory))
        write_new_json(retained / "attempt-start.json", {"status": "started_incomplete", "attempt": str(retained),
                       "stage_id": stage_id, "job": job, "observed_at": datetime.now().astimezone().isoformat()}, immutable=True)
        try:
            check_runtime(state)
            destination = directory / f"live-trace-{job}"
            if destination.exists() or destination.is_symlink():
                result = verify_installed(directory, stage, stage_id)
            else:
                closed = bind_closed_attempt(directory, job, retained)
                bundle = Path(closed["retained_copies"][closed["bundle"]])
                with tarfile.open(bundle) as source:
                    pointers = [member for member in source if member.issym()
                                and str(PurePosixPath(member.name)) == "progress.json"]
                if len(pointers) != 1:
                    raise ValueError("Require one original archived progress symlink")
                proof = query_alias(stage, stage_id, pointers[0].linkname, Path(closed["bundle"]), retained)
                if proof["original_bundle_sha256"] != digest(bundle):
                    raise ValueError("Original downloaded bundle changed before query")
                write_new_json(retained / "samefile-proof.json", proof, immutable=True)
                recovered = retained / "verified-stage"
                recovered.mkdir()
                run_archiver(recovered, bundle, proof)
                result = install_verified(directory, recovered, retained, stage, stage_id, proof, closed, state)
                result["already_SHA_archived"] = False
        except BaseException as error:
            (retained / "failure.txt").write_text(traceback.format_exc())
            write_new_json(retained / "attempt-status.json", {"status": "failed_preserved", "error": repr(error),
                           "attempt": str(retained), "raw_exists_is_not_success": True})
            try:
                retain_report_paths(retained, directory, job)
            except Exception as relocation_error:
                write_new_json(retained / "report-path-diagnosis.json", {"error": repr(relocation_error),
                               "primary_failure_preserved": str(retained / "failure.txt")})
            raise
        retain_report_paths(retained, directory, job)
        write_new_json(retained / "attempt-status.json", {"status": "complete_verified", "attempt": str(retained),
                       "installation_receipt": result["installation_receipt"]})
        return result


def verify_performance_identity(root, expected=None):
    path = regular_file(root / "local-monitor-identity.json")
    before = file_witness(path)
    identity = json.loads(path.read_text())
    if expected is not None and identity != expected:
        raise ValueError("Performance identity changed; explicit revalidation required")
    pid, argv = identity.get("pid"), identity.get("argv")
    start = identity.get("actual_process_start")
    if (type(pid) is not int or pid <= 0 or not isinstance(argv, list)
            or len(argv) < 2 or not all(isinstance(argument, str) and argument for argument in argv)
            or argv.count("--root") != 1 or argv.index("--root") + 1 >= len(argv)
            or argv[argv.index("--root") + 1] != str(root)
            or not isinstance(start, str) or not start.strip() or identity.get("monitor_interval_s") != 300
            or Path(argv[1]).name not in ("performance_campaign.py", "performance_followup.py", "performance_path_recovery.py")):
        raise ValueError("Require actual performance PID/argv/root/start and 300s identity")
    os.kill(pid, 0)
    actual = {}
    for field in ("command", "lstart", "uid", "state"):
        result = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", field + "="],
                                text=True, capture_output=True, timeout=10)
        if result.returncode or not result.stdout.strip():
            raise ValueError("Actual performance process identity unavailable")
        actual[field] = result.stdout.strip()
    if (shlex.split(actual["command"]) != argv or " ".join(actual["lstart"].split()) != " ".join(start.split())
            or actual["uid"] != str(os.getuid()) or actual["state"].startswith("Z")):
        raise ValueError("Actual performance argv/start/owner mismatch or reused PID")
    if file_witness(path) != before:
        raise ValueError("Performance identity changed during validation")
    return identity, {"pid": pid, "argv": argv, "root": str(root), "actual_process_start": actual["lstart"],
                      "uid": actual["uid"], "state": actual["state"], "identity_sha256": before["sha256"]}


def follow(root, polls):
    root = Path(root).resolve()
    if type(polls) is not int or not 1 <= polls <= 288:
        raise ValueError("Require a bounded1..288 polling count")
    previous = None
    expected_identity = None
    for ordinal in range(1, polls + 1):
        started = time.monotonic()
        row = {"observed_at": datetime.now().astimezone().isoformat(), "ordinal": ordinal,
               "target_interval_s": 300, "actual_gap_s": None if previous is None else started - previous,
               "recoveries": [], "runtime_changed": False, "GPU_submissions": False}
        previous = started
        try:
            identity, actual_identity = verify_performance_identity(root, expected_identity)
            expected_identity = identity
            row["performance_pid"] = identity["pid"]
            row["performance_identity"] = actual_identity
            state = json.loads((root / "campaign-state.json").read_text())
            check_runtime(state)
            for stage_id, stage in state["stages"].items():
                row["stage_id"] = stage_id
                if stage.get("independent_verification_error") or "diagnosis_required" in stage.get("phase", ""):
                    raise ValueError("Current stage requires diagnosis: " + str(stage.get("independent_verification_error")))
                if stage.get("terminal") != "COMPLETED" or stage.get("accounting_exit_code") != "0:0":
                    continue
                if stage.get("phase") == "verified":
                    row.setdefault("closed_stages_not_reverified", []).append(stage_id)
                    continue
                directory = root / "stages" / stage_id
                raw = directory / f"live-trace-{stage['job']}"
                if raw.exists() or raw.is_symlink():
                    with local_install_lock(directory):
                        row["recoveries"].append(verify_installed(directory, stage, stage_id))
                else:
                    latest_failure_log(directory, stage["job"])
                    verify_performance_identity(root, expected_identity)
                    row["recovery_attempted"] = True
                    row["recoveries"].append(recover(root, stage_id))
                    verify_performance_identity(root, expected_identity)
            row["status"] = "observed_not_quality_verified"
        except ProcessLookupError:
            row["status"] = "stopped_performance_pid_absent"
        except Exception as error:
            row.update({"status": "failed_preserved_stopped", "error": repr(error), "traceback": traceback.format_exc()})
            failure = Path(tempfile.mkdtemp(prefix="progress-alias-follow-failure-", dir=root))
            write_new_json(failure / "diagnosis.json", row)
            row["diagnosis"] = str(failure / "diagnosis.json")
        row["elapsed_s"] = time.monotonic() - started
        if row["status"] == "observed_not_quality_verified" and not state.get("research_goal_complete"):
            row["next_delay_s"] = (300 if row.get("recovery_attempted") or row["elapsed_s"] >= 300
                                   else max(0, 300 - row["elapsed_s"]))
        with (root / "progress-alias-follow.jsonl").open("a") as output:
            output.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if row["status"] == "failed_preserved_stopped":
            raise RuntimeError("Alias follow stopped; permanent diagnosis: " + row["diagnosis"])
        if row["status"] == "stopped_performance_pid_absent":
            return
        if state.get("research_goal_complete") is True:
            return
        time.sleep(row["next_delay_s"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Host-only samefile proof; never change active runtime")
    parser.add_argument("--root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stage-id")
    mode.add_argument("--follow", action="store_true")
    parser.add_argument("--polls", type=int, default=288)
    args = parser.parse_args()
    if args.follow:
        if not 1 <= args.polls <= 288:
            parser.error("Require a bounded1..288 polling count")
        follow(args.root, args.polls)
    else:
        print(json.dumps(recover(args.root, args.stage_id), indent=2))
