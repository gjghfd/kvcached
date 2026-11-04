import argparse
import ast
import hashlib
import inspect
import json
import re
import shlex
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import performance_followup as followup
import rebalance_campaign
import stress_campaign
import verify_trace
import workaware_evidence
from archive_trace_job import verify_raw


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SOURCE_SHA256_AT_IMPORT = digest(__file__)


def verify_terminal_logs(directory, job, archive):
    if (type(job) is not int or job <= 0 or type(archive.get("job")) is not int
            or archive.get("job") != job or archive.get("missing_job_logs") != []
            or set(archive.get("log_sha256", {})) != {"out", "err"}):
        raise ValueError("Require both original job logs and their complete SHA inventory")
    for suffix, checksum in archive["log_sha256"].items():
        path = Path(directory) / f"live-trace-{job}.{suffix}"
        if path.is_symlink() or not path.is_file() or digest(path) != checksum:
            raise ValueError("Original terminal job log SHA changed")
    return archive["log_sha256"]


def verify_caller_binding(entry, receipt):
    if (type(entry.get("job")) is not int or type(receipt.get("job")) is not int
            or entry["job"] != receipt.get("job")
            or entry.get("stage_id") != receipt.get("stage_id")):
        raise ValueError("Alias receipt does not match the caller job/stage")


def verify_host_manifest(path, repo, loaded_risk_sha256=None):
    expected = json.loads(Path(path).read_text())
    names = {"experiments/dp_opt/performance_path_recovery.py", "experiments/dp_opt/risk_certificate_portability.py"}
    if set(expected) != names or digest(__file__) != SOURCE_SHA256_AT_IMPORT:
        raise ValueError("Require the exact independently frozen loaded host-source manifest")
    followup.check_hashes(Path(repo), expected)
    if expected["experiments/dp_opt/performance_path_recovery.py"] != SOURCE_SHA256_AT_IMPORT:
        raise ValueError("Host manifest does not describe the loaded recovery source")
    if loaded_risk_sha256 is not None and expected["experiments/dp_opt/risk_certificate_portability.py"] != loaded_risk_sha256:
        raise ValueError("Host manifest does not describe the loaded risk source")
    return digest(path)


def bind_risk_stage_verification(root, portability, manifest_sha256):
    original = followup.verify_stage
    root = Path(root).resolve()

    def verifier(actual_root, entry):
        if Path(actual_root).resolve() != root:
            raise ValueError("Risk verification root differs from the explicit campaign")
        stage_id = followup.safe_stage_id(entry["stage_id"])
        ledger = json.loads((root / "campaign-state.json").read_text())["stages"][stage_id]
        if (type(entry.get("job")) is not int or entry["job"] != ledger["job"] or entry["plan"] != ledger["plan"]):
            raise ValueError("Risk verification caller differs from the stage ledger")
        directory = root / "stages" / stage_id
        archive = json.loads((directory / f"archive-{entry['job']}.json").read_text())
        log_hashes = verify_terminal_logs(directory, entry["job"], archive)
        binding = {"root": str(root), "job": entry["job"], "stage_id": stage_id,
                   "phase": entry["plan"]["phase_name"], "trace_sha256": entry["plan"]["trace_sha256"],
                   "raw_manifest_sha256": digest(directory / f"raw-{entry['job']}.sha256"),
                   "archive_report_sha256": digest(directory / f"archive-{entry['job']}.json"),
                   "log_sha256": log_hashes, "host_source_manifest_sha256": manifest_sha256}
        with portability.verification_scope(binding):
            return original(actual_root, entry)

    followup.verify_stage = verifier


def normalized_inventory(phase, runtime, expected, alias_root, canonical_root):
    phase, runtime = Path(phase), Path(runtime)
    alias_root, canonical_root = str(alias_root).rstrip("/"), str(canonical_root).rstrip("/")
    if (not alias_root.startswith("/") or not canonical_root.startswith("/")
            or ".." in Path(alias_root).parts or ".." in Path(canonical_root).parts
            or alias_root == canonical_root):
        raise ValueError("Require independently confirmed distinct absolute runtime roots")
    output, aliases, seen = [], {}, {}
    for line in (phase / "source.sha256").read_text().splitlines():
        checksum, source = line.split(maxsplit=1)
        if not re.fullmatch(r"[a-f0-9]{64}", checksum) or not source.startswith("/"):
            raise ValueError("Invalid absolute launch source SHA")
        if source in seen and seen[source] != checksum:
            raise ValueError("Conflicting launch source SHA")
        seen[source] = checksum
        canonical = source
        if source.startswith(alias_root + "/"):
            relative = source[len(alias_root) + 1:]
            if ".." in Path(relative).parts or expected.get(relative) != checksum:
                raise ValueError("Alias launch file differs from immutable runtime manifest")
            if digest(runtime / relative) != checksum:
                raise ValueError("Actual runtime bytes differ from alias launch SHA")
            canonical = canonical_root + "/" + relative
            aliases[source] = canonical
        output.append(checksum + "  " + canonical)
    if not aliases:
        raise ValueError("No launch aliases require repair")
    inventory = {}
    for line in output:
        checksum, source = line.split(maxsplit=1)
        if source in inventory and inventory[source] != checksum:
            raise ValueError("Conflicting canonical launch source SHA")
        inventory[source] = checksum
    for group in ("runtime-source", "workaware-runtime-source"):
        if group == "workaware-runtime-source" and not (phase / (group + ".json")).exists():
            continue
        manifest = json.loads((phase / (group + ".json")).read_text())
        for name, source in manifest["sources"].items():
            actual = digest(phase / group / name)
            if actual != source["sha256"] or inventory.get(source["path"]) != actual:
                raise ValueError("Canonical source identity is not fully proved by preserved bytes")
    return "\n".join(output) + "\n", aliases


def prove_aliases(root, state, stage):
    directory = root / "stages" / stage["stage_id"]
    job = stage["job"]
    archive = json.loads((directory / f"archive-{job}.json").read_text())
    verify_terminal_logs(directory, job, archive)
    raw = directory / f"live-trace-{job}"
    if archive["state"] != "FAILED" or (raw / "exit-status").read_text().strip() != "1":
        raise ValueError("Only the exact postcheck FAILED allocation is eligible")
    phase = raw / stage["plan"]["phase_name"]
    if ((phase / "exit-status").read_text().strip() != "0"
            or json.loads((phase / "summary.json").read_text()).get("status") != "passed"):
        raise ValueError("All four original policies must independently drain before alias recovery")
    error = (directory / f"live-trace-{job}.err").read_text()
    if not error.rstrip().endswith("ValueError: Amortized runtime archive differs from launch source SHA"):
        raise ValueError("Do not reuse an unrelated failed experiment")
    verify_raw(raw, (directory / f"raw-{job}.sha256").read_text())
    followup.check_hashes(raw / "runtime", state["frozen_runtime_sha256"])
    command = ("set -eu\nprintf 'OWNER|%s\\n' \"$(id -un)\"\nprintf 'HOME|%s\\n' \"$HOME\"\nprintf 'DATA|%s\\n' \"$(readlink -f \"$HOME/run/kvcached-paracloud\")\"\n"
               + f"sacct -n -X -P -j {job} --format=JobIDRaw,User,JobName%128,Partition,State%64\n")
    output = rebalance_campaign.checked_ssh(command)
    records = [line.split("|", 1)[1] for line in output.splitlines() if line.startswith("HOME|")]
    roots = [line.split("|", 1)[1] for line in output.splitlines() if line.startswith("DATA|")]
    owners = [line.split("|", 1)[1] for line in output.splitlines() if line.startswith("OWNER|")]
    rows = [line.rstrip("|").split("|") for line in output.splitlines() if line.startswith(str(job) + "|")]
    if (records != [stage["remote_home"]] or len(roots) != 1 or len(rows) != 1
            or len(owners) != 1 or rows[0][1] != owners[0]
            or rows[0][2:] != [followup.JOB_NAME, "gpu_a800", "FAILED"]):
        raise ValueError("Actual home/accounting/name/A800/FAILED identity differs")
    suffix = f"/state/live-trace-{job}/runtime"
    alias_root = records[0] + "/run/kvcached-paracloud" + suffix
    canonical_root = roots[0] + suffix
    normalized, aliases = normalized_inventory(phase, raw / "runtime", state["frozen_runtime_sha256"],
                                               alias_root, canonical_root)
    commands = ["set -eu"]
    for source, canonical in aliases.items():
        commands.extend([f"test \"$(readlink -f {shlex.quote(source)})\" = {shlex.quote(canonical)}",
                         f"sha256sum {shlex.quote(source)}"])
    confirmed = rebalance_campaign.checked_ssh("\n".join(commands))
    expected = {line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
                for line in (phase / "source.sha256").read_text().splitlines()}
    confirmed_sources = set()
    for line in confirmed.splitlines():
        checksum, source = line.split(maxsplit=1)
        if source not in aliases or source in confirmed_sources or expected.get(source) != checksum:
            raise ValueError("Remote alias bytes changed after terminal archive")
        confirmed_sources.add(source)
    if confirmed_sources != set(aliases):
        raise ValueError("Remote alias verification did not cover every source")
    receipt = {"job": job, "stage_id": stage["stage_id"], "actual_terminal": "FAILED",
               "inner_phase_exit_status": 0, "original_allocation_not_reclassified": True,
               "original_source_inventory_sha256": digest(phase / "source.sha256"),
               "normalized_source_inventory_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
               "raw_manifest_sha256": digest(directory / f"raw-{job}.sha256"),
               "alias_root": alias_root, "canonical_root": canonical_root, "aliases": aliases,
               "accounting_and_data_root_stdout": output, "readlink_and_SHA_stdout": confirmed}
    path = directory / "source-path-alias-audit.json"
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise ValueError("Preserve the first independently confirmed alias audit")
    if not path.exists():
        followup.write_json(path, receipt)
    return phase, raw, receipt


def audit_binding(root, state, phase):
        phase = Path(phase)
        for stage in state["stages"].values():
            directory = root / "stages" / stage["stage_id"]
            receipt_path = directory / "source-path-alias-audit.json"
            if "job" not in stage or not receipt_path.exists():
                continue
            raw = directory / f"live-trace-{stage['job']}"
            original_phase = raw / stage["plan"]["phase_name"]
            receipt = json.loads(receipt_path.read_text())
            if digest(phase / "source.sha256") != receipt["original_source_inventory_sha256"]:
                continue
            if (receipt["job"] != stage["job"] or receipt["stage_id"] != stage["stage_id"]
                    or receipt["actual_terminal"] != "FAILED"
                    or digest(directory / f"raw-{stage['job']}.sha256") != receipt["raw_manifest_sha256"]
                    or followup.file_hashes(phase) != followup.file_hashes(original_phase)):
                raise ValueError("Alias receipt is not bound to the immutable complete phase")
            verify_raw(raw, (directory / f"raw-{stage['job']}.sha256").read_text())
            verify_terminal_logs(directory, stage["job"], json.loads((directory / f"archive-{stage['job']}.json").read_text()))
            normalized, aliases = normalized_inventory(phase, raw / "runtime", state["frozen_runtime_sha256"],
                                                       receipt["alias_root"], receipt["canonical_root"])
            if (aliases != receipt["aliases"] or hashlib.sha256(normalized.encode()).hexdigest()
                    != receipt["normalized_source_inventory_sha256"]):
                raise ValueError("Confirmed alias normalization changed")
            return normalized, receipt
        return None


def alias_stage_function(original, terminal_check):
    tree = ast.parse(inspect.getsource(original))
    messages = {"Failed or partial allocations are not reclassified as success": 0,
                "Raw terminal status or stage/request binding differs": 0}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or len(node.body) != 1 or not isinstance(node.body[0], ast.Raise):
            continue
        error = node.body[0].exc
        if not isinstance(error, ast.Call) or len(error.args) != 1 or not isinstance(error.args[0], ast.Constant):
            continue
        message = error.args[0].value
        if message not in messages:
            continue
        messages[message] += 1
        if message.startswith("Failed or partial"):
            node.test = ast.parse("not _confirmed_alias_terminal(root, entry)", mode="eval").body
        else:
            node.test = ast.parse('read_json(raw / "stage.json") != entry["plan"]', mode="eval").body
    if set(messages.values()) != {1}:
        raise ValueError("Require exactly two known outer-only terminal gates")
    namespace = {**original.__globals__, "_confirmed_alias_terminal": terminal_check}
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(original), "exec"), namespace)
    return namespace[original.__name__]


def install_verifier(root, state):
    original_runtime = verify_trace.verify_amortized_runtime
    original_workaware = workaware_evidence.verify_runtime
    original_stage = followup.verify_stage

    def runtime_verifier(phase, headroom=False):
        binding = audit_binding(root, state, phase)
        if binding is None:
            return original_runtime(phase, headroom)
        normalized, receipt = binding
        with tempfile.TemporaryDirectory(prefix="independent-source-alias-") as temporary:
            audit = Path(temporary)
            shutil.copytree(Path(phase) / "runtime-source", audit / "runtime-source")
            shutil.copy2(Path(phase) / "runtime-source.json", audit / "runtime-source.json")
            (audit / "source.sha256").write_text(normalized)
            return original_runtime(audit, headroom)

    def terminal_check(actual_root, entry):
        raw = Path(actual_root) / "stages" / entry["stage_id"] / f"live-trace-{entry['job']}"
        binding = audit_binding(root, state, raw / entry["plan"]["phase_name"])
        if binding is not None:
            verify_caller_binding(entry, binding[1])
        return (Path(actual_root).resolve() == root.resolve() and binding is not None
                and entry.get("terminal") == "FAILED" and entry.get("accounting_exit_code") == "1:0"
                and (raw / "exit-status").read_text().strip() == "1")

    def workaware_verifier(phase):
        binding = audit_binding(root, state, phase)
        if binding is None:
            return original_workaware(phase)
        normalized, receipt = binding
        with tempfile.TemporaryDirectory(prefix="independent-workaware-source-alias-") as temporary:
            audit = Path(temporary)
            shutil.copytree(Path(phase) / "workaware-runtime-source", audit / "workaware-runtime-source")
            shutil.copy2(Path(phase) / "workaware-runtime-source.json", audit / "workaware-runtime-source.json")
            (audit / "source.sha256").write_text(normalized)
            return original_workaware(audit)

    amended_stage = alias_stage_function(original_stage, terminal_check)

    def stage_verifier(actual_root, entry):
        directory = Path(actual_root) / "stages" / entry["stage_id"]
        if not (directory / "source-path-alias-audit.json").exists():
            return original_stage(actual_root, entry)
        verify_caller_binding(entry, json.loads((directory / "source-path-alias-audit.json").read_text()))
        result = amended_stage(actual_root, entry)
        result.update({"source_path_aliases_independently_verified": True,
                       "original_allocation_terminal": "FAILED", "original_allocation_not_reclassified": True})
        return result

    verify_trace.verify_amortized_runtime = runtime_verifier
    workaware_evidence.verify_runtime = workaware_verifier
    followup.verify_stage = stage_verifier


def main():
    parser = argparse.ArgumentParser(description="Explicit postcheck path-alias recovery; GPU sources remain frozen")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--repair-stage")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--risk-portability-audit", action="store_true")
    parser.add_argument("--host-source-manifest", type=Path)
    parser.add_argument("--verify-completed-stage")
    args = parser.parse_args()
    root = args.root.resolve()
    state = json.loads((root / "campaign-state.json").read_text())
    followup.check_frozen_sources(Path(state["repo"]), state)
    if args.risk_portability_audit:
        if args.host_source_manifest is None:
            parser.error("NEW risk recovery requires an independent loaded host-source manifest")
        import risk_certificate_portability

        manifest_sha256 = verify_host_manifest(args.host_source_manifest, state["repo"], risk_certificate_portability.SOURCE_AT_IMPORT_SHA256)
        risk_certificate_portability.install_verifier(root / "risk-portability-audit.jsonl")
    elif args.host_source_manifest or args.verify_completed_stage:
        parser.error("Host-source/complete-stage recovery requires the risk audit flag")
    if args.repair_stage:
        with followup.campaign_lock(root):
            stage = state["stages"][followup.safe_stage_id(args.repair_stage)]
            prove_aliases(root, state, stage)
            install_verifier(root, state)
            if args.risk_portability_audit:
                bind_risk_stage_verification(root, risk_certificate_portability, manifest_sha256)
            result = followup.verify_stage(root, stage)
            stage.update({"phase": "verified", "verification": result,
                          "original_allocation_terminal": "FAILED", "postcheck_alias_recovery": True})
            state["path_recovery_previous_error"] = state.get("error")
            state["phase"] = "missing_native_cases_rescue_one_case_per_allocation"
            stress_campaign.save_state(root, state)
    else:
        install_verifier(root, state)
        if args.risk_portability_audit:
            bind_risk_stage_verification(root, risk_certificate_portability, manifest_sha256)
    if args.verify_completed_stage:
        stage_id = followup.safe_stage_id(args.verify_completed_stage)
        stage = state["stages"][stage_id]
        if stage.get("terminal") != "COMPLETED" or stage.get("accounting_exit_code") != "0:0":
            raise ValueError("Only an actual completed0:0 allocation may use this audit")
        result = followup.verify_stage(root, stage)
        result.update({"host_source_manifest_sha256": manifest_sha256,
                       "original_allocation_terminal": "COMPLETED", "runtime_or_trace_changed": False})
        followup.write_json(root / "stages" / stage_id / f"independent-performance-verification-host-{manifest_sha256[:12]}.json", result)
        return
    if args.audit_only:
        return
    original_ssh = rebalance_campaign.checked_ssh

    def canonical_ssh(command, *arguments, **keywords):
        prefix = 'export KVCACHED_DATA_ROOT="$(readlink -f "$HOME/run/kvcached-paracloud")"\n'
        return original_ssh(prefix + command, *arguments, **keywords)

    rebalance_campaign.checked_ssh = canonical_ssh
    try:
        followup.run(root, args.native_root.resolve(), state["native_job"], state["debug_job"], state["debug_name"],
                     False, False, False, state["policy_walltime"], state["allow_checkpoint_cancelled"], None, False)
    except Exception as error:
        failure = {"error": repr(error), "recorded_at": datetime.now().astimezone().isoformat(),
                   "automatic_duplicate_submission_authorized": False, "recovery_source_sha256": digest(__file__)}
        followup.write_json(root / f"path-recovery-error-{time.time_ns()}.json", failure)
        current = json.loads((root / "campaign-state.json").read_text())
        current.update({"failure_phase": current["phase"], "phase": "diagnosis_required_followup", "error": repr(error)})
        stress_campaign.save_state(root, current)
        raise


if __name__ == "__main__":
    main()
