import argparse
import fcntl
import hashlib
import inspect
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath


JOB_NAME = "qwen-upstream-first-layer-probe"
PERFORMANCE_NAME = "qwen-public-perf-followup"
ACTIVE = "PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|RESIZING|REQUEUED|REQUEUE_FED|REQUEUE_HOLD|SIGNALING|STAGE_OUT"
TERMINAL = "COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED"
NEW_FILES = ("experiments/dp_opt/upstream_first_layer_campaign.py",
             "experiments/dp_opt/upstream-first-layer-probe.sbatch",
             "tests/test_dp_opt_upstream_first_layer_campaign.py")
PROBE_FILES = ("experiments/dp_opt/upstream_first_layer_worker.py",
               "experiments/dp_opt/upstream_first_layer_probe.py",
               "tests/test_dp_opt_upstream_first_layer_probe.py")
COPY_ONLY = "kvcached/integration/vllm/patches.py"
INSTALLED_SCHEMA = "upstream-first-layer-installed-provenance-v1"
INSTALLED_SYMBOLS = {"linear_class", "linear_forward", "linear_base", "column_parallel_linear",
                     "quant_method_class", "quant_method_base", "quant_method_apply",
                     "cudagraph_wrapper", "cudagraph_call", "model_runner", "model_forward",
                     "runner_model_class", "model_class"}
LAYOUTS = ["balanced", "corank0", "corank1", "corank1", "corank0", "balanced"]
SCOPE = {"prompt_tokens": 149, "positions": [52, 53, 54], "candidates": [12822, 20632],
         "P74_first_layer_coverage": [], "P74_known_uncovered_positions": [139, 442]}
CONTROLS = {"partition": "gpu_a800", "gpus": 2, "nodes": 1, "walltime": "03:00:00",
            "monitor_interval_s": 300, "layouts": LAYOUTS, "scientific_scope": SCOPE,
            "policies": ["no_move"], "dtype": "bfloat16", "head_dtype": "float32",
            "max_model_len": 131072, "cache_bytes": 25769803776, "kv_budget_mib": 10240,
            "gpu_memory_utilization": .98, "pause_sync_interval": 32,
            "copies_allowed": False, "pause_allowed": False, "is_algorithm_benefit": False}
SSH_OPTIONS = ["-o", "ConnectTimeout=30", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3"]
SUBMIT_REGISTER_BOUND_S = 60
PERFORMANCE_POLL_FRESH_S = 30
PERFORMANCE_POLL_MARGIN_S = 30


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def positive_job(value):
    if not re.fullmatch(r"[1-9][0-9]*", str(value)):
        raise ValueError("Require a canonical positive job ID")
    return int(value)


def relative_path(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError("Unsafe immutable relative path")
    return name


def hashes(directory):
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise ValueError("Immutable payload cannot contain symlinks")
        if path.is_file():
            result[str(path.relative_to(directory))] = digest(path)
    return result


def check_hashes(directory, expected, exact=False):
    for name, checksum in expected.items():
        relative_path(name)
        path = Path(directory) / name
        if path.is_symlink() or not path.is_file() or digest(path) != checksum:
            raise ValueError("Frozen SHA mismatch: " + str(path))
    if exact and hashes(directory) != expected:
        raise ValueError("Immutable file set differs")


def manifest_hashes(text):
    result = {}
    for line in text.splitlines():
        checksum, name = line.split(maxsplit=1)
        name = name.removeprefix("./")
        relative_path(name)
        if not re.fullmatch(r"[0-9a-f]{64}", checksum) or name in result:
            raise ValueError("Invalid or duplicate raw SHA manifest entry")
        result[name] = checksum
    if not result:
        raise ValueError("Empty immutable manifest")
    return result


def read_registry(performance, observed=None):
    root = Path(performance["debug_registry"]["path"])
    if root != Path(performance["campaign_root"]) / "authorized-debug":
        raise ValueError("DEBUG registry not bound to explicit performance root")
    paths = sorted(root.glob("[0-9]*.json"))
    identities = []
    previous = None
    for sequence, path in enumerate(paths):
        record = read(path)
        if (path.name != f"{sequence:08d}.json" or record["sequence"] != sequence
                or record["previous_sha256"] != previous
                or record["campaign_root"] != performance["campaign_root"]):
            raise ValueError("DEBUG authorization SHA chain differs")
        checksum = digest(path)
        if sequence == 0 and checksum != performance["debug_registry"]["initial_sha256"]:
            raise ValueError("Initial DEBUG authorization changed")
        if sequence and (record.get("operator_explicit_registration") is not True
                         or record.get("remote_verified") is not True):
            raise ValueError("DEBUG allocation lacks operator registration")
        if observed and sequence == observed["sequence"] and checksum != observed["sha256"]:
            raise ValueError("Observed DEBUG authorization changed")
        for identity in record["identities"]:
            positive_job(identity["job"])
            if not re.fullmatch(r"qwen-[a-zA-Z0-9_.-]+", identity["name"]):
                raise ValueError("Unsafe explicit DEBUG identity")
            identities.append(identity)
        previous = checksum
    if not paths or observed and len(paths) <= observed["sequence"]:
        raise ValueError("DEBUG authorization missing or rolled back")
    if len({identity["job"] for identity in identities}) != len(identities):
        raise ValueError("Duplicate DEBUG identity")
    return identities, {"sequence": len(paths) - 1, "sha256": previous}


def performance_state(root, observed=None):
    root = Path(root).resolve()
    state = read(root / "campaign-state.json")
    if state.get("performance_independent_of_debug") is not True or "debug_registry" not in state:
        raise ValueError("Require prepared independent r5 performance coordinator")
    state["campaign_root"] = str(root)
    identities, head = read_registry(state, observed)
    active = [entry["job"] for entry in state["stages"].values()
              if "job" in entry and entry["phase"] != "verified"]
    if len(active) > 1:
        raise ValueError("Reconcile multiple unclosed performance allocations")
    state["active_performance_job"] = positive_job(active[0]) if active else None
    return state, identities, head


def queue_guard(performance_job, identities, debug_limit):
    if type(debug_limit) is not int or debug_limit not in (0, 1):
        raise ValueError("Only zero or one explicitly authorized active DEBUG allocation")
    mapping = {positive_job(entry["job"]): entry["name"] for entry in identities}
    owners = {positive_job(entry["job"]): entry.get("owner", "") for entry in identities}
    if performance_job is not None:
        if performance_job in mapping:
            raise ValueError("DEBUG cannot own the performance allocation")
        mapping[positive_job(performance_job)] = PERFORMANCE_NAME
    if any(not re.fullmatch(r"qwen-[a-zA-Z0-9_.-]+", name) for name in mapping.values()):
        raise ValueError("Unsafe queue name")
    serialized = ";".join(f"{job}:{name}" for job, name in mapping.items())
    script = ['set -eu', 'up_owner="$(id -un)"', 'test "$up_owner" = "$USER"',
              'up_queue="$(squeue -h -u "$up_owner" -o \'%i|%u|%j|%P|%T\')"',
              f"printf '%s\\n' \"$up_queue\" | awk -F'|' -v owner=\"$up_owner\" -v mapping={shlex.quote(serialized)} '",
              'BEGIN { split(mapping, pairs, ";"); for (pair_index in pairs) { split(pairs[pair_index], pair, ":"); names[pair[1]]=pair[2] } }',
              '/^[[:space:]]*$/ { next }',
              '{ for (field=1; field<=NF; field++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)',
              f'  if (NF != 5 || !($1 in names) || seen[$1]++ || $2 != owner || $3 != names[$1] || $4 != "gpu_a800" || $5 !~ /^({ACTIVE})$/) invalid=1 }}',
              "END { exit invalid }'", 'up_debug_active=0']
    for job, name in mapping.items():
        script.extend([f'up_account="$(sacct -n -X -P -j {job} --format=JobIDRaw,User,JobName%128,Partition,State%64)"',
            f"up_active=\"$(printf '%s\\n' \"$up_account\" | awk -F'|' -v owner=\"$up_owner\" -v registered_owner={shlex.quote(owners.get(job, ''))} -v job={job} -v name={shlex.quote(name)} '",
            '/^[[:space:]]*$/ { next }',
            '{ sub(/\\|$/, "", $0); for (field=1; field<=NF; field++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)',
            '  sub(/[ +].*$/, "", $5); count++',
            f'  if (NF != 5 || $1 != job || $2 != owner || (registered_owner != "" && $2 != registered_owner) || $3 != name || $4 != "gpu_a800" || $5 !~ /^({ACTIVE}|{TERMINAL})$/) invalid=1',
            f'  if ($5 ~ /^({ACTIVE})$/) active=1 }}',
            "END { if (invalid || count != 1) exit 1; print (active + 0) }')\""])
        if job != performance_job:
            script.append('up_debug_active=$((up_debug_active + up_active))')
    script.append(f'test "$up_debug_active" -le {debug_limit}')
    return "\n".join(script)


def parent_evidence(parent):
    state = read(parent / "campaign-state.json")
    archive = read(parent / "archive-278529.json")
    if (state.get("job") != 278529 or state.get("state_last_observed") != "COMPLETED"
            or state.get("GPU_diagnostic_complete") is not True
            or archive["job"] != 278529 or archive["state"] != "COMPLETED" or archive["missing_job_logs"]):
        raise ValueError("Require the real SHA-archived completed 278529 DEBUG parent")
    raw = parent / "live-trace-278529"
    check_hashes(raw, manifest_hashes((parent / "raw-278529.sha256").read_text()), exact=True)
    for suffix in ("out", "err"):
        if digest(parent / f"live-trace-278529.{suffix}") != archive["log_sha256"][suffix]:
            raise ValueError("Parent job log SHA differs")
    report = read(parent / "independent-head-hidden-verification.json")
    if report != read(raw / "independent-head-hidden-verification.json"):
        raise ValueError("Independent parent head/hidden evidence differs")
    requests = {request["request_id"]: request for request in read(raw / "trace.json")["requests"]}
    attributable = [row for row in report["projection_comparisons"]
        if requests[row["request_id"]]["prompt_tokens"] == 149 and row.get("first_top1_attribution_available") is True]
    if (len(attributable) != 4 or {(row["repeat"], row["layout"]) for row in attributable}
            != {(1, "corank0"), (2, "corank1"), (3, "corank1"), (4, "corank0")}
            or any(row["first_divergence_position"] != 53 or row.get("full_common_prefix") is not True
                   or row["hidden_changed_elements"] <= 0 or not math.isfinite(row["hidden_contribution"])
                   or abs(row["hidden_contribution"]) < 1e-12 for row in attributable)):
        raise ValueError("Require actual P149/53 common-prefix upstream-hidden evidence")
    return state, raw


def source_manifest_diff(parent_hashes, performance_hashes):
    shared = set(parent_hashes) & set(performance_hashes)
    conflicts = {name: {"parent_sha256": parent_hashes[name], "performance_sha256": performance_hashes[name]}
                 for name in sorted(shared) if parent_hashes[name] != performance_hashes[name]}
    return {"parent_source_count": len(parent_hashes), "performance_source_count": len(performance_hashes),
            "shared_source_count": len(shared), "union_source_count": len(set(parent_hashes) | set(performance_hashes)),
            "conflicts": conflicts, "old_manifests_modified": False}


def source_union(parent_hashes, performance_hashes):
    report = source_manifest_diff(parent_hashes, performance_hashes)
    if report["conflicts"]:
        raise ValueError("Explicit parent/performance source epoch conflicts: " + json.dumps(report, sort_keys=True))
    return {**performance_hashes, **parent_hashes}, report


def validate_source_epochs(parent, performance_root, repo):
    parent_state, raw = parent_evidence(Path(parent).resolve())
    performance, identities, head = performance_state(performance_root)
    if Path(performance["repo"]).resolve() != Path(repo).resolve():
        raise ValueError("Bind source-only validation to the explicit performance repository")
    sources, report = source_union(parent_state["frozen_runtime_sha256"], performance["frozen_runtime_sha256"])
    check_hashes(repo, sources)
    return {**report, "local_union_source_sha_verified": True, "parent_job": 278529,
            "parent_raw_file_count": len(hashes(raw)), "active_performance_job": performance["active_performance_job"],
            "prepare_performed": False, "ssh_performed": False}


def prepare(root, parent, performance_root, repo):
    root, parent, performance_root, repo = map(lambda path: Path(path).resolve(), (root, parent, performance_root, repo))
    parent_state, raw = parent_evidence(parent)
    performance, identities, head = performance_state(performance_root)
    if performance.get("controls", {}).get("monitor_interval_s") != 300:
        raise ValueError("Unsupported performance poll interval; do not guess a submission window")
    if Path(performance["repo"]).resolve() != repo:
        raise ValueError("Bind performance source root to the same repository")
    source_hashes, union_report = source_union(parent_state["frozen_runtime_sha256"], performance["frozen_runtime_sha256"])
    implementation_hashes = {name: digest(repo / name) for name in PROBE_FILES}
    source_hashes.update(implementation_hashes)
    source_hashes.update({name: digest(repo / name) for name in NEW_FILES})
    source_hashes.setdefault(COPY_ONLY, digest(repo / COPY_ONLY))
    check_hashes(repo, source_hashes)
    root.mkdir(parents=True, exist_ok=True)
    upload = root / "upload"
    upload.mkdir()
    runtime = upload / "runtime"
    for name in source_hashes:
        target = runtime / relative_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)
    shutil.copytree(raw / "frozen", upload / "frozen")
    shutil.copy2(raw / "placement-fit.json", upload / "placement-fit.json")
    provenance = upload / "provenance"
    provenance.mkdir()
    save(provenance / "parent-performance-source-union.json", union_report)
    for source, name in ((parent / "campaign-state.json", "parent-campaign-state.json"),
                         (performance_root / "campaign-state.json", "performance-campaign-state.json"),
                         (parent / "independent-head-hidden-verification.json", "parent-independent-verification.json"),
                         (parent / "raw-278529.sha256", "parent-raw.sha256")):
        shutil.copy2(source, provenance / name)
    entry = performance_root / "local-monitor-identity.json"
    shutil.copy2(entry, provenance / "performance-coordinator-identity.json")
    save(upload / "runtime-manifest.json", source_hashes)
    save(upload / "plan.json", {"controls": CONTROLS, "parent_job": 278529,
         "trace_sha256": digest(upload / "frozen/trace.json"), "frozen_sha256": hashes(upload / "frozen"),
         "placement_fit_sha256": digest(upload / "placement-fit.json"),
         "performance_root": str(performance_root), "protected_patch_copy_only": COPY_ONLY,
         "first_layer_implementation_sha256": implementation_hashes})
    check_hashes(runtime, source_hashes, exact=True)
    check_hashes(repo, source_hashes)
    payload = hashes(upload)
    with (root / "payload.sha256").open("x") as output:
        output.write("".join(f"{checksum}  {name}\n" for name, checksum in payload.items()))
    with tarfile.open(root / "payload.tar.gz", "w:gz") as archive:
        for name in payload:
            archive.add(upload / name, arcname=name, recursive=False)
    for path in runtime.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    state = {"repo": str(repo), "parent": str(parent), "concurrent_performance_root": str(performance_root),
             "phase": "prepared", "controls": CONTROLS, "runtime_sha256": source_hashes,
             "performance_runtime_sha256": performance["frozen_runtime_sha256"],
             "parent_runtime_sha256": parent_state["frozen_runtime_sha256"],
             "first_layer_implementation_sha256": implementation_hashes,
             "performance_registry": performance["debug_registry"], "authorization_observed": head,
             "payload_sha256": digest(root / "payload.tar.gz"), "payload_files_sha256": payload,
             "stage_name": "upstream-first-layer-" + uuid.uuid4().hex,
             "is_algorithm_benefit": False, "publication_authorized": False,
             "performance_independent_of_debug": True, "GPU_diagnostic_complete": False}
    save(root / "campaign-state.json", state)
    return state


def validate_local(root, state):
    if state["controls"] != CONTROLS:
        raise ValueError("Preserve immutable scientific controls")
    check_hashes(Path(state["repo"]), state["runtime_sha256"])
    check_hashes(root / "upload", state["payload_files_sha256"], exact=True)
    if digest(root / "payload.tar.gz") != state["payload_sha256"]:
        raise ValueError("Immutable package changed")
    performance, identities, head = performance_state(state["concurrent_performance_root"], state["authorization_observed"])
    if (performance["frozen_runtime_sha256"] != state["performance_runtime_sha256"]
            or performance["debug_registry"] != state["performance_registry"]):
        raise ValueError("Performance runtime/registry binding changed")
    return performance, identities, head


def ssh(root, label, script, stdin_path=None, timeout=240):
    stamp = str(time.time_ns())
    command = ["ssh", *SSH_OPTIONS, "paracloud", script if stdin_path else "bash -s"]
    started = time.monotonic()
    try:
        if stdin_path:
            with Path(stdin_path).open("rb") as source:
                result = subprocess.run(command, stdin=source, text=True, capture_output=True, timeout=timeout)
        else:
            result = subprocess.run(command, input=script, text=True, capture_output=True, timeout=timeout)
        save(root / f"{label}-{stamp}.json", {"ssh_exit": result.returncode, "stdout": result.stdout,
             "stderr": result.stderr, "elapsed_monotonic_s": time.monotonic() - started})
        if result.returncode:
            raise RuntimeError(f"{label}: SSH exit {result.returncode}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired as error:
        save(root / f"{label}-{stamp}.json", {"ssh_exit": None, "timeout": True,
             "error": str(error), "elapsed_monotonic_s": time.monotonic() - started})
        raise


def protected_sync(root, state):
    performance, identities, head = validate_local(root, state)
    guard = queue_guard(performance["active_performance_job"], identities, 0)
    home = ssh(root, "before-sync", guard + '\nprintf "HOME_RESULT|%s\\n" "$HOME"')
    lines = [line.split("|", 1)[1] for line in home.splitlines() if line.startswith("HOME_RESULT|")]
    if len(lines) != 1 or not lines[0].startswith("/") or "\n" in lines[0]:
        raise ValueError("Invalid remote home")
    stage = lines[0] + "/run/kvcached-paracloud/sync/" + state["stage_name"]
    ssh(root, "stage-create", guard + "\nmkdir -p " + shlex.quote(stage + "/backup"))
    ssh(root, "stage-upload", "cat > " + shlex.quote(stage + "/incoming.tar.gz"), root / "payload.tar.gz", 1800)
    commands = [guard, f'cd {shlex.quote(stage)}',
                f'test "$(sha256sum incoming.tar.gz | cut -d\' \' -f1)" = {state["payload_sha256"]}',
                'test ! -e incoming; mkdir incoming; tar -xzf incoming.tar.gz -C incoming',
                'mkdir -p upload']
    for name, checksum in state["payload_files_sha256"].items():
        quoted = shlex.quote(name)
        directory = shlex.quote(str(PurePosixPath(name).parent))
        commands.extend([f'test "$(sha256sum incoming/{quoted} | cut -d\' \' -f1)" = {checksum}',
            f'mkdir -p upload/{directory} backup/{directory}',
            f'if test -e upload/{quoted}; then test -f upload/{quoted}; test ! -L upload/{quoted}; '
            f'sha256sum upload/{quoted} >> before.sha256; cp -p upload/{quoted} backup/{quoted}; fi',
            f'cp -p incoming/{quoted} upload/{quoted}.new; mv upload/{quoted}.new upload/{quoted}',
            f'test "$(sha256sum upload/{quoted} | cut -d\' \' -f1)" = {checksum}',
            f'sha256sum upload/{quoted} >> after.sha256'])
    commands.extend(['find upload/runtime -type f -exec chmod a-w {} +', guard])
    ssh(root, "protected-copy-only-install", "\n".join(commands), timeout=540)
    state.update({"remote_home": lines[0], "remote_stage": stage,
                  "submission_ledger": stage + "/submitted-debug-job.txt", "phase": "staged"})
    save(root / "campaign-state.json", state)


def submit_held(root, state):
    performance, identities, head = validate_local(root, state)
    window = performance_poll_window(state["concurrent_performance_root"])
    if window["receipt_sha256"] != state.get("performance_poll_window", {}).get("receipt_sha256"):
        raise ValueError("Performance poll changed after the bounded submission window was granted")
    remaining = state["submit_register_deadline_epoch"] - time.time()
    if not 0 < remaining <= SUBMIT_REGISTER_BOUND_S:
        raise ValueError("Submission/registration deadline expired or unsupported")
    ledger = shlex.quote(state["submission_ledger"])
    stage = shlex.quote(state["remote_stage"])
    script = queue_guard(performance["active_performance_job"], identities, 0) + "\n" + (
        f"test ! -e {ledger}; mkdir {stage}/submission-intent; "
        f"export DP_OPT_UPSTREAM_UPLOAD={shlex.quote(state['remote_stage'] + '/upload')}; "
        'mkdir -p "$HOME/run/kvcached-paracloud/logs"; '
        f'sbatch --hold --parsable --partition=gpu_a800 --nodes=1 --gpus=2 --time=03:00:00 '
        f'--job-name={JOB_NAME} --export=ALL '
        '--output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
        '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
        f'{shlex.quote(state["remote_stage"] + "/upload/runtime/experiments/dp_opt/upstream-first-layer-probe.sbatch")} '
        f'> {ledger}; cat {ledger}')
    output = ssh(root, "submit-held", script, timeout=remaining)
    job = positive_job(output.split(";", 1)[0])
    return job


def receipt(root, job):
    actual = positive_job((root / "submission-receipt.txt").read_text().strip())
    if actual != positive_job(job):
        raise ValueError("Job is not the campaign's exact submission receipt")


def release_registered(root, state, job):
    receipt(root, job)
    performance, identities, head = validate_local(root, state)
    registered = [entry for entry in identities if entry["job"] == job and entry["name"] == JOB_NAME]
    if len(registered) != 1 or not registered[0].get("owner"):
        raise ValueError("New DEBUG job lacks the performance coordinator's explicit owner authorization")
    guard = queue_guard(performance["active_performance_job"], identities, 1)
    ssh(root, "release-registered", guard + f'\nscontrol release {job}')


def cancel_owned_held(root, state, job):
    receipt(root, job)
    script = f'''set -eu
owner="$(id -un)"
test "$owner" = "$USER"
queue="$(squeue -h -j {job} -o '%i|%u|%j|%P|%T')"
printf '%s\n' "$queue" | awk -F'|' -v job={job} -v owner="$owner" '
{{ for (field=1; field<=NF; field++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)
   count++; if (NF != 5 || $1 != job || $2 != owner || $3 != "{JOB_NAME}" || $4 != "gpu_a800" || $5 != "PENDING") bad=1 }}
END {{ exit (bad || count != 1) }}'
control="$(scontrol show job {job} -o)"
case " $control " in *" JobState=PENDING "*) ;; *) exit 2;; esac
case " $control " in *" Reason=JobHeldUser "*) ;; *) exit 2;; esac
scancel {job}
'''
    ssh(root, "cancel-only-owned-held-debug", script, timeout=20)


def authorization_chain(root, state):
    root = Path(root).resolve()
    python = shlex.quote(sys.executable)
    campaign = shlex.quote(str(root / "upload/runtime/experiments/dp_opt/upstream_first_layer_campaign.py"))
    performance = shlex.quote(str(Path(state["repo"]) / "experiments/dp_opt/performance_followup.py"))
    root_arg = shlex.quote(str(root))
    perf_arg = shlex.quote(state["concurrent_performance_root"])
    receipt_path = shlex.quote(str(root / "submission-receipt.txt"))
    registration = shlex.quote(str(root / "performance-debug-registration.json"))
    return f'''set -euo pipefail
released=0
job=""
cleanup() {{
    result=$?
    trap - EXIT
    if test "$result" -ne 0 && test "$released" = 0 && test -n "$job"; then
        {python} {campaign} --root {root_arg} --cancel-owned-held-job "$job" || true
    fi
    exit "$result"
}}
trap cleanup EXIT
trap 'exit 124' TERM
trap 'exit 130' INT
set -o noclobber
{python} {campaign} --root {root_arg} --submit-held > {receipt_path}
job="$(cat {receipt_path})"
{python} {performance} --root {perf_arg} --register-debug-job "$job" --register-debug-name {JOB_NAME} > {registration}
{python} {campaign} --root {root_arg} --release-registered-job "$job"
released=1
'''


def performance_poll_window(performance_root, now=None):
    now = time.time() if now is None else now
    performance, identities, head = performance_state(performance_root)
    if performance.get("controls", {}).get("monitor_interval_s") != 300:
        raise ValueError("Unsupported performance poll interval; do not guess a submission window")
    active = [entry for entry in performance["stages"].values()
              if "job" in entry and entry["phase"] != "verified"]
    if len(active) != 1 or active[0]["phase"] != "monitoring":
        raise ValueError("Require one actively monitored performance stage for DEBUG submission")
    entry = active[0]
    stage_id = relative_path(entry["stage_id"])
    if "/" in stage_id:
        raise ValueError("Unsupported performance stage ID")
    job = positive_job(entry["job"])
    path = Path(performance_root) / "stages" / stage_id / f"poll-{job}.log"
    before = path.stat()
    text = path.read_text()
    after = path.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise ValueError("Performance poll receipt is still being written")
    headers = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith('{"at":'):
            continue
        record = json.loads(line)
        if {"at", "ssh_exit", "debug_authorization_SHA", "monotonic_gap_s"} <= record.keys():
            headers.append((index, record))
    if not headers:
        raise ValueError("Missing installed performance poll receipt header")
    index, receipt_record = headers[-1]
    at = datetime.fromisoformat(receipt_record["at"])
    if (at.tzinfo is None or receipt_record["at"] != entry.get("last_poll_at")
            or receipt_record["ssh_exit"] != 0 or entry.get("last_poll_ssh_exit") != 0
            or receipt_record["debug_authorization_SHA"] != head):
        raise ValueError("Performance receipt/state/SSH/authorization binding differs")
    rows = [line.split("|") for line in lines[index + 1:] if line.startswith(str(job) + "|")]
    if len(rows) != 1 or len(rows[0]) < 7 or rows[0][1].split()[0].rstrip("+") not in ACTIVE.split("|"):
        raise ValueError("Performance receipt does not prove the bound allocation is active")
    started, completed = at.timestamp(), after.st_mtime
    remaining = started + 300 - now
    if (not started <= completed <= now or now - completed > PERFORMANCE_POLL_FRESH_S
            or remaining < SUBMIT_REGISTER_BOUND_S + PERFORMANCE_POLL_MARGIN_S):
        raise ValueError("Wait for a newly completed performance poll with sufficient 300s-cycle headroom: "
                         f"started={started}, mtime={completed}, now={now}, remaining={remaining}")
    return {"performance_job": job, "stage_id": stage_id, "receipt_path": str(path.resolve()),
            "receipt_sha256": hashlib.sha256(text.encode()).hexdigest(), "poll_at": receipt_record["at"],
            "receipt_mtime_ns": after.st_mtime_ns, "next_poll_epoch": started + 300,
            "completed_age_s": now - completed, "remaining_cycle_s": remaining,
            "max_submission_registration_s": SUBMIT_REGISTER_BOUND_S,
            "minimum_next_poll_margin_s": PERFORMANCE_POLL_MARGIN_S, "atomic_authorization": False}


def wait_for_performance_window(root, state):
    started = time.monotonic()
    while True:
        try:
            window = performance_poll_window(state["concurrent_performance_root"])
            save(root / "performance-pre-submit-poll-window.json", window)
            return window
        except (ValueError, FileNotFoundError, KeyError) as error:
            if time.monotonic() - started >= 360:
                raise RuntimeError("No fresh safe performance poll window within 360s; do not submit DEBUG") from error
            time.sleep(1)


def bounded_authorization(root, state):
    process = subprocess.Popen(["bash", "-c", authorization_chain(root, state)], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=max(0, state["submit_register_deadline_epoch"] - time.time()))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        save(root / "submit-register-chain.json", {"exit": process.returncode, "stdout": stdout,
             "stderr": stderr, "timed_out_after_s": SUBMIT_REGISTER_BOUND_S, "atomic_authorization": False})
        receipt_path = root / "submission-receipt.txt"
        if receipt_path.exists() and receipt_path.read_text().strip():
            try:
                job = positive_job(receipt_path.read_text().strip())
                cancel_owned_held(root, state, job)
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                save(root / "held-cleanup-reconciliation-required.json", {"error": str(error),
                     "performance_cancelled": False, "resubmission_allowed": False})
        else:
            save(root / "held-cleanup-reconciliation-required.json", {"submission_receipt_missing": True,
                 "remote_submission_ledger": state.get("submission_ledger"),
                 "performance_cancelled": False, "resubmission_allowed": False})
        raise RuntimeError("60s submit/registration bound exceeded; preserve intent and reconcile only owned held DEBUG")
    save(root / "submit-register-chain.json", {"exit": process.returncode, "stdout": stdout,
         "stderr": stderr, "max_submission_registration_s": SUBMIT_REGISTER_BOUND_S, "atomic_authorization": False})
    if process.returncode:
        raise RuntimeError("Held submit/registration/release failed; preserve ledgers and do not duplicate")


def submit_and_register(root, state):
    identity = read(root / "upload/provenance/performance-coordinator-identity.json")
    live = read(Path(state["concurrent_performance_root"]) / "local-monitor-identity.json")
    if identity != live or not identity.get("pid"):
        raise ValueError("Explicit performance coordinator identity changed")
    argv = identity.get("argv", [])
    if "--root" not in argv or argv[argv.index("--root") + 1] != state["concurrent_performance_root"]:
        raise ValueError("Performance process argv is not bound to the explicit root")
    os.kill(identity["pid"], 0)
    process = subprocess.run(["ps", "-ww", "-p", str(identity["pid"]), "-o", "command="], text=True, capture_output=True)
    if process.returncode or shlex.split(process.stdout.strip()) != argv:
        raise ValueError("Performance coordinator PID/actual argv differs")
    state["performance_poll_window"] = wait_for_performance_window(root, state)
    state["submit_register_deadline_epoch"] = time.time() + SUBMIT_REGISTER_BOUND_S
    state["phase"] = "submission_intent_do_not_duplicate"
    save(root / "campaign-state.json", state)
    bounded_authorization(root, state)
    job = positive_job((root / "submission-receipt.txt").read_text().strip())
    state.update({"job": job, "phase": "monitoring", "registered_before_release": True})
    save(root / "campaign-state.json", state)


def poll(root, state):
    job = positive_job(state["job"])
    output = ssh(root, "poll", f'''set -eu
owner="$(id -un)"
printf 'OWNER|%s\n' "$owner"
sacct -n -X -P -j {job} --format=JobIDRaw,User,JobName%128,Partition,State%64
squeue -h -j {job} -o '%i|%u|%j|%P|%T'
''')
    owners = [line.split("|", 1)[1] for line in output.splitlines() if line.startswith("OWNER|")]
    rows = [line.rstrip("|").split("|") for line in output.splitlines() if line.startswith(str(job) + "|")]
    if len(owners) != 1 or not rows:
        raise ValueError("Missing actual DEBUG accounting/owner")
    statuses = []
    for row in rows:
        row = [field.strip() for field in row]
        if len(row) != 5 or row[:4] != [str(job), owners[0], JOB_NAME, "gpu_a800"]:
            raise ValueError("Actual DEBUG job owner/name/partition differs")
        status = row[4].split()[0].rstrip("+")
        if status not in (ACTIVE + "|" + TERMINAL).split("|"):
            raise ValueError("Unsupported actual DEBUG state")
        statuses.append(status)
    status = next((value for value in statuses if value in ACTIVE.split("|")), statuses[0])
    state.update({"last_poll_at": datetime.now().astimezone().isoformat(), "last_poll_ssh_exit": 0,
                  "state_last_observed": status, "actual_accounting_queue_states": statuses})
    save(root / "campaign-state.json", state)
    return status


def archive_and_verify(root, state):
    scripts = root / "upload/runtime/experiments/dp_opt"
    job = state["job"]
    command = [sys.executable, str(scripts / "archive_trace_job.py"), str(job), "--archive", str(root)]
    if (root / f"live-trace-{job}").exists():
        command.append("--resume")
    result = subprocess.run(command, text=True, capture_output=True, timeout=3600)
    save(root / f"archive-attempt-{time.time_ns()}.json", {"exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    if result.returncode:
        state["archive_retry_count"] = state.get("archive_retry_count", 0) + 1
        save(root / "campaign-state.json", state)
        if state["archive_retry_count"] >= 12:
            raise RuntimeError("Twelve terminal SHA archive attempts failed; preserve evidence and reconcile, never resubmit")
        return False
    report = read(root / f"archive-{job}.json")
    if report["job"] != job or report["missing_job_logs"]:
        raise ValueError("Independent canonical SHA archive lacks both real job logs")
    for suffix in ("out", "err"):
        if digest(root / f"live-trace-{job}.{suffix}") != report["log_sha256"][suffix]:
            raise ValueError("Independently downloaded job log SHA differs")
    raw = root / f"live-trace-{job}"
    check_hashes(raw, manifest_hashes((root / f"raw-{job}.sha256").read_text()), exact=True)
    check_hashes(raw / "runtime", state["runtime_sha256"], exact=True)
    if report["state"] != "COMPLETED":
        save(root / "failed-terminal-diagnostic.json", {"job": job, "state": report["state"],
             "terminal_raw_and_logs_SHA_verified": True, "runtime_SHA_verified": True,
             "exit_status": (raw / "exit-status").read_text().strip(),
             "startup_log_present": (raw / "upstream-startup-tests.log").is_file(),
             "frozen_inputs_present": (raw / "frozen").is_dir(),
             "GPU_diagnostic_complete": False, "automatic_resubmission_authorized": False})
        raise RuntimeError("DEBUG allocation " + report["state"] + "; inspect preserved startup/compute logs")
    check_hashes(raw / "frozen", read(root / "upload/plan.json")["frozen_sha256"], exact=True)
    installed = verify_installed_provenance(raw, job)
    save(root / "independent-installed-upstream-verification.json", installed)
    if installed != read(raw / "compute-installed-upstream-verification.json"):
        raise ValueError("Compute and independent installed-source verification differ")
    output = root / "independent-upstream-first-layer-verification.json"
    environment = {**os.environ, "PYTHONPATH": str(scripts), "PYTHONDONTWRITEBYTECODE": "1"}
    for name in ("ENABLE_KVCACHED", "KVCACHED_AUTOPATCH", "DP_OPT_LIVE_PLACEMENT"):
        environment.pop(name, None)
    if not output.exists():
        result = subprocess.run([sys.executable, str(scripts / "upstream_first_layer_probe.py"), "--verify", str(raw),
                                 "--output", str(output)], env=environment, text=True, capture_output=True, timeout=600)
        save(root / "upstream-independent-verifier-process.json", {"exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError("Independent upstream verification failed; no resubmission")
    verified = read(output)
    if verified != read(raw / "compute-upstream-first-layer-verification.json"):
        raise ValueError("Compute and independent upstream verification differ")
    if report["state"] != "COMPLETED" or verified["all_in_scope_target_attributions_available"] is not True:
        raise RuntimeError("Allocation did not close complete in-scope attribution; preserve true terminal state")
    state.update({"phase": "completed_observation_not_benefit", "GPU_diagnostic_complete": True,
                  "all_trace_first_divergences_attributed": False, "quality_equivalence_proven": False})
    save(root / "campaign-state.json", state)
    return True


@contextmanager
def campaign_lock(root):
    with (root / "upstream-campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run(root, parent, performance_root, repo, prepare_only=False, once=False):
    root, parent, performance_root, repo = map(lambda path: Path(path).resolve(), (root, parent, performance_root, repo))
    root.mkdir(parents=True, exist_ok=True)
    with campaign_lock(root):
        state = read(root / "campaign-state.json") if (root / "campaign-state.json").exists() else prepare(root, parent, performance_root, repo)
        if (state["repo"], state["parent"], state["concurrent_performance_root"]) != (str(repo), str(parent), str(performance_root)):
            raise ValueError("Preserve explicit parent/performance/campaign identity")
        validate_local(root, state)
        if prepare_only:
            if state["phase"] != "prepared" or "job" in state:
                raise ValueError("Prepare-only is only an unsubmitted local checkpoint")
            return
        if state["GPU_diagnostic_complete"]:
            return
        if state["phase"].startswith("diagnosis_required") or state["phase"] == "submission_intent_do_not_duplicate":
            raise RuntimeError("Reconcile prior intent/failure; never submit a duplicate DEBUG job")
        try:
            if state["phase"] == "prepared":
                protected_sync(root, state)
            if state["phase"] == "staged":
                submit_and_register(root, state)
            previous = None
            while True:
                started = time.monotonic()
                try:
                    status = poll(root, state)
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    state.update({"last_poll_ssh_exit": "failed_or_timeout", "last_poll_error": str(error)})
                    save(root / "campaign-state.json", state)
                    status = None
                if previous is not None:
                    state["last_poll_monotonic_gap_s"] = started - previous
                    save(root / "campaign-state.json", state)
                previous = started
                if status in TERMINAL.split("|"):
                    if archive_and_verify(root, state):
                        return
                if once:
                    return
                time.sleep(max(0, 300 - (time.monotonic() - started)))
        except BaseException as error:
            state.update({"failure_phase": state["phase"], "phase": "diagnosis_required_upstream", "error": repr(error)})
            save(root / "campaign-state.json", state)
            raise


def make_launch(template):
    substitutions = {
        'repo="${KVCACHED_REPO_ROOT:-$HOME/kvcached/kvcached}"': 'repo="${DP_OPT_UPSTREAM_RUNTIME:?Supply immutable runtime}"',
        '"$repo/experiments/dp_opt/live_trace.py"': '"$evidence/upstream_first_layer_entry.py"',
        '--fit "$root/state/sparse-fit-231467.json"': '--fit "$DP_OPT_UPSTREAM_PLACEMENT_FIT"',
        '"$root/state/sparse-fit-231467.json" "${DP_OPT_TRACE_PATH:?Set DP_OPT_TRACE_PATH}"':
            '"$DP_OPT_UPSTREAM_PLACEMENT_FIT" "${DP_OPT_TRACE_PATH:?Set DP_OPT_TRACE_PATH}"',
    }
    for original, replacement in substitutions.items():
        if template.count(original) != 1:
            raise ValueError("Installed frozen shell entry contract differs: " + original)
        template = template.replace(original, replacement)
    return template.replace("python -m pytest -q", "python -m pytest -q -p no:cacheprovider")


def require_upstream_compute():
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("DP_OPT_UPSTREAM_COMPUTE_CONFIRMED") != "1":
        raise RuntimeError("All runtime imports/model startup/capture must run on the allocated compute node")
    return positive_job(os.environ["SLURM_JOB_ID"])


def qualified(symbol):
    return symbol.__module__ + "." + symbol.__qualname__


def installed_source(symbol, directory, root):
    symbol = inspect.unwrap(symbol)
    source_name = inspect.getsourcefile(symbol)
    if source_name is None:
        raise RuntimeError("Installed Python source file unavailable: " + qualified(symbol))
    source = Path(source_name).resolve(strict=True)
    content = source.read_bytes()
    checksum = hashlib.sha256(content).hexdigest()
    target = directory / (checksum + "-" + source.name)
    if target.exists():
        if digest(target) != checksum:
            raise RuntimeError("Installed source copy changed during collection")
    else:
        with target.open("xb") as output:
            output.write(content)
    if digest(source) != checksum:
        raise RuntimeError("Installed source changed during collection")
    result = {"qualified_name": qualified(symbol), "installed_path": str(source),
              "file_sha256": checksum, "copied_path": str(target.relative_to(root)),
              "symbol_source_sha256": None, "source_first_line": None, "source_line_count": None}
    try:
        lines, first = inspect.getsourcelines(symbol)
    except (OSError, TypeError):
        return result
    result.update({"symbol_source_sha256": hashlib.sha256("".join(lines).encode()).hexdigest(),
                   "source_first_line": first, "source_line_count": len(lines)})
    return result


def collect_installed_provenance(worker, graph_type=None):
    job = require_upstream_compute()
    if graph_type is None:
        try:
            from vllm.compilation.cuda_graph import CUDAGraphWrapper
        except ImportError as error:
            raise RuntimeError("Installed CUDAGraphWrapper source unavailable; quality unproven") from error

        graph_type = CUDAGraphWrapper
    root = Path(os.environ["DP_OPT_EVIDENCE_ROOT"])
    service = os.environ["DP_OPT_SERVICE"]
    rank = worker.parallel_config.data_parallel_rank
    if service not in ("instance-1", "instance-2") or type(rank) is not int or rank not in (0, 1):
        raise RuntimeError("Installed-source worker identity unsupported")
    identity = f"{service}-dp{rank}"
    runner = worker.model_runner
    try:
        model = runner.get_model()
        projection = model.model.layers[0].self_attn.qkv_proj
        wrapper = runner.model
        quant_method = projection.quant_method
        linear_mro = {base.__name__: base for base in type(projection).__mro__}
        quant_mro = {base.__name__: base for base in type(quant_method).__mro__}
        symbols = {"linear_class": type(projection), "linear_forward": projection.forward,
            "linear_base": linear_mro["LinearBase"], "column_parallel_linear": linear_mro["ColumnParallelLinear"],
            "quant_method_class": type(quant_method), "quant_method_base": quant_mro["LinearMethodBase"],
            "quant_method_apply": quant_method.apply, "cudagraph_wrapper": graph_type,
            "cudagraph_call": graph_type.__call__, "model_runner": type(runner),
            "model_forward": type(runner)._model_forward, "runner_model_class": type(wrapper),
            "model_class": type(model)}
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise RuntimeError("Unsupported installed linear/base/apply/runner.model mapping; quality unproven") from error
    if (not isinstance(wrapper, graph_type) or type(projection).__name__ != "QKVParallelLinear"
            or type(quant_method).__name__ != "UnquantizedLinearMethod"):
        raise RuntimeError("Unsupported installed QKV/quantization/CUDAGraphWrapper binding; quality unproven")
    worker_path = root / f"upstream-first-layer-worker-{identity}.json"
    metadata = read(worker_path)
    if (metadata["service"] != service or metadata["dp_rank"] != rank
            or metadata["model_class"] != qualified(type(model))
            or metadata["projection"]["weight_dtype"] != str(projection.weight.dtype)
            or metadata["projection"]["weight_shape"] != list(projection.weight.shape)):
        raise RuntimeError("Installed-source binding differs from actual first-layer worker")
    directory = root / "installed-upstream-sources" / identity
    directory.mkdir(parents=True)
    sources = {name: installed_source(symbol, directory, root) for name, symbol in symbols.items()}
    record = {name: metadata[name] for name in ("service", "dp_rank", "checkpoint", "model_class",
              "model_dtype", "attention_tp", "dp", "ep", "pp", "projection", "cudagraph_mode")}
    record.update({"schema": INSTALLED_SCHEMA, "slurm_job_id": job, "compute_hostname": os.uname().nodename,
        "worker_metadata_sha256": digest(worker_path), "runtime_worker_class": qualified(type(worker)),
        "runner_model_class": qualified(type(wrapper)), "runner_model_is_cudagraph_wrapper": True,
        "linear_mro": [qualified(base) for base in type(projection).__mro__],
        "quant_method_mro": [qualified(base) for base in type(quant_method).__mro__], "sources": sources,
        "specific_kernel_proven": False, "quality_equivalence_proven": False, "is_algorithm_benefit": False})
    path = root / f"installed-upstream-linear-graph-{identity}.json"
    with path.open("x") as output:
        output.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def verify_installed_provenance(root, job):
    root = Path(root)
    expected = {f"installed-upstream-linear-graph-{service}-dp{rank}.json": (service, rank)
                for service in ("instance-1", "instance-2") for rank in (0, 1)}
    paths = sorted(root.glob("installed-upstream-linear-graph-*.json"))
    if {path.name for path in paths} != set(expected):
        raise ValueError("Require installed linear/apply/graph sources from all four workers")
    stable = []
    for path in paths:
        record = read(path)
        service, rank = expected[path.name]
        worker_path = root / f"upstream-first-layer-worker-{service}-dp{rank}.json"
        worker = read(worker_path)
        if (record["schema"] != INSTALLED_SCHEMA or (record["service"], record["dp_rank"]) != (service, rank)
                or type(record["dp_rank"]) is not int or type(record["slurm_job_id"]) is not int
                or record["slurm_job_id"] != positive_job(job) or not record["compute_hostname"]
                or record["worker_metadata_sha256"] != digest(worker_path)
                or record["runner_model_is_cudagraph_wrapper"] is not True
                or any(record.get(name) is not False for name in
                       ("specific_kernel_proven", "quality_equivalence_proven", "is_algorithm_benefit"))
                or any(record[name] != worker[name] for name in
                       ("service", "dp_rank", "checkpoint", "model_class", "model_dtype", "attention_tp",
                        "dp", "ep", "pp", "projection", "cudagraph_mode"))
                or record["model_dtype"] != "torch.bfloat16" or record["attention_tp"] != 1
                or (record["dp"], record["ep"], record["pp"]) != (2, 2, 1)
                or record["cudagraph_mode"] != "FULL_DECODE_ONLY" or set(record["sources"]) != INSTALLED_SYMBOLS):
            raise ValueError("Installed-source identity/model/weights/compute binding differs")
        normalized = {}
        for name, source in record["sources"].items():
            copied = root / relative_path(source["copied_path"])
            if (not source["copied_path"].startswith(f"installed-upstream-sources/{service}-dp{rank}/")
                    or copied.is_symlink() or digest(copied) != source["file_sha256"]):
                raise ValueError("Installed source file SHA or per-worker path differs")
            if source["symbol_source_sha256"] is not None:
                first, count = source["source_first_line"], source["source_line_count"]
                if type(first) is not int or type(count) is not int or first <= 0 or count <= 0:
                    raise ValueError("Installed symbol source line binding invalid")
                lines = copied.read_text().splitlines(keepends=True)
                if (first + count - 1 > len(lines) or hashlib.sha256(
                        "".join(lines[first - 1:first - 1 + count]).encode()).hexdigest() != source["symbol_source_sha256"]):
                    raise ValueError("Installed symbol source SHA differs")
            normalized[name] = {key: value for key, value in source.items() if key != "copied_path"}
        sources = record["sources"]
        if (sources["linear_class"]["qualified_name"] != record["linear_mro"][0]
                or sources["quant_method_class"]["qualified_name"] != record["quant_method_mro"][0]
                or not record["linear_mro"][0].endswith(".QKVParallelLinear")
                or not record["quant_method_mro"][0].endswith(".UnquantizedLinearMethod")
                or any(sources[name]["qualified_name"] not in record["linear_mro"]
                       for name in ("linear_base", "column_parallel_linear"))
                or sources["quant_method_base"]["qualified_name"] not in record["quant_method_mro"]
                or sources["runner_model_class"]["qualified_name"] != record["runner_model_class"]
                or sources["model_class"]["qualified_name"] != record["model_class"]
                or sources["linear_class"]["file_sha256"] != worker["source_sha256"]["qkv"]
                or sources["model_runner"]["file_sha256"] != worker["source_sha256"]["runner"]):
            raise ValueError("Installed linear/base/apply/graph class binding differs")
        stable.append({"checkpoint": record["checkpoint"], "model_class": record["model_class"],
            "projection": record["projection"], "runner_model_class": record["runner_model_class"],
            "linear_mro": record["linear_mro"], "quant_method_mro": record["quant_method_mro"], "sources": normalized})
    if any(record != stable[0] for record in stable):
        raise ValueError("Four workers have inconsistent installed linear/apply/graph sources or weights")
    return {"schema": INSTALLED_SCHEMA, "job": positive_job(job), "workers_verified": 4,
            "installed_sources_verified": True, "source_file_sha256": {
                name: source["file_sha256"] for name, source in stable[0]["sources"].items()},
            "specific_kernel_proven": False, "quality_equivalence_proven": False, "is_algorithm_benefit": False}


def _initialize_installed_worker(self, kv_cache_config):
    require_upstream_compute()
    result = super(globals()["InstalledProvenanceUpstreamWorker"], self).initialize_from_config(kv_cache_config)
    collect_installed_provenance(self)
    return result


def __getattr__(name):
    if name != "InstalledProvenanceUpstreamWorker":
        raise AttributeError(name)
    require_upstream_compute()
    from upstream_first_layer_worker import UpstreamFirstLayerVerificationWorker

    worker_type = type(name, (UpstreamFirstLayerVerificationWorker,), {"__module__": __name__,
        "__qualname__": name, "initialize_from_config": _initialize_installed_worker})
    globals()[name] = worker_type
    return worker_type


def compute_main():
    require_upstream_compute()
    import head_hidden_probe
    import live_trace
    import upstream_first_layer_probe

    original = head_hidden_probe.instrument

    @contextmanager
    def instrument():
        with original():
            previous = live_trace.engine_arguments

            def engine_arguments(*args, **kwargs):
                arguments = previous(*args, **kwargs)
                arguments.worker_cls = "upstream_first_layer_campaign.InstalledProvenanceUpstreamWorker"
                return arguments

            live_trace.engine_arguments = engine_arguments
            try:
                yield
            finally:
                live_trace.engine_arguments = previous

    head_hidden_probe.instrument = instrument
    try:
        head_hidden_probe.main()
    finally:
        head_hidden_probe.instrument = original
    root = Path(sys.argv[sys.argv.index("--output") + 1])
    for name in ("upstream_first_layer_worker.py", "upstream_first_layer_probe.py"):
        with (root / name).open("xb") as output:
            output.write(Path(__file__).with_name(name).read_bytes())
    save(root / "compute-upstream-first-layer-verification.json", upstream_first_layer_probe.analyze(root))
    save(root / "compute-installed-upstream-verification.json", verify_installed_provenance(root, require_upstream_compute()))


def compute(upload, evidence):
    require_upstream_compute()
    plan = read(upload / "plan.json")
    if plan["controls"] != CONTROLS:
        raise ValueError("Compute scientific controls changed")
    runtime = evidence / "runtime"
    check_hashes(runtime, read(upload / "runtime-manifest.json"), exact=True)
    shutil.copytree(upload / "frozen", evidence / "frozen")
    shutil.copytree(upload / "provenance", evidence / "provenance")
    shutil.copy2(upload / "placement-fit.json", evidence / "placement-fit.json")
    shutil.copy2(upload / "plan.json", evidence / "plan.json")
    check_hashes(evidence / "frozen", plan["frozen_sha256"], exact=True)
    if digest(evidence / "frozen/trace.json") != plan["trace_sha256"] or digest(evidence / "placement-fit.json") != plan["placement_fit_sha256"]:
        raise ValueError("Compute trace/fit differs from actual parent")
    with (evidence / "upstream_first_layer_entry.py").open("x") as output:
        output.write('from upstream_first_layer_campaign import compute_main\n\nif __name__ == "__main__":\n    compute_main()\n')
    with (evidence / "upstream-launch.sh").open("x") as output:
        output.write(make_launch((runtime / "experiments/dp_opt/trace-benefit.sbatch").read_text()))
    environment = {name: value for name, value in os.environ.items() if not name.startswith("DP_OPT_")}
    environment.pop("VLLM_BATCH_INVARIANT", None)
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTEST_ADDOPTS": "-p no:cacheprovider",
        "DP_OPT_UPSTREAM_RUNTIME": str(runtime),
        "DP_OPT_UPSTREAM_PLACEMENT_FIT": str(upload / "placement-fit.json"),
        "DP_OPT_UPSTREAM_COMPUTE_CONFIRMED": "1",
        "KVCACHED_REPO_ROOT": str(runtime), "DP_OPT_PHASE_EVIDENCE_ROOT": str(evidence),
        "DP_OPT_MAX_MODEL_LEN": "131072", "DP_OPT_KV_CACHE_BYTES": "25769803776", "KVCACHED_GPU_UTILIZATION": "0.98",
        "DP_OPT_TRACE_BUDGET_MIB": "10240", "DP_OPT_REQUEST_TIMEOUT_S": "7200", "DP_OPT_DEADLINE_S": "120",
        "DP_OPT_RELEASE_UNUSED_CACHE": "1", "DP_OPT_RANK_LOCAL_ADMISSION": "1", "DP_OPT_CONCURRENT_ADMISSION": "1",
        "DP_OPT_REVALIDATE_PREPARED": "1", "DP_OPT_PREPLAN": "1", "DP_OPT_PAUSE_SYNC_INTERVAL": "32",
        "DP_OPT_KV_TRANSPORT": "dp_echo", "DP_OPT_TRANSFER_CHUNK_MIB": "32",
        "DP_OPT_TRACE_REPEATS": "6", "DP_OPT_TRACE_POLICIES": "no_move", "DP_OPT_WORKAWARE_OPTIMIZER": "point",
        "DP_OPT_PROMPT_BANK": str(evidence / "frozen/inputs/prompt-bank.json"),
        "DP_OPT_TRACE_PATH": str(evidence / "frozen/trace.json")})
    save(evidence / "upstream-runtime-environment.json", {name: value for name, value in environment.items()
         if name.startswith(("DP_OPT_", "KVCACHED_", "VLLM_"))})
    result = subprocess.run(["bash", str(evidence / "upstream-launch.sh")], env=environment)
    if result.returncode:
        raise RuntimeError("Compute upstream launch failed; preserve original exit status")
    check_hashes(runtime, read(upload / "runtime-manifest.json"), exact=True)


def main():
    parser = argparse.ArgumentParser(description="Independent P149 upstream DEBUG; prepare-only never SSH")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--concurrent-performance-root", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Read-only parent/runtime SHA diff; no root creation or SSH")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--compute-upload", type=Path)
    parser.add_argument("--compute-evidence", type=Path)
    helpers = parser.add_mutually_exclusive_group()
    helpers.add_argument("--submit-held", action="store_true", help=argparse.SUPPRESS)
    helpers.add_argument("--release-registered-job", type=int, help=argparse.SUPPRESS)
    helpers.add_argument("--cancel-owned-held-job", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    helper = args.submit_held or args.release_registered_job is not None or args.cancel_owned_held_job is not None
    if args.compute_upload:
        if not args.compute_evidence or args.root or args.parent or args.concurrent_performance_root or helper or args.prepare_only or args.validate_only or args.once:
            parser.error("Compute requires only --compute-upload and --compute-evidence")
        compute(args.compute_upload.resolve(), args.compute_evidence.resolve())
    elif helper:
        if not args.root or args.parent or args.concurrent_performance_root or args.prepare_only or args.validate_only or args.once or args.compute_evidence:
            parser.error("Internal allocation helper requires only its explicit root")
        root = args.root.resolve()
        state = read(root / "campaign-state.json")
        if state["phase"] != "submission_intent_do_not_duplicate":
            parser.error("Allocation helper requires existing immutable submission intent")
        if args.submit_held:
            print(submit_held(root, state))
        elif args.release_registered_job is not None:
            release_registered(root, state, positive_job(args.release_registered_job))
        else:
            cancel_owned_held(root, state, positive_job(args.cancel_owned_held_job))
    elif args.validate_only:
        if args.root or not args.parent or not args.concurrent_performance_root or args.prepare_only or args.once or args.compute_evidence:
            parser.error("Read-only validation requires only --parent and --concurrent-performance-root")
        print(json.dumps(validate_source_epochs(args.parent, args.concurrent_performance_root,
                         Path(__file__).resolve().parents[2]), indent=2))
    else:
        if not args.root or not args.parent or not args.concurrent_performance_root or args.compute_evidence:
            parser.error("Supply --root, --parent and explicit --concurrent-performance-root")
        run(args.root, args.parent, args.concurrent_performance_root, Path(__file__).resolve().parents[2], args.prepare_only, args.once)


if __name__ == "__main__":
    main()
