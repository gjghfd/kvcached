"""Opt-in followup; preparing never contacts the cluster.

Use --prepare-only to freeze identity/runtime, --once for one checkpoint or
allocation, and --native-only to stop at the all-six-case CPU rate freeze.
CANCELLED rescue needs --allow-checkpoint-cancelled. Explicit DEBUG rotation
uses --register-debug-job/--register-debug-name against the prepared root;
it verifies remote accounting before appending authorization. The original local
coordinator must exit; this driver never stops it. Longer policy walltimes
must be declared on the first invocation and remain immutable on resume.
"""

import argparse
import ast
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import rebalance_campaign
import stress_campaign
from amortized_campaign import check_frozen_sources
from archive_trace_job import verify_raw
from performance_numerics import digest, verify
from performance_public import make_trace_launch
from performance_followup_analysis import (
    COST_SCOPE, analyze_calibration, check_hashes, file_hashes, finalize_native_workspace,
    freeze_evaluation, join_phase, merge_native_case, native_names, poisson, prepare_workspace,
    read_json, search, split_public_analysis, validate_inputs, write_json,
)
from verify_workaware_profile import same_payload, verify_profile


FOLLOWUP_FILES = ("experiments/dp_opt/performance_followup.py",
                  "experiments/dp_opt/performance_followup_analysis.py",
                  "experiments/dp_opt/performance-followup.sbatch",
                  "tests/test_dp_opt_performance_followup.py")
NEW_DEPLOYMENT_MODULES = ("experiments/dp_opt/public_poisson_followup.py",
                          "experiments/dp_opt/public_population_calibration.py",
                          "experiments/dp_opt/public_quality_followup.py")
SYNC_FILES = FOLLOWUP_FILES + NEW_DEPLOYMENT_MODULES
COPY_ONLY_FILES = ("kvcached/integration/vllm/patches.py",)
EMPTY_QUEUE_GUARD = 'test -z "$(squeue -h -u "$USER")"'
JOB_NAME = "qwen-public-perf-followup"
ACTIVE_STATES = "PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|RESIZING|REQUEUED|REQUEUE_FED|REQUEUE_HOLD|SIGNALING|STAGE_OUT"
TERMINAL_STATES = "COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED"
CONTROLS = {"partition": "gpu_a800", "gpus": 2, "walltime": "04:30:00", "monitor_interval_s": 300,
            "max_model_len": 131072, "cache_bytes": 25769803776, "kv_budget_mib": 10240,
            "gpu_memory_utilization": .98, "pause_sync_interval": 32, "dtype": "bfloat16",
            "head": "default", "sparse_graph": "default", "hidden_head_hooks": False,
            "publication_authorized": False, "quality_equivalence_not_proven": True}


def positive_job_id(value):
    if not re.fullmatch(r"[1-9][0-9]*", str(value)):
        raise ValueError("Require an exact positive integer job ID")
    return int(value)


def safe_stage_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
        raise ValueError("Unsafe followup stage identity")
    return value


def policy_walltime(value):
    match = re.fullmatch(r"([0-9]+):([0-5][0-9]):([0-5][0-9])", str(value))
    if match is None:
        raise ValueError("Supply explicit single-policy walltime as HH:MM:SS")
    hours, minutes, seconds = map(int, match.groups())
    if hours * 3600 + minutes * 60 + seconds < 16200:
        raise ValueError("Do not shrink the default 04:30:00 allocation budget")
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def queue_guard(debug_job=None, debug_name="qwen-head-hidden-probe", performance_job=None, authorized_debug=None):
    if not re.fullmatch(r"qwen-[a-zA-Z0-9_.-]+", debug_name):
        raise ValueError("Require the explicitly verified debug job name")
    debug = authorized_debug if authorized_debug is not None else (
        [{"job": positive_job_id(debug_job), "name": debug_name}] if debug_job is not None else [])
    jobs = []
    for identity in debug:
        if not re.fullmatch(r"qwen-[a-zA-Z0-9_.-]+", identity["name"]):
            raise ValueError("Require each operator-authorized DEBUG job's exact name")
        jobs.append((positive_job_id(identity["job"]), identity["name"], identity.get("owner", "")))
    if performance_job is not None:
        performance_job = positive_job_id(performance_job)
        jobs.append((performance_job, JOB_NAME, ""))
    if len({job for job, name, owner in jobs}) != len(jobs):
        raise ValueError("DEBUG and performance must be separate allocations")
    mapping = ";".join(f"{job}:{name}" for job, name, owner in jobs)
    script = ["pf_guard_fail() { printf '%s\\n' 'PERFORMANCE_FOLLOWUP_GUARD_REJECTED' >&2; exit 73; }",
              'pf_owner="$(id -un)"', 'test "$pf_owner" = "$USER" || pf_guard_fail',
              'pf_queue="$(squeue -h -u "$pf_owner" -o \'%i|%u|%j|%P|%T\')" || pf_guard_fail',
              f"printf '%s\\n' \"$pf_queue\" | awk -F'|' -v owner=\"$pf_owner\" -v mapping={shlex.quote(mapping)} -v performance_job={shlex.quote(str(performance_job or ''))} '",
              'BEGIN { split(mapping, pairs, ";"); for (pair_index in pairs) { split(pairs[pair_index], pair, ":"); names[pair[1]]=pair[2] } }',
              '/^[[:space:]]*$/ { next }',
              '{ for (field=1; field<=NF; field++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)',
              f'  if (NF != 5 || !($1 in names) || seen[$1]++ || $2 != owner || $3 != names[$1] || $4 != "gpu_a800" || $5 !~ /^({ACTIVE_STATES})$/) invalid=1 }}',
              '  $1 != performance_job { debug_active++ }',
              "END { exit (invalid || debug_active > 1) }' || pf_guard_fail", 'pf_debug_active=0']
    for job, name, bound_owner in jobs:
        script.extend([
            f'pf_account="$(sacct -n -X -P -j {job} --format=JobIDRaw,User,JobName%128,Partition,State%64)" || pf_guard_fail',
            f"pf_active=\"$(printf '%s\\n' \"$pf_account\" | awk -F'|' -v owner=\"$pf_owner\" -v bound_owner={shlex.quote(bound_owner)} -v job={job} -v name={shlex.quote(name)} '",
            '/^[[:space:]]*$/ { next }',
            '{ sub(/\\|$/, "", $0); for (field=1; field<=NF; field++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)',
            '  sub(/[ +].*$/, "", $5); count++',
            f'  if (NF != 5 || $1 != job || $2 != owner || (bound_owner != "" && $2 != bound_owner) || $3 != name || $4 != "gpu_a800" || $5 !~ /^({ACTIVE_STATES}|{TERMINAL_STATES})$/) invalid=1',
            f'  if ($5 ~ /^({ACTIVE_STATES})$/) active=1 }}',
            "END { if (invalid || count != 1) exit 1; print (active + 0) }')\" || pf_guard_fail"])
        if job != performance_job:
            script.extend(['pf_debug_active=$((pf_debug_active + pf_active))', 'test "$pf_debug_active" -le 1 || pf_guard_fail'])
    return "\n".join(script)


def initialize_debug_registry(root, job, name):
    directory = root / "authorized-debug"
    if directory.is_symlink():
        raise ValueError("DEBUG registry cannot be a symbolic link")
    directory.mkdir(mode=0o700, exist_ok=True)
    initial = directory / "00000000.json"
    write_json(initial, {"campaign_root": str(root), "sequence": 0, "previous_sha256": None,
        "identities": [{"job": job, "name": name}] if job is not None else [],
        "authorization": "initial_explicit_identity_owner_name_partition_verified_by_every_remote_guard"})
    return {"path": str(directory), "initial_sha256": digest(initial)}


def read_debug_registry(state):
    binding = state["debug_registry"]
    directory = Path(binding["path"])
    if directory.is_symlink():
        raise ValueError("DEBUG registry cannot be a symbolic link")
    files = sorted(directory.glob("*.json"))
    identities, previous = [], None
    observed = state.get("debug_registry_observed")
    for sequence, path in enumerate(files):
        if path.name != f"{sequence:08d}.json" or path.is_symlink():
            raise ValueError("Preserve the contiguous append-only DEBUG authorization ledger")
        record = read_json(path)
        if (record["sequence"] != sequence or record["previous_sha256"] != previous
                or record["campaign_root"] != str(directory.parent)):
            raise ValueError("DEBUG authorization SHA chain or bound campaign root changed")
        previous = digest(path)
        if sequence == 0 and previous != binding["initial_sha256"]:
            raise ValueError("Initial DEBUG authorization SHA changed")
        if sequence and (record.get("operator_explicit_registration") is not True
                         or record.get("remote_verified") is not True):
            raise ValueError("New DEBUG IDs require explicit independently verified operator registration")
        if observed and sequence == observed["sequence"] and previous != observed["sha256"]:
            raise ValueError("Previously observed DEBUG authorization SHA changed")
        identities.extend(record["identities"])
    if not files or observed and len(files) - 1 < observed["sequence"]:
        raise ValueError("DEBUG authorization ledger removed or rolled back")
    if len({identity["job"] for identity in identities}) != len(identities):
        raise ValueError("A DEBUG identity can only be registered once")
    head = {"sequence": len(files) - 1, "sha256": previous}
    return identities, head


def campaign_queue_guard(state, performance_job=None):
    if "debug_registry" not in state:
        return queue_guard(state["debug_job"], state["debug_name"], performance_job)
    identities, head = read_debug_registry(state)
    state["debug_registry_observed"] = head
    return queue_guard(performance_job=performance_job, authorized_debug=identities)


def register_debug(root, job, name):
    root = Path(root).resolve()
    job = positive_job_id(job)
    if not re.fullmatch(r"qwen-[a-zA-Z0-9_.-]+", name):
        raise ValueError("Operator must supply the new DEBUG job's exact name")
    state = read_json(root / "campaign-state.json")
    if "debug_registry" not in state or Path(state["debug_registry"]["path"]) != root / "authorized-debug":
        raise ValueError("Prepare this driver with its bound DEBUG registry before registration")
    check_frozen_sources(Path(state["repo"]), state)
    with (root / "authorized-debug/.registration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        identities, head = read_debug_registry(state)
        previous = next((identity for identity in identities if identity["job"] == job), None)
        if previous:
            if previous["name"] != name:
                raise ValueError("The registered DEBUG ID cannot change its exact name")
            return head
        active = [entry["job"] for entry in state["stages"].values()
                  if "job" in entry and entry["phase"] != "verified"]
        if len(active) > 1:
            raise ValueError("Reconcile unclosed performance allocations before DEBUG registration")
        guard = queue_guard(performance_job=active[0] if active else None,
                            authorized_debug=[*identities, {"job": job, "name": name}])
        output = rebalance_campaign.checked_ssh(guard + '\nprintf "DEBUG_REGISTRATION_OWNER|%s\\n" "$pf_owner"\n'
            + f"sacct -n -X -P -j {job} --format=JobIDRaw,User,JobName%128,Partition,State%64")
        owners = [line.split("|", 1)[1].strip() for line in output.splitlines() if line.startswith("DEBUG_REGISTRATION_OWNER|")]
        rows = [[field.strip() for field in line.rstrip("|").split("|")]
                for line in output.splitlines() if line.startswith(str(job) + "|")]
        if (len(owners) != 1 or len(rows) != 1 or len(rows[0]) != 5
                or rows[0][:4] != [str(job), owners[0], name, "gpu_a800"]
                or rows[0][4].split()[0].rstrip("+") not in (ACTIVE_STATES + "|" + TERMINAL_STATES).split("|")):
            raise ValueError("Independent DEBUG registration accounting does not match owner/name/A800/ID")
        sequence = head["sequence"] + 1
        path = root / "authorized-debug" / f"{sequence:08d}.json"
        write_json(path, {"campaign_root": str(root), "sequence": sequence, "previous_sha256": head["sha256"],
            "identities": [{"job": job, "name": name, "owner": owners[0]}],
            "operator_explicit_registration": True, "remote_verified": True,
            "verification_stdout": output, "authorized_at": datetime.now().astimezone().isoformat()})
        return {"sequence": sequence, "sha256": digest(path)}


@contextmanager
def sync_scope(debug_job, debug_name, state=None):
    original_string = rebalance_campaign.checked_ssh
    original_stdin = rebalance_campaign.checked_ssh_script

    def string(command, *args, **kwargs):
        guard = campaign_queue_guard(state) if state is not None else queue_guard(debug_job, debug_name)
        return original_string(command.replace(EMPTY_QUEUE_GUARD, guard), *args, **kwargs)

    def stdin(script, *args, **kwargs):
        guard = campaign_queue_guard(state) if state is not None else queue_guard(debug_job, debug_name)
        return original_stdin(script.replace(EMPTY_QUEUE_GUARD, guard), *args, **kwargs)

    rebalance_campaign.checked_ssh = string
    rebalance_campaign.checked_ssh_script = stdin
    try:
        yield
    finally:
        rebalance_campaign.checked_ssh = original_string
        rebalance_campaign.checked_ssh_script = original_stdin


@contextmanager
def campaign_lock(root):
    with (root / "performance-followup.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another followup driver owns this ledger") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def source_modules(path):
    modules = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            modules.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            function = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else None
            if function == "import_module" and isinstance(node.args[0].value, str):
                modules.append(node.args[0].value.split(".")[0])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith(".py"):
            modules.append(Path(node.value).stem)
    return modules


def deployment_files(state):
    for name in COPY_ONLY_FILES:
        if name in state["frozen_runtime_sha256"] and state["frozen_runtime_sha256"][name] != state["protected_vllm_sha256"]:
            raise ValueError("Copy-only native patch must retain its protected SHA")
    return tuple(sorted((set(SYNC_FILES) | (set(state["frozen_runtime_sha256"])
                        - set(state["native_runtime_sha256"]))) - set(COPY_ONLY_FILES)))


def runtime_manifest(repo, parent_manifest):
    names = set(parent_manifest) | set(SYNC_FILES) | set(COPY_ONLY_FILES)
    launch = repo / "experiments/dp_opt/trace-benefit.sbatch"
    text = launch.read_text()
    literal = set(re.findall(r'\$repo/([A-Za-z0-9_./-]+\.(?:py|sbatch))', text))
    if any(not (repo / name).is_file() for name in literal):
        raise ValueError("Canonical shell launch references an absent repository file")
    names.update(literal)
    names.update(str(path.relative_to(repo)) for path in (repo / "experiments/dp_opt").glob("live_*.py"))
    for filename in re.findall(r'\b([A-Za-z0-9_.-]+\.(?:py|sbatch))\b', text):
        for directory in ("experiments/dp_opt", "tests"):
            path = repo / directory / filename
            if path.is_file():
                names.add(str(path.relative_to(repo)))
    for name in ("pyproject.toml", "conftest.py", "tests/conftest.py"):
        if (repo / name).is_file():
            names.add(name)
    pending = ["performance_followup", "public_poisson_followup"]
    for name in names:
        if name.endswith(".py"):
            pending.extend(source_modules(repo / name))
    for name in tuple(names):
        if name.endswith(".py"):
            for node in ast.walk(ast.parse((repo / name).read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith(".json"):
                    relative = Path(node.value)
                    if (not relative.is_absolute() and ".." not in relative.parts
                            and (repo / relative).is_file()):
                        names.add(str(relative))
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        path = repo / "experiments/dp_opt" / (name + ".py")
        if not path.is_file():
            continue
        names.add(str(path.relative_to(repo)))
        pending.extend(source_modules(path))
    manifest = {name: digest(repo / name) for name in sorted(names)}
    if any(manifest[name] != expected for name, expected in parent_manifest.items()):
        raise ValueError("The native frozen runtime changed; never deploy its replacement")
    return manifest


def native_coordinator_identity(parent, explicit_pid=None):
    identity_path = parent / "local-monitor-identity.json"
    if identity_path.exists():
        identity = read_json(identity_path)
        argv = identity.get("argv", [])
        if ("--root" not in argv or Path(argv[argv.index("--root") + 1]).resolve() != parent
                or not any(Path(argument).name == "performance_campaign.py" for argument in argv)):
            raise ValueError("Local native coordinator identity is not bound to this performance root")
        pid = positive_job_id(identity["pid"])
        if explicit_pid is not None and positive_job_id(explicit_pid) != pid:
            raise ValueError("Explicit native coordinator PID differs from the archived identity")
        return {"pid": pid, "identity_sha256": digest(identity_path), "explicit_binding": explicit_pid is not None}
    if explicit_pid is None:
        raise ValueError("Supply --native-coordinator-pid or a bound local-monitor-identity.json")
    return {"pid": positive_job_id(explicit_pid), "identity_sha256": None, "explicit_binding": True}


def native_coordinator_stopped(parent, identity):
    result = subprocess.run(["ps", "-ww", "-p", str(identity["pid"]), "-o", "state=", "-o", "command="],
                            text=True, capture_output=True)
    if result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip():
        return {**identity, "stopped": True, "query": "local_ps_process_absent_no_signals_sent"}
    if result.returncode or not result.stdout.strip():
        raise RuntimeError("Cannot independently verify native coordinator stopped; no rescue submission")
    if result.stdout.lstrip().startswith("Z"):
        return {**identity, "stopped": True, "query": "local_ps_zombie_no_signals_sent"}
    if "performance_campaign.py" not in result.stdout or str(parent) not in result.stdout:
        return {**identity, "stopped": True, "query": "original_bound_process_exited_PID_reused_no_signals_sent"}
    return None


def wait_native_archive(root, parent, state, once=False):
    job = state["native_job"]
    proof_path = root / "native-coordinator-exit.json"
    while True:
        stress_campaign.check_pause(root)
        stress_campaign.check_pause(parent)
        previous = read_json(parent / "campaign-state.json")
        ready = (previous.get("job") == job and previous.get("state_last_observed") in TERMINAL_STATES.split("|")
                 and all((parent / name).exists() for name in
                         (f"live-trace-{job}", f"raw-{job}.sha256", f"accounting-{job}.txt")))
        if ready:
            proof = read_json(proof_path) if proof_path.exists() else native_coordinator_stopped(
                parent, state["native_coordinator_identity"])
            if proof:
                if proof.get("stopped") is not True or any(proof.get(key) != value
                        for key, value in state["native_coordinator_identity"].items()):
                    raise ValueError("Native coordinator stop evidence changed")
                write_json(proof_path, proof)
                return proof
        with (root / "wait-native-local.log").open("a") as output:
            output.write(json.dumps({"at": datetime.now().astimezone().isoformat(),
                "native_phase": previous.get("phase"), "canonical_archive_ready": ready,
                "waiting_archive_and_coordinator_exit_not_native_complete_flag": True}) + "\n")
        if once:
            return None
        time.sleep(300)


def initial_state(root, parent, repo, native_job, debug_job, debug_name, walltime,
                  allow_checkpoint_cancelled=False, coordinator_identity=None):
    previous = read_json(parent / "campaign-state.json")
    if previous.get("job") != native_job:
        raise ValueError("Bind to the existing native performance job, not a fresh coordinator")
    return {"repo": str(repo), "native_root": str(parent), "native_job": native_job,
            "debug_job": debug_job, "debug_name": debug_name, "controls": CONTROLS,
            "debug_registry": initialize_debug_registry(root, debug_job, debug_name),
            "phase": "waiting_existing_native_archive", "stages": {},
            "frozen_runtime_sha256": runtime_manifest(repo, previous["frozen_runtime_sha256"]),
            "native_runtime_sha256": previous["frozen_runtime_sha256"],
            "protected_vllm_sha256": previous["protected_vllm_sha256"],
            "performance_independent_of_debug": True, "research_goal_complete": False,
            "cost_scope": COST_SCOPE, "policy_walltime": walltime,
            "native_rescue": "one_original_four_policy_900s_case_per_allocation_no_partial_timings",
            "native_rescue_budget": {"four_policy_lower_bound_s": 3600, "estimated_with_setup_reference_s": 4800,
                                     "loading_and_full_request_drain_not_a_walltime_guarantee": True},
            "allow_checkpoint_cancelled": allow_checkpoint_cancelled,
            "native_coordinator_identity": coordinator_identity,
            "walltime_risk": "heldout_10800s_plus_loading_and_full_7200s_request_drain_can_exceed_04h30_no_trace_truncation"}


def plan_for(root, stage_id, phase_name, kind, policy, trace_path, walltime="04:30:00"):
    safe_stage_id(stage_id)
    safe_stage_id(phase_name)
    trace = read_json(trace_path)
    if kind not in ("native", "rate", "explore", "heldout") or kind != "native" and policy not in poisson.POLICIES:
        raise ValueError("Unrecognized predeclared stage or policy")
    duration = 900 if kind == "native" else 10800 if kind == "heldout" else 3600
    if trace["duration_s"] != duration or kind == "rate" and policy != "no_move":
        raise ValueError("One allocation must contain one complete original rate/policy trace")
    native = read_json(root / "native-parent-review.json") if kind == "native" else read_json(root / "native-provenance.json")
    if kind == "native":
        if (phase_name not in native_names() or stage_id != phase_name or phase_name in native["available_cases"]
                or trace_path != root / "workspace/frozen" / (phase_name + ".json")):
            raise ValueError("Only missing original frozen native cases can receive a rescue allocation")
        search.validate_frozen(root / "workspace/frozen")
    policies = list(search.POLICIES) if kind == "native" else [policy]
    return {"stage_id": stage_id, "phase_name": phase_name, "kind": kind,
            "policy": "native_four_policies" if kind == "native" else policy, "policies": policies,
            "trace_relative": str(trace_path.relative_to(root)), "trace_sha256": digest(trace_path),
            "duration_s": duration, "seed": trace["seed"],
            "controller_seed": native["native_controller_seed"] if kind == "native" else trace["seed"],
            "rates_rps": [.001, .012] if kind == "native" else trace["model_rates_rps"],
            "fit_sha256": digest(root / "workspace/public-context-fit.json"), "controls": CONTROLS,
            "placement_fit_sha256": digest(root / "workspace/placement-fit.json"),
            "movement_costs": native["movement_costs"],
            "optimizer": "robust" if "kv_only_workaware" in policies else "point",
            "request_timeout_s": 7200, "walltime": "04:30:00" if kind in ("native", "rate") else policy_walltime(walltime)}


def stage_inputs(root, state, entry):
    repo = Path(state["repo"])
    stage_id = safe_stage_id(entry["stage_id"])
    directory = root / "stages" / stage_id
    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / "frozen"
    frozen.mkdir(exist_ok=True)
    trace = root / entry["plan"]["trace_relative"]
    if digest(trace) != entry["plan"]["trace_sha256"]:
        raise ValueError("Stage trace changed before protected sync")
    if not (frozen / "trace.json").exists():
        shutil.copy2(trace, frozen / "trace.json")
    if digest(frozen / "trace.json") != digest(trace):
        raise ValueError("Keep the original immutable stage trace")
    write_json(directory / "stage.json", entry["plan"])
    write_json(directory / "runtime-manifest.json", state["frozen_runtime_sha256"])
    copies = {"frozen/inputs": file_hashes(root / "workspace/frozen/inputs"),
              "public-context-calibration": file_hashes(root / "workspace/public-context-calibration")}
    write_json(directory / "copy-manifest.json", copies)
    check_frozen_sources(repo, state)
    with sync_scope(state["debug_job"], state["debug_name"], state):
        remote, home = rebalance_campaign.protected_sync(
            repo, directory, state,
            files=deployment_files(state),
            prefix="performance-followup-" + stage_id,
            upload_files=("stage.json", "runtime-manifest.json", "copy-manifest.json"))
    parent = home + f"/run/kvcached-paracloud/state/live-trace-{state['native_job']}"
    upload = remote + "/upload"
    commands = ["set -eu", campaign_queue_guard(state)]
    transfers = []
    for group, hashes in copies.items():
        transfers.extend((parent + "/" + group + "/" + name, upload + "/" + group + "/" + name, expected)
                         for name, expected in hashes.items())
    transfers.append((parent + "/public-context-fit.json", upload + "/public-context-fit.json", entry["plan"]["fit_sha256"]))
    placement_source = read_json(root / "native-parent-review.json")["placement_fit_source_phase"]
    transfers.append((parent + "/" + placement_source + "/placement-fit.json", upload + "/placement-fit.json",
                      entry["plan"]["placement_fit_sha256"]))
    transfers.extend((home + "/kvcached/kvcached/" + name, upload + "/runtime/" + name, expected)
                     for name, expected in state["frozen_runtime_sha256"].items())
    for source, target, expected in transfers:
        target_parent = str(Path(target).parent)
        source, target = shlex.quote(source), shlex.quote(target)
        commands.extend([f'test "$(sha256sum {source} | cut -d\' \' -f1)" = {expected}',
                         "mkdir -p " + shlex.quote(target_parent),
                         f"cp -p {source} {target}", f'test "$(sha256sum {target} | cut -d\' \' -f1)" = {expected}'])
    commands.append(campaign_queue_guard(state))
    rebalance_campaign.checked_ssh_script("\n".join(commands))
    entry.update({"remote_stage": remote, "remote_home": home, "phase": "staged",
                  "submission_ledger": remote + "/submitted-performance-followup-job.txt"})
    stress_campaign.save_state(root, state)


def submission_command(entry, state, recover=False):
    ledger = shlex.quote(entry["submission_ledger"])
    if recover:
        return f"set -eu; test -s {ledger}; cat {ledger}"
    upload = entry["remote_stage"] + "/upload"
    intent = shlex.quote(entry["submission_ledger"] + ".intent")
    return ('set -eu\npf_submit_lock="$HOME/run/kvcached-paracloud/performance-followup-submit.lock"\n'
            + 'mkdir -p "$(dirname "$pf_submit_lock")"\nmkdir "$pf_submit_lock"\n'
            + 'trap \'rmdir "$pf_submit_lock"\' EXIT\n'
            + campaign_queue_guard(state) + "\n"
            + f"if test -s {ledger}; then cat {ledger}; exit 0; fi\n"
            + f"mkdir {intent}\n"
            + f"export DP_OPT_PERFORMANCE_FOLLOWUP_UPLOAD={shlex.quote(upload)}\n"
            + 'sbatch --parsable --partition=gpu_a800 --nodes=1 --gpus=2 '
            + f"--time={policy_walltime(entry['plan']['walltime'])} --export=ALL "
            + f"--job-name={JOB_NAME} "
            + '--output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" '
            + '--error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" '
            + f"{shlex.quote(upload + '/runtime/experiments/dp_opt/performance-followup.sbatch')} > {ledger}.tmp\n"
            + f"mv {ledger}.tmp {ledger}\ncat {ledger}")


def reconcile_submission(root, state, entry):
    if "job" in entry:
        return entry["job"]
    recover = entry["phase"] == "submission_intent"
    if entry["phase"] not in ("staged", "submission_intent"):
        raise RuntimeError("Reconcile the recorded stage instead of requesting another allocation")
    entry["phase"] = "submission_intent"
    stress_campaign.save_state(root, state)
    output = rebalance_campaign.checked_ssh(submission_command(entry, state, recover))
    job = positive_job_id(output.strip().split(";")[0])
    entry.update({"job": job, "phase": "monitoring"})
    write_json(root / "stages" / entry["stage_id"] / "submission.json", {"job": job, "raw_response": output})
    stress_campaign.save_state(root, state)
    return job


def ensure_remote_watcher(root, state, entry):
    job = positive_job_id(entry["job"])
    remote = entry["remote_home"] + f"/run/kvcached-paracloud/state/live-trace-{job}"
    script = entry["remote_stage"] + "/upload/runtime/experiments/dp_opt/watch-trace-remote.sh"
    expected = state["frozen_runtime_sha256"]["experiments/dp_opt/watch-trace-remote.sh"]
    command = ("set -eu\n" + campaign_queue_guard(state, job) + "\n"
        + f"pf_root={shlex.quote(remote)}\nmkdir -p \"$pf_root\"\n"
        + 'for marker in remote-query.expected remote-query.started remote-query.exit; do '
        + 'test ! -L "$pf_root/$marker"; done\n'
        + 'if test ! -f "$pf_root/remote-query.started"; then\n'
        + '  if test ! -d "$pf_root/.performance-followup-watcher-intent"; then\n'
        + '    mkdir "$pf_root/.performance-followup-watcher-intent"\n'
        + f'    test "$(sha256sum {shlex.quote(script)} | cut -d\' \' -f1)" = {expected}\n'
        + f'    printf "job=%s\\n" {job} > "$pf_root/remote-query.expected"\n'
        + f'    nohup env DP_OPT_REMOTE_MAX_POLLS=1201 bash {shlex.quote(script)} {job} '
        + '>> "$pf_root/remote-query.log" 2>&1 < /dev/null &\n'
        + '  fi\n'
        + '  for attempt in $(seq 1 20); do test ! -f "$pf_root/remote-query.started" || break; sleep 1; done\n'
        + 'fi\ntest -f "$pf_root/remote-query.started"\ncat "$pf_root/remote-query.started"')
    output = rebalance_campaign.checked_ssh(command)
    fields = dict(line.split("=", 1) for line in output.splitlines())
    if (set(fields) != {"job", "hostname", "time", "pid", "run_id", "exit_status"}
            or positive_job_id(fields["job"]) != job or not fields["hostname"] or not fields["run_id"]
            or fields["exit_status"] != "running"):
        raise ValueError("Require the immutable query-only watcher's exact job/run identity")
    entry["remote_watcher_started"] = {**fields, "pid": positive_job_id(fields["pid"])}
    stress_campaign.save_state(root, state)


def poll_stage(root, state, entry):
    job = entry["job"]
    query = ("\ndate -Is\n" + f"sacct -n -X -P -j {job} --format=JobIDRaw,State,ExitCode,Start,End,Elapsed,NodeList\n"
               + f'pf_root="$HOME/run/kvcached-paracloud/state/live-trace-{job}"\n'
               + 'for name in phase.json progress.json error.json exit-status; do '
               + 'if test -f "$pf_root/$name"; then printf "\\n%s\\n" "$name"; cat "$pf_root/$name"; fi; done')
    previous = None
    while True:
        stress_campaign.check_pause(root)
        if state.get("native_root"):
            stress_campaign.check_pause(Path(state["native_root"]))
        started = time.monotonic()
        command = "set -eu\n" + campaign_queue_guard(state, job) + query
        authorization = state.get("debug_registry_observed")
        polled_at = datetime.now().astimezone()
        stamp = polled_at.isoformat()
        wall_gap = ((polled_at - datetime.fromisoformat(entry["last_poll_at"])).total_seconds()
                    if entry.get("last_poll_at") else None)
        try:
            result = stress_campaign.ssh(command)
            code, stdout, stderr = result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            code, stdout, stderr = 124, "", "SSH timeout; retain this job and retry at 300s"
        with (root / "stages" / entry["stage_id"] / f"poll-{job}.log").open("a") as output:
            output.write(json.dumps({"at": stamp, "ssh_exit": code,
                                     "debug_authorization_SHA": authorization,
                                     "wall_clock_gap_s": wall_gap,
                                     "monotonic_gap_s": started - previous if previous is not None else None})
                         + "\n" + stdout + stderr + "\n")
        previous = started
        entry.update({"last_poll_at": stamp, "last_poll_ssh_exit": code})
        stress_campaign.save_state(root, state)
        if code == 73 or "PERFORMANCE_FOLLOWUP_GUARD_REJECTED" in stderr:
            if "debug_registry" not in state or read_debug_registry(state)[1] == authorization:
                raise RuntimeError("Owner/name/queue guard rejected; preserve this job, never launch a third")
            time.sleep(max(0, 300 - (time.monotonic() - started)))
            continue
        if code == 0:
            rows = [line.split("|") for line in stdout.splitlines() if line.startswith(str(job) + "|")]
            if len(rows) == 1 and len(rows[0]) >= 7:
                terminal = rows[0][1].split()[0].rstrip("+")
                if terminal in TERMINAL_STATES.split("|"):
                    entry.update({"terminal": terminal, "accounting_exit_code": rows[0][2], "phase": "archiving"})
                    stress_campaign.save_state(root, state)
                    return terminal
        time.sleep(max(0, 300 - (time.monotonic() - started)))


def verify_stage(root, entry):
    stage_id = safe_stage_id(entry["stage_id"])
    archive = root / "stages" / stage_id
    if entry.get("terminal") != "COMPLETED" or entry.get("accounting_exit_code") != "0:0":
        raise ValueError("Failed or partial allocations are not reclassified as success")
    raw = archive / f"live-trace-{positive_job_id(entry['job'])}"
    verify_raw(raw, (archive / f"raw-{entry['job']}.sha256").read_text())
    if (raw / "exit-status").read_text().strip() != "0" or read_json(raw / "stage.json") != entry["plan"]:
        raise ValueError("Raw terminal status or stage/request binding differs")
    runtime = read_json(archive / "runtime-manifest.json")
    state = read_json(root / "campaign-state.json")
    if runtime != state["frozen_runtime_sha256"] or read_json(raw / "runtime-manifest.json") != runtime:
        raise ValueError("Executed frozen runtime manifest changed")
    check_hashes(raw / "runtime", runtime)
    copies = read_json(archive / "copy-manifest.json")
    expected_copies = {"frozen/inputs": file_hashes(root / "workspace/frozen/inputs"),
                       "public-context-calibration": file_hashes(root / "workspace/public-context-calibration")}
    if copies != expected_copies or read_json(raw / "copy-manifest.json") != copies:
        raise ValueError("Immutable copy provenance changed")
    for group, hashes in copies.items():
        if file_hashes(raw / group) != hashes:
            raise ValueError("Copied inputs or calibration source differ")
    validate_inputs(raw / "frozen/inputs")
    plan = entry["plan"]
    actual = raw / plan["phase_name"]
    config = read_json(actual / "config.json")
    if (digest(actual / "trace.json") != plan["trace_sha256"] or config["policies"] != plan["policies"]
            or config["repeats"] != 1 or config["max_model_len"] != 131072
            or config["cache_bytes"] != 25769803776 or config["kv_budget_mib"] != 10240
            or config["seed"] != plan["controller_seed"] or config["request_timeout_s"] != 7200
            or config.get("workaware_rate_priors") != plan["rates_rps"]
            or config["allocator_gpu_utilization"] != .98 or config["fairness_queue_threshold_s"] != 30
            or config.get("workaware_optimizer") != plan["optimizer"]
            or any(config[field] != value for field, value in plan["movement_costs"].items())
            or digest(actual / "placement-fit.json") != plan["placement_fit_sha256"]
            or config.get("workaware_fit_sha256") != plan["fit_sha256"]):
        raise ValueError("A shard changed the workload, model, budget, policy or shared costs")
    if (digest(raw / "public-context-fit.json") != plan["fit_sha256"]
            or digest(raw / "placement-fit.json") != plan["placement_fit_sha256"]):
        raise ValueError("Shard cost fit SHA changed")
    fit = verify_profile(raw / "public-context-calibration")
    if not same_payload(fit, read_json(raw / "public-context-fit.json")):
        raise ValueError("Shard cost sources do not reconstruct independently")
    return verify(actual, raw / "frozen/inputs/prompt-bank.json")


def execute_stage(root, state, plan):
    stage_id = plan["stage_id"]
    entry = state["stages"].setdefault(stage_id, {"stage_id": stage_id, "plan": plan, "phase": "prepared"})
    if entry["plan"] != plan:
        raise ValueError("The resumed stage cannot change its trace, rate, seed or setting")
    stress_campaign.save_state(root, state)
    if entry["phase"] == "verified":
        return entry, verify_stage(root, entry)
    if entry["phase"] == "diagnosis_required":
        raise RuntimeError("Keep the failed allocation; explicit diagnosis is required, not a retry")
    other = [name for name, item in state["stages"].items()
             if name != stage_id and "job" in item and item["phase"] != "verified"]
    if other:
        raise RuntimeError("Another performance allocation remains unclosed in the ledger")
    if entry["phase"] == "prepared":
        stage_inputs(root, state, entry)
    reconcile_submission(root, state, entry)
    if "terminal" not in entry:
        ensure_remote_watcher(root, state, entry)
        poll_stage(root, state, entry)
    archive = root / "stages" / stage_id
    stress_campaign.archive_job(archive, entry["job"], Path(state["repo"]) / "experiments/dp_opt", entry)
    verify_raw(archive / f"live-trace-{entry['job']}", (archive / f"raw-{entry['job']}.sha256").read_text())
    stress_campaign.save_state(root, state)
    if entry["terminal"] != "COMPLETED":
        entry["phase"] = "diagnosis_required"
        stress_campaign.save_state(root, state)
        raise RuntimeError("Archive retained; do not use incomplete work or resubmit a failed allocation")
    try:
        result = verify_stage(root, entry)
    except Exception as error:
        entry.update({"phase": "diagnosis_required", "independent_verification_error": repr(error)})
        stress_campaign.save_state(root, state)
        raise
    write_json(archive / "independent-performance-verification.json", result)
    entry["phase"] = "verified"
    stress_campaign.save_state(root, state)
    return entry, result


def compute(upload, evidence):
    plan = read_json(upload / "stage.json")
    if plan["controls"] != CONTROLS:
        raise ValueError("The immutable performance controls changed")
    runtime = read_json(upload / "runtime-manifest.json")
    check_hashes(evidence / "runtime", runtime)
    for name in ("stage.json", "runtime-manifest.json", "copy-manifest.json", "public-context-fit.json", "placement-fit.json"):
        shutil.copy2(upload / name, evidence / name)
    shutil.copytree(upload / "frozen", evidence / "frozen")
    shutil.copytree(upload / "public-context-calibration", evidence / "public-context-calibration")
    for group, hashes in read_json(evidence / "copy-manifest.json").items():
        if file_hashes(evidence / group) != hashes:
            raise ValueError("Compute-side immutable inputs/cost copy differs")
    validate_inputs(evidence / "frozen/inputs")
    if (digest(evidence / "frozen/trace.json") != plan["trace_sha256"]
            or digest(evidence / "public-context-fit.json") != plan["fit_sha256"]
            or digest(evidence / "placement-fit.json") != plan["placement_fit_sha256"]
            or not same_payload(verify_profile(evidence / "public-context-calibration"),
                                read_json(evidence / "public-context-fit.json"))):
        raise ValueError("Compute-side trace or verified default-head costs changed")
    environment = {name: value for name, value in os.environ.items() if not name.startswith("DP_OPT_")}
    environment.pop("VLLM_BATCH_INVARIANT", None)
    environment.update({"KVCACHED_REPO_ROOT": str(evidence / "runtime"), "KVCACHED_GPU_UTILIZATION": "0.98",
        "DP_OPT_MAX_MODEL_LEN": "131072", "DP_OPT_KV_CACHE_BYTES": "25769803776",
        "DP_OPT_TRACE_BUDGET_MIB": "10240", "DP_OPT_REQUEST_TIMEOUT_S": str(plan["request_timeout_s"]),
        "DP_OPT_DEADLINE_S": "120", "DP_OPT_RELEASE_UNUSED_CACHE": "1", "DP_OPT_RANK_LOCAL_ADMISSION": "1",
        "DP_OPT_CONCURRENT_ADMISSION": "1", "DP_OPT_REVALIDATE_PREPARED": "1", "DP_OPT_PREPLAN": "1",
        "DP_OPT_KV_TRANSPORT": "dp_echo", "DP_OPT_TRANSFER_CHUNK_MIB": "32", "DP_OPT_MAX_MOVES": "8",
        "DP_OPT_CANDIDATE_LIMIT": "8", "DP_OPT_PAUSE_SYNC_INTERVAL": "32", "DP_OPT_FAIRNESS_QUEUE_THRESHOLD_S": "30",
        "DP_OPT_WORKAWARE_PROFILES": "1", "DP_OPT_WORKAWARE_PHASE_PROFILES": "1", "DP_OPT_RANDOM_BASELINE": "1",
        "DP_OPT_WORKAWARE_OPTIMIZER": plan["optimizer"], "DP_OPT_REBALANCE_MAX_MOVES": "4",
        "DP_OPT_REBALANCE_CANDIDATE_LIMIT": "8", "DP_OPT_REBALANCE_MIN_GAIN_S": "1", "DP_OPT_REBALANCE_COOLDOWN_S": "10",
        "DP_OPT_REBALANCE_RETRY_S": "2", "DP_OPT_PERFORMANCE_NUMERICS": "observe", "DP_OPT_TRACE_REPEATS": "1",
        "DP_OPT_TRACE_POLICIES": " ".join(plan["policies"]), "DP_OPT_TRACE_SEED": str(plan["controller_seed"]),
        "DP_OPT_MOVE_FIXED_S": str(plan["movement_costs"]["move_fixed_s"]),
        "DP_OPT_MOVE_BANDWIDTH_GIB_S": str(plan["movement_costs"]["move_bandwidth_gib_s"]),
        "DP_OPT_PERFORMANCE_FOLLOWUP_PLACEMENT_FIT": str(evidence / "placement-fit.json"),
        "DP_OPT_WORKAWARE_RATE_PRIORS": " ".join(map(str, plan["rates_rps"])),
        "DP_OPT_TRACE_PATH": str(evidence / "frozen/trace.json"),
        "DP_OPT_PROMPT_BANK": str(evidence / "frozen/inputs/prompt-bank.json"),
        "DP_OPT_WORKAWARE_FIT": str(evidence / "public-context-fit.json"),
        "DP_OPT_WORKAWARE_FIT_SHA256": plan["fit_sha256"],
        "DP_OPT_WORKAWARE_SOURCE_ROOT": str(evidence / "public-context-calibration"),
        "DP_OPT_PHASE_EVIDENCE_ROOT": str(evidence / plan["phase_name"])})
    launch = evidence / "performance-followup-trace.sh"
    template = evidence / "performance-followup-entry-only.sh"
    make_trace_launch(evidence / "runtime", template)
    text = template.read_text()
    original_fit = '"$root/state/sparse-fit-231467.json"'
    if text.count(original_fit) != 2:
        raise ValueError("Require the exact canonical placement-fit argument and source SHA inventory")
    with launch.open("x") as output:
        output.write(text.replace(original_fit, '"$DP_OPT_PERFORMANCE_FOLLOWUP_PLACEMENT_FIT"'))
    write_json(evidence / "performance-followup-launch-provenance.json", {
        "entry_only_template_sha256": digest(template), "final_launch_sha256": digest(launch),
        "placement_fit_sha256": plan["placement_fit_sha256"],
        "only_entry_and_byte_identical_placement_fit_path_changed": True})
    write_json(evidence / "phase.json", {"phase": plan["phase_name"], "policies": plan["policies"],
                                         "stage_id": plan["stage_id"], "greedy_gate": "observation_only"})
    (evidence / "progress.json").symlink_to(evidence / plan["phase_name"] / "progress.json")
    subprocess.run(["bash", str(launch)], env=environment, check=True)
    result = verify(evidence / plan["phase_name"], evidence / "frozen/inputs/prompt-bank.json")
    write_json(evidence / "compute-performance-verification.json", result)
    write_json(evidence / "summary.json", {"status": "one_complete_native_case_or_performance_policy_trace",
                                           "quality_equivalence_not_proven": True, "research_goal_complete": False})


def run(root, parent, native_job=278514, debug_job=278529, debug_name="qwen-head-hidden-probe",
        prepare_only=False, once=False, rate_only=False, walltime="04:30:00",
        allow_checkpoint_cancelled=False, native_coordinator_pid=None, native_only=False):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    native_job = positive_job_id(native_job)
    if root == parent or parent.is_relative_to(root) or root.is_relative_to(parent / f"live-trace-{native_job}"):
        raise ValueError("Use an independent followup evidence directory, not the native coordinator's root")
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    debug_job = positive_job_id(debug_job) if debug_job is not None else None
    queue_guard(debug_job, debug_name)
    walltime = policy_walltime(walltime)
    with campaign_lock(root):
        path = root / "campaign-state.json"
        identity = native_coordinator_identity(parent, native_coordinator_pid)
        state = read_json(path) if path.exists() else initial_state(root, parent, repo, native_job, debug_job,
            debug_name, walltime, allow_checkpoint_cancelled, identity)
        if ((state["repo"], state["native_root"], state["native_job"], state["debug_job"], state["debug_name"])
                != (str(repo), str(parent), native_job, debug_job, debug_name) or state["controls"] != CONTROLS
                or state["policy_walltime"] != walltime or state["allow_checkpoint_cancelled"] != allow_checkpoint_cancelled
                or state["native_coordinator_identity"] != identity
                or state.get("debug_registry") != initialize_debug_registry(root, debug_job, debug_name)):
            raise ValueError("Preserve the driver, parent, DEBUG identity and declared controls")
        check_frozen_sources(repo, state)
        campaign_queue_guard(state)
        stress_campaign.check_pause(root)
        stress_campaign.check_pause(parent)
        stress_campaign.save_state(root, state)
        if prepare_only:
            return state
        stopped = wait_native_archive(root, parent, state, once)
        if stopped is None:
            return state
        workspace = prepare_workspace(root, parent, native_job, allow_checkpoint_cancelled, stopped)
        if read_json(parent / "campaign-state.json")["frozen_runtime_sha256"] != state["native_runtime_sha256"]:
            raise ValueError("The native coordinator's frozen manifest changed")
        review = read_json(root / "native-parent-review.json")
        state["native_parent_terminal_unchanged"] = review["terminal"]
        for name in native_names():
            if name in review["available_cases"]:
                continue
            stress_campaign.check_pause(root)
            stress_campaign.check_pause(parent)
            plan = plan_for(root, name, name, "native", None, workspace / "frozen" / (name + ".json"))
            already = state["stages"].get(name, {}).get("phase") == "verified"
            entry, result = execute_stage(root, state, plan)
            merge_native_case(root, entry, result)
            state["phase"] = "missing_native_cases_rescue_one_case_per_allocation"
            stress_campaign.save_state(root, state)
            if once and not already:
                return state
        finalize_native_workspace(root, parent, native_job, verify_stage)
        state["phase"] = "six_native_cases_independently_verified_original_terminals_preserved"
        stress_campaign.save_state(root, state)
        if native_only:
            return state
        for ordinal in range(3):
            stress_campaign.check_pause(root)
            stress_campaign.check_pause(parent)
            name = f"rate-{ordinal}"
            plan = plan_for(root, name, name, "rate", "no_move", workspace / "calibration-frozen" / (name + ".json"))
            already = state["stages"].get(name, {}).get("phase") == "verified"
            entry, result = execute_stage(root, state, plan)
            source = root / "stages" / name / f"live-trace-{entry['job']}" / name
            target = workspace / name
            if not target.exists():
                shutil.copytree(source, target)
            if file_hashes(target) != file_hashes(source):
                raise ValueError("Merged rate evidence differs from its raw SHA archive")
            write_json(root / (name + "-provenance.json"), {"stage_id": name, "job": entry["job"], "verification": result,
                "raw_manifest_sha256": digest(root / "stages" / name / f"raw-{entry['job']}.sha256")})
            if once and not already:
                return state
        calibration = analyze_calibration(root)
        state["phase"] = "rate_calibration_independently_verified"
        stress_campaign.save_state(root, state)
        if calibration["candidate_case"] is None:
            state["phase"] = "no_qualified_rate_negative_results_retained"
            stress_campaign.save_state(root, state)
            return state
        if rate_only:
            return state
        config = freeze_evaluation(root)
        evaluation = root / "evaluation"
        for ordinal in range(len(config["grid"])):
            name = f"explore-{ordinal}"
            entries = {}
            for policy in poisson.POLICIES:
                stress_campaign.check_pause(root)
                stress_campaign.check_pause(parent)
                stage_id = name + "-" + policy
                plan = plan_for(root, stage_id, name, "explore", policy, evaluation / "frozen" / (name + ".json"), walltime)
                already = state["stages"].get(stage_id, {}).get("phase") == "verified"
                entries[policy], result = execute_stage(root, state, plan)
                if once and not already:
                    return state
            join_phase(root, name, entries)
        with split_public_analysis(root, verify_stage):
            if (evaluation / "selection.json").exists():
                selection = poisson.select(evaluation)
                if selection != read_json(evaluation / "selection.json"):
                    raise ValueError("The predeclared selection changed on resume")
            else:
                selection = poisson.freeze_heldout(evaluation)
        if selection["best_setting"] is not None:
            setting = selection["best_setting"]
            expected = poisson.make_poisson(evaluation / "frozen/inputs", poisson.HELDOUT_SEED,
                                           setting["conv_rate"], poisson.HELDOUT_S, setting["code_rate"])
            write_json(evaluation / "heldout-trace.json", expected)
            entries = {}
            for policy in poisson.POLICIES:
                stage_id = "heldout-" + policy
                stress_campaign.check_pause(root)
                stress_campaign.check_pause(parent)
                plan = plan_for(root, stage_id, "heldout", "heldout", policy, evaluation / "heldout-trace.json", walltime)
                if stage_id not in state["stages"]:
                    print("Heldout budget risk: 10800s trace + model loading + full 7200s request drain may exceed "
                          + plan["walltime"] + "; no requests or seeds are changed.", file=sys.stderr, flush=True)
                already = state["stages"].get(stage_id, {}).get("phase") == "verified"
                entries[policy], result = execute_stage(root, state, plan)
                if once and not already:
                    return state
            join_phase(root, "heldout", entries)
        with split_public_analysis(root, verify_stage):
            result = poisson.analyze(evaluation)
        write_json(root / "independent-evaluation-analysis.json", result)
        state["phase"] = "split_exploration_and_optional_heldout_SHA_verified"
        stress_campaign.save_state(root, state)
        return state


def main():
    parser = argparse.ArgumentParser(description="Independent native rescue -> rate -> one-policy-per-A800-job followup")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--native-root", type=Path)
    parser.add_argument("--native-job", type=int, default=278514)
    parser.add_argument("--debug-job", type=int, default=278529)
    parser.add_argument("--debug-name", default="qwen-head-hidden-probe")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--rate-only", action="store_true")
    parser.add_argument("--native-only", action="store_true", help="Stop after six native cases and immutable CPU rate freeze")
    parser.add_argument("--native-coordinator-pid", type=int, help="Explicit local native coordinator PID if identity file is absent")
    parser.add_argument("--allow-checkpoint-cancelled", action="store_true",
                        help="Authorize reuse of complete exit0 cases in scientifically checkpointed CANCELLED native job")
    parser.add_argument("--policy-walltime", default="04:30:00",
                        help="Explicit longer single-policy allocation budget; heldout loading/drain may exceed 4h30")
    parser.add_argument("--compute-upload", type=Path)
    parser.add_argument("--compute-evidence", type=Path)
    parser.add_argument("--register-debug-job", type=int, help="Operator-only: validate and append one exact DEBUG allocation")
    parser.add_argument("--register-debug-name", help="Exact Slurm name required for explicit DEBUG registration")
    args = parser.parse_args()
    if args.register_debug_job is not None or args.register_debug_name is not None:
        if (args.root is None or args.register_debug_job is None or args.register_debug_name is None
                or args.native_root or args.compute_upload or args.compute_evidence or args.prepare_only
                or args.once or args.native_only or args.rate_only):
            parser.error("DEBUG registration requires --root, --register-debug-job and --register-debug-name only")
        print(json.dumps(register_debug(args.root, args.register_debug_job, args.register_debug_name)))
    elif args.compute_upload:
        if args.compute_evidence is None or args.root or args.native_root or not os.environ.get("SLURM_JOB_ID"):
            parser.error("Compute execution requires an allocation and explicit immutable upload/evidence")
        compute(args.compute_upload, args.compute_evidence)
    else:
        if args.root is None or args.native_root is None or args.compute_evidence:
            parser.error("Supply independent --root and existing --native-root")
        try:
            run(args.root, args.native_root, args.native_job, args.debug_job, args.debug_name,
                args.prepare_only, args.once, args.rate_only, args.policy_walltime,
                args.allow_checkpoint_cancelled, args.native_coordinator_pid, args.native_only)
        except Exception as error:
            root = args.root.resolve()
            root.mkdir(parents=True, exist_ok=True)
            failure = {"error": repr(error), "recorded_at": datetime.now().astimezone().isoformat(),
                       "automatic_duplicate_submission_authorized": False}
            write_json(root / ("coordinator-error-" + str(time.time_ns()) + ".json"), failure)
            state_path = root / "campaign-state.json"
            if state_path.exists():
                state = read_json(state_path)
                state.update({"failure_phase": state["phase"], "phase": "diagnosis_required_followup",
                              "error": repr(error)})
                stress_campaign.save_state(root, state)
            raise


if __name__ == "__main__":
    main()
