import argparse
import ast
import csv
import fcntl
import hashlib
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
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath

import upstream_first_layer_campaign as upstream
import upstream_qkv_mirror_probe as mirror_probe


SCHEMA = "upstream-qkv-mirror-campaign-v1"
MATRIX_SCHEMA = "upstream-first-layer-full-matrices-v1"
JOB_NAME = "qwen-upstream-qkv-mirror"
SCOPE = upstream.SCOPE
MAX_MATRIX_BYTES = 128 * 1024 * 1024
KERNEL_TRACE_DIRECTORY = "upstream-qkv-mirror-kernel-traces"
BASE_CONTROLS = {**upstream.CONTROLS, "mirror_repetitions": 3, "primary_operator": "quant_method.apply",
                 "parent_full_matrices_required": True, "atomic_authorization": False}
NEW_FILES = ("experiments/dp_opt/upstream_qkv_mirror_campaign.py", "experiments/dp_opt/upstream-qkv-mirror.sbatch",
             "tests/test_dp_opt_upstream_qkv_mirror_campaign.py")
MIRROR_FILES = ("experiments/dp_opt/upstream_qkv_mirror_worker.py", "experiments/dp_opt/upstream_qkv_mirror_probe.py",
                "tests/test_dp_opt_upstream_qkv_mirror_probe.py")
CLAIMS = {"numerical_root_cause_proven": False, "specific_kernel_proven": False,
          "specific_cublas_algorithm_proven": False, "downstream_first_top1_causality_proven": False,
          "quality_equivalence_proven": False, "is_algorithm_benefit": False, "publication_authorized": False}
read = upstream.read
save = upstream.save
digest = upstream.digest
hashes = upstream.hashes
check_hashes = upstream.check_hashes
positive_job = upstream.positive_job


def campaign_controls(kernel_activity=False):
    policy = mirror_probe.kernel_activity_policy(kernel_activity)
    reviewed = {"maximum_observations_per_worker": 1, "maximum_shadow_calls_per_worker": 6,
                "maximum_trace_bytes_per_call": 4 * 1024 * 1024, "maximum_trace_events_per_call": 4096,
                "selection_rule": "first_in_scope_trial_graph_replay_per_worker", "operator": "quant_method.apply",
                "extra_shadow_calls_only": True, "native_cuda_graph_profiled": False, "record_shapes": False,
                "profile_memory": False, "with_stack": False, "with_flops": False,
                "unsupported_is_nonfatal": True, "profiler_buffer_and_export_peak_not_hard_bounded": True}
    if any(mirror_probe.canonical_sha(policy.get(name)) != mirror_probe.canonical_sha(value)
           for name, value in reviewed.items()):
        raise ValueError("Kernel policy exceeds the reviewed shadow/capture/retention contract")
    return {**BASE_CONTROLS, "kernel_activity": policy, "kernel_trace_directory": KERNEL_TRACE_DIRECTORY,
            "kernel_retention_budget": {"workers": 4, "maximum_trace_files": 24,
                                        "maximum_trace_bytes": 96 * 1024 * 1024},
            "kernel_shadow_timings_are_native_performance": False}


CONTROLS = campaign_controls()


def validate_kernel_plan(plan, manifest):
    controls = plan.get("controls")
    if not isinstance(controls, dict) or not isinstance(controls.get("kernel_activity"), dict):
        raise ValueError("Immutable kernel controls missing; require a NEW prepare")
    expected = campaign_controls(controls["kernel_activity"].get("enabled"))
    if plan.get("schema") != SCHEMA or mirror_probe.canonical_sha(controls) != mirror_probe.canonical_sha(expected):
        raise ValueError("Immutable mirror/kernel controls or budgets differ")
    sources = {name: manifest.get(name) for name in MIRROR_FILES[:2]}
    if (any(not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum) for checksum in sources.values())
            or plan.get("kernel_activity_source_sha256") != sources
            or plan.get("kernel_activity_policy_sha256") != mirror_probe.canonical_sha(expected["kernel_activity"])):
        raise ValueError("Immutable kernel policy/source SHA binding differs")
    return expected


def verify_kernel_campaign(root, plan):
    root = Path(root)
    manifest = read(root / "runtime-manifest.json")
    controls = validate_kernel_plan(plan, manifest)
    sources = plan["kernel_activity_source_sha256"]
    check_hashes(root / "runtime", sources)
    local_path(root, controls["kernel_trace_directory"])
    workers = {}
    for service in ("instance-1", "instance-2"):
        for rank in (0, 1):
            name = f"upstream-qkv-mirror-worker-{service}-dp{rank}.json"
            path = local_path(root, name)
            worker = read(path)
            if (worker.get("service") != service or type(worker.get("dp_rank")) is not int or worker["dp_rank"] != rank
                    or worker.get("repetitions") != controls["mirror_repetitions"]
                    or mirror_probe.canonical_sha(worker.get("kernel_activity_policy"))
                       != plan["kernel_activity_policy_sha256"]
                    or any(worker.get("source_sha256", {}).get(symbol) != sources[source]
                           for symbol, source in zip(("mirror_worker", "mirror_probe"), MIRROR_FILES[:2]))):
                raise ValueError("Actual worker kernel opt-in/policy/source differs from the immutable campaign")
            workers[name] = digest(path)
    return {"schema": "upstream-qkv-mirror-kernel-campaign-binding-v1", "controls": controls,
            "kernel_activity_source_sha256": sources, "kernel_activity_policy_sha256": plan["kernel_activity_policy_sha256"],
            "worker_metadata_sha256": workers, "unsupported_activity_is_partial_not_kernel_attribution": True,
            "native_graph_kernel_attribution_proven": False, **CLAIMS}


def local_path(root, name):
    relative = upstream.relative_path(name)
    path = Path(root) / relative
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Evidence path escapes its immutable root")
    return path


def source_symbol(root, source):
    path = local_path(root, source["copied_path"])
    if digest(path) != source["file_sha256"]:
        raise ValueError("Installed source file SHA differs")
    first, count = source["source_first_line"], source["source_line_count"]
    if type(first) is not int or type(count) is not int or min(first, count) <= 0:
        raise ValueError("Installed Python symbol source extent unavailable")
    lines = path.read_text().splitlines(keepends=True)
    text = "".join(lines[first - 1:first - 1 + count])
    if first + count - 1 > len(lines) or hashlib.sha256(text.encode()).hexdigest() != source["symbol_source_sha256"]:
        raise ValueError("Installed symbol SHA/extent differs")
    return text


def review_apply_abi(text):
    import textwrap

    tree = ast.parse(textwrap.dedent(text))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "apply"]
    if len(functions) != 1:
        raise ValueError("Require the actual bound quant_method.apply source")
    function = functions[0]
    names = [argument.arg for argument in function.args.args]
    if (len(names) != 4 or names[:2] != ["self", "layer"] or names[3] != "bias"
            or function.args.posonlyargs or function.args.kwonlyargs or function.args.vararg or function.args.kwarg):
        raise ValueError("Unsupported installed apply signature; layer proxy ABI unproven")
    parents = {child: node for node in ast.walk(function) for child in ast.iter_child_nodes(node)}
    fields = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id == "layer":
            parent = parents.get(node)
            if not isinstance(parent, ast.Attribute) or parent.value is not node or parent.attr not in ("weight", "bias"):
                raise ValueError("Layer escapes into an unreviewed callee; weight/bias-only proxy ABI unproven")
            if not isinstance(parent.ctx, ast.Load):
                raise ValueError("Installed apply mutates the layer proxy")
            fields.add(parent.attr)
    if "weight" not in fields:
        raise ValueError("Installed apply has no verified weight operand")
    return {"layer_fields": sorted(fields), "layer_escapes": False,
            "apply_symbol_sha256": hashlib.sha256(text.encode()).hexdigest(), **CLAIMS}


def installed_parent(root, job):
    expected = {(service, rank) for service in ("instance-1", "instance-2") for rank in (0, 1)}
    records = {}
    for path in root.glob("installed-upstream-linear-graph-*.json"):
        record = read(path)
        identity = (record["service"], record["dp_rank"])
        if identity not in expected or identity in records:
            raise ValueError("Installed parent worker coverage differs")
        worker_path = root / f"upstream-first-layer-worker-{identity[0]}-dp{identity[1]}.json"
        worker = read(worker_path)
        if (record["schema"] != upstream.INSTALLED_SCHEMA or type(record["dp_rank"]) is not int
                or record["slurm_job_id"] != job or type(record["slurm_job_id"]) is not int
                or not record["compute_hostname"] or record["runner_model_is_cudagraph_wrapper"] is not True
                or record["worker_metadata_sha256"] != digest(worker_path)
                or set(record["sources"]) != upstream.INSTALLED_SYMBOLS
                or any(record[name] != worker[name] for name in ("checkpoint", "model_class", "model_dtype", "projection",
                       "attention_tp", "dp", "ep", "pp", "cudagraph_mode"))
                or record["model_dtype"] != "torch.bfloat16" or record["attention_tp"] != 1
                or (record["dp"], record["ep"], record["pp"]) != (2, 2, 1)
                or record["cudagraph_mode"] != "FULL_DECODE_ONLY"
                or any(record.get(name) is not False for name in ("specific_kernel_proven", "quality_equivalence_proven", "is_algorithm_benefit"))):
            raise ValueError("Parent installed provenance is not real same-job CUDA evidence")
        sources = record["sources"]
        if (sources["linear_class"]["qualified_name"] != record["linear_mro"][0]
                or not record["linear_mro"][0].endswith(".QKVParallelLinear")
                or sources["quant_method_class"]["qualified_name"] != record["quant_method_mro"][0]
                or not record["quant_method_mro"][0].endswith(".UnquantizedLinearMethod")
                or any(sources[name]["qualified_name"] not in record["linear_mro"] for name in ("linear_base", "column_parallel_linear"))
                or sources["quant_method_base"]["qualified_name"] not in record["quant_method_mro"]
                or sources["runner_model_class"]["qualified_name"] != record["runner_model_class"]
                or sources["model_class"]["qualified_name"] != record["model_class"]
                or sources["quant_method_apply"]["qualified_name"] != record["quant_method_mro"][0] + ".apply"
                or sources["cudagraph_call"]["qualified_name"] != sources["cudagraph_wrapper"]["qualified_name"] + ".__call__"):
            raise ValueError("Installed MRO/bound apply/CUDAGraph source identity differs")
        for source in record["sources"].values():
            if not source["copied_path"].startswith(f"installed-upstream-sources/{identity[0]}-dp{identity[1]}/"):
                raise ValueError("Installed source lacks per-worker path binding")
            source_symbol(root, source)
        for source_name, worker_name in (("linear_class", "qkv"), ("model_runner", "runner")):
            if record["sources"][source_name]["symbol_source_sha256"] != worker["source_sha256"][worker_name]:
                raise ValueError("Parent symbol SHA differs; never equate fileSHA with symbolSHA")
        abi = review_apply_abi(source_symbol(root, record["sources"]["quant_method_apply"]))
        stable = {"checkpoint": record["checkpoint"], "model_class": record["model_class"],
                  "linear_mro": record["linear_mro"], "quant_method_mro": record["quant_method_mro"],
                  "runner_model_class": record["runner_model_class"], "projection": record["projection"], "sources": {
            name: {key: value for key, value in source.items() if key != "copied_path"}
            for name, source in record["sources"].items()}, "abi": abi}
        records[identity] = stable
    if set(records) != expected or any(value != next(iter(records.values())) for value in records.values()):
        raise ValueError("Four parent workers have inconsistent installed ABI/source/weights")
    return {"workers_verified": 4, **next(iter(records.values()))}


def full_parent_matrices(raw, job):
    index_path = raw / "upstream-first-layer-full-matrices.json"
    if not index_path.is_file():
        raise ValueError("Parent is row-only: complete B1/padded QKV matrices unavailable; submit forbidden")
    index = read(index_path)
    if (index["schema"] != MATRIX_SCHEMA or index["job"] != job or index["scientific_scope"] != SCOPE
            or index["captured_on_compute"] is not True or index["source_frame_before_parent_consume"] is not True):
        raise ValueError("Full parent matrices lack compute/same-forward provenance")
    snapshots = {str(path.relative_to(raw)): read(path) for path in (raw / "upstream-first-layer-snapshots").glob("*.json")
                 if read(path)["forward_mode"] == "FULL"}
    seen, centers = set(), []
    for record in index["records"]:
        name = record["snapshot_path"]
        if name not in snapshots or name in seen or digest(local_path(raw, name)) != record["snapshot_sha256"]:
            raise ValueError("Full matrix source snapshot SHA/coverage differs")
        seen.add(name)
        snapshot = snapshots[name]
        if (snapshot["prompt_tokens"] != 149 or snapshot["position"] not in (52, 53, 54)
                or record["refresh_witness"] != snapshot["refresh_witness"]
                or snapshot["refresh_witness"]["generation_after"] != snapshot["refresh_witness"]["generation_before"] + 1
                or set(record["matrices"]) != {"qkv_input", "qkv_output"}):
            raise ValueError("Full matrices have stale generation or wrong scientific scope")
        for buffer_name, matrix in record["matrices"].items():
            buffer = snapshot["buffers"][buffer_name]
            path = local_path(raw, matrix["path"])
            shape = buffer["shape"]
            token_row = snapshot["token_row_index"]
            if (len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape)
                    or shape[0] != snapshot["num_tokens_padded"] or type(token_row) is not int
                    or not 0 <= token_row < shape[0] or shape[0] * shape[1] * 2 > MAX_MATRIX_BYTES
                    or path.stat().st_size != shape[0] * shape[1] * 2):
                raise ValueError("Full matrix shape/row/bounded byte length unsupported")
            rows, width = shape
            content = path.read_bytes()
            if (matrix["shape"] != shape or matrix["dtype"] != "torch.bfloat16"
                    or matrix["encoding"] != "bf16-little-endian" or len(content) != rows * width * 2
                    or hashlib.sha256(content).hexdigest() != matrix["sha256"]
                    or hashlib.sha256(content[token_row * width * 2:(token_row + 1) * width * 2]).hexdigest() != buffer["row_sha256"]):
                raise ValueError("Complete BF16 matrix bytes/shape/actual target row SHA differs")
        if snapshot["position"] == 53:
            centers.append(snapshot)
    if not snapshots or seen != set(snapshots):
        raise ValueError("Require full matrices for every accepted in-scope graph snapshot")
    pairs = [(left, right) for left in centers for right in centers
             if left["num_tokens_padded"] == 1 and right["num_tokens_padded"] > 1
             and left["context_token_ids"] == right["context_token_ids"] and left["projection"] == right["projection"]]
    if not pairs:
        raise ValueError("Parent has no common-prefix p53 B1/padded complete-output pair")
    return {"index_sha256": digest(index_path), "full_matrix_snapshots": len(seen), "common_prefix_B1_padded_pairs": len(pairs),
            "same_QKV_input_rows_in_all_pairs": all(left["buffers"]["qkv_input"]["row_sha256"] == right["buffers"]["qkv_input"]["row_sha256"]
                                                   for left, right in pairs), **CLAIMS}


def offline_parent(raw, destination):
    scripts = raw / "runtime/experiments/dp_opt"
    environment = {**os.environ, "PYTHONPATH": str(scripts), "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_ADDOPTS": "-p no:cacheprovider"}
    for name in ("ENABLE_KVCACHED", "KVCACHED_AUTOPATCH", "DP_OPT_LIVE_PLACEMENT"):
        environment.pop(name, None)
    result = subprocess.run([sys.executable, str(scripts / "upstream_first_layer_probe.py"), "--verify", str(raw),
                             "--output", str(destination)], env=environment, cwd=destination.parent,
                            text=True, capture_output=True, timeout=600)
    if result.returncode:
        raise ValueError("Independent frozen parent analyzer failed: " + result.stderr[-2000:])
    return read(destination)


def parent_evidence(parent):
    parent = Path(parent).resolve()
    state = read(parent / "campaign-state.json")
    job = positive_job(state["job"])
    archive = read(parent / f"archive-{job}.json")
    if (state.get("GPU_diagnostic_complete") is not True or state.get("state_last_observed") != "COMPLETED"
            or archive["job"] != job or archive["state"] != "COMPLETED" or archive["missing_job_logs"]
            or state["controls"] != upstream.CONTROLS):
        raise ValueError("Require a real successful first-layer GPU parent, not head-hidden or startup-only")
    raw = parent / f"live-trace-{job}"
    check_hashes(raw, upstream.manifest_hashes((parent / f"raw-{job}.sha256").read_text()), exact=True)
    for suffix in ("out", "err"):
        if digest(parent / f"live-trace-{job}.{suffix}") != archive["log_sha256"][suffix]:
            raise ValueError("Parent Slurm log SHA differs")
    runtime = read(raw / "runtime-manifest.json")
    if not runtime or runtime != state["runtime_sha256"]:
        raise ValueError("Parent must bind its actual immutable runtime manifest, not an assumed source count")
    check_hashes(raw / "runtime", runtime, exact=True)
    if (raw / "exit-status").read_text().strip() != "0":
        raise ValueError("Parent compute exit status is not success")
    accounting = (parent / f"accounting-{job}.txt").read_text().splitlines()
    rows = [[field.strip() for field in line.rstrip("|").split("|")] for line in accounting if line.startswith(str(job) + "|")]
    if (len(rows) != 1 or len(rows[0]) < 4 or rows[0][1:3] != ["COMPLETED", "0:0"]
            or not re.search(r"(?:^|,)gres/gpu=2(?:,|$)", rows[0][-1])):
        raise ValueError("Parent actual Slurm accounting lacks completed two-GPU allocation")
    gpu_rows = list(csv.reader((raw / "gpu-memory-before-load.csv").read_text().splitlines()))
    if (len(gpu_rows) != 2 or any(len(row) != 4 or "A800" not in row[1] or not row[0].strip().startswith("GPU-") for row in gpu_rows)
            or gpu_rows[0][0].strip() == gpu_rows[1][0].strip()):
        raise ValueError("Parent actual two-A800 UUID provenance missing")
    report = read(parent / "independent-upstream-first-layer-verification.json")
    if (report != read(raw / "compute-upstream-first-layer-verification.json")
            or report["status"] != "completed_first_layer_observation_not_benefit"
            or report["all_in_scope_target_attributions_available"] is not True or report["four_worker_QKV_weights_verified"] is not True
            or report["scientific_scope"] != SCOPE or report["numerical_root_cause_proven"] is not False):
        raise ValueError("Parent does not prove full in-scope first-layer GPU observation")
    with tempfile.TemporaryDirectory(prefix="qkv-parent-verify-") as temporary:
        if offline_parent(raw, Path(temporary) / "verification.json") != report:
            raise ValueError("Independent parent reconstruction differs from archived report")
    installed = installed_parent(raw, job)
    matrices = full_parent_matrices(raw, job)
    check_hashes(raw, upstream.manifest_hashes((parent / f"raw-{job}.sha256").read_text()), exact=True)
    anchors = ("campaign-state.json", f"raw-{job}.sha256", f"archive-{job}.json", f"accounting-{job}.txt",
               "independent-upstream-first-layer-verification.json", f"live-trace-{job}.out", f"live-trace-{job}.err")
    return state, raw, {"parent_job": job, "anchor_sha256": {name: digest(parent / name) for name in anchors},
                        "raw_manifest_sha256": digest(parent / f"raw-{job}.sha256"),
                        "installed": installed, "matrices": matrices, **CLAIMS}


def driver_alive(performance_root, expected=None):
    root = Path(performance_root).resolve()
    identity = read(root / "local-monitor-identity.json")
    if expected is not None and identity != expected:
        raise ValueError("Performance driver identity changed; immutable reprepare required")
    pid = identity.get("pid")
    argv = identity.get("argv", [])
    if (type(pid) is not int or pid <= 0 or not isinstance(argv, list) or "--root" not in argv
            or argv.index("--root") + 1 == len(argv) or argv[argv.index("--root") + 1] != str(root)
            or not identity.get("actual_process_start") or identity.get("monitor_interval_s") != 300):
        raise ValueError("Require recorded actual performance PID/start/argv, not a launcher alias")
    os.kill(pid, 0)
    actual = {}
    for field in ("command", "lstart", "uid"):
        result = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", field + "="], text=True, capture_output=True, timeout=10)
        if result.returncode or not result.stdout.strip():
            raise ValueError("Performance driver PID is not alive")
        actual[field] = result.stdout.strip()
    if (shlex.split(actual["command"]) != argv or " ".join(actual["lstart"].split()) != " ".join(identity["actual_process_start"].split())
            or actual["uid"] != str(os.getuid())):
        raise ValueError("Performance actual owner/start/argv mismatch or PID reuse")
    return identity


def prepare(root, parent, performance_root, repo, kernel_activity=False):
    controls = campaign_controls(kernel_activity)
    root, parent, performance_root, repo = [Path(path).resolve() for path in (root, parent, performance_root, repo)]
    isolated_root(root, parent, performance_root, repo)
    if root.exists() and any(path.name != "mirror-campaign.lock" for path in root.iterdir()):
        raise ValueError("New campaign root must be empty; retain partial prepare in place")
    parent_state, raw, gate = parent_evidence(parent)
    performance, identities, head = upstream.performance_state(performance_root)
    identity = driver_alive(performance_root)
    if Path(performance["repo"]).resolve() != repo or not performance["frozen_runtime_sha256"]:
        raise ValueError("Bind to the caller's actual live performance runtime manifest/source root")
    report = upstream.source_manifest_diff(parent_state["runtime_sha256"], performance["frozen_runtime_sha256"])
    root.mkdir(parents=True, exist_ok=True)
    save(root / "source-union-review.json", report)
    if report["conflicts"]:
        raise ValueError("Explicit parent/performance intersection conflicts: " + json.dumps(report["conflicts"], sort_keys=True))
    union = {**performance["frozen_runtime_sha256"], **parent_state["runtime_sha256"]}
    if set(union) & set(NEW_FILES + MIRROR_FILES):
        raise ValueError("Never overwrite a parent/performance frozen file with mirror implementation")
    union.update({name: digest(repo / name) for name in NEW_FILES + MIRROR_FILES})
    union.setdefault(upstream.COPY_ONLY, digest(repo / upstream.COPY_ONLY))
    for name in union:
        upstream.relative_path(name)
    upload = root / "upload"
    upload.mkdir()
    for name, checksum in union.items():
        source = raw / "runtime" / name if name in parent_state["runtime_sha256"] else repo / name
        if digest(source) != checksum:
            raise ValueError("Readonly source snapshot SHA mismatch: " + name)
        target = upload / "runtime" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copytree(raw / "frozen", upload / "frozen")
    shutil.copy2(raw / "placement-fit.json", upload / "placement-fit.json")
    provenance = upload / "provenance"
    provenance.mkdir()
    save(provenance / "source-union-review.json", report)
    for source, name in ((parent / "campaign-state.json", "parent-campaign-state.json"),
                         (parent / f"raw-{gate['parent_job']}.sha256", "parent-raw.sha256"),
                         (parent / "independent-upstream-first-layer-verification.json", "parent-verification.json"),
                         (performance_root / "campaign-state.json", "performance-campaign-state.json"),
                         (performance_root / "local-monitor-identity.json", "performance-driver-identity.json")):
        shutil.copy2(source, provenance / name)
    save(provenance / "parent-gate.json", gate)
    save(upload / "runtime-manifest.json", union)
    save(upload / "plan.json", {"schema": SCHEMA, "controls": controls, "parent_job": gate["parent_job"],
        "kernel_activity_source_sha256": {name: union[name] for name in MIRROR_FILES[:2]},
        "kernel_activity_policy_sha256": mirror_probe.canonical_sha(controls["kernel_activity"]),
        "parent_gate_sha256": digest(provenance / "parent-gate.json"), "trace_sha256": digest(upload / "frozen/trace.json"),
        "frozen_sha256": hashes(upload / "frozen"), "placement_fit_sha256": digest(upload / "placement-fit.json"),
        "parent_runtime_sha256": parent_state["runtime_sha256"], "performance_runtime_sha256": performance["frozen_runtime_sha256"],
        "protected_patch_copy_only": upstream.COPY_ONLY})
    check_hashes(upload / "runtime", union, exact=True)
    save(upload / "compute-integrity.json", hashes(upload))
    payload = hashes(upload)
    with (root / "payload.tar.gz").open("xb") as output:
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            for name in payload:
                archive.add(upload / name, arcname=name, recursive=False)
    for path in upload.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    state = {"schema": SCHEMA, "repo": str(repo), "parent": str(parent), "concurrent_performance_root": str(performance_root),
        "phase": "prepared", "controls": controls, "parent_gate": gate, "driver_identity": identity,
        "runtime_sha256": union, "performance_runtime_sha256": performance["frozen_runtime_sha256"],
        "performance_registry": performance["debug_registry"], "authorization_observed": head,
        "performance_job_at_prepare": positive_job(performance["active_performance_job"]),
        "payload_files_sha256": payload, "payload_sha256": digest(root / "payload.tar.gz"),
        "stage_name": "upstream-qkv-mirror-" + uuid.uuid4().hex, "GPU_diagnostic_complete": False, **CLAIMS}
    save(root / "campaign-state.json", state)
    return state


def isolated_root(root, parent, performance_root, repo):
    root = Path(root).resolve()
    if any(root == Path(path).resolve() or root.is_relative_to(Path(path).resolve())
           or Path(path).resolve().is_relative_to(root) for path in (parent, performance_root, repo)):
        raise ValueError("Campaign must be isolated from parent/performance/source roots")


def frozen_host(root, state):
    scripts = root / "upload/runtime/experiments/dp_opt"
    for module, name in ((sys.modules[__name__], "upstream_qkv_mirror_campaign.py"), (upstream, "upstream_first_layer_campaign.py"),
                         (mirror_probe, "upstream_qkv_mirror_probe.py")):
        expected = scripts / name
        if (Path(module.__file__).resolve() != expected.resolve()
                or digest(expected) != state["runtime_sha256"]["experiments/dp_opt/" + name]):
            raise ValueError("Network operations require the copied immutable campaign/helper, not working source")


def validate_local(root, state, submit=False, full_parent=False):
    root = Path(root).resolve()
    isolated_root(root, state["parent"], state["concurrent_performance_root"], state["repo"])
    if state["schema"] != SCHEMA or any(state.get(name) is not value for name, value in CLAIMS.items()):
        raise ValueError("Immutable mirror controls/claims differ")
    if not re.fullmatch(r"upstream-qkv-mirror-[0-9a-f]{32}", state["stage_name"]):
        raise ValueError("Unsupported unique isolated stage identity")
    check_hashes(root / "upload", state["payload_files_sha256"], exact=True)
    if digest(root / "payload.tar.gz") != state["payload_sha256"]:
        raise ValueError("Immutable payload changed")
    plan = read(root / "upload/plan.json")
    manifest = read(root / "upload/runtime-manifest.json")
    controls = validate_kernel_plan(plan, manifest)
    if controls != state["controls"] or manifest != state["runtime_sha256"]:
        raise ValueError("Master mirror controls/runtime differ from the immutable plan")
    performance, identities, head = upstream.performance_state(state["concurrent_performance_root"], state["authorization_observed"])
    if (performance["frozen_runtime_sha256"] != state["performance_runtime_sha256"]
            or performance["debug_registry"] != state["performance_registry"]):
        raise ValueError("Performance source/registry epoch changed")
    if submit:
        frozen_host(root, state)
        driver_alive(state["concurrent_performance_root"], state["driver_identity"])
        if positive_job(performance["active_performance_job"]) != state["performance_job_at_prepare"]:
            raise ValueError("Performance allocation changed; immutable reprepare required")
        check_hashes(Path(state["parent"]), state["parent_gate"]["anchor_sha256"])
        if read(root / "upload/provenance/parent-gate.json") != state["parent_gate"]:
            raise ValueError("Immutable verified parent gate differs")
    if full_parent:
        parent_state, raw, gate = parent_evidence(state["parent"])
        if gate != state["parent_gate"]:
            raise ValueError("Actual verified parent evidence changed")
    return performance, identities, head


def ssh(root, label, script, stdin_path=None, timeout=240):
    if os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Host-only coordinator may not SSH/submit from compute")
    state = read(Path(root) / "campaign-state.json")
    frozen_host(Path(root), state)
    return upstream.ssh(root, label, script, stdin_path, timeout)


def guard(state, debug_limit):
    performance, identities, head = upstream.performance_state(state["concurrent_performance_root"], state["authorization_observed"])
    job = positive_job(performance["active_performance_job"])
    return upstream.queue_guard(job, identities, debug_limit) + f'''\nmirror_perf="$(squeue -h -j {job} -o '%i|%u|%j|%P|%T')"
printf '%s\\n' "$mirror_perf" | awk -F'|' -v job={job} -v owner="$up_owner" '
{{ count++; if (NF != 5 || $1 != job || $2 != owner || $3 != "{upstream.PERFORMANCE_NAME}" || $4 != "gpu_a800" || $5 !~ /^({upstream.ACTIVE})$/) bad=1 }}
END {{ exit (bad || count != 1) }}' '''


def protected_sync(root, state):
    validate_local(root, state, submit=True)
    script = guard(state, 0)
    output = ssh(root, "before-sync", script + '\nprintf "HOME_RESULT|%s\\n" "$HOME"')
    homes = [line.split("|", 1)[1] for line in output.splitlines() if line.startswith("HOME_RESULT|")]
    if len(homes) != 1 or not homes[0].startswith("/") or "\n" in homes[0]:
        raise ValueError("Remote home mapping unavailable")
    stage = homes[0] + "/run/kvcached-paracloud/sync/" + state["stage_name"]
    quoted = shlex.quote(stage)
    ssh(root, "stage-create", script + f"\ntest ! -e {quoted}; mkdir -p {quoted}/backup")
    ssh(root, "stage-upload", "cat > " + shlex.quote(stage + "/incoming.tar.gz"), root / "payload.tar.gz", 1800)
    commands = [script, f"cd {quoted}", f'test "$(sha256sum incoming.tar.gz | cut -d\' \' -f1)" = {state["payload_sha256"]}',
                "mkdir incoming upload; tar -xzf incoming.tar.gz -C incoming"]
    for name, checksum in state["payload_files_sha256"].items():
        file_name, directory = shlex.quote(name), shlex.quote(str(PurePosixPath(name).parent))
        commands.extend([f'test "$(sha256sum incoming/{file_name} | cut -d\' \' -f1)" = {checksum}',
            f"mkdir -p upload/{directory} backup/{directory}",
            f"if test -e upload/{file_name}; then test -f upload/{file_name}; test ! -L upload/{file_name}; "
            f"sha256sum upload/{file_name} >> before.sha256; cp -p upload/{file_name} backup/{file_name}; "
            f"else printf '%s\\n' {shlex.quote('MISSING  upload/' + name)} >> before.sha256; fi",
            f"cp -p incoming/{file_name} upload/{file_name}.new; mv upload/{file_name}.new upload/{file_name}",
            f'test "$(sha256sum upload/{file_name} | cut -d\' \' -f1)" = {checksum}',
            f"sha256sum upload/{file_name} >> after.sha256"])
    commands.extend(["find upload -type f -exec chmod a-w {} +", script])
    ssh(root, "protected-copy-only", "\n".join(commands), timeout=600)
    state.update({"remote_stage": stage, "submission_ledger": stage + "/submitted-mirror-job.txt", "phase": "staged"})
    save(root / "campaign-state.json", state)


def submit_held(root, state):
    validate_local(root, state, submit=True)
    window = upstream.performance_poll_window(state["concurrent_performance_root"])
    if window["receipt_sha256"] != state["performance_poll_window"]["receipt_sha256"]:
        raise ValueError("Poll receipt changed after the bounded authorization window")
    remaining = state["submit_register_deadline_epoch"] - time.time()
    if not 0 < remaining <= 60:
        raise ValueError("Submission/registration 60s deadline expired")
    ledger, stage = shlex.quote(state["submission_ledger"]), state["remote_stage"]
    script = guard(state, 0) + f'''\ntest ! -e {ledger}; mkdir {shlex.quote(stage + '/submission-intent')}
export DP_OPT_QKV_MIRROR_UPLOAD={shlex.quote(stage + '/upload')}
mkdir -p "$HOME/run/kvcached-paracloud/logs"
sbatch --hold --parsable --partition=gpu_a800 --nodes=1 --gpus=2 --time=03:00:00 --job-name={JOB_NAME} --export=ALL \\
 --output="$HOME/run/kvcached-paracloud/logs/live-trace-%j.out" --error="$HOME/run/kvcached-paracloud/logs/live-trace-%j.err" \\
 {shlex.quote(stage + '/upload/runtime/experiments/dp_opt/upstream-qkv-mirror.sbatch')} > {ledger}
cat {ledger}'''
    return positive_job(ssh(root, "submit-held", script, timeout=remaining).split(";", 1)[0])


def receipt(root, state, job):
    if positive_job((root / "submission-receipt.txt").read_text().strip()) != job:
        raise ValueError("Held job is not the exact local submission receipt")
    if job in (state["performance_job_at_prepare"], state["parent_gate"]["parent_job"]):
        raise ValueError("DEBUG cannot operate on performance/parent allocations")


def release_registered(root, state, job):
    receipt(root, state, job)
    performance, identities, head = validate_local(root, state, submit=True)
    matches = [entry for entry in identities if entry["job"] == job and entry["name"] == JOB_NAME and entry.get("owner")]
    if len(matches) != 1:
        raise ValueError("Held mirror lacks explicit performance registration")
    remaining = state["submit_register_deadline_epoch"] - time.time()
    if not 0 < remaining <= 60:
        raise ValueError("Release forbidden after the bounded authorization deadline")
    ssh(root, "release-registered", guard(state, 1) + held_identity(state, job) + f"\nscontrol release {job}", timeout=min(20, remaining))


def held_identity(state, job):
    command = shlex.quote(state["remote_stage"] + "/upload/runtime/experiments/dp_opt/upstream-qkv-mirror.sbatch")
    return f'''
control="$(scontrol show job {job} -o)"
case " $control " in *" JobState=PENDING "*) ;; *) exit 2;; esac
case " $control " in *" Reason=JobHeldUser "*) ;; *) exit 2;; esac
case " $control " in *" Command="{command}" "*) ;; *) exit 2;; esac
'''


def cancel_owned_held(root, state, job):
    receipt(root, state, job)
    ledger = shlex.quote(state["submission_ledger"])
    script = f'''set -eu
owner="$(id -un)"; test "$owner" = "$USER"
test "$(cut -d';' -f1 {ledger})" = {job}
queue="$(squeue -h -j {job} -o '%i|%u|%j|%P|%T')"
if test -z "$queue"; then
 account="$(sacct -n -X -P -j {job} --format=JobIDRaw,User,JobName%128,Partition,State%64)"
 printf '%s\\n' "$account" | awk -F'|' -v job={job} -v owner="$owner" '
 {{ sub(/\\|$/, "", $0); count++; if (NF != 5 || $1 != job || $2 != owner || $3 != "{JOB_NAME}" || $4 != "gpu_a800" || $5 !~ /^CANCELLED([ +].*)?$/) bad=1 }}
 END {{ exit (bad || count != 1) }}'
 exit 0
fi
test "$queue" = "{job}|$owner|{JOB_NAME}|gpu_a800|PENDING"
''' + held_identity(state, job) + f"\nscancel {job}"
    ssh(root, "cancel-only-owned-held-mirror", script, timeout=20)


def authorization_chain(root, state):
    python = shlex.quote(sys.executable)
    scripts = root / "upload/runtime/experiments/dp_opt"
    mirror = shlex.quote(str(scripts / "upstream_qkv_mirror_campaign.py"))
    performance = shlex.quote(str(scripts / "performance_followup.py"))
    root_arg, perf_arg = shlex.quote(str(root)), shlex.quote(state["concurrent_performance_root"])
    return f'''set -euo pipefail
released=0; job=""
cleanup() {{
 result=$?; trap - EXIT
 if test "$result" -ne 0 && test "$released" = 0 && test -n "$job"; then
  {python} {mirror} --root {root_arg} --cancel-owned-held-job "$job" || true
 fi
 exit "$result"
}}
trap cleanup EXIT; trap 'exit 124' TERM; trap 'exit 130' INT
set -o noclobber
{python} {mirror} --root {root_arg} --submit-held > {shlex.quote(str(root / 'submission-receipt.txt'))}
job="$(cat {shlex.quote(str(root / 'submission-receipt.txt'))})"
{python} {performance} --root {perf_arg} --register-debug-job "$job" --register-debug-name {JOB_NAME} > {shlex.quote(str(root / 'performance-debug-registration.json'))}
{python} {mirror} --root {root_arg} --release-registered-job "$job"
released=1
'''


def submit_and_register(root, state):
    validate_local(root, state, submit=True, full_parent=True)
    preflight_epoch = time.time()
    while True:
        if time.time() - preflight_epoch > 600:
            raise ValueError("No fresh poll within 600s of full parent revalidation; no submit")
        try:
            window = upstream.performance_poll_window(state["concurrent_performance_root"])
            break
        except (ValueError, FileNotFoundError) as error:
            driver_alive(state["concurrent_performance_root"], state["driver_identity"])
            save(root / f"waiting-perf-poll-{time.time_ns()}.json", {"reason": str(error), "ssh_performed": False})
            time.sleep(5)
    state.update({"performance_poll_window": window, "submit_register_deadline_epoch": time.time() + 60,
                  "full_parent_preflight_epoch": preflight_epoch,
                  "phase": "submission_intent_do_not_duplicate"})
    save(root / "campaign-state.json", state)
    chain_started = time.monotonic()
    process = subprocess.Popen(["bash", "-c", authorization_chain(root, state)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=max(0, state["submit_register_deadline_epoch"] - time.time()))
    except subprocess.TimeoutExpired:
        timed_out = True
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
    late = time.time() > state["submit_register_deadline_epoch"]
    save(root / "submit-register-chain.json", {"exit": process.returncode, "stdout": stdout, "stderr": stderr,
        "timed_out": timed_out, "deadline_exceeded": late, "elapsed_monotonic_s": time.monotonic() - chain_started,
        "max_submission_registration_s": 60, "atomic_authorization": False})
    if timed_out or late or process.returncode:
        try:
            path = root / "submission-receipt.txt"
            if not path.exists() or not path.read_text().strip():
                output = ssh(root, "recover-exact-submission-ledger", "cat " + shlex.quote(state["submission_ledger"]), timeout=20)
                job = positive_job(output.split(";", 1)[0])
                path.write_text(str(job) + "\n")
            cancel_owned_held(root, state, positive_job(path.read_text().strip()))
        except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
            save(root / "held-cleanup-reconciliation-required.json", {"error": str(error), "performance_cancelled": False,
                                                                      "resubmission_allowed": False})
        raise RuntimeError("Bounded DEBUG authorization failed; preserve ledgers, never cancel performance or duplicate submit")
    job = positive_job((root / "submission-receipt.txt").read_text().strip())
    state.update({"job": job, "phase": "monitoring", "registered_before_release": True})
    save(root / "campaign-state.json", state)


def poll(root, state):
    job = positive_job(state["job"])
    output = ssh(root, "poll", f'''set -eu
printf 'OWNER|%s\\n' "$(id -un)"
sacct -n -X -P -j {job} --format=JobIDRaw,User,JobName%128,Partition,State%64,ExitCode,AllocTRES%256
squeue -h -j {job} -o '%i|%u|%j|%P|%T'
''')
    owners = [line.split("|", 1)[1] for line in output.splitlines() if line.startswith("OWNER|")]
    rows = [line.rstrip("|").split("|") for line in output.splitlines() if line.startswith(str(job) + "|")]
    if len(owners) != 1 or not rows:
        raise ValueError("Missing actual mirror accounting/owner")
    states = []
    for fields in rows:
        fields = [field.strip() for field in fields]
        if len(fields) not in (5, 7) or not fields[4].split():
            raise ValueError("Malformed mirror accounting row")
        status = fields[4].split()[0].rstrip("+")
        if (fields[:4] != [str(job), owners[0], JOB_NAME, "gpu_a800"]
                or status not in (upstream.ACTIVE + "|" + upstream.TERMINAL).split("|")):
            raise ValueError("Actual mirror job identity/state differs")
        if len(fields) == 7:
            state["accounting_exit_code"] = fields[5]
            state["accounting_gpu_resources"] = fields[6]
        states.append(status)
    status = next((value for value in states if value in upstream.ACTIVE.split("|")), states[0])
    state.update({"state_last_observed": status, "last_poll_at": datetime.now().astimezone().isoformat(), "last_poll_ssh_exit": 0})
    save(root / "campaign-state.json", state)
    return status


def extract_terminal(bundle, staging, manifest, job, terminal):
    import archive_trace_job

    if terminal not in upstream.TERMINAL.split("|"):
        raise ValueError("Require actual terminal state for archive extraction")
    positive_job(job)
    pointers = []
    with tarfile.open(bundle) as archive:
        members = archive.getmembers()
        names = [str(PurePosixPath(member.name)) for member in members]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate archive member")
        regular = {str(PurePosixPath(member.name)) for member in members if member.isfile()}
        selected = []
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not (staging / member.name).resolve().is_relative_to(staging.resolve()):
                raise ValueError("Unsafe terminal archive member")
            if member.issym() and str(name) == "progress.json":
                target = PurePosixPath(member.linkname)
                anchor = ("run", "kvcached-paracloud", "state", f"live-trace-{job}")
                matches = [index for index in range(len(target.parts)) if target.parts[index:index + 4] == anchor]
                if not target.is_absolute() or ".." in target.parts or len(matches) != 1:
                    raise ValueError("Unbound terminal progress pointer")
                relative = PurePosixPath(*target.parts[matches[0] + 4:])
                if len(relative.parts) != 2 or relative.name != "progress.json" or terminal == "COMPLETED" and str(relative) not in regular:
                    raise ValueError("Invalid/missing completed progress target")
                pointers.append({"remote_target": member.linkname, "target_present": str(relative) in regular,
                                 "actual_terminal": terminal, "exit_marker_not_fabricated": True})
            elif member.isfile() or member.isdir():
                selected.append(member)
            else:
                raise ValueError("Unsafe archive link/member")
        archive.extractall(staging, members=selected)
    archive_trace_job.verify_raw(staging, manifest)
    return pointers


def archive_owned(root, state):
    import archive_trace_job

    frozen_host(root, state)
    terminal = poll(root, state)
    if terminal not in upstream.TERMINAL.split("|"):
        raise ValueError("Never archive an active mirror allocation")
    job = positive_job(state["job"])
    receipt(root, state, job)
    original, argv = archive_trace_job.extract_raw, sys.argv
    def extract_owned(bundle, staging, manifest, observed_job, allow_missing_failed_progress=False):
        if observed_job != job:
            raise ValueError("Wrong archive job")
        return extract_terminal(bundle, staging, manifest, observed_job, terminal)

    archive_trace_job.extract_raw = extract_owned
    sys.argv = [argv[0], str(job), "--archive", str(root)] + (["--resume"] if (root / f"live-trace-{job}").exists() else [])
    try:
        archive_trace_job.main()
    finally:
        archive_trace_job.extract_raw, sys.argv = original, argv


def archive_and_verify(root, state):
    scripts = root / "upload/runtime/experiments/dp_opt"
    process = subprocess.run([sys.executable, str(scripts / "upstream_qkv_mirror_campaign.py"), "--root", str(root), "--archive-owned-job"],
                             text=True, capture_output=True, timeout=3600)
    save(root / f"archive-attempt-{time.time_ns()}.json", {"exit": process.returncode, "stdout": process.stdout, "stderr": process.stderr})
    if process.returncode:
        state["archive_retry_count"] = state.get("archive_retry_count", 0) + 1
        save(root / "campaign-state.json", state)
        if state["archive_retry_count"] >= 12:
            raise RuntimeError("Terminal archive failed twelve times; retain bundle/ledgers, no resubmission")
        return False
    job = positive_job(state["job"])
    archive = read(root / f"archive-{job}.json")
    raw = root / f"live-trace-{job}"
    check_hashes(raw, upstream.manifest_hashes((root / f"raw-{job}.sha256").read_text()), exact=True)
    for suffix in ("out", "err"):
        if digest(root / f"live-trace-{job}.{suffix}") != archive["log_sha256"][suffix]:
            raise ValueError("Mirror terminal log SHA differs")
    if archive["missing_job_logs"] or archive["state"] != "COMPLETED" or state["accounting_exit_code"] != "0:0":
        state.update({"phase": "diagnosis_required_mirror_terminal", "GPU_diagnostic_complete": False})
        save(root / "campaign-state.json", state)
        raise RuntimeError("Actual mirror failure preserved; do not label startup/partial evidence successful")
    check_hashes(raw / "runtime", state["runtime_sha256"], exact=True)
    plan = read(root / "upload/plan.json")
    check_hashes(raw / "frozen", plan["frozen_sha256"], exact=True)
    if (read(raw / "runtime-manifest.json") != state["runtime_sha256"] or read(raw / "plan.json") != plan
            or hashes(raw / "provenance") != hashes(root / "upload/provenance")
            or (raw / "exit-status").read_text().strip() != "0"
            or digest(raw / "placement-fit.json") != plan["placement_fit_sha256"]
            or not re.search(r"(?:^|,)gres/gpu=2(?:,|$)", state["accounting_gpu_resources"])):
        raise ValueError("Mirror terminal runtime/input/parent/accounting provenance differs")
    if installed_parent(raw, job) != state["parent_gate"]["installed"]:
        raise ValueError("Actual mirror installed ABI/weights differ from verified parent")
    output = root / "independent-upstream-qkv-mirror-verification.json"
    environment = {**os.environ, "PYTHONPATH": str(scripts), "PYTHONDONTWRITEBYTECODE": "1"}
    for name in ("ENABLE_KVCACHED", "KVCACHED_AUTOPATCH", "DP_OPT_LIVE_PLACEMENT"):
        environment.pop(name, None)
    if not output.exists():
        result = subprocess.run([sys.executable, str(scripts / "upstream_qkv_mirror_probe.py"), "--verify", str(raw), "--output", str(output)],
                                env=environment, cwd=root, text=True, capture_output=True, timeout=600)
        save(root / "independent-verifier-process.json", {"exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError("Independent mirror verifier failed; raw/log SHA preserved")
    result = read(output)
    if (result != read(raw / "compute-upstream-qkv-mirror-verification.json")
            or result["all_in_scope_graph_sources_mirrored"] is not True
            or result["status"] != "bounded_qkv_mirror_observation_not_quality_fix" or result["scientific_scope"] != SCOPE
            or any(result.get(name) is not False for name in ("specific_kernel_proven", "specific_cublas_algorithm_proven",
                                                            "downstream_first_top1_causality_proven", "quality_equivalence_proven", "is_algorithm_benefit"))):
        raise ValueError("Compute/independent mirror reconstruction differs")
    if verify_kernel_campaign(raw, plan) != read(raw / "compute-qkv-kernel-campaign-binding.json"):
        raise ValueError("Compute/independent kernel campaign source/opt-in binding differs")
    state.update({"phase": "completed_observation_not_root_cause", "GPU_diagnostic_complete": True, **CLAIMS})
    save(root / "campaign-state.json", state)
    return True


@contextmanager
def campaign_lock(root):
    with (root / "mirror-campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run(root, parent, performance_root, repo, prepare_only=False, once=False, kernel_activity=False):
    campaign_controls(kernel_activity)
    if kernel_activity and not prepare_only:
        raise ValueError("Kernel opt-in is only declared at NEW --prepare-only, never at submission/resume")
    root, parent, performance_root, repo = [Path(path).resolve() for path in (root, parent, performance_root, repo)]
    isolated_root(root, parent, performance_root, repo)
    if prepare_only and (root / "campaign-state.json").is_file():
        existing = read(root / "campaign-state.json")
        if existing.get("controls") != campaign_controls(kernel_activity):
            raise ValueError("Prepare opt-in differs from the immutable root; require a NEW root")
    root.mkdir(parents=True, exist_ok=True)
    with campaign_lock(root):
        if not (root / "campaign-state.json").exists():
            if not prepare_only:
                raise ValueError("Run requires an explicit prior local --prepare-only and copied immutable launcher")
            prepare(root, parent, performance_root, repo, kernel_activity)
        state = read(root / "campaign-state.json")
        if (state["parent"], state["concurrent_performance_root"], state["repo"]) != (str(parent), str(performance_root), str(repo)):
            raise ValueError("Preserve explicit campaign/parent/performance identities")
        validate_local(root, state)
        if prepare_only:
            if state["controls"] != campaign_controls(kernel_activity):
                raise ValueError("Prepare opt-in differs from the immutable root; require a NEW root")
            if state["phase"] != "prepared" or "job" in state:
                raise ValueError("Prepare-only cannot operate a staged/submitted job")
            return
        if state["GPU_diagnostic_complete"]:
            return
        if state["phase"].startswith("diagnosis_required") or state["phase"] == "submission_intent_do_not_duplicate":
            raise RuntimeError("Reconcile prior intent/failure; never automatically resubmit")
        try:
            if state["phase"] == "prepared":
                protected_sync(root, state)
            if state["phase"] == "staged":
                submit_and_register(root, state)
            previous = None
            while True:
                started, epoch = time.monotonic(), time.time()
                try:
                    status = poll(root, state)
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    state.update({"last_poll_ssh_exit": "failed_or_timeout", "last_poll_error": str(error)})
                    status = None
                save(root / f"monitor-{time.time_ns()}.json", {"started_epoch": epoch, "monotonic_gap_s": None if previous is None else started - previous,
                    "wall_gap_s": None if state.get("previous_poll_epoch") is None else epoch - state["previous_poll_epoch"],
                    "ssh_exit": state.get("last_poll_ssh_exit"), "status": status})
                previous = started
                state["previous_poll_epoch"] = epoch
                save(root / "campaign-state.json", state)
                if status in upstream.TERMINAL.split("|") and archive_and_verify(root, state):
                    return
                if once:
                    return
                time.sleep(max(0, 300 - (time.monotonic() - started)))
        except BaseException as error:
            state.update({"failure_phase": state["phase"], "phase": "diagnosis_required_mirror", "error": repr(error),
                          "automatic_resubmission_authorized": False, "performance_cancelled": False})
            save(root / "campaign-state.json", state)
            raise


def require_compute():
    upstream.require_upstream_compute()
    if os.environ.get("DP_OPT_QKV_MIRROR_COMPUTE_CONFIRMED") != "1" or os.environ.get("PYTEST_ADDOPTS") != "-p no:cacheprovider":
        raise RuntimeError("Mirror compute opt-in/cache-free environment missing")


def _initialize_installed(self, kv_cache_config):
    result = super(globals()["InstalledProvenanceMirrorWorker"], self).initialize_from_config(kv_cache_config)
    upstream.collect_installed_provenance(self)
    return result


def __getattr__(name):
    if name != "InstalledProvenanceMirrorWorker":
        raise AttributeError(name)
    require_compute()
    from upstream_qkv_mirror_worker import UpstreamQKVMirrorVerificationWorker

    worker = type(name, (UpstreamQKVMirrorVerificationWorker,), {"__module__": __name__, "__qualname__": name,
                                                              "initialize_from_config": _initialize_installed})
    globals()[name] = worker
    return worker


def compute_main():
    require_compute()
    import head_hidden_probe
    import live_trace
    import upstream_qkv_mirror_probe

    original = head_hidden_probe.instrument

    @contextmanager
    def instrument():
        with original():
            previous = live_trace.engine_arguments

            def arguments(*args, **kwargs):
                result = previous(*args, **kwargs)
                result.worker_cls = "upstream_qkv_mirror_campaign.InstalledProvenanceMirrorWorker"
                return result

            live_trace.engine_arguments = arguments
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
    result = upstream_qkv_mirror_probe.analyze(root)
    save(root / "compute-qkv-kernel-campaign-binding.json", verify_kernel_campaign(root, read(root / "plan.json")))
    save(root / "compute-upstream-qkv-mirror-verification.json", result)


def make_launch(template):
    result = upstream.make_launch(template)
    previous_fit = '"$evidence/placement-fit.json"'
    input_fit = '"$DP_OPT_UPSTREAM_PLACEMENT_FIT"'
    if result.count(previous_fit) == 2:
        result = result.replace(previous_fit, input_fit)
    if (previous_fit in result or result.count(input_fit) != 2
            or result.count('--fit ' + input_fit) != 1
            or result.count(input_fit + ' "${DP_OPT_TRACE_PATH:?Set DP_OPT_TRACE_PATH}"') != 1):
        raise ValueError("Frozen fit CLI/SHA inventory must both bind the distinct immutable input")
    original = '"$evidence/upstream_first_layer_entry.py"'
    if result.count(original) != 1 or result.count('mkdir -p "$evidence"') != 1:
        raise ValueError("Frozen launch entry/cwd contract differs")
    return result.replace(original, '"$evidence/upstream_qkv_mirror_entry.py"').replace(
        'mkdir -p "$evidence"', 'mkdir -p "$evidence"\ncd "$evidence"')


def placement_fit_input(upload, evidence):
    source = local_path(upload, "placement-fit.json")
    target = local_path(evidence, "placement-fit.json")
    if (not source.is_file() or source.resolve() == target.resolve()
            or target.exists() and source.samefile(target)):
        raise ValueError("Placement-fit input and output archive must be distinct files")
    return source


def compute_environment(runtime, evidence, upload):
    controls = validate_kernel_plan(read(upload / "plan.json"), read(upload / "runtime-manifest.json"))
    local_path(evidence, controls["kernel_trace_directory"])
    environment = {name: value for name, value in os.environ.items() if not name.startswith("DP_OPT_")}
    environment.pop("VLLM_BATCH_INVARIANT", None)
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTEST_ADDOPTS": "-p no:cacheprovider",
        "DP_OPT_UPSTREAM_RUNTIME": str(runtime), "DP_OPT_UPSTREAM_COMPUTE_CONFIRMED": "1",
        "DP_OPT_UPSTREAM_PLACEMENT_FIT": str(placement_fit_input(upload, evidence)),
        "DP_OPT_QKV_MIRROR_COMPUTE_CONFIRMED": "1", "DP_OPT_QKV_MIRROR_REPETITIONS": str(controls["mirror_repetitions"]),
        "DP_OPT_QKV_MIRROR_KERNEL_ACTIVITY": "1" if controls["kernel_activity"]["enabled"] else "0",
        "KVCACHED_REPO_ROOT": str(runtime), "DP_OPT_PHASE_EVIDENCE_ROOT": str(evidence),
        "DP_OPT_MAX_MODEL_LEN": "131072", "DP_OPT_KV_CACHE_BYTES": "25769803776", "KVCACHED_GPU_UTILIZATION": "0.98",
        "DP_OPT_TRACE_BUDGET_MIB": "10240", "DP_OPT_REQUEST_TIMEOUT_S": "7200", "DP_OPT_DEADLINE_S": "120",
        "DP_OPT_RELEASE_UNUSED_CACHE": "1", "DP_OPT_RANK_LOCAL_ADMISSION": "1", "DP_OPT_CONCURRENT_ADMISSION": "1",
        "DP_OPT_REVALIDATE_PREPARED": "1", "DP_OPT_PREPLAN": "1", "DP_OPT_PAUSE_SYNC_INTERVAL": "32",
        "DP_OPT_KV_TRANSPORT": "dp_echo", "DP_OPT_TRANSFER_CHUNK_MIB": "32", "DP_OPT_TRACE_REPEATS": "6",
        "DP_OPT_TRACE_POLICIES": "no_move", "DP_OPT_WORKAWARE_OPTIMIZER": "point",
        "DP_OPT_PROMPT_BANK": str(evidence / "frozen/inputs/prompt-bank.json"), "DP_OPT_TRACE_PATH": str(evidence / "frozen/trace.json")})
    return environment


def write_compute_entry(evidence):
    entry = Path(evidence) / "upstream_qkv_mirror_entry.py"
    with entry.open("x") as output:
        output.write('if __name__ == "__main__":\n'
                     '    from upstream_qkv_mirror_campaign import compute_main\n'
                     '    compute_main()\n')
    return entry


def compute(upload, evidence):
    compute_preflight(upload, evidence)
    plan = read(upload / "plan.json")
    runtime = evidence / "runtime"
    manifest = read(upload / "runtime-manifest.json")
    check_hashes(runtime, manifest, exact=True)
    fit_input = placement_fit_input(upload, evidence)
    if (evidence / "placement-fit.json").exists():
        raise ValueError("Require a fresh output fit archive; do not pre-copy a readonly destination")
    shutil.copytree(upload / "frozen", evidence / "frozen")
    shutil.copytree(upload / "provenance", evidence / "provenance")
    shutil.copy2(upload / "plan.json", evidence / "plan.json")
    shutil.copy2(upload / "runtime-manifest.json", evidence / "runtime-manifest.json")
    check_hashes(evidence / "frozen", plan["frozen_sha256"], exact=True)
    if (digest(evidence / "frozen/trace.json") != plan["trace_sha256"] or digest(fit_input) != plan["placement_fit_sha256"]
            or digest(evidence / "provenance/parent-gate.json") != plan["parent_gate_sha256"]):
        raise ValueError("Compute input/parent-gate SHA differs")
    write_compute_entry(evidence)
    with (evidence / "mirror-launch.sh").open("x") as output:
        output.write(make_launch((runtime / "experiments/dp_opt/trace-benefit.sbatch").read_text()))
    environment = compute_environment(runtime, evidence, upload)
    save(evidence / "mirror-runtime-environment.json", {name: value for name, value in environment.items() if name.startswith(("DP_OPT_", "KVCACHED_", "VLLM_", "PYTEST_"))})
    result = subprocess.run(["bash", str(evidence / "mirror-launch.sh")], env=environment, cwd=evidence)
    if result.returncode:
        raise RuntimeError("Actual compute mirror launch failed; preserve original exit status")
    if digest(evidence / "placement-fit.json") != plan["placement_fit_sha256"] or digest(fit_input) != plan["placement_fit_sha256"]:
        raise ValueError("Actual live_trace input/output fit SHA differs after distinct-file copy")
    check_hashes(runtime, manifest, exact=True)
    if installed_parent(evidence, positive_job(os.environ["SLURM_JOB_ID"])) != read(upload / "provenance/parent-gate.json")["installed"]:
        raise ValueError("Actual compute installed source/weights differ from the independently verified parent")


def compute_preflight(upload, evidence):
    require_compute()
    placement_fit_input(upload, evidence)
    plan = read(upload / "plan.json")
    validate_kernel_plan(plan, read(upload / "runtime-manifest.json"))
    local_path(evidence, plan["controls"]["kernel_trace_directory"])
    expected = read(upload / "compute-integrity.json")
    if hashes(upload) != {**expected, "compute-integrity.json": digest(upload / "compute-integrity.json")}:
        raise ValueError("Compute upload integrity/file set differs")
    check_hashes(evidence / "runtime", read(upload / "runtime-manifest.json"), exact=True)
    check_hashes(upload / "frozen", plan["frozen_sha256"], exact=True)
    if (digest(upload / "provenance/parent-gate.json") != plan["parent_gate_sha256"]
            or digest(upload / "placement-fit.json") != plan["placement_fit_sha256"]
            or digest(upload / "frozen/trace.json") != plan["trace_sha256"]):
        raise ValueError("Compute pre-model parent/trace/fit SHA differs")
    for source in read(upload / "provenance/parent-gate.json")["installed"]["sources"].values():
        path = Path(source["installed_path"])
        if not path.is_absolute() or not path.is_file() or digest(path) != source["file_sha256"]:
            raise ValueError("Current compute installed linear/graph/source epoch differs from archived parent")


def main():
    parser = argparse.ArgumentParser(description="Host-only gated P149 QKV mirror campaign; prepare/validate never SSH")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--concurrent-performance-root", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--kernel-activity", action="store_true",
                        help="Explicit NEW prepare-only opt-in to bounded eager true-apply shadow profiling; default disabled")
    parser.add_argument("--validate-parent", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--compute-upload", type=Path)
    parser.add_argument("--compute-evidence", type=Path)
    parser.add_argument("--compute-preflight", action="store_true")
    helpers = parser.add_mutually_exclusive_group()
    helpers.add_argument("--submit-held", action="store_true", help=argparse.SUPPRESS)
    helpers.add_argument("--release-registered-job", type=int, help=argparse.SUPPRESS)
    helpers.add_argument("--cancel-owned-held-job", type=int, help=argparse.SUPPRESS)
    helpers.add_argument("--archive-owned-job", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    helper = args.submit_held or args.release_registered_job is not None or args.cancel_owned_held_job is not None or args.archive_owned_job
    if args.kernel_activity and (not args.prepare_only or args.compute_upload or args.validate_parent or helper):
        parser.error("--kernel-activity is only permitted with a NEW --prepare-only")
    if args.compute_upload:
        if (not args.compute_evidence or args.root or args.parent or args.concurrent_performance_root or helper
                or args.prepare_only or args.validate_parent or args.once or args.repo):
            parser.error("Compute requires only explicit upload/evidence")
        action = compute_preflight if args.compute_preflight else compute
        action(args.compute_upload.resolve(), args.compute_evidence.resolve())
    elif args.validate_parent:
        if (not args.parent or args.root or args.concurrent_performance_root or helper or args.compute_evidence
                or args.prepare_only or args.once or args.compute_preflight or args.repo):
            parser.error("Read-only parent validation requires only --parent")
        print(json.dumps(parent_evidence(args.parent)[2], indent=2))
    elif helper:
        if (not args.root or args.parent or args.concurrent_performance_root or args.compute_evidence
                or args.prepare_only or args.once or args.compute_preflight or args.repo):
            parser.error("Allocation helper requires only the exact root")
        root = args.root.resolve()
        state = read(root / "campaign-state.json")
        if args.cancel_owned_held_job is not None:
            frozen_host(root, state)
        else:
            validate_local(root, state)
        if not args.archive_owned_job and state["phase"] != "submission_intent_do_not_duplicate":
            parser.error("Require immutable submission intent")
        if args.submit_held:
            print(submit_held(root, state))
        elif args.release_registered_job is not None:
            release_registered(root, state, positive_job(args.release_registered_job))
        elif args.cancel_owned_held_job is not None:
            cancel_owned_held(root, state, positive_job(args.cancel_owned_held_job))
        else:
            archive_owned(root, state)
    else:
        if (not args.root or not args.parent or not args.concurrent_performance_root or args.compute_evidence
                or args.compute_preflight or args.prepare_only and args.once):
            parser.error("Require isolated --root, real --parent and explicit --concurrent-performance-root")
        repo = args.repo
        if repo is None:
            state_path = args.root / "campaign-state.json"
            repo = Path(read(state_path)["repo"]) if state_path.is_file() else Path(__file__).resolve().parents[2]
        run(args.root, args.parent, args.concurrent_performance_root, repo, args.prepare_only, args.once, args.kernel_activity)


if __name__ == "__main__":
    main()
