import argparse
import ast
import asyncio
import builtins
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "experiments/dp_opt"
POLICIES = ["no_move", "own_model", "kv_only", "no_move_fair", "random_no_return",
            "kv_only_headroom", "kv_only_workaware"]
CAPTURE_SIZES = [1, 8, 16, 32, 64, 128, 256]


def source_namespace(filename, functions=(), assignments=()):
    source = SOURCE_ROOT / filename
    tree = ast.parse(source.read_text(), filename=str(source))
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in functions
                or isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in assignments for target in node.targets)]
    namespace = {"argparse": argparse, "asyncio": asyncio, "math": math, "os": os, "Path": Path}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def append_capture(kind, **values):
    with Path(os.environ["CPU_CLI_CAPTURE"]).open("a") as output:
        output.write(json.dumps({"kind": kind, **values}) + "\n")


def captured_engine_arguments(arguments):
    namespace = source_namespace("live_benchmark.py", assignments=("DEFAULT_MAX_MODEL_LEN",))

    def isolated_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm.engine.arg_utils":
            return SimpleNamespace(AsyncEngineArgs=lambda **values: values)
        raise AssertionError(f"Inference import is forbidden in CPU CLI capture: {name}")

    namespace["__builtins__"] = {**vars(builtins), "__import__": isolated_import}
    source = SOURCE_ROOT / "live_benchmark.py"
    definition = next(node for node in ast.parse(source.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == "engine_arguments")
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["engine_arguments"](
        arguments.checkpoint, arguments.port, arguments.cache_bytes, arguments.max_model_len)


def capture_live_cli(arguments):
    defaults = source_namespace("live_benchmark.py", assignments=("DEFAULT_MAX_MODEL_LEN", "DEFAULT_KV_CACHE_BYTES"))
    defaults.update(source_namespace("queue_fairness.py", assignments=("DEFAULT_QUEUE_THRESHOLD_S",)))
    namespace = source_namespace("live_trace.py", functions=("main",), assignments=("POLICIES", "TRACE_POLICIES"))
    namespace.update({name: value for name, value in defaults.items() if name.startswith("DEFAULT_")})

    async def capture(parsed):
        payload = {name: str(value) if isinstance(value, Path) else value for name, value in vars(parsed).items()}
        environment = {name: value for name, value in os.environ.items()
                       if name.startswith("DP_OPT_") or name == "KVCACHED_GPU_UTILIZATION"}
        append_capture("live", phase=parsed.output.name, arguments=payload,
                       environment=environment, engine=captured_engine_arguments(parsed))
        (parsed.output / "progress.json").write_text("{}")

    namespace["run"] = capture
    sys.argv = [str(SOURCE_ROOT / "live_trace.py"), *arguments]
    namespace["main"]()


def capture_bundle_cli(arguments):
    source = SOURCE_ROOT / "workaware_profile_bundle.py"
    definition = next(node for node in ast.parse(source.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == "main")
    statements = []
    for statement in definition.body:
        if isinstance(statement, ast.ImportFrom):
            continue
        statements.append(statement)
        if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "args" for target in statement.targets):
            break
    sys.argv = [str(source), *arguments]
    namespace = {"argparse": argparse, "Path": Path}
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(source), "exec"), namespace)
    parsed = namespace["args"]
    append_capture("bundle", legacy=str(parsed.legacy), supplement=str(parsed.supplement),
                   output=str(parsed.output), anchor_supplement=parsed.anchor_supplement)
    if os.environ.get("CPU_CLI_FAIL_STAGE") == "bundle":
        raise SystemExit(17)
    parsed.output.mkdir()
    (parsed.output / "fit.json").write_text(json.dumps({"scope": "CPU_only_fake_composite_not_GPU_evidence"}))


def dispatch_python(arguments):
    if arguments[0] == "-c":
        if "from qwen_common import checkpoint_path" in arguments[1]:
            print("CPU-only-unused-checkpoint")
        elif "from workaware_evidence import runtime_files" in arguments[1]:
            namespace = source_namespace("workaware_evidence.py", functions=("runtime_files",), assignments=(
                "RUNTIME_FILES", "PROFILE_RUNTIME_FILES", "RANDOM_RUNTIME_FILES", "PHASE_RUNTIME_FILES",
                "PHASE_RANDOM_RUNTIME_FILES", "ROBUST_RUNTIME_FILES"))
            print("\n".join(namespace["runtime_files"]()))
        else:
            sys.argv = ["-c", *arguments[2:]]
            exec(arguments[1], {"__name__": "__main__"})
        return
    name = Path(arguments[0]).name
    if name == "live_trace.py":
        capture_live_cli(arguments[1:])
    elif name == "workaware_profile_bundle.py":
        capture_bundle_cli(arguments[1:])
    elif name == "public_population_calibration.py":
        append_capture("verify_composite", arguments=arguments[1:])
        if os.environ.get("CPU_CLI_FAIL_STAGE") == "verify_composite":
            raise SystemExit(19)
        print(json.dumps({"scope": "CPU_only_mocked_raw_verification"}))
    elif name == "public_poisson_followup.py":
        action = arguments[arguments.index("--action") + 1]
        root = Path(arguments[arguments.index("--root") + 1])
        append_capture("followup", action=action)
        if action == "select":
            (root / "heldout-trace.json").write_text(json.dumps({
                "seed": 2026103105, "model_rates_rps": [.0005, .055], "requests": []}))
        print("{}")
    elif name == "verify_trace.py":
        append_capture("verify_trace", phase=Path(arguments[1]).name)
        Path(arguments[arguments.index("--output") + 1]).write_text("{}")
    else:
        raise AssertionError(f"Unexpected Python command: {arguments}")


def run_campaign(root, fail_stage=None):
    home, data, tools, upload = (root / name for name in ("home", "data", "tools", "upload"))
    for directory in (home, data / "state", tools, upload / "frozen/inputs"):
        directory.mkdir(parents=True)
    common = home / "aegaeon/paracloud/scripts/common.sh"
    common.parent.mkdir(parents=True)
    common.write_text("require_compute() { :; }\nload_toolchains() { "
                      f"export ENV_PREFIX={shlex.quote(str(root / 'env'))} "
                      f"DEPLOY_ROOT={shlex.quote(str(root / 'deploy'))}; }}\n")
    driver = tools / "cpu-tool"
    driver.write_text(f"#!{sys.executable} -S\n" + """
import os
import sys
from pathlib import Path

name, arguments = Path(sys.argv[0]).name, sys.argv[1:]
if name == "sha256sum":
    import hashlib
    for filename in arguments:
        print(hashlib.sha256(Path(filename).read_bytes()).hexdigest(), "  ", filename, sep="")
elif name == "nvidia-smi":
    if any("memory.free" in argument for argument in arguments):
        print("GPU-CPU-0, A800 CPU fixture, 81920, 80000\\nGPU-CPU-1, A800 CPU fixture, 81920, 80000")
    else:
        print("CPU-only GPU command fixture")
elif name == "date":
    print("2026-10-31T00:00:00+00:00")
elif name == "mktemp":
    import tempfile
    print(tempfile.mkdtemp(prefix="transfer-", dir=os.environ["CPU_CLI_TEMP"]))
elif name == "python":
    if arguments[:2] == ["-m", "pytest"]:
        sys.exit(0)
    import runpy
    from types import SimpleNamespace
    sys.modules["pytest"] = SimpleNamespace(
        fixture=lambda **kwargs: lambda function: function,
        mark=SimpleNamespace(parametrize=lambda *args, **kwargs: lambda function: function))
    namespace = runpy.run_path(os.environ["CPU_CLI_TEST_SOURCE"])
    namespace["dispatch_python"](arguments)
else:
    raise AssertionError(f"Forbidden external command in CPU CLI test: {name}")
""")
    driver.chmod(0o755)
    for name in ("python", "sha256sum", "nvidia-smi", "date", "mktemp", "ssh", "sbatch", "squeue"):
        (tools / name).symlink_to(driver)
    parent = data / "state/live-trace-70001"
    parent.mkdir()
    parent_payload = b'{"scope":"CPU_only_fake_parent_not_GPU_evidence"}\n'
    (parent / "public-context-fit.json").write_bytes(parent_payload)
    (parent / "exit-status").write_text("0\n")
    (data / "state/sparse-fit-231467.json").write_text("{}")
    frozen = upload / "frozen"
    (frozen / "inputs/prompt-bank.json").write_text("{}")
    population = source_namespace("public_population_calibration.py", functions=("calibration_plan",),
                                  assignments=("POPULATIONS", "PROMPT_LIMITS"))
    (frozen / "population-calibration-plan.json").write_text(json.dumps(population["calibration_plan"]()))
    (frozen / "evaluation-config.json").write_text(json.dumps({"grid": [{}, {}]}))
    for ordinal, rates in enumerate(([.0005, .04], [.001, .045])):
        (frozen / f"explore-{ordinal}.json").write_text(json.dumps({
            "seed": 2026103104, "model_rates_rps": rates, "requests": []}))
    shutil.copy2(frozen / "explore-0.json", frozen / "trace.json")
    capture = root / "capture.jsonl"
    environment = {name: value for name, value in os.environ.items() if not name.startswith((
        "DP_OPT_", "KVCACHED_", "VLLM_", "ENABLE_KVCACHED", "CPU_CLI_"))}
    environment.update({"HOME": str(home), "PATH": str(tools) + os.pathsep + os.environ["PATH"],
                        "PYTHONDONTWRITEBYTECODE": "1", "KVCACHED_DATA_ROOT": str(data),
                        "KVCACHED_REPO_ROOT": str(SOURCE_ROOT.parents[1]), "SLURM_JOB_ID": "70002",
                        "CPU_CLI_TEST_SOURCE": str(Path(__file__).resolve()), "CPU_CLI_CAPTURE": str(capture),
                        "CPU_CLI_TEMP": str(root), "DP_OPT_PUBLIC_FIT_JOB": "70001",
                        "DP_OPT_PUBLIC_FIT_SHA256": hashlib.sha256(parent_payload).hexdigest(),
                        "DP_OPT_PUBLIC_POISSON_UPLOAD": str(upload), "DP_OPT_WORKAWARE_FIT": "inherited-fit",
                        "DP_OPT_WORKAWARE_FIT_SHA256": "inherited-sha", "DP_OPT_WORKAWARE_SOURCE_ROOT": "inherited-source",
                        "DP_OPT_WORKAWARE_CALIBRATION_PLAN": "inherited-plan", "DP_OPT_PAUSE_SYNC_INTERVAL": "7",
                        "KVCACHED_GPU_UTILIZATION": "0.5"})
    if fail_stage is not None:
        environment["CPU_CLI_FAIL_STAGE"] = fail_stage
    result = subprocess.run(["bash", str(SOURCE_ROOT / "public-poisson-search.sbatch")],
                            cwd=root, env=environment, text=True, capture_output=True, timeout=90)
    records = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
    return SimpleNamespace(result=result, records=records, evidence=data / "state/live-trace-70002", parent=parent)


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    result = run_campaign(tmp_path_factory.mktemp("public-population-cli"))
    assert result.result.returncode == 0, result.result.stdout + result.result.stderr
    return result


def phase_record(campaign, phase):
    matches = [record for record in campaign.records if record["kind"] == "live" and record["phase"] == phase]
    assert len(matches) == 1
    return matches[0]


def test_cost_real_cli_has_point_no_move_pilot_plan_and_no_fit(campaign):
    record = phase_record(campaign, "public-population-calibration")
    arguments, environment = record["arguments"], record["environment"]
    assert arguments["policies"] == ["no_move"]
    assert arguments["workaware_optimizer"] == "point"
    assert arguments["workaware_cadence_pilot"] is True
    plan = campaign.evidence / "frozen/population-calibration-plan.json"
    assert arguments["workaware_calibration_plan"] == str(plan)
    assert [entry["population"] for entry in json.loads(plan.read_text())["populations"]] == [5, 6, 7, 8, 9]
    for name in ("workaware_fit", "workaware_fit_sha256", "workaware_source_root", "workaware_rate_priors"):
        assert arguments[name] is None
    for name in ("DP_OPT_WORKAWARE_FIT", "DP_OPT_WORKAWARE_FIT_SHA256", "DP_OPT_WORKAWARE_SOURCE_ROOT"):
        assert name not in environment
    assert environment["DP_OPT_WORKAWARE_CADENCE_PILOT"] == "1"
    assert environment["DP_OPT_WORKAWARE_CALIBRATION_PLAN"] == str(plan)


@pytest.mark.parametrize("phase", ("explore-0", "explore-1", "heldout"))
def test_seven_policy_real_cli_uses_frozen_composite_after_cost(campaign, phase):
    record = phase_record(campaign, phase)
    arguments, environment = record["arguments"], record["environment"]
    composite = campaign.evidence / "public-population-profile"
    fit = composite / "fit.json"
    assert arguments["policies"] == POLICIES
    assert arguments["workaware_optimizer"] == "robust"
    assert arguments["workaware_cadence_pilot"] is False
    assert arguments["workaware_calibration_plan"] is None
    assert arguments["workaware_fit"] == str(fit)
    assert arguments["workaware_fit_sha256"] == hashlib.sha256(fit.read_bytes()).hexdigest()
    assert arguments["workaware_source_root"] == str(composite)
    assert arguments["workaware_fit"] != str(campaign.parent / "public-context-fit.json")
    assert "DP_OPT_WORKAWARE_CALIBRATION_PLAN" not in environment
    assert environment["DP_OPT_WORKAWARE_CADENCE_PILOT"] == "0"
    trace = json.loads(Path(arguments["trace"]).read_text())
    assert arguments["workaware_rate_priors"] == trace["model_rates_rps"]
    assert arguments["seed"] == trace["seed"]


@pytest.mark.parametrize("phase", ("public-population-calibration", "explore-0", "explore-1", "heldout"))
def test_real_cli_and_engine_builder_preserve_serving_defaults(campaign, phase):
    record = phase_record(campaign, phase)
    arguments, environment, engine = record["arguments"], record["environment"], record["engine"]
    assert arguments["kv_budget_mib"] == 10240
    assert arguments["max_model_len"] == 131072
    assert arguments["cache_bytes"] == 24 * 1024**3
    assert arguments["request_timeout_s"] == 18000
    assert arguments["deadline_s"] == 120
    assert arguments["rebalance_max_moves"] == 4
    assert arguments["rebalance_candidate_limit"] == 8
    assert environment["DP_OPT_PAUSE_SYNC_INTERVAL"] == "32"
    assert float(environment["KVCACHED_GPU_UTILIZATION"]) == .98
    assert environment["DP_OPT_KV_TRANSPORT"] == "dp_echo"
    assert engine["dtype"] == "bfloat16"
    assert engine["max_model_len"] == 131072
    assert engine["kv_cache_memory_bytes"] == 24 * 1024**3
    assert engine["compilation_config"]["cudagraph_capture_sizes"] == CAPTURE_SIZES
    assert engine["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    assert engine["enable_dbo"] is False


def test_composite_is_built_and_verified_before_any_seven_policy_cli(campaign):
    stages = [record for record in campaign.records if record["kind"] in ("live", "bundle", "verify_composite")]
    assert [record["kind"] for record in stages] == ["live", "bundle", "verify_composite", "live", "live", "live"]
    bundle = stages[1]
    assert bundle["legacy"] == str(campaign.parent / "public-context-calibration")
    assert bundle["supplement"] == str(campaign.evidence / "public-population-calibration")
    assert bundle["output"] == str(campaign.evidence / "public-population-profile")
    assert bundle["anchor_supplement"] is True
    assert [record["phase"] for record in campaign.records if record["kind"] == "verify_trace"] == [
        "explore-0", "explore-1", "heldout"]
    assert (campaign.evidence / "exit-status").read_text().strip() == "0"


@pytest.mark.parametrize("stage,exit_code", (("bundle", 17), ("verify_composite", 19)))
def test_failed_composite_stage_cannot_launch_robust_cli(tmp_path, stage, exit_code):
    campaign = run_campaign(tmp_path, fail_stage=stage)
    assert campaign.result.returncode == exit_code, campaign.result.stdout + campaign.result.stderr
    assert [record["phase"] for record in campaign.records if record["kind"] == "live"] == [
        "public-population-calibration"]
    assert (campaign.evidence / "exit-status").read_text().strip() == str(exit_code)
    assert not (campaign.evidence / "summary.json").exists()
