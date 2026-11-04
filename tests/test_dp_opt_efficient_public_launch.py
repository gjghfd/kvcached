import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import efficient_public_continuation as continuation
import efficient_public_launch as launch


def stage_files(destination, files):
    repo = Path(__file__).resolve().parents[1]
    for name in files:
        source, target = repo / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def isolated_command(script, arguments, directory):
    environment = {name: value for name, value in os.environ.items()
                   if name not in ("PYTHONPATH", "ENABLE_KVCACHED", "KVCACHED_AUTOPATCH", "DP_OPT_LIVE_PLACEMENT")}
    return subprocess.run([sys.executable, "-E", str(script), *arguments], cwd=directory,
                          env=environment, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("role", ["search", "poisson"])
def test_launch_generation_needs_only_stdlib_and_launch_sources(tmp_path, role):
    staged = tmp_path / "staged"
    canonical, wrapper, count = launch.LAUNCHES[role]
    stage_files(staged, ("experiments/dp_opt/efficient_public_launch.py",
                         "experiments/dp_opt/" + canonical, "experiments/dp_opt/" + wrapper,
                         "experiments/dp_opt/trace-benefit.sbatch"))
    script = staged / "experiments/dp_opt/efficient_public_launch.py"
    output = staged / "live-trace-274999/launch-sources"
    result = isolated_command(script, ["--make-launch", role, "--output", str(output)], tmp_path)
    assert result.returncode == 0, result.stderr
    generated = Path(result.stdout.strip())
    assert generated == output / "efficient-public-launch.sh"
    assert generated.read_text().count('if test "$DP_OPT_WORKAWARE_OPTIMIZER" = robust;') == count
    assert "efficient_public_continuation.py" not in generated.read_text()
    receipt = json.loads((output / "efficient-public-launch-source.json").read_text())
    assert receipt["entry_sha256"] == launch.digest(script)
    assert receipt["original_models_graphs_budget_trace_tokens_and_objective_unchanged"] is True


@pytest.mark.parametrize("role", ["search", "poisson"])
def test_actual_sync_set_can_import_its_compute_analysis_without_local_checkout(tmp_path, role):
    staged = tmp_path / "staged"
    module = continuation.public_workload_campaign if role == "search" else continuation.public_poisson_followup
    stage_files(staged, tuple(dict.fromkeys((*module.SYNC_FILES, *continuation.EXTRA_FILES))))
    script = staged / "experiments/dp_opt/efficient_public_launch.py"
    root = staged / "live-trace-274999"
    result = isolated_command(script, ["--make-launch", role, "--output", str(root / "launch-sources")], tmp_path)
    assert result.returncode == 0, result.stderr
    result = isolated_command(script, ["--module", role, "--root", str(root), "--help"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "--action" in result.stdout
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "import efficient_workaware_trace, verify_efficient_trace, verify_queue_fairness")
    result = isolated_command("-c", [code, str(staged / "experiments/dp_opt")], tmp_path)
    assert result.returncode == 0, result.stderr


def test_launch_import_does_not_load_local_coordinators_or_gpu_modules(tmp_path):
    staged = tmp_path / "staged"
    stage_files(staged, ("experiments/dp_opt/efficient_public_launch.py",))
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import efficient_public_launch; "
            "assert not any(name in sys.modules for name in "
            "('public_poisson_followup','public_workload_campaign','stress_campaign','torch','vllm'))")
    result = subprocess.run([sys.executable, "-E", "-c", code, str(staged / "experiments/dp_opt")],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("arguments", [[], ["--verify-phase", "phase"],
                                      ["--verify-phase", "phase", "--make-launch", "search"]])
def test_compute_entry_rejects_ambiguous_or_incomplete_action(tmp_path, arguments):
    script = Path(launch.__file__)
    result = isolated_command(script, [*arguments, "--output", str(tmp_path / "output")], tmp_path)
    assert result.returncode != 0
    assert not (tmp_path / "output").exists()
