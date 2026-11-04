import argparse
import hashlib
import importlib
import json
import shlex
import subprocess
import sys
from pathlib import Path


LAUNCHES = {
    "search": ("public-workload-search.sbatch", "efficient-public-search.sbatch", 1),
    "poisson": ("public-poisson-search.sbatch", "efficient-public-poisson.sbatch", 1),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_phase(root, prompt_bank):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    if config.get("workaware_optimizer") == "robust":
        from verify_efficient_trace import verify

        return verify(root, prompt_bank)
    if config.get("workaware_optimizer") != "point" or (root / "solver-efficiency-source.json").exists():
        raise ValueError("Require declared point baseline or SHA-certified efficient robust phase")
    from verify_trace import verify

    return verify(root, prompt_bank)


def launch_payload(repo, output, role):
    output = Path(output)
    directory = Path(repo) / "experiments/dp_opt"
    source_name, wrapper_name, expected_calls = LAUNCHES[role]
    source, trace = directory / source_name, directory / "trace-benefit.sbatch"
    original_call = 'bash "$repo/experiments/dp_opt/trace-benefit.sbatch"'
    if source.read_text().count(original_call) != expected_calls or trace.read_text().count('/live_trace.py"') != 1:
        raise ValueError("Canonical launch structure changed; diagnose before generating an opt-in launch")
    trace_output = output / "efficient-trace-launch.sh"
    script_output = output / "efficient-public-launch.sh"
    replacement = ('if test "$DP_OPT_WORKAWARE_OPTIMIZER" = robust; then bash '
                   + shlex.quote(str(trace_output)) + '; else ' + original_call + '; fi')
    script = source.read_text().replace(original_call, replacement)
    entry = 'python "$repo/experiments/dp_opt/efficient_public_launch.py"'
    script = script.replace('python "$repo/experiments/dp_opt/verify_trace.py"', entry + ' --verify-phase')
    module_name = "public_workload_search" if role == "search" else "public_poisson_followup"
    original_entry = f'python "$repo/experiments/dp_opt/{module_name}.py"'
    if original_entry not in script:
        raise ValueError("Canonical independent analysis entry changed")
    script = script.replace(original_entry, entry + " --module " + role)
    generated = {
        trace_output: trace.read_text().replace('/live_trace.py"', '/efficient_workaware_trace.py"'),
        script_output: script,
    }
    receipt = {"role": role, "output_directory": str(output), "canonical_source_sha256": digest(source),
               "canonical_trace_sha256": digest(trace), "entry_sha256": digest(__file__),
               "wrapper_sha256": digest(directory / wrapper_name),
               "generated_sha256": {path.name: hashlib.sha256(content.encode()).hexdigest()
                                    for path, content in generated.items()},
               "original_models_graphs_budget_trace_tokens_and_objective_unchanged": True}
    return generated, receipt


def make_launch(repo, output, role):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    generated, receipt = launch_payload(repo, output, role)
    if any(path.exists() for path in generated) or (output / "efficient-public-launch-source.json").exists():
        raise RuntimeError("Preserve an existing opt-in launch; never overwrite its executed source")
    for path, content in generated.items():
        with path.open("x") as handle:
            handle.write(content)
        subprocess.run(["bash", "-n", str(path)], check=True)
    with (output / "efficient-public-launch-source.json").open("x") as handle:
        handle.write(json.dumps(receipt, indent=2) + "\n")
    return output / "efficient-public-launch.sh"


def verify_launch(root, repo, role):
    root = Path(root)
    directory = root / "launch-sources"
    receipt = json.loads((directory / "efficient-public-launch-source.json").read_text())
    remote_output = Path(receipt["output_directory"])
    if (not remote_output.is_absolute() or remote_output.name != "launch-sources"
            or remote_output.parent.name != root.name or not root.name.startswith("live-trace-")):
        raise ValueError("Executed public launch must belong to this immutable job")
    generated, expected = launch_payload(repo, remote_output, role)
    if receipt != expected:
        raise ValueError("Public launch source receipt differs from the frozen opt-in transformation")
    for path, content in generated.items():
        if (directory / path.name).read_bytes() != content.encode():
            raise ValueError("Executed public launch differs from its independent reconstruction")
    return receipt


def main():
    if "--module" in sys.argv:
        index = sys.argv.index("--module")
        role = sys.argv[index + 1]
        del sys.argv[index:index + 2]
        if "--root" not in sys.argv:
            raise ValueError("Independent public action requires its immutable job root")
        verify_launch(Path(sys.argv[sys.argv.index("--root") + 1]), Path(__file__).resolve().parents[2], role)
        module = importlib.import_module({"search": "public_workload_search", "poisson": "public_poisson_followup"}[role])
        original = module.verify
        module.verify = verify_phase
        try:
            module.main()
        finally:
            module.verify = original
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-launch", choices=LAUNCHES)
    parser.add_argument("--verify-phase", type=Path)
    parser.add_argument("--prompt-bank", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.make_launch is None) == (args.verify_phase is None):
        parser.error("Choose launch generation or strict phase verification")
    if args.make_launch is not None:
        print(make_launch(Path(__file__).resolve().parents[2], args.output, args.make_launch))
    else:
        if args.prompt_bank is None:
            parser.error("Strict phase verification requires the frozen prompt bank")
        result = verify_phase(args.verify_phase, args.prompt_bank)
        with args.output.open("x") as handle:
            handle.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
