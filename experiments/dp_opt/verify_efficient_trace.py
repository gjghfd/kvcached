import argparse
import hashlib
import json
from pathlib import Path

import workaware_robust_adapter
from verify_solver_efficiency import verify_plan
from verify_trace import verify as verify_trace


SOURCES = ("efficient_workaware_trace.py", "solver_efficiency.py", "grouped_robust_uncertainty.py",
           "prune_headroom_scenarios.py", "verify_solver_efficiency.py")


def verify(root, prompt_bank):
    root = Path(root)
    manifest = json.loads((root / "solver-efficiency-source.json").read_text())
    if (set(manifest.get("source_sha256", {})) != set(SOURCES)
            or manifest.get("objective_and_uncertainty_unchanged") is not True
            or manifest.get("original_strict_greedy_token_checks_unchanged") is not True):
        raise ValueError("Require a complete explicit equivalent-solver source manifest")
    for name, expected in manifest["source_sha256"].items():
        if hashlib.sha256((root / (name + ".executed")).read_bytes()).hexdigest() != expected:
            raise ValueError("Executed equivalent-solver source differs from its recorded SHA")
    config = json.loads((root / "config.json").read_text())
    if config.get("workaware_optimizer") != "robust":
        raise ValueError("Equivalent-solver trace must explicitly use robust workaware optimization")
    original = workaware_robust_adapter.verify_raw_plan
    workaware_robust_adapter.verify_raw_plan = verify_plan
    try:
        result = verify_trace(root, prompt_bank)
    finally:
        workaware_robust_adapter.verify_raw_plan = original
    return {**result, "equivalent_solver_sources_SHA_verified": manifest["source_sha256"],
            "original_full_serial_and_paired_token_checks_preserved": True,
            "pruning_certificates_use_independent_physical_and_rational_replay": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, args.prompt_bank)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
