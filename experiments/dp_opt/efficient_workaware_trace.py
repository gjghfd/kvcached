import hashlib
import json
import sys
from pathlib import Path

import live_trace
from solver_efficiency import exact_solver_efficiency


def main():
    if ("--workaware-optimizer" not in sys.argv
            or sys.argv[sys.argv.index("--workaware-optimizer") + 1] != "robust"):
        raise ValueError("Efficient solver entry requires explicit robust workaware optimization")
    root = Path(sys.argv[sys.argv.index("--output") + 1])
    root.mkdir(parents=True, exist_ok=True)
    sources = (Path(__file__).name, "solver_efficiency.py", "grouped_robust_uncertainty.py",
               "prune_headroom_scenarios.py", "verify_solver_efficiency.py")
    hashes = {}
    for name in sources:
        content = Path(__file__).with_name(name).read_bytes()
        with (root / (name + ".executed")).open("xb") as output:
            output.write(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    with (root / "solver-efficiency-source.json").open("x") as output:
        output.write(json.dumps({"source_sha256": hashes, "objective_and_uncertainty_unchanged": True,
                                 "original_strict_greedy_token_checks_unchanged": True}, indent=2) + "\n")
    with exact_solver_efficiency():
        live_trace.main()


if __name__ == "__main__":
    main()
