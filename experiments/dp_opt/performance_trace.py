import hashlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import live_trace
from performance_numerics import runtime_observation
from solver_efficiency import exact_solver_efficiency


def main():
    if os.environ.get("DP_OPT_PERFORMANCE_NUMERICS") != "observe":
        raise ValueError("Performance numerical observation requires explicit opt-in")
    root = Path(sys.argv[sys.argv.index("--output") + 1])
    root.mkdir(parents=True, exist_ok=True)
    robust = ("--workaware-optimizer" in sys.argv and
              sys.argv[sys.argv.index("--workaware-optimizer") + 1] == "robust")
    sources = ("performance_trace.py", "performance_numerics.py", "solver_efficiency.py",
               "grouped_robust_uncertainty.py", "prune_headroom_scenarios.py", "verify_solver_efficiency.py")
    hashes = {}
    for name in sources:
        content = Path(__file__).with_name(name).read_bytes()
        with (root / (name + ".executed")).open("xb") as output:
            output.write(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    with (root / "performance-source.json").open("x") as output:
        output.write(json.dumps({"source_sha256": hashes, "equivalent_robust_solver_enabled": robust,
                                "original_strict_greedy_token_checks_unchanged": False,
                                "greedy_divergence_observed_not_fatal": True,
                                "objective_and_uncertainty_unchanged": True}, indent=2) + "\n")
    with runtime_observation(root), exact_solver_efficiency() if robust else nullcontext():
        live_trace.main()


if __name__ == "__main__":
    main()
