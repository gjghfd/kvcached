"""Merge trials from several short jobs into one trace root (symlinks) so report_main.py compares them together.

Duplicate policy names get an @<job> suffix (e.g. ca_full@283933); the first occurrence keeps the plain name.
Also prints each job's node-health probe so cross-job comparisons can be qualified.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", type=Path, nargs="+", help="evidence/xplace-<tag>/job-<id> directories")
    parser.add_argument("--output", type=Path, required=True, help="new combined trace root")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seen = set()
    for job in args.jobs:
        phases = json.loads((job / "out" / "phases.json").read_text())["phases"]
        probe = next((p for p in phases if p["phase"] == "probe"), {})
        node = (job / "topology.txt").exists()
        trials = sorted(path for path in (job / "out" / "trace").iterdir() if (path / "summary.json").exists())
        for trial in trials:
            summary = json.loads((trial / "summary.json").read_text())
            name = trial.name if trial.name not in seen else f"{trial.name}@{job.name.split('-')[-1]}"
            seen.add(trial.name)
            (args.output / name).symlink_to(trial.resolve())
            print(f"{job.name} {trial.name:12s} -> {name:22s} status {summary['status']:6s} "
                  f"probe {probe.get('step_ms', float('nan')):.1f} ms ({probe.get('status', 'n/a')})"
                  f"{'' if node else ' [no topology]'}")


if __name__ == "__main__":
    main()
