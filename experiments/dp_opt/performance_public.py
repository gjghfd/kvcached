import argparse
import importlib
import json
from contextlib import contextmanager
from pathlib import Path

from performance_numerics import digest, verify


@contextmanager
def adapted(module):
    original = module.verify
    module.verify = verify
    try:
        yield
    finally:
        module.verify = original


def make_trace_launch(repo, output):
    source = Path(repo) / "experiments/dp_opt/trace-benefit.sbatch"
    text = source.read_text()
    if text.count('/live_trace.py"') != 1:
        raise ValueError("Canonical trace entry changed")
    generated = text.replace('/live_trace.py"', '/performance_trace.py"')
    output = Path(output)
    with output.open("x") as handle:
        handle.write(generated)
    receipt = {"canonical_trace_sha256": digest(source), "generated_sha256": digest(output),
               "only_entry_changed": True, "greedy_gate_observed_not_fatal": True,
               "model_graph_budget_request_arrival_and_migration_configuration_unchanged": True}
    with output.with_suffix(".source.json").open("x") as handle:
        handle.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-phase", type=Path)
    parser.add_argument("--prompt-bank", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-drained-greedy-failure", action="store_true")
    parser.add_argument("--make-trace-launch", type=Path)
    parser.add_argument("--module", choices=("search", "poisson"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--action")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.verify_phase, args.make_trace_launch, args.module)) != 1:
        parser.error("Choose exactly one explicit performance action")
    if args.make_trace_launch:
        result = make_trace_launch(Path(__file__).resolve().parents[2], args.make_trace_launch)
    elif args.verify_phase:
        result = verify(args.verify_phase, args.prompt_bank, args.allow_drained_greedy_failure)
    else:
        module = importlib.import_module("public_workload_search" if args.module == "search" else "public_poisson_followup")
        if args.root is None or args.action is None or args.allow_drained_greedy_failure:
            parser.error("Public analysis requires an immutable root/action; no legacy failure bypass")
        actions = ({"freeze-calibration": module.freeze_calibration, "analyze": module.analyze}
                   if args.module == "search" else {"analyze": module.analyze})
        if args.action not in actions:
            parser.error("Unsupported performance public analysis action")
        with adapted(module):
            result = actions[args.action](args.root)
    if args.output:
        with args.output.open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
