"""Check a time-multiplexing (*_tm) trial: holds, Code solo speed, Conv progress while held, correctness."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trial", type=Path, help=".../out/trace/<policy>")
    parser.add_argument("--trace", type=Path, help="frozen trace json (exact-length check)")
    args = parser.parse_args()
    root = args.trial
    origin = json.loads((root / "workload.json").read_text())["origin"]
    requests = json.loads((root / "requests.json").read_text())
    holds = json.loads((root / "holds.json").read_text()) if (root / "holds.json").exists() else []
    migrations = json.loads((root / "migrations.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    print(f"{root.name}: status {summary['status']} requests {summary['requests']}/{summary['expected']} "
          f"moves {summary['moves']} holds {len(holds)} held_s {summary.get('held_s', 0):.1f}")
    spans = [(hold["paused_at"] - origin, hold["resumed_at"] - origin) for hold in holds]
    for hold, (start, end) in zip(holds, spans):
        reach = hold.get("pause_reach_s")
        print(f"  hold t=[{start:7.1f}, {end:7.1f}] {end - start:5.1f}s conv_at_pause {hold.get('requests_at_pause')}"
              f" from_txn {bool(hold.get('from_transaction'))} reach_s {max(reach) if reach else float('nan'):.2f}")
    for request in sorted((r for r in requests if r["model"] == 0), key=lambda r: r["arrival_s"]):
        reserved, first, done = (request[key] - origin for key in ("reserved_at", "first_token_at", "completed_at"))
        held = sum(max(0.0, min(first, e) - max(reserved, s)) for s, e in spans)
        print(f"  code {request['trace_request_id']} arr {request['arrival_s']:6.0f} queue {reserved - request['arrival_s'] - 0:6.1f}"
              f" prefill {first - reserved:5.1f}s (Conv held {held:5.1f}s of it) decode {1e3 * (done - first) / 63:6.1f}ms/token"
              f" ttft {first - request['arrival_s']:6.1f}s path {request.get('admission_path')}")
    leaked = 0
    for request in requests:
        if request["model"] != 1:
            continue
        cadence = [(t - origin, count) for t, count in request["cadence"]]
        for (t0, c0), (t1, c1) in zip(cadence, cadence[1:]):
            for s, e in spans:
                if s + 1.0 <= t0 and t1 <= e - 1.0:
                    leaked += c1 - c0
    print(f"  Conv tokens produced strictly inside holds: {leaked}")
    print(f"  migrations {len(migrations)} bitwise_verified {sum(bool(m.get('bitwise_verified')) for m in migrations)}"
          f" move_s median {statistics.median(m['elapsed_s'] for m in migrations) if migrations else float('nan'):.2f}")
    if args.trace:
        expected = {r["request_id"]: r["output_tokens"] for r in json.loads(args.trace.read_text())["requests"]}
        bad = [r["trace_request_id"] for r in requests if expected.get(r["trace_request_id"]) != r["output_tokens"]]
        print(f"  exact output lengths: {len(requests) - len(bad)}/{len(requests)} ok; missing "
              f"{len(set(expected) - {r['trace_request_id'] for r in requests})}")
    gaps = sorted(r["max_token_gap_s"] for r in requests if r["model"] == 1)
    print(f"  Conv max token gap p50 {gaps[len(gaps) // 2]:.1f}s max {gaps[-1]:.1f}s")


if __name__ == "__main__":
    main()
