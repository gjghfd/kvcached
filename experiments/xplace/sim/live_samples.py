"""Extract live step-time samples from a trace run (pilot or main) for simulator calibration.

Per 1s tick: per-rank Conv count/context (from request ranks + migrations + cadence), Code state, pause state,
and the measured Conv step time (median per-token time of cadence segments covering the tick; all Conv
requests share one step clock because DP ranks step together).
"""
import bisect
import json
import math
import statistics
from pathlib import Path

CAPTURE = (1, 8, 16, 32, 64, 128, 256)


def bucket(count):
    return next(size for size in CAPTURE if count <= size) if count > 0 else 0


def load_run(root):
    root = Path(root)
    workload = json.loads((root / "workload.json").read_text())
    requests = json.loads((root / "requests.json").read_text())
    transactions = json.loads((root / "transactions.json").read_text())
    migrations = json.loads((root / "migrations.json").read_text())
    return workload, requests, transactions, migrations


def load_holds(root):
    path = Path(root) / "holds.json"
    return json.loads(path.read_text()) if path.exists() else []


def pause_intervals(transactions, origin, model=1, holds=()):
    spans = []
    for item in transactions:
        if model not in item.get("pause_models", (0, 1)):
            continue
        receipts = (item.get("pause_receipts") or [None, None])[model] or []
        reached = [receipt["reached_at"] for receipt in receipts if receipt and receipt.get("reached_at")]
        start = min(reached) if reached else item["started_at"] + item.get("pause_wait_s", 0.0)
        spans.append((start - origin, item["started_at"] + item["control_s"] - origin))
    spans += [(hold["paused_at"] - origin, hold["resumed_at"] - origin) for hold in holds if hold.get("model") == model]
    return spans


class Timeline:
    """Rank and generated-token count of every request as a function of trace-relative time."""

    def __init__(self, requests, migrations, origin):
        self.origin = origin
        moves = {}
        for event in migrations:
            # vLLM appends "-<8 hex>" to engine-side request ids
            moves.setdefault(event["request_id"].rsplit("-", 1)[0], []).append(event)
        self.items = []
        for request in requests:
            history = sorted(moves.get(request["request_id"], []), key=lambda event: event["started_at"])
            ranks = [(request["reserved_at"] - origin, request["rank_initial"])]
            ranks += [(event["started_at"] - origin, event["destination"]) for event in history]
            cadence = [(t - origin, count) for t, count in request["cadence"]]
            self.items.append({"request": request, "model": request["model"], "ranks": ranks, "cadence": cadence,
                               "start": request["reserved_at"] - origin, "end": request["completed_at"] - origin,
                               "first": request["first_token_at"] - origin})

    @staticmethod
    def rank_at(item, t):
        rank = item["ranks"][0][1]
        for when, value in item["ranks"]:
            if when <= t:
                rank = value
        return rank

    @staticmethod
    def tokens_at(item, t):
        cadence = item["cadence"]
        if t < cadence[0][0]:
            return 0
        times = [point[0] for point in cadence]
        index = bisect.bisect_right(times, t) - 1
        if index >= len(cadence) - 1:
            return cadence[-1][1]
        (t0, c0), (t1, c1) = cadence[index], cadence[index + 1]
        return c0 + (c1 - c0) * (t - t0) / (t1 - t0)

    def layout(self, model, t, num_ranks=4):
        counts, contexts = [0] * num_ranks, [0.0] * num_ranks
        for item in self.items:
            if item["model"] != model or not item["start"] <= t < item["end"]:
                continue
            rank = self.rank_at(item, t)
            counts[rank] += 1
            prompt = item["request"]["prompt_tokens"]
            contexts[rank] += prompt + self.tokens_at(item, t) if t >= item["first"] else 0.0
        return counts, contexts


def code_phases(requests, origin):
    prefill, decode = [], []
    for request in requests:
        if request["model"] != 0:
            continue
        prefill.append((request["reserved_at"] - origin, request["first_token_at"] - origin, request))
        decode.append((request["first_token_at"] - origin, request["completed_at"] - origin, request))
    return prefill, decode


def span_state(t0, t1, prefill, decode, paused):
    """State of [t0, t1] if one state covers it entirely; 'transition' if it straddles a boundary."""
    def overlap(spans):
        return sum(max(0.0, min(t1, e) - max(t0, s)) for s, e, *_ in spans)

    def covered(spans):
        points = [t0 + (t1 - t0) * k / 8 for k in range(9)]
        return all(any(s <= p <= e for s, e, *_ in spans) for p in points)

    if overlap(paused) > 0:
        return "paused"
    if overlap(prefill) == 0 and overlap(decode) == 0:
        return "none"
    if covered(prefill):
        return "prefill"
    if covered(decode) and overlap(prefill) == 0:
        return "decode"
    return "transition"


def conv_step_samples(root, window=(0.0, math.inf), tick=1.0):
    """One sample per tick with the measured Conv step and layout features."""
    workload, requests, transactions, migrations = load_run(root)
    origin = workload["origin"]
    timeline = Timeline(requests, migrations, origin)
    prefill, decode = code_phases(requests, origin)
    paused = pause_intervals(transactions, origin, holds=load_holds(root))
    segments = []
    for item in timeline.items:
        if item["model"] != 1:
            continue
        for (t0, c0), (t1, c1) in zip(item["cadence"], item["cadence"][1:]):
            if c1 > c0:
                segments.append((t0, t1, (t1 - t0) / (c1 - c0)))
    segments.sort()
    starts = [segment[0] for segment in segments]
    end = min(window[1], max(item["end"] for item in timeline.items))
    samples = []
    t = window[0]
    while t < end:
        lo = bisect.bisect_left(starts, t - 60.0)
        covering = [segment for segment in segments[lo:bisect.bisect_right(starts, t)] if segment[0] <= t < segment[1]]
        if covering:
            span = (min(segment[0] for segment in covering), max(segment[1] for segment in covering))
            inside = lambda spans: sum(s <= t < e for s, e, *_ in spans)
            counts, contexts = timeline.layout(1, t)
            samples.append({"t": t, "step_s": statistics.median(segment[2] for segment in covering),
                            "state": span_state(*span, prefill, decode, paused),
                            "code_prefill": inside(prefill), "code_decode": inside(decode),
                            "counts": counts, "max_count": max(counts), "total": sum(counts),
                            "bucket": bucket(max(counts)), "max_context": max(contexts),
                            "full_bucket": all(count == bucket(max(counts)) for count in counts)})
        t += tick
    return samples


def code_samples(root):
    """Per-Code prefill duration and decode TPOT with the concurrent Conv layout and other Codes."""
    workload, requests, transactions, migrations = load_run(root)
    origin = workload["origin"]
    timeline = Timeline(requests, migrations, origin)
    prefill, decode = code_phases(requests, origin)
    rows = []
    for start, end, request in prefill:
        middle = (start + end) / 2
        counts, _ = timeline.layout(1, middle)
        others_prefill = sum(s < end and e > start for s, e, r in prefill if r is not request)
        others_decode = sum(s < end and e > start for s, e, r in decode if r is not request)
        first, done = request["first_token_at"] - origin, request["completed_at"] - origin
        overlap_prefill = sum(max(0.0, min(done, e) - max(first, s)) for s, e, r in prefill if r is not request)
        rows.append({"id": request["trace_request_id"], "arrival_s": request["arrival_s"],
                     "queue_s": request["reserved_at"] - request["arrived_at"], "prefill_s": end - start,
                     "prompt": request["prompt_tokens"], "conv_counts_mid": counts, "conv_total_mid": sum(counts),
                     "others_prefill": others_prefill, "others_decode": others_decode,
                     "decode_tpot_s": (done - first) / max(1, request["output_tokens"] - 1),
                     "decode_overlap_other_prefill_s": overlap_prefill})
    return rows
