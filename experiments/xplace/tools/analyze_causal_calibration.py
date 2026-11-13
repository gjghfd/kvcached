import argparse
import bisect
from collections import Counter
import copy
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as element_tree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from causal_observation import causal_joint_state
from batch_link import valid_features


def digest(filename):
    return hashlib.sha256(Path(filename).read_bytes()).hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def nonnegative_integer(value):
    return type(value) is int and value >= 0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(filename):
    return json.loads(Path(filename).read_text())


def write(filename, value):
    with Path(filename).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def retention_status(report):
    errors = []
    result = dict(status="unavailable", errors=errors, collection_dropped=None, post_close_rejected=None,
                  raw_counts=copy.deepcopy(report.get("counts")), raw_retention_lossless=report.get("retention_lossless"))
    try:
        require(report.get("kind") == "causal_collection_v1", "wrong_collection_kind")
        protocol, events, counts = report["protocol"], report["events"], report["counts"]
        require(isinstance(events, list) and events, "missing_events")
        for name in ("attempted", "retained", "dropped", "invalid", "bytes_retained", "rounds", "incomplete_rounds", "missed_ticks"):
            require(nonnegative_integer(counts.get(name)), "invalid_counter:" + name)
        require(counts["retained"] == len(events), "retained_count_mismatch")
        require(counts["attempted"] == counts["retained"] + counts["dropped"], "attempt_conservation_failure")
        require(type(report.get("retention_lossless")) is bool
                and report["retention_lossless"] == (counts["dropped"] == 0), "retention_flag_differs_from_counts")
        previous_sequence, previous_time = 0, -math.inf
        for event in events:
            require(type(event.get("sequence")) is int and previous_sequence < event["sequence"] <= counts["attempted"],
                    "invalid_event_sequence")
            require(number(event.get("recorded_at")) and event["recorded_at"] >= previous_time, "invalid_event_time")
            require(event.get("case_id") == protocol["case_id"] and event.get("clock_id") == protocol["clock_id"],
                    "event_identity_mismatch")
            previous_sequence, previous_time = event["sequence"], event["recorded_at"]
        encoded_bytes = sum(len(json.dumps(event, allow_nan=False, separators=(",", ":")).encode()) for event in events)
        require(counts["bytes_retained"] == encoded_bytes, "retained_bytes_mismatch")
        require(sum(event["kind"] == "collection_start" for event in events) == 1
                and events[0]["kind"] == "collection_start" and events[0]["sequence"] == 1, "missing_collection_start")
        require(events[0].get("protocol") == protocol, "collection_start_protocol_differs")
        require(sum(event["kind"] == "collection_stop" for event in events) == 1
                and events[-1]["kind"] == "collection_stop" and report.get("stopped") is True,
                "missing_terminal_collection_stop")
        cutoffs = [event for event in events if event["kind"] == "trace_cutoff"]
        require(len(cutoffs) == 1 and {key: cutoffs[0].get(key) for key in ("at", "censored_at", "cohort")}
                == report.get("trace_cutoff"), "immutable_cutoff_mismatch")
        require(cutoff_time(report) <= cutoffs[0]["at"] <= cutoffs[0]["recorded_at"], "invalid_cutoff_time")
        result["collection_dropped"] = events[-1]["sequence"] - len(events)
        result["post_close_rejected"] = counts["attempted"] - events[-1]["sequence"]
        require(result["collection_dropped"] + result["post_close_rejected"] == counts["dropped"], "drop_partition_failure")
        require(result["collection_dropped"] == 0, "collection_event_loss")
        require(counts["invalid"] == 0, "invalid_events")
        require(report.get("sampler_error") is None and report.get("task_ended") is True, "sampler_not_cleanly_closed")
        result["status"] = "complete_collection_stream"
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        errors.append(str(error))
    return result


def cutoff_time(report):
    cutoff = report["trace_cutoff"]
    require(isinstance(cutoff, dict) and number(cutoff.get("at")), "missing_immutable_cutoff")
    value = cutoff.get("censored_at")
    if value is not None:
        require(number(value) and value <= cutoff["at"], "invalid_censor_boundary")
    return cutoff["at"] if value is None else value


class ReceiptIndex:
    def __init__(self, report):
        self.cutoff = cutoff_time(report)
        self.receipts, self.terminals, self.arrivals, self.submissions = {}, {}, {}, {}
        self.invalid = set()
        for event in report["events"]:
            kind, identity = event["kind"], event.get("request_id")
            if kind == "arrival":
                require(isinstance(identity, str) and identity not in self.arrivals, "duplicate_or_invalid_arrival")
                require(number(event.get("arrived_at")) and event["arrived_at"] <= event["recorded_at"]
                        and event.get("count") == 0 and event.get("submitted_at") is None, "invalid_arrival_state")
                self.arrivals[identity] = event
                self.receipts[identity] = []
                self.terminals[identity] = []
            elif kind == "submission":
                require(identity in self.arrivals and identity not in self.submissions, "invalid_submission_identity")
                require(number(event.get("submitted_at"))
                        and self.arrivals[identity]["arrived_at"] <= event["submitted_at"] <= event["recorded_at"],
                        "invalid_submission_time")
                require(all(event.get(key) == self.arrivals[identity].get(key) for key in ("model", "prompt_tokens", "arrived_at")),
                        "submission_identity_changed")
                self.submissions[identity] = event
            elif kind == "output_receipt":
                require(identity in self.submissions, "receipt_without_submission")
                previous = self.receipts[identity][-1] if self.receipts[identity] else None
                require(nonnegative_integer(event.get("count")) and nonnegative_integer(event.get("previous_count")),
                        "invalid_receipt_count")
                require(event["previous_count"] == (previous["count"] if previous else 0)
                        and event["count"] >= event["previous_count"], "receipt_count_discontinuity")
                require(number(event.get("received_at")) and event["received_at"] <= event["recorded_at"]
                        and event["received_at"] >= (previous["received_at"] if previous else self.submissions[identity]["submitted_at"]),
                        "invalid_receipt_timestamp")
                require(type(event.get("finished")) is bool and event.get("model") == self.arrivals[identity]["model"],
                        "receipt_identity_or_terminal_mismatch")
                self.receipts[identity].append(event)
                if event["finished"]:
                    self.terminals[identity].append(dict(at=event["received_at"], kind="final_output", reason=event.get("finish_reason")))
            elif kind == "stream_terminal":
                require(identity in self.arrivals and number(event.get("received_at"))
                        and self.arrivals[identity]["arrived_at"] <= event["received_at"] <= event["recorded_at"]
                        and event.get("terminal") in ("returned", "cancelled", "error"), "terminal_without_valid_arrival")
                self.terminals[identity].append(dict(at=event["received_at"], kind=event["terminal"], reason=event.get("error_type")))
            elif kind in ("receipt_unavailable", "submission_unavailable", "registration_unavailable"):
                self.invalid.add(identity)
        self.counts = {identity: [event["count"] for event in rows] for identity, rows in self.receipts.items()}
        self.times = {identity: [event["received_at"] for event in rows] for identity, rows in self.receipts.items()}
        self.validate_cohort(report["trace_cutoff"]["cohort"], report["trace_cutoff"]["at"])

    def validate_cohort(self, cohort, as_of):
        identities = {state["request_id"] for state in cohort}
        visible = {identity for identity, event in self.arrivals.items() if event["recorded_at"] <= as_of}
        require(len(identities) == len(cohort) and identities == visible, "cohort_differs_from_all_visible_arrivals")
        for state in cohort:
            identity = state["request_id"]
            require(all(state.get(key) == self.arrivals[identity].get(key) for key in ("model", "prompt_tokens", "arrived_at")),
                    "cohort_arrival_identity_changed")
            submission = self.submissions.get(identity)
            submitted = submission if submission is not None and submission["recorded_at"] <= as_of else None
            require(state.get("submitted_at") == (submitted["submitted_at"] if submitted else None)
                    and state.get("rank_initial") == (submitted.get("rank_initial") if submitted else None),
                    "cohort_submission_differs_from_events")
            prefix = bisect.bisect_right(self.times[identity], as_of)
            receipt = self.receipts[identity][prefix - 1] if prefix else None
            require(state["count"] == (receipt["count"] if receipt else 0)
                    and state.get("last_received_at") == (receipt["received_at"] if receipt else None),
                    "cohort_receipts_differ_from_events")
            terminals = [item for item in self.terminals[identity] if item["at"] <= as_of]
            stream = [item for item in terminals if item["kind"] != "final_output"]
            require(state.get("final_output_received") == any(item["kind"] == "final_output" for item in terminals)
                    and state.get("stream_terminal") == (stream[-1]["kind"] if stream else None),
                    "cohort_terminal_differs_from_events")

    def target(self, state, as_of, increment):
        identity = state["request_id"]
        needed = state["count"] + increment
        result = dict(request_id=identity, model=state["model"], needed_count=needed, status="unavailable",
                      residual_s=None, lower_bound_s=None, receipt_at=None)
        if identity not in self.receipts or identity in self.invalid:
            return dict(result, reason="invalid_or_missing_request_stream")
        if self.arrivals[identity]["recorded_at"] > as_of:
            return dict(result, reason="arrival_not_available_at_query")
        if any(terminal["at"] <= as_of for terminal in self.terminals[identity]):
            return dict(result, reason="terminal_already_received_at_query")
        prefix = bisect.bisect_right(self.times[identity], as_of)
        received_count = self.counts[identity][prefix - 1] if prefix else 0
        if received_count != state["count"]:
            return dict(result, reason="cohort_count_differs_from_asof_receipts")
        offset = bisect.bisect_left(self.counts[identity], needed, lo=prefix)
        hit = self.receipts[identity][offset] if offset < len(self.receipts[identity]) else None
        terminals = [terminal for terminal in self.terminals[identity] if as_of < terminal["at"] <= self.cutoff]
        terminal = min(terminals, key=lambda item: item["at"]) if terminals else None
        if hit is not None and as_of < hit["received_at"] <= self.cutoff and (terminal is None or hit["received_at"] <= terminal["at"]):
            return dict(result, status="reached", residual_s=hit["received_at"] - as_of,
                        lower_bound_s=hit["received_at"] - as_of, receipt_at=hit["received_at"],
                        reached_count=hit["count"], coalesced=hit["count"] - hit["previous_count"] > 1)
        if terminal is not None:
            return dict(result, status="terminal_before_target", terminal=terminal,
                        lower_bound_s=terminal["at"] - as_of)
        return dict(result, status="right_censored", lower_bound_s=self.cutoff - as_of)


def unfinished(state):
    return state.get("final_output_received") is False and state.get("stream_terminal") is None


def cohort_features(cohort, as_of):
    rows, identities = [], set()
    for state in cohort:
        identity = state.get("request_id")
        require(isinstance(identity, str) and identity not in identities, "invalid_or_duplicate_cohort_identity")
        identities.add(identity)
        require(type(state.get("model")) is int and state["model"] in (0, 1), "invalid_cohort_model")
        require(nonnegative_integer(state.get("count")) and type(state.get("prompt_tokens")) is int
                and state["prompt_tokens"] > 0, "invalid_cohort_counters")
        require(number(state.get("arrived_at")) and state["arrived_at"] <= as_of, "future_or_missing_arrival")
        require(type(state.get("final_output_received")) is bool
                and state.get("stream_terminal") in (None, "returned", "cancelled", "error"), "invalid_cohort_terminal")
        if not unfinished(state):
            continue
        submitted, received = state.get("submitted_at"), state.get("last_received_at")
        require(submitted is None or number(submitted) and state["arrived_at"] <= submitted <= as_of,
                "future_or_invalid_submission")
        require(received is None or number(received) and submitted is not None and submitted <= received <= as_of,
                "future_or_invalid_receipt")
        rows.append(dict(request_id=identity, model=state["model"], prompt_tokens=state["prompt_tokens"],
                         received_count=state["count"], arrival_age_s=as_of - state["arrived_at"],
                         queued=submitted is None, submission_age_s=None if submitted is None else as_of - submitted,
                         last_receipt_age_s=None if received is None else as_of - received,
                         rank_initial=state.get("rank_initial"), receipt_status=state.get("receipt_status")))
    rows.sort(key=lambda item: item["request_id"])
    return dict(requests=rows, counts=[sum(item["model"] == model for item in rows) for model in (0, 1)],
                queued_counts=[sum(item["model"] == model and item["queued"] for item in rows) for model in (0, 1)],
                received_counts=[sum(item["received_count"] for item in rows if item["model"] == model) for model in (0, 1)])


def source_features(sample, protocol, attested_workers=None):
    joined = causal_joint_state(sample.get("transport"), as_of=sample["as_of"],
                                max_age_s=protocol["period_s"] + protocol["deadline_s"],
                                max_skew_s=protocol["deadline_s"], clock_id=protocol["clock_id"],
                                num_ranks=2, tensor_parallel_size=2)
    if joined["status"] != "available":
        return dict(status="unavailable", errors=joined["errors"])
    if attested_workers is not None:
        observed = {(item["model"], item["dp_rank"], item["tp_rank"]): item["producer_pid"] for item in joined["workers"]}
        if observed != attested_workers:
            return dict(status="unavailable", errors=["worker_sources_differ_from_attestation"])
    workers = [dict(model=item["model"], dp_rank=item["dp_rank"], tp_rank=item["tp_rank"], host_call=item["host_call"],
                    producer_pid=item["producer_pid"], producer_id=item["producer_id"],
                    host_age_s=None if item["host_call_started_at"] is None else sample["as_of"] - item["host_call_started_at"],
                    sample_age_s=sample["as_of"] - item["observed_at"], failed_calls=item["failed_calls"])
               for item in joined["workers"]]
    links = link_features(sample.get("batch_links"), joined, sample["as_of"], protocol)
    return dict(status="available" if links["status"] == "available" else "unavailable", workers=workers,
                directional_indices=joined["directional_indices"], rank_waves=joined["rank_waves"],
                scheduler_receipts=joined["scheduler_receipts"], batch_links=links, device_progress="unknown")


def link_features(transport, joined, as_of, protocol):
    try:
        require(isinstance(transport, dict) and transport.get("kind") == "live_batch_link_transport_v1"
                and transport.get("status") == "observed" and transport.get("clock_id") == protocol["clock_id"],
                "missing_or_invalid_link_transport")
        require(number(transport.get("collected_at")) and transport["collected_at"] <= as_of, "future_link_transport")
        expected = {(model, rank, lane) for model in (0, 1) for rank in (0, 1) for lane in (None, 0, 1)}
        pids = {(row["model"], row["dp_rank"], row.get("tp_rank")): row["producer_pid"]
                for row in joined["rank_waves"] + joined["workers"]}
        indexed, normalized = {}, []
        for row in transport["records"]:
            identity = (row.get("model"), row.get("dp_rank"), row.get("tp_rank"))
            require(identity in expected and identity not in indexed
                    and all(type(value) is int for value in identity[:2])
                    and (identity[2] is None or type(identity[2]) is int), "invalid_or_duplicate_link_lane")
            require(type(row.get("producer_pid")) is int and row["producer_pid"] == pids[identity]
                    and row.get("role") == ("engine" if identity[2] is None else "worker")
                    and row.get("schema") == 1 and row.get("status") == "observed"
                    and row.get("clock_id") == protocol["clock_id"] and row.get("kind") == "causal_batch_link_v1"
                    and row.get("reason") is None and row.get("history_retained") is False, "link_source_mismatch")
            require(number(row.get("observed_at")) and number(row.get("available_at"))
                    and row["observed_at"] <= row["available_at"] <= transport["collected_at"]
                    and as_of - row["observed_at"] <= protocol["period_s"] + protocol["deadline_s"], "stale_or_future_link")
            require(type(row.get("active")) is bool and all(nonnegative_integer(row.get(key))
                    for key in ("revision", "completed", "failures")), "invalid_link_counters")
            latest = row.get("latest")
            if latest is not None:
                require(isinstance(latest, dict) and latest.get("model") == identity[0] and latest.get("dp_rank") == identity[1],
                        "link_operation_identity_mismatch")
                require(type(latest.get("sequence")) is int and latest["sequence"] > 0
                        and isinstance(latest.get("engine_epoch"), str) and len(latest["engine_epoch"]) == 32
                        and all(character in "0123456789abcdef" for character in latest["engine_epoch"])
                        and type(latest.get("engine_pid")) is int and latest["engine_pid"] > 0, "invalid_link_origin")
                require(number(latest.get("started_at")) and latest["started_at"] <= row["observed_at"], "future_link_start")
                if latest.get("rpc") is not None:
                    require(latest["rpc"] in ("execute_model", "sample_tokens", "execute_dummy_batch")
                            and valid_features(latest.get("features"), latest["rpc"]), "invalid_batch_features")
                else:
                    require(latest.get("features") is None, "features_without_rpc")
                for key in ("step_counter", "current_wave", "prefill_schedule_interval"):
                    require(key not in latest or nonnegative_integer(latest[key]), "invalid_dp_cadence_feature")
                for key in ("returned_at", "rpc_dispatched_at", "step_returned_at"):
                    require(latest.get(key) is None or number(latest[key]) and latest["started_at"] <= latest[key] <= row["observed_at"],
                            "future_link_boundary")
            indexed[identity] = row
        require(set(indexed) == expected, "missing_link_lane")
        for identity in sorted(indexed, key=lambda key: (key[0], key[1], -1 if key[2] is None else key[2])):
            row, latest = indexed[identity], indexed[identity].get("latest")
            engine = indexed[(identity[0], identity[1], None)]
            origin = engine.get("latest")
            matched = latest is not None and origin is not None and all(
                latest.get(key) == origin.get(key) for key in ("engine_pid", "engine_epoch", "sequence"))
            if origin is not None:
                require(origin["engine_pid"] == engine["producer_pid"], "engine_origin_pid_mismatch")
                require(latest is None or all(latest.get(key) == origin[key] for key in ("engine_pid", "engine_epoch")),
                        "foreign_link_origin")
            if matched and latest.get("rpc") is not None and origin.get("rpc") is not None:
                require(latest.get("features") == origin.get("features"), "matched_batch_membership_differs")
            latest_fields = ("engine_epoch", "engine_pid", "model", "dp_rank", "sequence", "operation", "started_at",
                             "step_counter", "current_wave", "prefill_schedule_interval", "rpc", "rpc_dispatched_at",
                             "features", "returned_at", "failed", "step_returned_at")
            safe_latest = None if latest is None else {key: copy.deepcopy(latest[key]) for key in latest_fields if key in latest}
            normalized.append(dict(model=identity[0], dp_rank=identity[1], tp_rank=identity[2], active=row["active"],
                                   revision=row["revision"], completed=row["completed"], failures=row["failures"],
                                   same_latest_engine_operation=matched, latest=safe_latest,
                                   observation_age_s=as_of - row["observed_at"]))
        return dict(status="available", lanes=normalized, complete_history=False, common_rank_barrier=False)
    except (KeyError, TypeError, ValueError) as error:
        return dict(status="unavailable", reason=str(error))


def label_queries(report, increment=16, attested_workers=None):
    require(type(increment) is int and increment == 16, "Only frozen K16 is supported")
    retention = retention_status(report)
    cutoff, protocol = cutoff_time(report), report["protocol"]
    starts = [event for event in report["events"] if event["kind"] == "collection_start"]
    require(len(starts) == 1, "Cannot define sample grid without collection origin")
    origin = starts[0]["origin"]
    require(number(origin) and origin <= cutoff and number(protocol["period_s"]) and protocol["period_s"] > 0,
            "invalid_collection_grid")
    count = int(((Decimal(str(cutoff)) - Decimal(str(origin))) / Decimal(str(protocol["period_s"]))).to_integral_value(rounding=ROUND_FLOOR)) + 1
    require(0 < count <= 100000, "Unbounded sample grid")
    samples = {}
    for event in report["events"]:
        if event["kind"] == "sample":
            require(nonnegative_integer(event.get("tick")) and event["tick"] not in samples, "duplicate_or_invalid_sample_tick")
            samples[event["tick"]] = event
    try:
        index, stream_error = ReceiptIndex(report), None
    except (KeyError, ValueError, TypeError) as error:
        index, stream_error = None, str(error)
    for tick in range(count):
        scheduled = float(Decimal(str(origin)) + tick * Decimal(str(protocol["period_s"])))
        row = dict(tick=tick, scheduled_at=scheduled, as_of=None, status="missing_sample", features=None,
                   source_features=None, targets=[], joint_target_s=None, joint_lower_bound_s=None,
                   target_available=False, feature_available=False, prediction_gate_passed=False)
        sample = samples.get(tick)
        if sample is None:
            yield row
            continue
        row.update(as_of=sample.get("as_of"), source_event_sequence=sample["sequence"],
                   actual_scheduled_at=sample.get("scheduled_at"), sample_started_at=sample.get("started_at"))
        if not (number(sample.get("as_of")) and number(sample.get("started_at"))
                and number(sample.get("scheduled_at"))
                and abs(scheduled - sample["scheduled_at"]) <= 4 * math.ulp(scheduled)
                and sample["scheduled_at"] <= sample["started_at"] <= sample["as_of"] <= sample["recorded_at"]):
            yield dict(row, status="invalid_sample_times")
            continue
        if sample["as_of"] > cutoff:
            yield dict(row, status="sample_crosses_cutoff")
            continue
        try:
            row["features"] = cohort_features(sample["cohort_at_completion"], sample["as_of"])
            row["source_features"] = source_features(sample, protocol, attested_workers)
        except (KeyError, TypeError, ValueError) as error:
            yield dict(row, status="invalid_asof_features", reason=str(error))
            continue
        if retention["status"] != "complete_collection_stream" or index is None:
            yield dict(row, status="unavailable_event_stream", reason=stream_error or retention["errors"])
            continue
        try:
            index.validate_cohort(sample["cohort_at_completion"], sample["as_of"])
        except (KeyError, TypeError, ValueError) as error:
            yield dict(row, status="invalid_asof_cohort", reason=str(error))
            continue
        row["feature_available"] = (row["source_features"]["status"] == "available"
                                    and sample.get("transport_complete") is True
                                    and sample.get("cancelled") is False and sample.get("error_type") is None)
        cohort = [state for state in sample["cohort_at_completion"] if unfinished(state)]
        if not cohort:
            yield dict(row, status="empty_cohort")
            continue
        row["targets"] = [index.target(state, sample["as_of"], increment) for state in cohort]
        bounds = [target["lower_bound_s"] for target in row["targets"]]
        row["joint_lower_bound_s"] = max(bounds) if all(number(bound) for bound in bounds) else None
        row["target_available"] = all(target["status"] == "reached" for target in row["targets"])
        row["joint_target_s"] = max(target["residual_s"] for target in row["targets"]) if row["target_available"] else None
        row["status"] = "labeled" if row["target_available"] else "partial_or_censored_target"
        yield row


def collection_coverage(report, rules, attested_workers=None):
    retention = retention_status(report)
    cutoff = cutoff_time(report)
    samples = [event for event in report["events"] if event["kind"] == "sample"]
    active = [event for event in samples if number(event.get("as_of")) and event["as_of"] <= cutoff
              and any(unfinished(state) for state in event.get("cohort_at_start", []))]
    independently_complete = []
    stream = None
    try:
        stream = ReceiptIndex(report)
    except (KeyError, TypeError, ValueError):
        pass
    for event in active:
        try:
            require(stream is not None, "invalid_receipt_stream")
            origin = report["events"][0]["origin"]
            require(nonnegative_integer(event.get("tick")), "invalid_sample_tick")
            scheduled = origin + event["tick"] * report["protocol"]["period_s"]
            require(number(event.get("scheduled_at")) and abs(scheduled - event["scheduled_at"]) <= 4 * math.ulp(scheduled)
                    and scheduled <= event["started_at"] <= event["as_of"] <= event["recorded_at"], "invalid_sample_times")
            stream.validate_cohort(event["cohort_at_start"], event["started_at"])
            stream.validate_cohort(event["cohort_at_completion"], event["as_of"])
            independently_complete.append(event.get("transport_complete") is True
                                          and event.get("cancelled") is False and event.get("error_type") is None
                                          and source_features(event, report["protocol"], attested_workers)["status"] == "available")
        except (KeyError, TypeError, ValueError):
            independently_complete.append(False)
    complete = sum(independently_complete)
    counts = report["counts"]
    denominator = counts["rounds"] + counts["missed_ticks"]
    missed_fraction = counts["missed_ticks"] / denominator if denominator else None
    fraction = complete / len(active) if active else None
    errors = list(retention["errors"])
    if counts["rounds"] != len(samples):
        errors.append("round_count_mismatch")
    if counts["incomplete_rounds"] != sum(event.get("transport_complete") is not True for event in samples):
        errors.append("incomplete_round_count_mismatch")
    try:
        accounted = set()
        missed = 0
        for event in report["events"]:
            if event["kind"] == "sample":
                first, size = event.get("tick"), 1
            elif event["kind"] == "missed_sample_ticks":
                first, size = event.get("first_tick"), event.get("count")
                require(nonnegative_integer(size) and size > 0, "invalid_missed_tick_count")
                missed += size
            else:
                continue
            require(nonnegative_integer(first) and first + size <= 100000, "invalid_tick_accounting")
            new_ticks = set(range(first, first + size))
            require(not accounted.intersection(new_ticks), "duplicate_tick_accounting")
            accounted.update(new_ticks)
        require(accounted and accounted == set(range(max(accounted) + 1)), "unaccounted_sampler_ticks")
        require(missed == counts["missed_ticks"], "missed_tick_count_mismatch")
    except (TypeError, ValueError) as error:
        errors.append(str(error))
    if fraction is None or fraction < rules["minimum_active_round_transport_complete_fraction"]:
        errors.append("active_transport_coverage_below_gate")
    if missed_fraction is None or missed_fraction > rules["maximum_missed_tick_fraction"]:
        errors.append("missed_tick_fraction_above_gate")
    return dict(status="passed" if not errors else "not_passed", errors=errors, retention=retention,
                active_rounds=len(active), complete_active_rounds=complete, active_transport_fraction=fraction,
                missed_tick_fraction=missed_fraction, prediction_gate_passed=False)


def token_bracket(row, boundary):
    count = row.get("output_tokens")
    require(nonnegative_integer(count), "invalid_output_count")
    observed_first = row.get("first_token_observed", not row.get("censored", False) or count > 0)
    if not observed_first or count == 0:
        return [0, 0]
    first = row.get("first_token_at")
    require(number(first), "missing_first_receipt")
    if first > boundary:
        return [0, 0]
    cadence = row.get("cadence")
    require(isinstance(cadence, list) and cadence, "missing_cadence")
    previous_time, previous_count = -math.inf, 0
    lower, upper = 0, count
    for timestamp, tokens in cadence:
        require(number(timestamp) and timestamp >= previous_time and nonnegative_integer(tokens)
                and previous_count <= tokens <= count, "invalid_cadence")
        if timestamp <= boundary:
            lower = max(lower, tokens)
        else:
            upper = min(upper, tokens)
        previous_time, previous_count = timestamp, tokens
    last = row.get("completed_at") if row.get("censored") else cadence[-1][0]
    require(number(last) and last >= first and cadence[-1][0] <= last, "invalid_last_receipt")
    if last <= boundary:
        lower = upper = count
    return [lower, upper]


def interval(lower, upper=None):
    upper = lower if upper is None else upper
    require(number(lower) and number(upper) and 0 <= lower <= upper, "invalid_metric_interval")
    return dict(lower=lower, upper=upper, point=lower if lower == upper else None)


def request_metrics(case, workload, summary, completed, incomplete, rules):
    metrics = {name: None for name in rules["primary_metrics"]}
    result = dict(status="unavailable", metrics=metrics, errors=[], requests=[], ignored_post_cutoff_completions=[])
    try:
        require(workload.get("policy") == "no_move" and summary.get("policy") == "no_move", "non_paired_policy")
        require(summary.get("status") in ("passed", "censored") and not summary.get("errors"), "failed_case")
        origin = workload["origin"]
        require(number(origin), "invalid_trace_origin")
        boundary = origin + rules["window_s"]
        if summary.get("censored_at_s") is not None:
            require(number(summary["censored_at_s"]) and rules["window_s"] <= summary["censored_at_s"]
                    <= rules["window_s"] + rules["window_overshoot_limit_s"], "censor_boundary_outside_protocol")
        selected = {}
        for rows, is_censored in ((completed, False), (incomplete, True)):
            identities = set()
            for row in rows:
                identity = row["trace_request_id"]
                require(identity not in identities and bool(row.get("censored")) == is_censored, "duplicate_or_wrong_request_kind")
                identities.add(identity)
                if identity in selected:
                    require(is_censored, "duplicate_completed_identity")
                    result["ignored_post_cutoff_completions"].append(identity)
                selected[identity] = row
        expected = {row["request_id"]: row for row in case["requests"]}
        require(len(expected) == len(case["requests"]) and set(selected) == set(expected), "missing_or_extra_planned_requests")
        require(summary.get("expected") == len(expected), "planned_request_count_mismatch")
        conv_brackets, conv_gaps, code_ttfts = [], [], []
        conv_expected = sum(mark["model"] == 1 for mark in expected.values())
        code_expected = sum(mark["model"] == 0 for mark in expected.values())
        for identity, mark in expected.items():
            row = selected[identity]
            if row.get("censored"):
                require(number(summary.get("censored_at_s")) and number(row.get("censored_at"))
                        and abs(row["censored_at"] - origin - summary["censored_at_s"]) <= 1e-6,
                        "request_censor_time_differs_from_summary")
                require(number(row.get("completed_at")) and row["completed_at"] <= row["censored_at"],
                        "receipt_after_immutable_censor")
            require(row.get("model") == mark["model"] and row.get("prompt_tokens") == mark["prompt_tokens"],
                    "request_identity_or_prompt_changed")
            require(number(row.get("arrived_at")) and abs(row["arrived_at"] - (origin + mark["arrival_s"])) <= 1e-6,
                    "request_arrival_changed")
            require((row.get("target_output_tokens") if row.get("censored") else row.get("output_tokens")) == mark["output_tokens"],
                    "original_output_length_changed")
            bracket = token_bracket(row, boundary)
            item = dict(request_id=identity, engine_request_id=row.get("request_id"), model=mark["model"],
                        prompt_tokens=mark["prompt_tokens"], arrived_at=row["arrived_at"],
                        received_at_window=bracket, censored=bool(row.get("censored")))
            first = row.get("first_token_at")
            first_observed = row.get("first_token_observed", not row.get("censored", False) or row["output_tokens"] > 0)
            if mark["model"] == 1:
                conv_brackets.append(bracket)
                last = row.get("completed_at") if row.get("censored") else row["cadence"][-1][0]
                if (first_observed and row["output_tokens"] >= 2 and number(first) and number(last)
                        and first <= last <= boundary + rules["window_overshoot_limit_s"]):
                    gap = (last - first) / (row["output_tokens"] - 1)
                    conv_gaps.append(gap)
                    item.update(observed_prefix_tpot_s=gap, prefix_last_receipt_at=last)
                else:
                    item["tpot_unavailable"] = True
            else:
                if first_observed and number(first) and row["arrived_at"] <= first <= boundary:
                    code_ttfts.append(first - row["arrived_at"])
                    item["ttft_s"] = first - row["arrived_at"]
                else:
                    item["ttft_unavailable"] = True
            result["requests"].append(item)
        lower = sum(row[0] for row in conv_brackets)
        upper = sum(row[1] for row in conv_brackets)
        if lower > 0 and conv_expected:
            metrics["conv_inverse_total_received_tokens_per_window_second"] = interval(rules["window_s"] / upper,
                                                                                       rules["window_s"] / lower)
        if conv_expected and len(conv_gaps) == conv_expected:
            metrics["conv_mean_observed_interreceipt_time_per_token"] = interval(statistics.mean(conv_gaps))
        if code_expected and len(code_ttfts) == code_expected:
            metrics["code_mean_ttft"] = interval(statistics.mean(code_ttfts))
        result["status"] = "available" if all(value is not None for value in metrics.values()) else "partial_metrics"
        result["metric_coverage"] = dict(conv_gap=len(conv_gaps), conv_planned=conv_expected,
                                          code_ttft=len(code_ttfts), code_planned=code_expected)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        result["errors"].append(str(error))
        result["metrics"] = {name: None for name in rules["primary_metrics"]}
    return result


def compare_blocks(blocks, case_ids, rules):
    require(len(case_ids) == 4 and len(set(case_ids)) == 4, "requires_four_unique_planned_cases")
    by_id = {block["block"]: block for block in blocks}
    require(len(by_id) == len(blocks) and set(by_id) <= {0, 1}, "duplicate_or_unknown_overhead_block")
    result = dict(passed=False, missing_blocks=sorted({0, 1} - set(by_id)), pairs=[], metrics={}, errors=[],
                  prediction_gate_passed=False)
    values = {name: [] for name in rules["primary_metrics"]}
    for block_id in (0, 1):
        for case_id in case_ids:
            block = by_id.get(block_id)
            pair = dict(block=block_id, case=case_id, metrics={name: None for name in values})
            for name in values:
                try:
                    off = block["cases"]["off"][case_id]["performance"]["metrics"][name]
                    collect = block["cases"]["collect"][case_id]["performance"]["metrics"][name]
                    require(off is not None and collect is not None and off["lower"] > 0, "undefined_paired_metric")
                    interval(off["lower"], off["upper"])
                    interval(collect["lower"], collect["upper"])
                    loss = dict(lower=collect["lower"] / off["upper"] - 1,
                                upper=collect["upper"] / off["lower"] - 1)
                    pair["metrics"][name] = loss
                    values[name].append(loss)
                except (KeyError, TypeError, ValueError):
                    result["errors"].append(f"missing_pair:{block_id}:{case_id}:{name}")
            result["pairs"].append(pair)
    expected_pairs = 2 * len(case_ids)
    for name, rows in values.items():
        summary = dict(available_pairs=len(rows), expected_pairs=expected_pairs, passed=False)
        if rows:
            summary.update(median_loss_lower=statistics.median(row["lower"] for row in rows),
                           median_loss_upper=statistics.median(row["upper"] for row in rows),
                           worst_loss_upper=max(row["upper"] for row in rows))
            summary["passed"] = (len(rows) == expected_pairs and summary["median_loss_upper"] <= rules["per_metric_median_loss_limit"]
                                 and summary["worst_loss_upper"] <= rules["per_metric_worst_case_loss_limit"])
        result["metrics"][name] = summary
    result["passed"] = (not result["missing_blocks"] and not result["errors"]
                        and all(item["passed"] for item in result["metrics"].values()))
    return result


def summarize_process_usage(filename):
    if not filename.exists():
        return dict(status="missing", exact_cumulative_cpu=False, true_peak_rss=False)
    maxima, peak_rss, samples, read_failures = {}, 0, 0, 0
    try:
        with filename.open() as handle:
            for line in handle:
                row = json.loads(line)
                samples += 1
                read_failures += len(row["read_failures"])
                peak_rss = max(peak_rss, sum(process["rss_pages"] for process in row["processes"]) * row["page_bytes"])
                for process in row["processes"]:
                    identity = (process["pid"], process["start_ticks"])
                    maxima[identity] = max(maxima.get(identity, 0), process["cpu_ticks"] / row["clock_ticks_per_s"])
        require(samples > 0 and maxima, "empty_process_usage")
        return dict(status="sampled", samples=samples, maximum_sampled_group_rss_bytes=peak_rss,
                    sum_last_observed_process_cpu_s=sum(maxima.values()), sampled_processes=len(maxima),
                    read_failures=read_failures, exact_cumulative_cpu=False, true_peak_rss=False)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        return dict(status="invalid", error=str(error), exact_cumulative_cpu=False, true_peak_rss=False)


def audit_startup(job_root, protocol):
    filename = job_root / "inputs/startup-test-contract.json"
    require(protocol.get("schema") == 2 and digest(filename) == protocol.get("startup_test_contract_sha256"),
            "startup_contract_digest_differs")
    contract = read(filename)
    count, expected = contract.get("expected_count"), contract.get("expected_testcases")
    nodes = contract.get("nodes")
    require(contract.get("schema") == 2 and contract.get("non_timesliced_only") is True
            and type(count) is int and count > 0 and isinstance(expected, list) and len(expected) == count
            and isinstance(nodes, list) and nodes and all(isinstance(node, str) and node.startswith("tests/test_")
            and ".." not in node and "timesliced" not in node for node in nodes), "startup_contract_invalid")
    require(all(isinstance(identity, dict) and set(identity) == {"classname", "name"}
                and all(isinstance(value, str) and value for value in identity.values()) for identity in expected),
            "startup_expected_identity_invalid")
    expected_ids = [(identity["classname"], identity["name"]) for identity in expected]
    require(len(set(expected_ids)) == count, "startup_expected_identities_duplicate")
    document = element_tree.parse(job_root / "startup-tests.xml")
    suites = list(document.getroot().iter("testsuite"))
    require(suites and all(all(key in suite.attrib for key in ("tests", "failures", "errors", "skipped"))
                          and all(int(suite.attrib[key]) == 0 for key in ("errors", "failures", "skipped"))
                          and int(suite.attrib["tests"]) >= 0 for suite in suites), "startup_tests_failed_or_skipped")
    testcases = list(document.getroot().iter("testcase"))
    actual = [(item.attrib.get("classname"), item.attrib.get("name")) for item in testcases]
    require(len(actual) == count == sum(int(suite.attrib["tests"]) for suite in suites)
            and Counter(actual) == Counter(expected_ids), "startup_exact_identities_differ")
    require(not any(item.tag in ("failure", "error", "skipped") for case in testcases for item in case.iter()),
            "startup_hidden_failure_or_skip")
    return dict(status="passed", expected_count=count, observed_count=len(actual),
                contract_sha256=digest(filename), junit_sha256=digest(job_root / "startup-tests.xml"))


def audit_compute_payload(job_root, analysis_protocol):
    expected = analysis_protocol.get("compute_payload_sha256")
    require(isinstance(expected, dict) and expected and all(
        isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts
        and len(Path(name).parts) >= 2 and Path(name).parts[0] in ("runtime", "tests", "inputs")
        and isinstance(identity, str) and len(identity) == 64
        and all(character in "0123456789abcdef" for character in identity)
        for name, identity in expected.items()), "compute_payload_contract_invalid")
    actual = {}
    for name in ("runtime", "tests", "inputs"):
        directory = job_root / name
        require(not directory.is_symlink(), "compute_payload_symlink")
        for filename in directory.rglob("*"):
            relative = filename.relative_to(job_root)
            require(not filename.is_symlink(), "compute_payload_symlink")
            if filename.is_file() and "__pycache__" not in relative.parts and filename.suffix != ".pyc":
                actual[str(relative)] = digest(filename)
    require(actual == expected, "compute_payload_files_or_digests_differ")
    return dict(status="passed", files=len(actual))


def audit_paired_budget(output, plan, devices):
    filename = output / "budget.json"
    budget = read(filename)
    required, natural = plan.get("paired_budget_gib"), budget.get("natural_budget_gib")
    vectors = (required, natural, budget.get("budget_gib"), budget.get("required_budget_gib"))
    require(all(isinstance(values, list) and len(values) == 2
                and all(number(value) and value > 0 for value in values) for values in vectors),
            "paired_budget_vector_invalid")
    require(budget.get("paired_budget_status") == "verified" and budget["budget_gib"] == required
            and budget["required_budget_gib"] == required and budget.get("margin_gib") == plan.get("budget_margin_gib") == 3.0
            and all(actual >= target and actual % 0.25 == 0
                    for actual, target in zip(natural, required)), "paired_budget_unavailable_or_changed")
    memory = budget.get("memory")
    require(isinstance(memory, dict) and set(memory) == {"0", "1"}, "paired_budget_memory_missing")
    require(set(devices) == {(model, rank, lane) for model in (1, 2) for rank in (0, 1) for lane in (0, 1)}
            and all(isinstance(identity, str) and identity for identity in devices.values())
            and len(set(devices.values())) == 4, "paired_budget_attested_devices_invalid")
    for model in (0, 1):
        ranks = memory[str(model)]
        require(isinstance(ranks, list) and len(ranks) == 2, "paired_budget_memory_ranks_differ")
        for rank, lanes in enumerate(ranks):
            require(isinstance(lanes, list) and len(lanes) == 2, "paired_budget_memory_lanes_differ")
            for lane, row in enumerate(lanes):
                require(isinstance(row, dict) and all(type(row.get(key)) is int for key in ("rank", "tp_rank", "tp_size"))
                        and (row["rank"], row["tp_rank"], row["tp_size"]) == (rank, lane, 2)
                        and row.get("gpu_uuid") == devices.get((model + 1, rank, lane))
                        and row.get("released_unused_allocator_cache") is True, "paired_budget_device_evidence_differs")
                free, total = row.get("free_mib"), row.get("total_mib")
                require(number(free) and number(total) and 0 <= free <= total and total > 0
                        and natural[rank] <= total / 1024 - budget["margin_gib"], "paired_budget_memory_values_invalid")
    require(budget.get("capacity_schema") == 1 and budget.get("num_ranks") == 2
            and budget.get("tensor_parallel_size") == 2, "capacity_calculation_schema_invalid")
    summaries, calculations = budget.get("summary_snapshots"), budget.get("rank_calculations")
    require(isinstance(summaries, list) and len(summaries) == 2
            and all(isinstance(ranks, list) and len(ranks) == 2 for ranks in summaries)
            and isinstance(calculations, list) and len(calculations) == 2, "capacity_calculation_inputs_missing")
    times = budget.get("measurement_times", {})
    ordered_times = [times.get(key) for key in ("memory_requested_at", "memory_received_at", "summaries_received_at")]
    require(all(number(value) and value >= 0 for value in ordered_times)
            and ordered_times == sorted(ordered_times), "capacity_measurement_order_invalid")
    for rank in (0, 1):
        require(all(isinstance(summaries[model][rank], dict) for model in (0, 1)), "capacity_summary_invalid")
        mapped = [summaries[model][rank].get("mapped_bytes") for model in (0, 1)]
        require(all(type(value) is int and value >= 0 for value in mapped), "capacity_mapped_bytes_invalid")
        free = min(memory[str(model)][rank][lane]["free_mib"] for model in (0, 1) for lane in (0, 1)) * 1024**2
        unrounded = (sum(mapped) + free) / 1024**3 - budget["margin_gib"]
        rounded = math.floor(unrounded * 4) / 4
        expected = dict(rank=rank, minimum_free_bytes=free, mapped_bytes_by_model=mapped,
                        mapped_bytes=sum(mapped), unrounded_budget_gib=unrounded, rounded_budget_gib=rounded)
        require(calculations[rank] == expected and natural[rank] == rounded, "capacity_independent_recomputation_differs")
    return dict(status="passed", budget_gib=required, natural_budget_gib=natural, sha256=digest(filename),
                memory_lanes=8, natural_budget_independently_recomputed=True,
                note="Recomputed from recorded sequential memory and mapped-KV snapshots; not an atomic capacity guarantee.")


def audit_block(job_root, block_id, manifest, protocol, plan, analysis_protocol):

    errors, clocks, gpu_sets, child_pids = [], [], [], []
    startup, payload, budgets = None, None, {}
    try:
        payload = audit_compute_payload(job_root, analysis_protocol)
        require((job_root / "exit-status").read_text().strip() == "0", "job_exit_not_zero")
        status = read(job_root / "out/causal-block/status.json")
        require(status["status"] == "collected" and status["block"] == block_id
                and status["planned"] == protocol["blocks"][block_id] and not status["not_executed"], "block_incomplete")
        require(len(status["completed"]) == 2 and all(row["status"] == "passed" for row in status["completed"]),
                "condition_driver_failed")
        require(status["started"] == protocol["blocks"][block_id]
                and [row["mode"] for row in status["completed"]] == status["started"], "condition_order_changed")
        startup = audit_startup(job_root, protocol)
        actual = {str(filename.relative_to(job_root / "runtime")): digest(filename)
                  for filename in (job_root / "runtime").rglob("*.py")}
        require(actual == manifest["runtime_sha256"], "archived_runtime_differs")
        manifest_sha = digest(job_root / "inputs/deployment-manifest.json")
        for mode in protocol["blocks"][block_id]:
            root = job_root / "out/causal-block" / mode
            process = read(root / "process.json")
            require(process["status"] == "passed" and process["returncode"] == 0
                    and not process["forced_cleanup"] and not process["timeout"], "unclean_condition_process")
            child_pids.append(process["pid"])
            output = root / "out"
            front = read(output / "causal-frontend-attestation.json")
            final = read(output / "causal-final-attestation.json")
            for record in (front, final):
                require(record["status"] == "verified" and record["manifest_sha256"] == manifest_sha
                        and record["native_checked"] is True and record["versions"] == manifest["versions"],
                        "frontend_attestation_failed")
            require(front["clock_id"] == final["clock_id"] and front["pid"] == final["pid"] == process["pid"],
                    "frontend_identity_changed")
            clocks.append(front["clock_id"])
            aggregate = read(output / "causal-worker-attestations.json")
            require(aggregate["status"] == "verified" and aggregate["clock_id"] == front["clock_id"], "worker_aggregate_invalid")
            aggregated = {(item["model"], item["dp_rank"], item["tp_rank"]): item["pid"] for item in aggregate["workers"]}
            require(len(aggregated) == len(aggregate["workers"]) == 8, "worker_aggregate_count_differs")
            workers, devices = [], {}
            for model in (1, 2):
                for rank in (0, 1):
                    for lane in (0, 1):
                        row = read(output / f"worker-instance-{model}-dp{rank}-tp{lane}.json")
                        attestation = row["causal_deployment"]
                        require(attestation["status"] == "verified" and attestation["manifest_sha256"] == manifest_sha
                                and attestation["configuration"] == manifest["configuration"]
                                and attestation["clock_id"] == front["clock_id"]
                                and attestation["pid"] == row["pid"], "worker_attestation_failed")
                        require(row["service"] == f"instance-{model}" and row["dp_rank"] == rank
                                and row["groups"]["dp"] == dict(rank=rank, world_size=2)
                                and row["groups"]["tp"] == dict(rank=lane, world_size=2)
                                and "A800" in row["gpu_name"], "worker_lane_or_device_mismatch")
                        workers.append(attestation["pid"])
                        require(aggregated[(model - 1, rank, lane)] == attestation["pid"], "worker_aggregate_pid_differs")
                        devices[(model, rank, lane)] = row["gpu_uuid"]
            require(len(set(workers)) == 8, "worker_processes_not_distinct")
            require(len(set(devices.values())) == 4 and all(devices[(1, rank, lane)] == devices[(2, rank, lane)]
                                                           for rank in (0, 1) for lane in (0, 1)), "gpu_colocation_differs")
            gpu_sets.append(sorted(set(devices.values())))
            budgets[mode] = audit_paired_budget(output, plan, devices)
            require(read(output / "calibration/dataplane.json")["status"] == "passed", "dataplane_failed")
            phases = read(output / "phases.json")["phases"]
            require(not any(row["status"] == "failed" for row in phases)
                    and any(row["phase"] == "job" and row["status"] == "passed" for row in phases), "main_phase_failed")
            started = [row for row in phases if row["phase"] == "startup"]
            require([row["status"] for row in started] == ["started", "passed"]
                    and started[1]["natural_budget_gib"] == budgets[mode]["natural_budget_gib"]
                    and started[1]["trace_budget_gib"] == plan["paired_budget_gib"], "startup_budget_record_differs")
        require(len(set(clocks)) == 1 and gpu_sets[0] == gpu_sets[1] and len(set(child_pids)) == 2,
                "paired_conditions_not_common_host_restarted_engines")
    except (OSError, KeyError, TypeError, ValueError, IndexError, element_tree.ParseError) as error:
        errors.append(str(error))
    return dict(status="passed" if not errors else "not_passed", errors=errors, startup=startup, payload=payload, budgets=budgets,
                slurm_accounting_independently_required=True, prediction_gate_passed=False)


def analyze_case(root, case, mode, protocol, query_file, plan):
    result = dict(case=case["case"], mode=mode, status="unavailable", input_sha256={}, error=None,
                  performance=dict(metrics={key: None for key in protocol["overhead"]["primary_metrics"]}),
                  collection=None, labels=None)
    try:
        required = [root / "case.json", root / "trace/workload.json", root / "trace/summary.json", root / "trace/requests.json"]
        for filename in required:
            result["input_sha256"][str(filename)] = digest(filename)
        require(read(root / "case.json") == case, "case_differs_from_frozen_plan")
        incomplete_file = root / "trace/incomplete.json"
        incomplete = read(incomplete_file) if incomplete_file.exists() else []
        if incomplete_file.exists():
            result["input_sha256"][str(incomplete_file)] = digest(incomplete_file)
        workload = read(root / "trace/workload.json")
        result["workload_conditions"] = {key: workload.get(key) for key in
                                         ("budgets_gib", "num_ranks", "max_inflight", "kv_mode", "reserve_extra_tokens",
                                          "trace_sha256", "duration_s", "requests", "native", "policy_switches")}
        result["performance"] = request_metrics(case, workload,
                                                 read(root / "trace/summary.json"), read(root / "trace/requests.json"),
                                                 incomplete, protocol["overhead"])
        require(workload.get("budgets_gib") == plan["paired_budget_gib"]
                and workload.get("num_ranks") == 2 and workload.get("policy") == "no_move"
                and all(workload.get(key) == plan[key] for key in ("max_inflight", "kv_mode", "reserve_extra_tokens")),
                "workload_differs_from_fixed_budget_or_admission_plan")
        if mode == "collect":
            collection_file = root / "collection.json"
            result["input_sha256"][str(collection_file)] = digest(collection_file)
            report = read(collection_file)
            require(all(report["protocol"].get(key) == value for key, value in protocol["collection"].items()),
                    "collection_configuration_differs")
            summary = read(root / "trace/summary.json")
            if summary.get("censored_at_s") is not None:
                require(abs(cutoff_time(report) - workload["origin"] - summary["censored_at_s"]) <= 1e-6,
                        "collection_trace_cutoff_differs")
            stream = ReceiptIndex(report)
            planned_arrivals = {row["engine_request_id"]: row for row in result["performance"]["requests"]
                                if row["arrived_at"] <= cutoff_time(report)}
            require(None not in planned_arrivals and len(planned_arrivals) == len(result["performance"]["requests"])
                    and set(planned_arrivals) == set(stream.arrivals), "collection_planned_arrivals_differ")
            for identity, arrival in stream.arrivals.items():
                require(all(arrival.get(key) == planned_arrivals[identity][key]
                            for key in ("model", "prompt_tokens", "arrived_at")), "collection_arrival_metadata_differs")
            attestation_file = root.parents[2] / "causal-worker-attestations.json"
            front_file = root.parents[2] / "causal-frontend-attestation.json"
            attestation = read(attestation_file)
            require(attestation["status"] == "verified"
                    and attestation["clock_id"] == report["protocol"]["clock_id"] == read(front_file)["clock_id"]
                    and report["protocol"]["case_id"] == case["case"], "collection_attestation_identity_differs")
            attested_workers = {(item["model"], item["dp_rank"], item["tp_rank"]): item["pid"] for item in attestation["workers"]}
            require(len(attested_workers) == len(attestation["workers"]) == 8, "collection_worker_count_differs")
            result["input_sha256"][str(attestation_file)] = digest(attestation_file)
            result["input_sha256"][str(front_file)] = digest(front_file)
            result["collection"] = collection_coverage(report, protocol["coverage"], attested_workers)
            counts, targets, jointly_eligible, total = Counter(), Counter(), 0, 0
            with query_file.open("x") as handle:
                for query in label_queries(report, protocol["reference"]["k"], attested_workers):
                    handle.write(json.dumps(query, allow_nan=False, separators=(",", ":")) + "\n")
                    total += 1
                    counts[query["status"]] += 1
                    targets.update(target["status"] for target in query["targets"])
                    jointly_eligible += int(query["target_available"] and query["feature_available"])
            result["labels"] = dict(queries=total, status_counts=dict(counts), member_outcomes=dict(targets),
                                    label_and_source_available=jointly_eligible, file=str(query_file), sha256=digest(query_file),
                                    prediction_gate_passed=False)
        result["status"] = "analyzed"
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        result["error"] = dict(type=type(error).__name__, reason=str(error))
    return result


def analyze_jobs(jobs, output, analysis_protocol):
    require(jobs and len({block for block, root in jobs}) == len(jobs), "empty_or_duplicate_order_blocks")
    measurement_revision = analysis_protocol.get("measurement_revision")
    require(type(measurement_revision) is int and measurement_revision in (2, 3),
            "analysis_measurement_revision_differs")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "analysis-protocol.json", analysis_protocol)
    blocks, fingerprints, case_ids, frozen = [], set(), None, None
    for block_id, job_root in jobs:
        require(type(block_id) is int and block_id in (0, 1), "unknown_order_block")
        manifest = read(job_root / "inputs/deployment-manifest.json")
        require(digest(job_root / "inputs/deployment-manifest.json") == analysis_protocol["deployment_manifest_sha256"],
                "manifest_differs_from_predeclared_analysis")
        protocol_file = job_root / "inputs/deployment-protocol.json"
        protocol = read(protocol_file)
        require(digest(protocol_file) == manifest["protocol_sha256"], "deployment_protocol_digest_differs")
        plan_file = job_root / "inputs/causal-training-plan-20261110-r0.json"
        require(digest(plan_file) == manifest["training_plan_sha256"] == protocol["training_plan_sha256"], "training_plan_digest_differs")
        require(digest(job_root / "inputs/prompt-bank.json") == manifest["prompt_bank_sha256"] == protocol["prompt_bank_sha256"],
                "prompt_bank_digest_differs")
        plan = read(plan_file)
        require(plan.get("schema") == protocol.get("schema") == 2
                and plan["paired_budget_gib"] == protocol["paired_budget_gib"]
                and plan["case_window_s"] == protocol["overhead"]["window_s"], "revised_measurement_plan_differs")
        require(len(plan["cases"]) == protocol["planned_cases_per_condition"] == 4
                and sum(len(case["requests"]) for case in plan["cases"]) == protocol["planned_requests_per_condition"],
                "planned_cases_or_requests_differ")
        require([list(order) for order in protocol["blocks"]] == [["off", "collect"], ["collect", "off"]], "mode_order_changed")
        fingerprints.add(digest(job_root / "inputs/deployment-manifest.json"))
        frozen = protocol
        current_ids = [case["case"] for case in plan["cases"]]
        require(case_ids is None or case_ids == current_ids, "different_case_lists_across_blocks")
        case_ids = current_ids
        block = dict(block=block_id, job_root=str(job_root), audit=audit_block(job_root, block_id, manifest, protocol, plan, analysis_protocol),
                     cases={"off": {}, "collect": {}}, usage={})
        for mode in ("off", "collect"):
            mode_root = job_root / "out/causal-block" / mode
            block["usage"][mode] = summarize_process_usage(mode_root / "process-usage.jsonl")
            try:
                require(read(mode_root / "out/calibration/causal/plan.json") == plan
                        and read(mode_root / "out/calibration/causal/mode.json")["mode"] == mode, "executed_plan_or_mode_differs")
            except (OSError, KeyError, TypeError, ValueError) as error:
                block["audit"]["errors"].append(str(error))
                block["audit"]["status"] = "not_passed"
            for case in plan["cases"]:
                case_root = mode_root / "out/calibration/causal" / case["case"]
                query_file = output / f"block{block_id}-{mode}-{case['case']}-queries.jsonl"
                block["cases"][mode][case["case"]] = analyze_case(case_root, case, mode, protocol, query_file, plan)
        for case_id in case_ids:
            off = block["cases"]["off"][case_id].get("workload_conditions")
            collect = block["cases"]["collect"][case_id].get("workload_conditions")
            if off is None or off != collect or any(value is None for value in off.values()):
                block["audit"]["errors"].append("paired_workload_conditions_differ:" + case_id)
                block["audit"]["status"] = "not_passed"
        blocks.append(block)
    require(blocks and len(fingerprints) == 1, "both_blocks_require_identical_source_configuration_manifest")
    overhead = compare_blocks(blocks, case_ids, frozen["overhead"])
    collection_passed = all(block["cases"]["collect"][case_id].get("collection", {}).get("status") == "passed"
                            if block["cases"]["collect"][case_id].get("collection") is not None else False
                            for block in blocks for case_id in case_ids)
    run_artifacts_passed = all(block["audit"]["status"] == "passed" for block in blocks)
    process_usage_available = all(usage["status"] == "sampled" for block in blocks for usage in block["usage"].values())
    result = dict(schema=1, blocks=blocks, overhead=overhead, collection_coverage_passed=collection_passed,
                  run_artifacts_passed=run_artifacts_passed,
                  process_usage_available=process_usage_available,
                  complete_overhead_and_collection=overhead["passed"] and collection_passed and run_artifacts_passed
                  and process_usage_available,
                  prediction_gate_passed=False, independent_validation=False,
                  note="Observed training targets and overhead only; Slurm/archive acceptance remains separate.")
    write(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", action="append", required=True, help="Order block id=terminal local job root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-protocol", type=Path, required=True)
    args = parser.parse_args()
    jobs = []
    for item in args.job:
        block, directory = item.split("=", 1)
        jobs.append((int(block), Path(directory)))
    require(len({block for block, _ in jobs}) == len(jobs), "duplicate_block_input")
    result = analyze_jobs(jobs, args.output, read(args.analysis_protocol))
    print(json.dumps({key: value for key, value in result.items() if key != "blocks"}, indent=2))


if __name__ == "__main__":
    main()
