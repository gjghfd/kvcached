import hashlib
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


POLICIES = ("random_no_return", "kv_only_workaware")


@pytest.fixture
def repayment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("analyze_return_repayment")


def conv_record(samples, origin=0, identifier="Conv"):
    return {
        "request_id": identifier,
        "model": 1,
        "prompt_tokens": 2048,
        "token_ids": list(range(samples[-1][0])),
        "cadence": [
            {"tokens": ordinal, "time": origin + elapsed} for ordinal, elapsed in samples
        ],
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload))


@pytest.fixture
def phase(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "exit-status").write_text("0\n")
    write_json(root / "trace.json", {
        "duration_s": 20,
        "requests": [{"request_id": "Conv"}, {"request_id": "Code"}],
    })
    for policy, origin, release, times in (
        (POLICIES[0], 10000, .5, list(range(1, 9))),
        (POLICIES[1], 100, 1.5, [1, 2, 3, 4.5, 6, 6.25, 6.5, 7]),
    ):
        directory = root / f"r0-{policy}"
        directory.mkdir()
        write_json(directory / "requests.json", [
            conv_record(list(enumerate(times, 1)), origin),
            {"request_id": "Code", "model": 0, "arrived_at": origin,
             "completed_at": origin + release},
        ])
        write_json(directory / "workload.json", {"origin": origin})
        decisions = [] if policy == POLICIES[0] else [{
            "path": "release_rebalance_ilp",
            "reason": "rebalanced",
            "started_at": origin + 4,
            "resume_completed_at": origin + 6,
            "moves": [{"model": 1}],
        }]
        write_json(directory / "decisions.json", decisions)
    return root


def append_code(root, identifier, intervals):
    trace_path = root / "trace.json"
    trace = json.loads(trace_path.read_text())
    trace["requests"].append({"request_id": identifier})
    write_json(trace_path, trace)
    for policy, (arrived, completed) in zip(POLICIES, intervals):
        directory = root / f"r0-{policy}"
        origin = json.loads((directory / "workload.json").read_text())["origin"]
        path = directory / "requests.json"
        requests = json.loads(path.read_text())
        requests.append({"request_id": identifier, "model": 0,
                         "arrived_at": origin + arrived, "completed_at": origin + completed})
        write_json(path, requests)


@pytest.mark.parametrize("arrivals,last_ordinal", [((7, 11), 6), ((11, 6.5), 6)])
def test_next_Code_boundary_uses_each_policy_clock_and_is_exclusive(
    repayment, phase, arrivals, last_ordinal
):
    append_code(phase, "next-Code", [(arrival, arrival + 1) for arrival in arrivals])
    result = repayment.analyze(phase)
    window = result["windows"][0]
    span = window["requests"][0]
    assert window["release_relative_times_s"] == [.5, 1.5]
    assert window["end_relative_times_s"] == list(arrivals)
    assert span["anchor_relative_times_s"] == [3, 3]
    assert span["last_token_ordinal"] == last_ordinal
    assert span["net_saved_s"] == pytest.approx(-.25)


@pytest.mark.parametrize("occupied_policy", [0, 1])
def test_Code_arriving_exactly_at_release_on_either_clock_has_no_absent_window(
    repayment, phase, occupied_policy
):
    intervals = [(11, 12), (11, 12)]
    intervals[occupied_policy] = ((.5, 1.5)[occupied_policy], 10)
    append_code(phase, "touching-Code", intervals)
    result = repayment.analyze(phase)
    assert result["included_spans"] == 0
    assert result["windows"][0]["exclusion"] == (
        "overlapping_Code_or_no_positive_Code_absent_window"
    )
    assert result["windows"][0]["end_relative_times_s"][occupied_policy] is None


def test_anchor_precedes_control_even_when_tokens_arrive_during_control(repayment, phase):
    span = repayment.analyze(phase)["windows"][0]["requests"][0]
    assert span["anchor_token_ordinal"] == 3
    assert span["anchor_relative_times_s"] == [3, 3]
    assert span["return_control_elapsed_s"] == 2
    assert span["treatment_duration_s"] == (4 - 3) + (6 - 4) + (7 - 6)
    assert span["baseline_duration_s"] == 8 - 3
    assert span["maximum_observed_debt_s"] == 1
    assert span["net_saved_s"] == 1


def test_batched_tokens_at_start_and_resume_count_intervals_not_prefix_samples(repayment):
    before = conv_record([(32, 1), (64, 2), (96, 3), (128, 4), (160, 5), (192, 6)], 100)
    after = conv_record([(32, 1), (64, 2), (96, 3), (128, 6), (160, 6), (192, 6)], 200)
    span = repayment.paired_repayment_span(
        before, after, [100, 200], {"start_s": 3, "resume_s": 6}, [3, 3], [10, 10]
    )
    assert span["anchor_token_ordinal"] == 96
    assert span["matched_prefix_samples"] == 3
    assert span["matched_token_intervals"] == 96
    assert span["baseline_duration_s"] == span["treatment_duration_s"] == 3
    assert span["maximum_observed_debt_s"] == 2
    assert span["first_repaid_prefix"]["token_ordinal"] == 192
    assert span["sustained_repaid_prefix"]["token_ordinal"] == 192


def test_disjoint_pre_return_batches_cannot_manufacture_a_common_anchor(repayment):
    before = conv_record([(32, 1), (96, 3), (128, 4), (192, 6)])
    after = conv_record([(64, 2), (128, 5), (192, 7)])
    span = repayment.paired_repayment_span(
        before, after, [0, 0], {"start_s": 3, "resume_s": 5}, [0, 0], [10, 10]
    )
    assert span == {"exclusion": "no_common_pre_return_ordinal_after_both_Code_releases"}


@pytest.mark.parametrize("tail,sustained_ordinal", [([8, 9], None), ([6.5, 7], 6)])
def test_exact_breakeven_before_later_debt_is_not_sustained_repayment(
    repayment, tail, sustained_ordinal
):
    before = conv_record(list(enumerate(range(1, 9), 1)))
    after = conv_record(list(enumerate([1, 2, 3, 4, 6, 6, *tail], 1)))
    span = repayment.paired_repayment_span(
        before, after, [0, 0], {"start_s": 3, "resume_s": 4}, [0, 0], [20, 20]
    )
    assert span["first_repaid_prefix"]["token_ordinal"] == 4
    assert span["first_repaid_prefix"]["net_saved_s"] == 0
    assert span["maximum_observed_debt_s"] == 1
    if sustained_ordinal is None:
        assert span["sustained_repaid_prefix"] is None
        assert span["net_saved_s"] == -1
    else:
        assert span["sustained_repaid_prefix"]["token_ordinal"] == sustained_ordinal
        assert span["sustained_repaid_prefix"]["net_saved_s"] == 0
        assert span["net_saved_s"] == 1


def test_complete_unselected_policy_cannot_fill_complementary_selected_subsets(repayment, phase):
    trace_path = phase / "trace.json"
    trace = json.loads(trace_path.read_text())
    trace["requests"].append({"request_id": "other-Conv"})
    write_json(trace_path, trace)
    complete = phase / "r0-kv_only_headroom"
    complete.mkdir()
    for policy, missing in zip(POLICIES, ("other-Conv", "Conv")):
        directory = phase / f"r0-{policy}"
        path = directory / "requests.json"
        requests = json.loads(path.read_text())
        requests.append({**requests[0], "request_id": "other-Conv"})
        if policy == POLICIES[0]:
            write_json(complete / "requests.json", requests)
            write_json(complete / "workload.json", {"origin": 10000})
            write_json(complete / "decisions.json", [])
        write_json(path, [record for record in requests if record["request_id"] != missing])
    with pytest.raises(ValueError, match="all offered paired requests"):
        repayment.analyze(phase)


@pytest.mark.parametrize("policy", POLICIES)
def test_same_length_duplicate_request_set_is_not_a_complete_selected_policy(
    repayment, phase, policy
):
    path = phase / f"r0-{policy}" / "requests.json"
    requests = json.loads(path.read_text())
    write_json(path, [requests[0], requests[0]])
    with pytest.raises(ValueError, match="all offered paired requests"):
        repayment.analyze(phase)


def test_unselected_incomplete_policy_is_not_read_or_pooled_into_the_pair(repayment, phase):
    directory = phase / "r0-no_move"
    directory.mkdir()
    (directory / "requests.json").write_text("incomplete JSON")
    result = repayment.analyze(phase)
    assert result["included_spans"] == 1
    assert result["observed_span_saved_request_s"] == 1
    assert len(result["source_files_sha256"]) == 8
    assert not any("r0-no_move" in path for path in result["source_files_sha256"])


def test_coincident_Code_releases_do_not_double_count_the_same_Conv_span(repayment, phase):
    append_code(phase, "co-released-Code", [(.25, .5), (.25, 1.5)])
    result = repayment.analyze(phase)
    assert (result["included_spans"], result["observed_span_saved_request_s"]) == (1, 1)
    assert len(result["windows"]) == 1
    assert set(result["windows"][0]["Code_request_ids"]) == {"Code", "co-released-Code"}


@pytest.fixture
def cli_paths(phase, monkeypatch):
    manifest = phase.parent / "raw.sha256"
    manifest.write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(phase)}\n"
        for path in sorted(phase.rglob("*")) if path.is_file()
    ))
    bank = phase.parent / "bank.json"
    output = phase.parent / "audit.json"
    monkeypatch.setattr(sys, "argv", [
        "repayment-audit", "--root", str(phase), "--manifest", str(manifest),
        "--prompt-bank", str(bank), "--output", str(output),
    ])
    return {"manifest": manifest, "bank": bank, "output": output}


def test_existing_output_is_preserved_before_even_reading_the_SHA_manifest(
    repayment, cli_paths, monkeypatch
):
    output = cli_paths["output"]
    sentinel = b"prior diagnostic must survive\x00\xff"
    output.write_bytes(sentinel)
    cli_paths["manifest"].unlink()
    guards = []
    for name in ("verify_raw", "verify", "analyze"):
        guard = Mock(side_effect=AssertionError(f"{name} must not run"))
        monkeypatch.setattr(repayment, name, guard)
        guards.append(guard)
    with pytest.raises(SystemExit) as error:
        repayment.main()
    assert error.value.code == 2
    assert output.read_bytes() == sentinel
    for guard in guards:
        guard.assert_not_called()


def test_real_SHA_failure_never_reaches_ownership_analysis_or_creates_output(
    repayment, phase, cli_paths, monkeypatch
):
    trace = phase / "trace.json"
    trace.write_bytes(trace.read_bytes() + b"\n")
    ownership = Mock(side_effect=AssertionError("ownership must follow SHA validation"))
    analysis = Mock(side_effect=AssertionError("analysis must follow SHA validation"))
    monkeypatch.setattr(repayment, "verify", ownership)
    monkeypatch.setattr(repayment, "analyze", analysis)
    with pytest.raises(ValueError, match="checksum mismatch"):
        repayment.main()
    ownership.assert_not_called()
    analysis.assert_not_called()
    assert not cli_paths["output"].exists()


def test_valid_SHA_but_failed_ownership_does_not_run_analysis_or_create_output(
    repayment, phase, cli_paths, monkeypatch
):
    ownership = Mock(side_effect=ValueError("ownership mismatch"))
    analysis = Mock(side_effect=AssertionError("analysis must follow ownership validation"))
    monkeypatch.setattr(repayment, "verify", ownership)
    monkeypatch.setattr(repayment, "analyze", analysis)
    with pytest.raises(ValueError, match="ownership mismatch"):
        repayment.main()
    ownership.assert_called_once_with(phase, cli_paths["bank"])
    analysis.assert_not_called()
    assert not cli_paths["output"].exists()


def test_output_created_during_verification_is_not_overwritten(
    repayment, phase, cli_paths, monkeypatch
):
    output = cli_paths["output"]
    sentinel = b"concurrent diagnostic"

    def finish_verification(root, bank):
        assert root == phase
        assert bank == cli_paths["bank"]
        output.write_bytes(sentinel)
        return {"status": "verified"}

    monkeypatch.setattr(repayment, "verify", finish_verification)
    with pytest.raises(FileExistsError):
        repayment.main()
    assert output.read_bytes() == sentinel
