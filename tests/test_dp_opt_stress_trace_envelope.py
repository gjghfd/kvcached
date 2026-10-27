import importlib
import random
from pathlib import Path

import pytest


@pytest.fixture
def envelope(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("stress_trace_envelope")


def request(identifier, blocks, residual, movable_after=0):
    return {"id": identifier, "blocks": blocks, "residual_s": residual, "movable_after_s": movable_after}


def test_shared_page_releases_only_after_last_owner(envelope):
    pool = envelope.make_pool([request("short", 32, 5), request("long", 32, 10)])
    assert envelope.pool_at(pool, 0)["pages"] == 2
    assert envelope.pool_at(pool, 5)["pages"] == 2
    assert envelope.pool_at(pool, 10)["pages"] == 1


def test_multi_request_consolidation_checks_destination_peak(envelope):
    pools = [envelope.make_pool([request("left", 128, 100)]),
             envelope.make_pool([request("right", 128, 120)])]
    plan = envelope.packing_plan(pools, 320, 768)
    assert plan["moves"] == 1 and not plan["direct"]
    before, after, plan = envelope.waiting_times(pools, 320, 768)
    assert before == 100 and after == 0
    assert envelope.packing_plan(pools, 320, 576) is None


def test_prefill_gate_waits_for_readiness_not_full_completion(envelope):
    pools = [envelope.make_pool([request("left", 128, 100, 2)]),
             envelope.make_pool([request("right", 128, 120)])]
    assert envelope.packing_plan(pools, 320, 768) is None
    before, after, plan = envelope.waiting_times(pools, 320, 768)
    assert before == 100 and after == 2


def test_code_larger_than_empty_rank_stays_unsupported(envelope):
    pools = [envelope.make_pool([]), envelope.make_pool([])]
    assert envelope.waiting_times(pools, 512, 768) == (None, None, None)


def test_reduction_is_latency_fraction_not_speedup(envelope):
    result = envelope.evaluate([{"prompt_tokens": 16, "output_tokens": 2}],
                               [{"prompt_tokens": 16, "output_tokens": 2}],
                               2, .01, .001, 768, samples=20)
    before = result["baseline_Code_latency_proxy_s"]["p95"]
    after = result["constructive_Code_latency_proxy_s"]["p95"]
    assert result["conditional_latency_reduction_fraction"]["p95"] == pytest.approx(1 - after / before)
    assert "not_GPU_result" in result["status"]


def test_compressed_page_geometry_matches_existing_block_oracle(envelope):
    other = importlib.import_module("mixed_workload_envelope")
    requests = [request("one", 31, 5), request("two", 64, 10), request("three", 129, 15)]
    pool = envelope.make_pool(requests)
    reference = other.make_pool([(entry["id"], entry["blocks"], entry["residual_s"], True) for entry in requests])
    for elapsed in (0, 5, 10, 15):
        remaining = reference
        for entry in requests:
            if entry["residual_s"] <= elapsed:
                remaining = other.remove_request(remaining, entry["id"])
        observed = envelope.pool_at(pool, elapsed)
        assert observed["pages"] == len(remaining["pages"])
        assert observed["free_slots"] == other.free_slots(remaining)


def test_offered_unsafe_states_are_counted_not_claimed_as_benefit(envelope):
    result = envelope.evaluate([{"prompt_tokens": 16, "output_tokens": 64}],
                               [{"prompt_tokens": 16, "output_tokens": 2}],
                               1024, .05, .001, 192, samples=100)
    assert result["counters"]["offered_over_budget"] > 0
    assert sum(result["counters"].get(key, 0) for key in
               ("safe_supported", "offered_over_budget", "unsupported_code_even_empty")) == 100


def test_service_and_control_sensitivity_preserve_sampled_states(envelope):
    conv_marks = [{"prompt_tokens": 32, "output_tokens": 128}]
    code_marks = [{"prompt_tokens": 512, "output_tokens": 16}]
    before = envelope.evaluate(conv_marks, code_marks, 128, .5, .001, 384, samples=100,
                               decode_step_s=.02)
    after = envelope.evaluate(conv_marks, code_marks, 128, .5, .001, 384, samples=100,
                              decode_step_s=.02, treatment_code_service_s=75, move_control_s=10)
    assert before["baseline_Code_latency_proxy_s"] == after["baseline_Code_latency_proxy_s"]
    assert after["constructive_Code_latency_proxy_s"]["mean"] >= (
        before["constructive_Code_latency_proxy_s"]["mean"] + 18)
    assert after["decode_step_proxy_s"] == .02
    assert after["prefill_proxy_s"] == .2


def test_random_event_waits_match_independent_block_owner_oracle(envelope):
    other = importlib.import_module("mixed_workload_envelope")
    rng = random.Random(20261024)
    for case in range(200):
        requests = [request(identifier, rng.randint(1, 257), rng.randint(1, 10), rng.randint(0, 2))
                    for identifier in range(rng.randint(0, 8))]
        pools = [envelope.make_pool(requests[rank::2]) for rank in (0, 1)]
        code_blocks, capacity = rng.randint(1, 512), rng.randrange(2, 20) * 96
        expected_baseline = expected_treatment = None
        events = sorted({0} | {record[key] for record in requests for key in ("residual_s", "movable_after_s")})
        for elapsed in events:
            code_pools = [other.make_pool([]), other.make_pool([])]
            conv_pools = []
            for rank in (0, 1):
                records = requests[rank::2]
                pool = other.make_pool([(record["id"], record["blocks"], record["residual_s"], True)
                                        for record in records])
                for record in records:
                    if record["residual_s"] <= elapsed:
                        pool = other.remove_request(pool, record["id"])
                conv_pools.append(pool)
            observations = [code_pools, conv_pools]
            if other.direct_fit(observations, code_blocks, capacity):
                if expected_baseline is None:
                    expected_baseline = elapsed
                if expected_treatment is None:
                    expected_treatment = elapsed
            remaining = [record for record in requests if record["residual_s"] > elapsed]
            if expected_treatment is None and all(record["movable_after_s"] <= elapsed for record in remaining):
                used = other.mapped_mib(observations)
                for source in (0, 1):
                    destination = 1 - source
                    blocks = sum(len(record["blocks"]) for record in conv_pools[source]["requests"])
                    destination_peak = used[destination] + other.increment_mib(conv_pools[destination], blocks)
                    empty_source = other.make_pool([])
                    source_after = (len(empty_source["pages"]) + len(code_pools[source]["pages"])) * 96
                    source_after += other.increment_mib(code_pools[source], code_blocks)
                    if max(destination_peak, source_after) <= capacity:
                        expected_treatment = elapsed
                        break
        baseline, treatment, plan = envelope.waiting_times(pools, code_blocks, capacity)
        assert (baseline, treatment) == (expected_baseline, expected_treatment), case
