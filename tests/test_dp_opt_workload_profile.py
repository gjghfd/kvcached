import importlib
from pathlib import Path

import pytest


@pytest.fixture
def profile(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("workload_profile")


def test_weighted_quantiles_are_nearest_rank_not_unique_mark_quantiles(profile):
    result = profile.histogram_statistics({8: 90, 4096: 9, 1000000: 1})
    assert result["requests"] == 100 and result["quantiles_nearest_rank"]["0.9"] == 8
    assert result["quantiles_nearest_rank"]["0.99"] == 4096
    assert result["fraction_ge"]["1000000"] == .01


def test_joint_duration_proxy_retains_code_short_output_and_conv_long_output(profile):
    code = profile.paired_statistics({(8192, 8): 10})
    conv = profile.paired_statistics({(2048, 512): 10})
    assert code["mean_logical_reserved_mib"] > conv["mean_logical_reserved_mib"]
    assert code["service_proxy"]["mean_resident_s"] < conv["service_proxy"]["mean_resident_s"]
    assert code["service_proxy"]["mean_full_reservation_mib_seconds"] < conv["service_proxy"]["mean_full_reservation_mib_seconds"]


def test_full_profile_reads_every_pair_and_reports_nonpositive_marks(profile, tmp_path):
    path = tmp_path / "test.csv"
    path.write_text("TIMESTAMP,ContextTokens,GeneratedTokens\nnow,8192,8\nnow,8192,8\nnow,32,0\n")
    joint, invalid = profile.read_azure_full(path)
    assert joint[(8192, 8)] == 2 and invalid == 1


def test_memory_is_not_sharded_by_expert_parallelism(profile):
    result = profile.memory_scale({"num_hidden_layers": 48, "num_key_value_heads": 4, "head_dim": 128})
    assert result["bytes_per_token"] == 98304
    assert result["logical_kv_gib"]["32768"] == 3
    assert result["logical_kv_gib"]["1000000"] > 91


def test_reference_answers_not_presented_as_generation_measurements(profile):
    records = [{"source": "LongBench-v2", "task": "Code Repository Understanding", "context_tokens": 1000000,
                "prompt_tokens": 1000010, "reference_answer_tokens": 1}]
    result = profile.longbench_profile(records)
    assert result["datasets"]["LongBench-v2"]["Code Repository Understanding"]["supported_prompt_plus_512_fraction"]["1000000"] == 0
    assert "not_measured_generation_length" in result["output_scope"]


def test_conv_migration_can_admit_large_shortdecode_code(profile):
    envelope = importlib.import_module("mixed_workload_envelope")
    code = [envelope.make_pool([]), envelope.make_pool([])]
    conv = [envelope.make_pool([("conv0", 128, 15, True), ("conv1", 128, 12, True)]),
            envelope.make_pool([("conv2", 128, 10, True), ("conv3", 128, 8, True)])]
    observations = [code, conv]
    capacity = 768
    assert envelope.mapped_mib(observations) == [576, 576]
    assert not envelope.direct_fit(observations, 192, capacity)
    move = envelope.conv_one_move(observations, 192, capacity, True)
    assert move is not None and move["released_source_mib"] == 192
    assert move["incoming_incremental_mib"] == 288
    assert max(move["final_after_admission_mib"]) <= capacity
    assert envelope.natural_wait(observations, 192, capacity) == 8
    assert envelope.conv_one_move(observations, 192, 672, True) is None
    snapshot = envelope.conservative_ilp_snapshot(observations, 192, capacity)
    solver = importlib.import_module("ilp_reference")
    assert solver.enumerate_targets(snapshot, max_moves=2)["winner"] is not None
    assert solver.enumerate_targets(dict(snapshot, movable=[]), max_moves=0)["winner"] is None


def test_conv_handoff_accounts_for_reusable_code_pool_space(profile):
    envelope = importlib.import_module("mixed_workload_envelope")
    observations = [[envelope.make_pool([]), envelope.make_pool([])],
                    [envelope.make_pool([("conv0-base", 33, 10, True), ("conv0-tail", 32, 12, True)]),
                     envelope.make_pool([("conv1", 64, 15, True)])]]
    assert envelope.mapped_mib(observations) == [288, 288]
    assert not envelope.direct_fit(observations, 120, 288)
    move = envelope.conv_one_move(observations, 120, 288, True)
    assert move is not None and move["incoming_incremental_mib"] == 96
    assert move["final_used_mib"] == [192, 288]
    assert move["final_after_admission_mib"] == [288, 288]
    assert move["final_used_mib"][move["target"]] + 120 * envelope.BLOCK_MIB > 288


def test_stationary_sampler_weights_conv_by_residence_not_arrival_count(profile):
    envelope = importlib.import_module("mixed_workload_envelope")
    sampler = envelope.MarkSampler([{"prompt_tokens": 1024, "output_tokens": 8, "count": 1},
                                    {"prompt_tokens": 1024, "output_tokens": 512, "count": 1}])
    assert sampler.residents[-1] > 20 and sampler.arrivals[-1] == 2
