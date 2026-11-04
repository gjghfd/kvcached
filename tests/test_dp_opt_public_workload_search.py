import importlib
import json
import random
import sys
from pathlib import Path

import pytest


@pytest.fixture
def search(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments/dp_opt"))
    return importlib.import_module("public_workload_search")


@pytest.fixture
def inputs(tmp_path):
    conv = [{"prompt_key": "public-one", "prompt_tokens": 100, "output_tokens": 16001,
             "original_output_tokens": 16001, "original_response_sha256": "a" * 64, "count": 1},
            {"prompt_key": "public-two", "prompt_tokens": 200, "output_tokens": 8525,
             "original_output_tokens": 8525, "original_response_sha256": "b" * 64, "count": 1}]
    code = [{"prompt_key": "swe-real", "prompt_tokens": 100001, "output_tokens": 64, "count": 1}]
    bank = {"prompts": {mark["prompt_key"]: {} for mark in code + conv}}
    for name, payload in (("conv-marks.json", conv), ("code-marks.json", code), ("prompt-bank.json", bank)):
        (tmp_path / name).write_text(json.dumps(payload))
    return tmp_path


@pytest.mark.parametrize("population", [2, 3, 4])
def test_native_public_grid_uses_original_response_lengths(search, inputs, population):
    trace = search.make_native(inputs, search.SEEDS[0], population)
    assert len(trace["requests"]) == population + 1
    assert trace["requests"][-1]["arrival_s"] == 90
    assert all(mark["output_tokens"] == mark["original_output_tokens"] for mark in trace["requests"][:-1])
    assert "synthetic_Conv_output_tokens" not in trace
    assert search.make_native(inputs, search.SEEDS[0], population) == trace


@pytest.mark.parametrize("seed,population", [(123, 2), (2026103101, 5)])
def test_no_outcome_dependent_native_reroll(search, inputs, seed, population):
    with pytest.raises(ValueError):
        search.make_native(inputs, seed, population)


def test_full_distribution_poisson_includes_shorter_original_marks(search, inputs):
    trace = search.make_poisson(inputs, search.CALIBRATION_SEED, .01)
    outputs = {mark["output_tokens"] for mark in trace["requests"] if mark["model"] == 1}
    assert outputs == {8525, 16001}
    assert trace["arrival"] == "two_independent_Poisson_streams"
    assert trace["allow_empty_tenants"] is True
    assert trace["model_names"][1] == "OpenThoughts3-original-reasoning"


def test_zero_code_draw_is_kept_not_rerolled_or_bootstrapped(search, inputs):
    trace = search.make_poisson(inputs, 123, .01, code_rate=1e-12)
    assert not any(mark["model"] == 0 for mark in trace["requests"])
    assert trace["requests"][0]["arrival_s"] > 0


@pytest.mark.parametrize("field,value", [("output_tokens", 24576), ("prompt_key", "missing"),
                                       ("synthetic_output_tokens", 16001)])
def test_trace_rejects_extended_response_or_synthetic_prompt(search, inputs, field, value):
    trace = search.make_native(inputs, search.SEEDS[0], 3)
    trace["requests"][0][field] = value
    with pytest.raises(ValueError):
        search.original_trace(trace, inputs)


def test_rate_prior_uses_original_joint_moment_and_measured_service(search):
    marks = [{"prompt_tokens": 100, "output_tokens": 1000}, {"prompt_tokens": 100, "output_tokens": 10000}]
    records = [{"model": 1, "first_token_at": 1., "completed_at": 3., "token_ids": [1, 2, 3]}]
    result = search.measured_rate_priors(marks, records, 9600)
    moment = (192 * 1000 + 960 * 10000) / 2
    assert result["bank_mean_independent_page_times_output_mib_tokens"] == moment
    assert result["conv_rates_rps"][1] == pytest.approx(5376 / moment)
    assert result["scope"].endswith("actual_pressure_verification")


def test_supplemental_cost_plan_covers_new_n4_without_changing_serving_budget(search, monkeypatch, tmp_path):
    monkeypatch.setattr(search, "validate", lambda path: {"dataset_label": "openthoughts3"})
    frozen = tmp_path / "frozen"
    (frozen / "inputs").mkdir(parents=True)
    (frozen / "inputs/input-manifest.json").write_text("{}")
    monkeypatch.setattr(search, "make_native", lambda inputs, seed, population: {"seed": seed, "population": population})
    monkeypatch.setattr(search, "validate_frozen", lambda path: None)
    config = search.prepare(tmp_path)
    assert config["cost_profile_screen"]["maximum_mapped_mib"] < 10240
    assert config["cost_profile_screen"]["cases"] == 144
    assert len(config["trace_sha256"]) == 6
    assert json.loads((frozen / "calibration-plan.json").read_text())["populations"][-1]["population"] == 4


@pytest.mark.parametrize("policies,pilot", [(["no_move"], True), (["no_move"], False),
    (["no_move", "random_no_return", "kv_only_headroom", "kv_only_workaware"], False)])
def test_actual_live_CLI_accepts_policy_specific_optimizer_without_importing_inference(
        search, monkeypatch, policies, pilot):
    live = importlib.import_module("live_trace")
    received = []
    arguments = ["live_trace.py", "--checkpoint", "unused", "--fit", "unused.json", "--trace", "unused.json",
                 "--output", "unused", "--policies", *policies,
                 "--workaware-optimizer", search.phase_optimizer(policies)]
    if "kv_only_workaware" in policies:
        arguments += ["--workaware-fit", "fit.json", "--workaware-fit-sha256", "a" * 64,
                      "--workaware-source-root", "source", "--workaware-rate-priors", ".001", ".012"]
    if pilot:
        arguments += ["--workaware-cadence-pilot"]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(live, "run", lambda args: args)
    monkeypatch.setattr(live.asyncio, "run", received.append)
    live.main()
    assert received[0].workaware_optimizer == ("robust" if "kv_only_workaware" in policies else "point")
    batch = (Path(search.__file__).parent / "public-workload-search.sbatch").read_text()
    assert 'phase_optimizer(os.environ["DP_OPT_TRACE_POLICIES"].split())' in batch
