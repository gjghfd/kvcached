import json

import pytest

from migration_roundtrip_control import exact_state_receipt
from verify_roundtrip_transaction import verify


def inventory(source, blocks):
    observed = [[{"requests": [], "pages": [], "mapped_bytes": 0, "page_bytes": 64 * 1024**2,
                  "block_bytes": 1024**2, "page_block_capacity": 64}
                 for rank in (0, 1)] for model in (0, 1)]
    page = blocks[0] // 64
    observed[1][source].update({"requests": [{"id": "request", "blocks": blocks, "movable": True,
        "computed_tokens": 1100, "output_tokens": 1024}], "pages": [{"id": page, "owners": ["request"]}],
        "mapped_bytes": 64 * 1024**2})
    return observed


def fixture(directory):
    original = {"request_id": "request", "request": b"serialized", "blocks": [64, 65]}
    states = [exact_state_receipt(original, {**original, "blocks": blocks})
              for blocks in ([128, 129], [192, 193])]
    hops = [{"request_id": "request", "model": 1, "source": source, "destination": 1 - source,
             "bitwise_verified": True, "integrity": "device_echo_full_bytewise", "kv_bytes": 2 * 1024**2,
             "copy_peak_gib": [.0625, .0625]} for source in (0, 1)]
    control = {"mode": "paused_roundtrip", "model": 1, "source": 0, "budgets_gib": [1, 1],
               "final_state": states[-1], "states": states, "hops": hops}
    files = {"control.json": control, "inventory-before.json": inventory(0, [64, 65]),
             "inventory-after-hop-1.json": inventory(1, [128, 129]),
             "inventory-after-hop-2.json": inventory(0, [192, 193]),
             "inventory-final.json": inventory(0, [192, 193])}
    for name, value in files.items():
        (directory / name).write_text(json.dumps(value))
    for name in ("request-before.pkl", "request-after-hop-1.pkl", "request-after-hop-2.pkl", "request-final.pkl"):
        (directory / name).write_bytes(b"serialized")
    return control


def test_independent_exact_state_and_owner_verification(tmp_path):
    fixture(tmp_path)
    result = verify(tmp_path)
    assert result["hops"] == 2 and result["is_benefit"] is False
    assert result["mixed_batch_correctness_proven"] is False


def test_corrupt_serialized_state_fails_without_tolerance(tmp_path):
    fixture(tmp_path)
    (tmp_path / "request-after-hop-1.pkl").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="serialized"):
        verify(tmp_path)


@pytest.mark.parametrize("case", ["peak", "nonfinite", "nan_peak", "progress"])
def test_reject_peak_invalid_budget_and_intermediate_decode(tmp_path, case):
    control = fixture(tmp_path)
    if case == "peak":
        control["hops"][0]["copy_peak_gib"][0] = 2
        (tmp_path / "control.json").write_text(json.dumps(control))
    elif case == "nan_peak":
        control["hops"][0]["copy_peak_gib"][0] = float("nan")
        (tmp_path / "control.json").write_text(json.dumps(control))
    elif case == "nonfinite":
        control["budgets_gib"][0] = float("nan")
        (tmp_path / "control.json").write_text(json.dumps(control))
    else:
        path = tmp_path / "inventory-after-hop-1.json"
        observed = json.loads(path.read_text())
        observed[1][1]["requests"][0]["output_tokens"] += 1
        path.write_text(json.dumps(observed))
    with pytest.raises(ValueError):
        verify(tmp_path)
