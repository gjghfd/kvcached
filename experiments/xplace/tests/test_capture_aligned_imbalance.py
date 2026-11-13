from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))

from imbalance_calibration import capture_aligned_blocks, scan_cases


def test_all_capture_boundaries_and_neighbors_are_predeclared():
    blocks = capture_aligned_blocks()
    assert len(blocks) == 6
    assert [block["series"][0]["batch_size"] for block in blocks] == [128, 256, 256, 128, 128, 256]
    for block in blocks:
        cases = scan_cases(block, 2, 2)
        batch = block["series"][0]["batch_size"]
        light_counts = {min(case["layout"]) for case in cases}
        for boundary in (0, 1, 8, 16, 32, 64, 128):
            for offset in (-1, 0, 1):
                count = boundary + offset
                if boundary <= batch // 2 and 0 <= count <= batch // 2:
                    assert count in light_counts
        assert all(sum(case["layout"]) == batch for case in cases)
        assert all(max(case["layout"]) <= 256 for case in cases)
        assert len(cases) == (14 if batch == 128 else 17)
        assert (len(cases) + 1) * block["case_timeout_s"] <= block["max_scan_s"]


def test_repeats_keep_inputs_and_rotate_heavy_rank():
    for batch in (128, 256):
        blocks = [block for block in capture_aligned_blocks() if block["series"][0]["batch_size"] == batch]
        scans = [scan_cases(block, 2, 2) for block in blocks]
        assert len({block["seed"] for block in blocks}) == 1
        assert all(block["context_tokens"] == 4096 and block["output_tokens"] == 1024 for block in blocks)
        reference = {case["delta"]: case["layout"] for case in scans[0]}
        for repeat, cases in enumerate(scans):
            for case in cases:
                layout = reference[case["delta"]]
                assert case["layout"] == (layout if repeat % 2 == 0 else list(reversed(layout)))
                assert case["capture_bucket"] == (batch // 2 if case["delta"] == 0 else batch)
