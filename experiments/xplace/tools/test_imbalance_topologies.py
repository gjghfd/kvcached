import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))

try:
    import matplotlib.pyplot as plt
    import imbalance_topologies as plotting
except ImportError:
    plotting = None


@unittest.skipIf(plotting is None, "Plot tests need the paper plotting environment")
class TopologyPlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paths = {4: self.root / "dp4.json", 2: self.root / "dp2.json"}
        self.reports = {}
        for ranks in (4, 2):
            points = []
            for batch in (32, 64, 128):
                for spread in (0, 2, batch // ranks):
                    step = 10 + batch / 8 + spread / 4 + ranks
                    points.append({"batch_size": batch, "imbalance": spread,
                                   "normalized_imbalance": spread / (batch / ranks),
                                   "mean_ms": step, "min_ms": step - 0.5, "max_ms": step + 0.5})
            report = {"status": "passed", "num_ranks": ranks, "tensor_parallel_size": 4 // ranks,
                      "expert_parallel_size": 4, "context_tokens": 4096, "output_tokens": 1024,
                      "points": points}
            self.reports[ranks] = report
            self.paths[ranks].write_text(json.dumps(report))
        self.addCleanup(plt.close, "all")

    def test_rejects_wrong_topology_missing_series_and_stale_analysis(self):
        for damage in ("topology", "series", "stale", "status"):
            with self.subTest(damage=damage):
                report = copy.deepcopy(self.reports[2])
                if damage == "topology":
                    report["tensor_parallel_size"] = 1
                elif damage == "series":
                    report["points"] = [point for point in report["points"] if point["batch_size"] != 128]
                elif damage == "status":
                    report["status"] = "incomplete"
                else:
                    report["points"][0]["mean_ms"] += 1
                self.paths[2].write_text(json.dumps(report))
                with patch.object(plotting, "analyze", return_value=self.reports[2]):
                    with self.assertRaises(ValueError):
                        plotting.verified_analysis(self.paths[2], (2, 2), self.root)

    def test_panels_preserve_all_points_topologies_and_min_max_bands(self):
        raw_roots = {4: self.root / "raw4", 2: self.root / "raw2"}

        def recompute(root):
            return self.reports[4 if root == raw_roots[4] else 2]

        with patch.object(plotting.evidence, "DENSE_ANALYSIS", self.paths[4]), \
                patch.object(plotting.evidence, "DENSE_ROOT", raw_roots[4]), \
                patch.object(plotting, "analyze", side_effect=recompute):
            figure = plotting.comparison_figure(self.paths[2], raw_roots[2])
        self.assertEqual(len(figure.axes), 2)
        for ranks, axis in zip((4, 2), figure.axes):
            self.assertEqual(len(axis.lines), 3)
            self.assertEqual(len(axis.collections), 3)
            self.assertFalse(axis.patches)
            self.assertEqual(len(axis.texts), 1)
            self.assertIn(f"DP{ranks} / TP{4 // ranks} / EP4", axis.texts[0].get_text())
            for batch, curve, band in zip((32, 64, 128), axis.lines, axis.collections):
                points = [point for point in self.reports[ranks]["points"] if point["batch_size"] == batch]
                self.assertEqual(curve.get_label(), f"BS{batch}")
                self.assertEqual(list(curve.get_xdata()), [point["normalized_imbalance"] for point in points])
                self.assertEqual(list(curve.get_ydata()), [point["mean_ms"] for point in points])
                vertices = band.get_paths()[0].vertices
                for point in points:
                    values = vertices[vertices[:, 0] == point["normalized_imbalance"], 1]
                    self.assertEqual(min(values), point["min_ms"])
                    self.assertEqual(max(values), point["max_ms"])
            self.assertEqual(len({curve.get_marker() for curve in axis.lines}), 3)
            self.assertEqual(len({curve.get_linestyle() for curve in axis.lines}), 3)
        self.assertTrue(figure.axes[0].get_shared_y_axes().joined(*figure.axes))


if __name__ == "__main__":
    unittest.main()
