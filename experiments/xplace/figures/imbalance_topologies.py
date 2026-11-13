import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import osdi
import paper_data as evidence
from analyze_imbalance import analyze


TP2_ROOT = evidence.TP2_ROOT


def verified_analysis(path, topology, raw_root):
    analysis = evidence.read(path)
    actual = (analysis["num_ranks"], analysis["tensor_parallel_size"])
    if analysis["status"] != "passed" or actual != topology or analysis["expert_parallel_size"] != 4:
        raise ValueError(f"Expected verified DP{topology[0]}/TP{topology[1]}/EP4 data: {path}")
    if {point["batch_size"] for point in analysis["points"]} != {32, 64, 128}:
        raise ValueError("All three batch sizes must be present")
    if analyze(raw_root) != analysis:
        raise ValueError(f"Analysis differs from the complete raw measurements: {path}")
    return analysis


def comparison_figure(tp2_path, tp2_root=TP2_ROOT):
    analyses = [verified_analysis(evidence.DENSE_ANALYSIS, (4, 1), evidence.DENSE_ROOT),
                verified_analysis(tp2_path, (2, 2), tp2_root)]
    if any((analysis["context_tokens"], analysis["output_tokens"]) != (4096, 1024) for analysis in analyses):
        raise ValueError("Both calibrations must use 4096 input and 1024 output tokens")
    figure, axes = plt.subplots(1, 2, figsize=(osdi.FULL, 2.15), sharey=True,
                                gridspec_kw={"wspace": 0.18})
    styles = [(32, osdi.COLOR["corun"], "o", "-"), (64, osdi.COLOR["ours"], "s", "--"),
              (128, "#009E73", "^", "-.")]
    for axis, analysis, panel in zip(axes, analyses, ("(a) DP4 / TP1 / EP4", "(b) DP2 / TP2 / EP4")):
        for batch_size, color, marker, linestyle in styles:
            points = sorted((point for point in analysis["points"] if point["batch_size"] == batch_size),
                            key=lambda point: point["normalized_imbalance"])
            spreads = [point["normalized_imbalance"] for point in points]
            axis.plot(spreads, [point["mean_ms"] for point in points], color=color, marker=marker,
                      linestyle=linestyle, markerfacecolor="white", markersize=3.5, label=f"BS{batch_size}")
            axis.fill_between(spreads, [point["min_ms"] for point in points],
                              [point["max_ms"] for point in points], color=color, alpha=0.15, linewidth=0)
        axis.set_xlabel("归一化不均衡度")
        axis.set_xticks([0, 0.5, 1, 1.5, 2])
        axis.set_xlim(-0.05, 2.05)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dddddd", linewidth=0.4, linestyle=":")
        osdi.subcaption(axis, panel, y=-0.36)
    axes[0].set_ylabel("decode 步时 (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.91), ncol=3)
    return figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp2-analysis", type=Path,
                        default=evidence.TP2_ANALYSIS)
    parser.add_argument("--tp2-root", type=Path, default=TP2_ROOT)
    args = parser.parse_args()
    osdi.apply()
    print(osdi.save(comparison_figure(args.tp2_analysis, args.tp2_root), "imbalance_topologies"))


if __name__ == "__main__":
    main()
