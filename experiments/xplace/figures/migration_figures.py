"""Manuscript schematics and plots using the migration-only evidence selection."""
import statistics

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import osdi
import paper_data as evidence
from fragmentation_pptx import render_fragmentation
from architecture_pptx import render_architecture

COLORS = {"B": osdi.COLOR["corun"], "X": osdi.COLOR["ours"]}


def migration1_fragmentation():
    return render_fragmentation(evidence.FIGURES[0])


def padding_figure():
    grid = {case["name"]: case for case in evidence.read(evidence.J1B_GRID) if case.get("step_s")}
    figure, axes = plt.subplots(1, 2, figsize=(osdi.FULL * 0.88, 1.85), gridspec_kw={"wspace": 0.32})
    axis = axes[0]
    balanced = sorted((case["layout"][0], 1000 * case["step_s"]) for name, case in grid.items()
                      if name.startswith("bal") and name.endswith("_c4k"))
    for bucket in (8, 16, 32):
        axis.axvline(bucket, color=osdi.COLOR["gray"], linestyle=":", linewidth=0.8)
        axis.text(bucket, 28.7, f"G{bucket}", ha="center", va="bottom", fontsize=6.5)
    axis.plot([count for count, _ in balanced], [step for _, step in balanced], marker="o", color=COLORS["B"])
    axis.set_xlabel("每个 rank 的 Conv 数")
    axis.set_ylabel("decode 步时 (ms)")
    axis.set_xticks([1, 4, 8, 12, 16, 20, 24, 32])
    axis.set_xlim(0, 33)
    axis.set_ylim(8, 28.5)
    osdi.subcaption(axis, "(a) 均衡布局的档位效应", y=-0.36)
    axis = axes[1]
    dense = evidence.read(evidence.DENSE_ANALYSIS)
    if dense["status"] != "passed" or (dense["num_ranks"], dense["tensor_parallel_size"]) != (4, 1):
        raise ValueError("Figure 2(b) requires verified DP4/TP1 calibration")
    for batch_size, color, marker, linestyle in ((32, COLORS["B"], "o", "-"), (64, COLORS["X"], "s", "--")):
        samples = sorted((point for point in dense["points"] if point["batch_size"] == batch_size),
                         key=lambda point: point["imbalance"])
        spreads = [point["imbalance"] for point in samples]
        axis.plot(spreads, [point["mean_ms"] for point in samples],
                  color=color, marker=marker, linestyle=linestyle, markerfacecolor="white",
                  markeredgewidth=1, label=f"BS{batch_size}")
        axis.fill_between(spreads, [point["min_ms"] for point in samples],
                          [point["max_ms"] for point in samples], color=color, alpha=0.15, linewidth=0)
    axis.set_xticks([0, 8, 16, 24, 32])
    axis.set_xlabel("不均衡度（请求数极差）")
    axis.set_ylabel("decode 步时 (ms)")
    axis.set_ylim(8, 28.5)
    axis.set_xlim(-1, 33)
    osdi.legend_top(axis, ncol=2)
    osdi.subcaption(axis, "(b) 固定请求总量的失衡代价", y=-0.36)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dddddd", linewidth=0.4, linestyle=":")
    return figure


def migration2_padding():
    return osdi.save(padding_figure(), evidence.FIGURES[1])


def migration3_design():
    return render_architecture(evidence.FIGURES[2])


def migration4_results():
    figure, axes = plt.subplots(1, 2, figsize=(osdi.FULL, 2.5), gridspec_kw={"wspace": 0.32})
    for trace, (axis, seed) in enumerate(zip(axes, evidence.SEEDS), 1):
        maximum = 0
        for sid, offset in (("B", -0.17), ("X", 0.17)):
            requests = sorted(evidence.cohort(seed, sid, 0), key=lambda request: request["arrival_s"])
            for index, request in enumerate(requests):
                wait = request["reserved_at"] - request["arrived_at"]
                total = request["first_token_at"] - request["arrived_at"]
                axis.barh(index + offset, wait, 0.29, color=COLORS[sid], alpha=0.35,
                          hatch="///", edgecolor=COLORS[sid], linewidth=0.5)
                axis.barh(index + offset, total - wait, 0.29, left=wait, color=COLORS[sid], linewidth=0)
                axis.text(total + 7, index + offset, f"{total:.1f}", va="center", fontsize=6.5)
                maximum = max(maximum, total)
        axis.set_yticks(range(4), ["420", "540", "660", "780"])
        axis.set_ylabel("Code 到达时间 (s)")
        axis.set_xlabel("TTFT (s)")
        axis.set_xlim(0, maximum * 1.17)
        axis.set_ylim(3.6, -0.6)
        axis.grid(False, axis="y")
        axis.grid(True, axis="x")
        osdi.subcaption(axis, f"({chr(96 + trace)}) 踪迹{trace}", y=-0.28)
    handles = []
    for sid in evidence.POLICIES:
        handles.extend([Patch(facecolor=COLORS[sid], alpha=0.35, hatch="///", label=f"{sid} 等待"),
                        Patch(facecolor=COLORS[sid], label=f"{sid} 准入后")])
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.90), ncol=4)
    return osdi.save(figure, evidence.FIGURES[3])


def migration5_mechanism():
    figure, axes = plt.subplots(1, 2, figsize=(osdi.FULL, 2.15), gridspec_kw={"wspace": 0.40})
    states = [("none", "无 Code", osdi.COLOR["gray"], ""),
              ("prefill", "Code prefill", COLORS["B"], "//"),
              ("decode", "Code decode", COLORS["X"], "\\\\")]
    axis = axes[0]
    for trace, seed in enumerate(evidence.SEEDS):
        for index, (sid, policy) in enumerate(evidence.POLICIES.items()):
            position, bottom = trace * 3 + index, 0.0
            parts = evidence.report(seed)["mechanism"][policy]["tpot_by_state_ms"]
            for state, label, color, hatch in states:
                axis.bar(position, parts[state], 0.66, bottom=bottom, color=color, hatch=hatch,
                         edgecolor="black", linewidth=0.5, label=label if position == 0 else None)
                bottom += parts[state]
            axis.text(position, bottom + 0.5, f"{bottom:.1f}", ha="center", fontsize=6.5)
    axis.set_xticks([0, 1, 3, 4], ["B\n踪迹1", "X\n踪迹1", "B\n踪迹2", "X\n踪迹2"])
    axis.set_ylabel("Conv TPOT 贡献 (ms)")
    axis.set_ylim(0, 47)
    osdi.legend_top(axis, ncol=3, fontsize=6.5)
    osdi.subcaption(axis, "(a) 真机：执行阶段分解", y=-0.33)
    axis = axes[1]
    pairs = evidence.imbalance_rows()
    axis.scatter([row["extra_spread"] for row in pairs], [row["b_pct"] for row in pairs],
                 color=COLORS["B"], alpha=0.55, s=12, linewidth=0)
    axis.axhline(0, color="black", linewidth=0.6)
    axis.set_xlabel("随机迁移多留下的 Conv 数差\n（无 Code 时段的 rank 最大−最小）")
    axis.set_ylabel("均衡感知组合相对随机对照\nConv TPOT 变化 (%)")
    axis.text(0.96, 0.08, f"仅模拟，{len(pairs)} 对", ha="right", transform=axis.transAxes)
    osdi.subcaption(axis, "(b) 模拟：组合方案对照", y=-0.40)
    return osdi.save(figure, evidence.FIGURES[4])


def range_curve(axis, positions, samples, color, label):
    medians = [statistics.median(values) for values in samples]
    axis.fill_between(positions, [min(values) for values in samples], [max(values) for values in samples],
                      color=color, alpha=0.14, linewidth=0)
    axis.plot(positions, medians, marker="o", color=color, label=label)


def migration6_factors():
    rows = evidence.sweep_rows()
    rates, budgets = (0.08, 0.10, 0.12, 0.14), (16, 20, 24)
    figure, axes = plt.subplots(1, 3, figsize=(osdi.FULL, 2.1), gridspec_kw={"wspace": 0.5})
    for sid, policy in evidence.POLICIES.items():
        samples = [[row["code_ttft_avg"] for row in rows
                    if row["policy"] == policy and row["budget"] == 16 and row["rate"] == rate] for rate in rates]
        range_curve(axes[0], rates, samples, COLORS[sid], f"{sid} 模拟")
        samples = [[row["code_ttft_avg"] for row in rows
                    if row["policy"] == policy and row["rate"] == 0.12 and row["budget"] == budget]
                   for budget in budgets]
        range_curve(axes[2], budgets, samples, COLORS[sid], f"{sid} 模拟")
        for seed in evidence.SEEDS:
            value = evidence.summary(seed, sid)["code_ttft_avg"]
            for axis, position in ((axes[0], 0.12), (axes[2], evidence.CAPACITY_GIB)):
                axis.scatter(position, value, marker="D", facecolor="none", edgecolor=COLORS[sid],
                             s=22, linewidth=0.8, zorder=5)
    samples = []
    for rate in rates:
        selected = [row for row in rows if row["budget"] == 16 and row["rate"] == rate]
        baseline = {(row["seed"], row["replica"]): row for row in selected if row["policy"] == evidence.POLICIES["B"]}
        samples.append([100 * (row["conv_tpot_avg"] / baseline[(row["seed"], row["replica"])]["conv_tpot_avg"] - 1)
                        for row in selected if row["policy"] == evidence.POLICIES["X"]])
    range_curve(axes[1], rates, samples, COLORS["X"], "X 相对 B")
    for seed in evidence.SEEDS:
        axes[1].scatter(0.12, 100 * evidence.comparison(seed)["conv_tpot"]["change"], marker="D",
                        facecolor="none", edgecolor=COLORS["X"], s=22, linewidth=0.8, zorder=5)
    axes[1].axhline(20, color="black", linestyle="--", linewidth=0.7)
    axes[1].text(0.081, 21, "20% 评估目标", fontsize=6.5)
    for axis in axes[:2]:
        axis.set_xlabel("Conv 到达率 (RPS)")
        axis.set_xticks(rates, [f"{rate:.2f}" for rate in rates])
    axes[0].set_ylabel("Code TTFT (s)")
    axes[1].set_ylabel("Conv TPOT 变化 (%)")
    axes[2].set_xlabel("每卡 KV 预算 (GiB)")
    axes[2].set_ylabel("Code TTFT (s)")
    axes[2].set_xticks(budgets)
    for axis, title in zip(axes, ("(a) 到达率：长请求", "(b) 到达率：共址代价", "(c) 容量：长请求")):
        osdi.subcaption(axis, title, y=-0.36)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="D", markerfacecolor="none", color="black", linestyle="none"))
    labels.append("真机（两踪迹）")
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.91), ncol=3)
    return osdi.save(figure, evidence.FIGURES[5])


def imbalance_topologies():
    from imbalance_topologies import comparison_figure

    return osdi.save(comparison_figure(evidence.TP2_ANALYSIS), "imbalance_topologies")


def main(names=None):
    osdi.apply()
    for name in names or evidence.FIGURES:
        if name not in evidence.FIGURES:
            raise ValueError(f"Figure outside the manuscript selection: {name}")
        print(globals()[name]())


if __name__ == "__main__":
    main()
