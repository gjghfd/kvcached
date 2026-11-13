"""图 7：收益从哪里来（V16 通用版）。

(a) Conv TPOT 按 Code 状态拆分（V16，seed 911，`report-s911.json` 的 mechanism）：placement 只改变"无 Code"一段，
    分时消掉并行的两段、换来暂停；
(b) 代价感知 placement 的收益与随机迁移多留下的失衡成正比：通用版模拟（非最终 seed 801–812，每个 trace 5 个随机
    种子；`b-vs-imbalance-general-20261107.json`）的点云与真机的配对（大点：V16 两个 seed 的 S9 → S6，分时 + 按需切片）。
    横轴为无 Code 时段各 rank Conv 数之差（随机减代价感知）。
"""
import json

import matplotlib.pyplot as plt
import numpy as np

import data as D
import osdi
from osdi import COLOR

SEED = 911
TRACE = D.TRACE
BARS = [("no_move_v2_nh_g", "S2\n受控基线"), ("ca_full_v2_g", "S3\n代价感知"), ("no_move_tm_slo_nh_g", "S4\n不迁移"),
        ("mig_random_tm_slo_g", "S9\n随机迁移"), ("ca_full_tm_v2_g", "S5\n无切片"), ("ca_full_tm_slo_g", "S6\nXPlace")]
CORUN_BARS = 2  # the first bars run in parallel, the rest time-multiplex
STATES = [("none", "无 Code", COLOR["corun"], ""), ("prefill", "Code prefill", COLOR["tm"], "//"),
          ("decode", "Code decode", COLOR["ours"], "\\\\\\\\"), ("paused", "Conv 暂停", "white", "xx")]
SIM_FILE = "b-vs-imbalance-general-20261107.json"
LIVE = ["V16-s911", "V16-s914"]  # same-job pairs S9 (random migration, no return) -> S6
RANDOM, COST = "mig_random_tm_slo_g", "ca_full_tm_slo_g"


def breakdown():
    report = D.v16_report(SEED)
    return {policy: report["mechanism"][policy]["tpot_by_state_ms"] for policy, _ in BARS}


def live_points(runs, random_policy, cost_policy):
    points = []
    for run in runs:
        extra = D.no_code_spread(run, random_policy) - D.no_code_spread(run, cost_policy)
        change = D.pair(run, random_policy, run, cost_policy, 1, "tpot_s")["change"]
        print(f"fig7(b): {run} {D.SID[random_policy]} -> {D.SID[cost_policy]}: extra spread {extra:.2f}, "
              f"Conv TPOT {100 * change:+.1f}%")
        points.append((extra, 100 * change))
    return points


def main():
    osdi.apply()
    fig, axes = plt.subplots(1, 2, figsize=(osdi.FULL, 1.95), gridspec_kw={"wspace": 0.28, "width_ratios": [1.25, 1]})

    ax = axes[0]
    parts = breakdown()
    for i, (policy, _) in enumerate(BARS):
        bottom = 0.0
        for key, _, color, hatch in STATES:
            value = parts[policy].get(key, 0.0)
            if value <= 0:
                continue
            ax.bar(i, value, 0.62, bottom=bottom, color=color, hatch=hatch, edgecolor="black", linewidth=0.5)
            bottom += value
        print(f"fig7(a): {D.SID[policy]} {policy}: total {bottom:.2f} ms = "
              + ", ".join(f"{key} {parts[policy].get(key, 0.0):.2f}" for key, _, _, _ in STATES))
        ax.text(i, bottom + 0.6, f"{bottom:.1f}", ha="center", va="bottom", fontsize=6.5,
                fontweight="bold" if policy == "ca_full_tm_slo_g" else "normal")
    ax.axvline(CORUN_BARS - 0.5, color=COLOR["gray"], linewidth=0.6)
    ax.text((CORUN_BARS - 1) / 2, 50.8, "并行", ha="center", fontsize=7)
    ax.text((CORUN_BARS + len(BARS) - 1) / 2, 50.8, "分时", ha="center", fontsize=7)
    ax.set_xticks(range(len(BARS)))
    ax.set_xticklabels([label for _, label in BARS], fontsize=6.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Conv TPOT (ms)")
    ax.set_ylim(0, 54)
    ax.set_xlim(-0.6, len(BARS) - 0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, hatch=h, edgecolor="black", linewidth=0.5) for _, _, c, h in STATES]
    osdi.legend_top(ax, handles, [label for _, label, _, _ in STATES], ncol=4, y=1.0, fontsize=6.5,
                    handlelength=1.2, columnspacing=0.8)
    osdi.subcaption(ax, f"(a) Conv TPOT 按 Code 状态拆分（{TRACE[SEED]}）", y=-0.33)

    ax = axes[1]
    sim = json.loads((D.SIM / SIM_FILE).read_text())["pairs"]
    for sched, color, label in (("corun", COLOR["corun"], "并行（模拟）"), ("tm", COLOR["tm"], "分时（模拟）")):
        xs = np.array([p["extra_spread"] for p in sim[sched]])
        ys = np.array([p["b_pct"] for p in sim[sched]])
        ax.scatter(xs, ys, s=7, color=color, alpha=0.35, linewidth=0, label=label)
        slope = float((xs * ys).sum() / (xs * xs).sum())
        print(f"fig7(b): sim {sched}: {len(xs)} pairs, slope through 0 {slope:+.2f}%/unit, "
              f"r = {np.corrcoef(xs, ys)[0, 1]:.2f}")
        grid = np.linspace(0, 7.5, 2)
        ax.plot(grid, slope * grid, color=color, linestyle="--", linewidth=0.8)
    points = live_points(LIVE, RANDOM, COST)
    ax.scatter([x for x, _ in points], [y for _, y in points], s=26, marker="o", facecolor=COLOR["ours"],
               edgecolor="black", linewidth=0.6, zorder=3, label="分时 + 切片（真机）")
    for (x, y), run in zip(points, LIVE):
        ax.annotate(TRACE[int(run.split("-s")[-1])], (x, y), xytext=(5, -2), textcoords="offset points", fontsize=6,
                    va="top")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("随机迁移多留下的 rank 间 Conv 数之差")
    ax.set_ylabel("代价感知相对随机的\nConv TPOT 变化 (%)")
    ax.set_xlim(-0.3, 7.5)
    ax.set_ylim(-14, 1.5)
    ax.legend(loc="lower left", ncol=1, fontsize=6, handletextpad=0.2, borderaxespad=0.3, labelspacing=0.2,
              frameon=True, facecolor="white", edgecolor="none", framealpha=0.85, borderpad=0.3)
    osdi.subcaption(ax, "(b) 收益与随机迁移留下的失衡成正比", y=-0.33)
    return osdi.save(fig, "fig7_mechanism")


if __name__ == "__main__":
    print(main())
