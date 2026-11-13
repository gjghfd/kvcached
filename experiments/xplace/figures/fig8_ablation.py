"""图 8：最终方案（XPlace，S6，`ca_full_tm_slo_g`）的逐项消融（V16 通用版）。

每行去掉或替换一个组件，画出相对同一作业、同一 seed 中 S6 的变化（正值表示去掉后变差，即该组件有益）。
(a) Code TTFT 均值；(b) Conv TPOT 均值；(c) Conv TTFT p99。实心为 seed 911，斜线为 seed 914；
误差棒为请求级配对 bootstrap 95% CI，不含轮次间波动。超出坐标范围的柱截断并标出数值。
"不迁移"用不留余量的版本（S4，`no_move_tm_slo_nh_g`）。周期切片与 v1 控制面没有在 V16 中重跑，不画。
"""
import random

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

import data as D
import osdi
from osdi import COLOR

REF = "ca_full_tm_slo_g"
GROUPS = [("调度", COLOR["tm"], [("ca_full_v2_g", "并行（去掉分时与切片）"), ("ca_full_tm_v2_g", "去掉切片")]),
          ("放置", COLOR["cost"], [("no_move_tm_slo_nh_g", "不迁移（不留余量）"),
                                        ("mig_random_tm_slo_g", "随机迁移、无回迁"),
                                        ("ca_return_tm_slo_g", "准入改为随机"), ("ca_admit_tm_slo_g", "去掉回迁")]),
          ("余量", COLOR["nomove"], [("ca_full_tm_slo_nh_g", "去掉增长余量")])]
METRICS = [("code", "(a) Code TTFT 均值", (-30, 150), [0, 50, 100]),
           ("tpot", "(b) Conv TPOT 均值", (-20, 50), [-10, 0, 20, 40]),
           ("p99", "(c) Conv TTFT p99", (-60, 150), [-50, 0, 50, 100])]
SEEDS = (911, 914)


def find_pair(seed, variant):
    """(XPlace run, variant run) from the latest job of this seed that ran both."""
    for job in reversed(D.JOBS[seed]):
        found = {p: next((run for run in job if (D.RUNS[run] / p / "requests.json").exists()), None)
                 for p in (REF, variant)}
        if found[REF] and found[variant]:
            return found[REF], found[variant]
    return None


def p99_change(ref_run, run, variant, rounds=2000, seed=7):
    values = np.array(D.pairs(ref_run, REF, run, variant, 1, "ttft_s"))
    point = np.percentile(values[:, 1], 99) / np.percentile(values[:, 0], 99) - 1
    rng = random.Random(seed)
    n = len(values)
    draws = []
    for _ in range(rounds):
        sample = values[[rng.randrange(n) for _ in range(n)]]
        draws.append(np.percentile(sample[:, 1], 99) / np.percentile(sample[:, 0], 99) - 1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, (lo, hi)


def change(seed, variant, metric):
    """(change, 95% CI) of the variant relative to S6 in the same V16 job of this seed."""
    found = find_pair(seed, variant)
    if not found:
        raise KeyError(f"no same-job pair of {REF} and {variant} for seed {seed}")
    ref_run, run = found
    if metric == "p99":
        return p99_change(ref_run, run, variant)
    model, key = (0, "ttft_s") if metric == "code" else (1, "tpot_s")
    result = D.pair(ref_run, REF, run, variant, model, key)
    return result["change"], tuple(result["ci95"])


def layout():
    """Top-to-bottom rows: (y, label, policy, group index)."""
    rows, y = [], 0.0
    for g, (_, _, variants) in enumerate(GROUPS):
        for policy, label in variants:
            rows.append((y, f"{D.SID[policy]} {label}", policy, g))
            y -= 1.0
        y -= 0.4
    return rows


def panel(ax, metric, limits, ticks, rows):
    lo_lim, hi_lim = limits
    for y, label, policy, g in rows:
        color = GROUPS[g][1]
        for k, seed in enumerate(SEEDS):
            value, ci = change(seed, policy, metric)
            print(f"fig8: {metric:4s} seed {seed} {label}: {100 * value:+.1f}% [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]")
            yy = y + (0.2 if k == 0 else -0.2)
            shown = min(max(100 * value, lo_lim), hi_lim)
            ax.barh(yy, shown, 0.38, color=color if k == 0 else "white", edgecolor=color, hatch="" if k == 0 else "////",
                    linewidth=0.6)
            if shown != 100 * value:
                ax.text(shown, yy, f"{100 * value:+.0f}%", ha="right" if shown > 0 else "left", va="center",
                        fontsize=5.5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3})
            else:
                ax.errorbar(shown, yy, xerr=[[shown - 100 * ci[0]], [100 * ci[1] - shown]], fmt="none",
                            ecolor="black", elinewidth=0.5, capsize=1.2, capthick=0.5)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlim(lo_lim, hi_lim)
    ax.set_xticks(ticks)
    ax.set_ylim(rows[-1][0] - 0.6, rows[0][0] + 0.6)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, color="#c8c8c8")


def group_brackets(fig, ax, rows):
    """Group names as vertical brackets left of the row labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left = min(t.get_window_extent(renderer).x0 for t in ax.get_yticklabels() if t.get_text())
    x = ax.transAxes.inverted().transform((left, 0))[0] - 0.03
    blend = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for g, (name, color, _) in enumerate(GROUPS):
        ys = [y for y, _, _, gg in rows if gg == g]
        top, bottom = max(ys) + 0.4, min(ys) - 0.4
        ax.plot([x, x], [bottom, top], transform=blend, clip_on=False, color=color, linewidth=1.2)
        ax.text(x - 0.03, (top + bottom) / 2, name, transform=blend, rotation=90, ha="right", va="center",
                fontsize=7, color="black")


def main():
    osdi.apply()
    rows = layout()
    fig, axes = plt.subplots(1, 3, figsize=(osdi.FULL, 2.2), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, (metric, caption, limits, ticks) in zip(axes, METRICS):
        panel(ax, metric, limits, ticks, rows)
        osdi.subcaption(ax, caption, y=-0.24)
    axes[1].set_xlabel("相对 XPlace（S6）的变化（%，正值表示去掉该组件后变差）")
    axes[0].set_yticks([y for y, _, _, _ in rows])
    axes[0].set_yticklabels([label for _, label, _, _ in rows])
    axes[0].tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#777777", edgecolor="#777777", linewidth=0.6),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#777777", hatch="////", linewidth=0.6)]
    osdi.legend_top(fig, handles, ["踪迹1", "踪迹2"], ncol=2, y=0.97)
    group_brackets(fig, axes[0], rows)
    return osdi.save(fig, "fig8_ablation")


if __name__ == "__main__":
    print(main())
