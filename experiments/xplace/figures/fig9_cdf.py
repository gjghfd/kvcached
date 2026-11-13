"""图 9：分时的代价与切片的作用（V16 通用版，两个 seed 合并，每个 seed 一轮，窗口内的 Conv）。

(a) Conv TTFT 的 CDF：全程分时（S5）让新 Conv 等整个独占期，按需切片（S6）把尾部拉回 SLO（5 s）以内；
(b) 每个 Conv 的最长 token 间隔的 CDF：分时与切片都会留下一次停顿，按需切片的停顿上限由 S_max（15 s）决定。
    受控基线（S2）在 seed 911 上有 3 个 Conv 被 vLLM 抢占重算，停顿 30–76 s（两个超出坐标范围，图中标注）。
"""
import matplotlib.pyplot as plt
import numpy as np

import data as D
import osdi
from osdi import COLOR

LINES = [("no_move_v2_nh_g", "S2 受控基线", COLOR["nomove"], "-."), ("ca_full_v2_g", "S3 并行", COLOR["corun"], "-"),
         ("ca_full_tm_v2_g", "S5 分时（无切片）", COLOR["tm"], "--"),
         ("ca_full_tm_slo_g", "S6 XPlace（按需切片）", COLOR["ours"], "-")]
OURS = "ca_full_tm_slo_g"
GAP_MAX = 32.0


def values(policy, key):
    out = []
    for seed in (911, 914):
        runs = D.runs_with(seed, policy)
        run = runs[-1]  # one run per seed so both seeds weigh equally
        conv, origin = D.window_conv(run, policy)
        for r in conv:
            out.append(r["first_token_at"] - r["arrived_at"] if key == "ttft" else r["max_token_gap_s"])
    return np.sort(np.array(out))


def cdf(ax, key):
    for policy, label, color, style in LINES:
        xs = values(policy, key)
        print(f"fig9: {key:4s} {label}: n={len(xs)}, p50 {np.percentile(xs, 50):.2f} s, p99 {np.percentile(xs, 99):.2f} s, "
              f"max {xs.max():.2f} s" + (f", share <= 5 s {np.mean(xs <= 5):.3f}" if key == "ttft" else ""))
        ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", color=color, linestyle=style, label=label,
                linewidth=1.3 if policy == OURS else 1.1)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("CDF")
    ax.grid(axis="x", linestyle="--", linewidth=0.4, color="#c8c8c8")


def main():
    osdi.apply()
    fig, axes = plt.subplots(1, 2, figsize=(osdi.FULL, 1.8), gridspec_kw={"wspace": 0.25})
    cdf(axes[0], "ttft")
    axes[0].set_xscale("log")
    axes[0].set_xlim(0.1, 40)
    axes[0].set_xticks([0.1, 0.5, 1, 5, 10, 30])
    axes[0].set_xticklabels(["0.1", "0.5", "1", "5", "10", "30"])
    axes[0].axvline(5, color=COLOR["gray"], linestyle=":", linewidth=0.8)
    axes[0].text(5.4, 0.08, "SLO 5 s", fontsize=6.5, color="#555555")
    axes[0].set_xlabel("Conv TTFT (s，对数轴)")
    osdi.subcaption(axes[0], "(a) Conv TTFT", y=-0.32)
    cdf(axes[1], "gap")
    axes[1].set_xlim(0, GAP_MAX)
    tail = values(LINES[0][0], "gap")
    stalls = tail[tail > 10]
    if len(stalls):  # Conv of the baseline preempted by vLLM (recompute); the longest stalls run off the axis
        off = int((stalls > GAP_MAX).sum())
        axes[1].text(3.5, 0.9, f"S2：{len(stalls)} 个被抢占的 Conv\n停顿 {'、'.join(f'{v:.0f}' for v in stalls)} s"
                     + (f"\n（{off} 个在图外）" if off else ""), ha="left", va="top", fontsize=6.5,
                     color=COLOR["nomove"], linespacing=1.15)
    axes[1].set_xlabel("每个 Conv 的最长 token 间隔 (s)")
    osdi.subcaption(axes[1], "(b) Conv 最长停顿", y=-0.32)
    osdi.legend_top(fig, ncol=4, y=0.97, handles=axes[0].get_legend_handles_labels()[0],
                    labels=axes[0].get_legend_handles_labels()[1])
    return osdi.save(fig, "fig9_cdf")


if __name__ == "__main__":
    print(main())
