"""图 3：同一个 Code（660 s 到达）驻留期间，三种调度下 Conv 的总 token 速率（V16 通用版，seed 911）。

并行（S3，ca_full_v2_g）：Code 与 Conv 同时运行，双方都被拖慢；分时（S5，ca_full_tm_v2_g）：长 prefill 的实例
（Code）独占、持有其他实例（Conv），Code 很快完成，但新 Conv 的首 token 要等整个独占期；分时 + 按需切片
（S6，ca_full_tm_slo_g）：被持有的实例有请求等首 token 时短暂放行它。
"""
import matplotlib.pyplot as plt
import numpy as np

import data as D
import osdi
from osdi import COLOR

RUN = "V16-s911"
ROWS = [("ca_full_v2_g", "并行（S3）"), ("ca_full_tm_v2_g", "分时（S5）"), ("ca_full_tm_slo_g", "分时 + 按需切片（S6）")]
T0, T1, BIN = 622.0, 850.0, 1.0


def running_parts(a, b, held):
    """[a, b] minus the Conv-held intervals: Conv only decodes while it is not paused."""
    parts = [(a, b)]
    for ha, hb in held:
        parts = [piece for lo, hi in parts
                 for piece in ((lo, min(hi, ha)), (max(lo, hb), hi)) if piece[1] > piece[0]]
    return parts or [(a, b)]


def conv_rate(run, policy, held):
    """Conv tokens/s in 1 s bins. Progress is logged every 32 tokens, so each interval's tokens are spread over
    the part of it during which Conv was running."""
    requests, origin, _, _ = D.run_data(run, policy)
    edges = np.arange(T0, T1 + BIN, BIN)
    tokens = np.zeros(len(edges) - 1)
    for r in requests:
        if r["model"] != 1:
            continue
        points = [(t - origin, c) for t, c in r["cadence"]]
        for (a, ca), (b, cb) in zip(points, points[1:]):
            if b <= T0 or a >= T1 or b <= a:
                continue
            parts = running_parts(a, b, held)
            rate = (cb - ca) / sum(hi - lo for lo, hi in parts)
            for lo, hi in parts:
                lo, hi = max(lo, T0), min(hi, T1)
                if hi <= lo:
                    continue
                first, last = int((lo - T0) // BIN), int(min((hi - T0) // BIN, len(tokens) - 1))
                for i in range(first, last + 1):
                    overlap = min(hi, edges[i + 1]) - max(lo, edges[i])
                    if overlap > 0:
                        tokens[i] += rate * overlap
    return edges[:-1] + BIN / 2, tokens / BIN


def spans(run, policy):
    requests, origin, transactions, holds = D.run_data(run, policy)
    codes = [(r["reserved_at"] - origin, r["first_token_at"] - origin, r["completed_at"] - origin, r["arrival_s"])
             for r in requests if r["model"] == 0]
    held = [(h["paused_at"] - origin, h["resumed_at"] - origin) for h in holds if h.get("model") == 1]
    moves = [(t["started_at"] - origin, t.get("path")) for t in transactions if t.get("moves")]
    return codes, held, moves


def main():
    osdi.apply()
    fig, axes = plt.subplots(len(ROWS), 1, figsize=(osdi.FULL, 2.55), sharex=True, gridspec_kw={"hspace": 0.18})
    info = [spans(RUN, policy) for policy, _ in ROWS]
    series = [conv_rate(RUN, policy, held) for (policy, _), (_, held, _) in zip(ROWS, info)]
    peak = max(rate.max() for _, rate in series)
    for ax, (policy, label), (centers, rate), (codes, held, moves) in zip(axes, ROWS, series, info):
        for start, first, end, arrival in codes:
            if end < T0 or start > T1:
                continue
            ax.axvspan(start, first, color=COLOR["tm"], alpha=0.22, linewidth=0)
            ax.axvspan(first, end, color=COLOR["ours"], alpha=0.12, linewidth=0)
            if arrival in (660.0, 780.0):
                print(f"fig3: {label} ({policy}) Code {arrival:.0f} s: TTFT {first - arrival:.1f} s, "
                      f"resident until {end:.0f} s")
                ax.annotate("", xy=(first, peak * 1.02), xytext=(arrival, peak * 1.02),
                            arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": "black"})
                ax.text((arrival + first) / 2, peak * 1.06, f"Code TTFT {first - arrival:.0f} s", ha="center",
                        va="bottom", fontsize=6.5)
        for a, b in held:
            if b > T0 and a < T1:
                ax.axvspan(max(a, T0), min(b, T1), ymin=0, ymax=0.06, facecolor="white", edgecolor="#555555",
                           hatch="////", linewidth=0.4)
        for t, path in moves:
            if T0 <= t <= T1:
                ax.plot(t, peak * 0.93, marker="v" if path == "admission" else "^", color="black", markersize=3)
        for item in D.json_file(RUN, policy, "slices.json"):
            if T0 <= item["t"] <= T1:
                ax.plot(item["t"], peak * 0.82, marker="D", color=COLOR["cost"], markersize=2.8)
        ax.plot(centers, rate, color=COLOR["corun"], linewidth=0.9)
        ax.fill_between(centers, 0, rate, color=COLOR["corun"], alpha=0.25, linewidth=0)
        ax.set_ylim(0, peak * 1.25)
        ax.set_xlim(T0, T1)
        ax.text(0.005, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    axes[1].set_ylabel("Conv 总 token 速率 (token/s)")
    axes[-1].set_xlabel("时间 (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR["tm"], alpha=0.35, linewidth=0),
               plt.Rectangle((0, 0), 1, 1, color=COLOR["ours"], alpha=0.2, linewidth=0),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="#555555", hatch="////", linewidth=0.4),
               plt.Line2D([], [], marker="v", color="black", linestyle="none", markersize=3),
               plt.Line2D([], [], marker="^", color="black", linestyle="none", markersize=3),
               plt.Line2D([], [], marker="D", color=COLOR["cost"], linestyle="none", markersize=2.8)]
    osdi.legend_top(axes[0], handles, ["Code prefill", "Code decode", "Conv 暂停", "迁移准入", "回迁", "切片放行 Conv"],
                    ncol=6, y=1.02)
    return osdi.save(fig, "fig3_timeline")


if __name__ == "__main__":
    print(main())
