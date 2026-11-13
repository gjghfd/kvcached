"""图 2：DP 填充放大失衡的每步代价（J1b Conv 步时网格，4K context，DP=4）。

(a) 每个 rank 的 Conv 数相同、逐步增加时的 decode 步时，虚线为 CUDA graph 档位（8、16、32）；
(b) Conv 总数相同（32 / 64 个）、最满 rank 越来越满时的步时。
"""
import json

import matplotlib.pyplot as plt

import data as D
import osdi
from osdi import COLOR


def main():
    osdi.apply()
    grid = {case["name"]: case for case in json.loads(D.J1B_GRID.read_text()) if case.get("step_s")}
    fig, axes = plt.subplots(1, 2, figsize=(osdi.FULL * 0.82, 1.75), gridspec_kw={"wspace": 0.3, "width_ratios": [1, 1.25]})

    ax = axes[0]
    balanced = sorted((case["layout"][0], 1e3 * case["step_s"]) for name, case in grid.items()
                      if name.startswith("bal") and name.endswith("_c4k"))
    for bucket in (8, 16, 32):
        ax.axvline(bucket, color=COLOR["gray"], linestyle=":", linewidth=0.8)
        ax.text(bucket, 28.6, f"G{bucket}", ha="center", va="bottom", fontsize=6.5, color="#555555")
    ax.plot([n for n, _ in balanced], [ms for _, ms in balanced], marker="o", color=COLOR["corun"])
    ax.set_xlabel("每个 rank 的 Conv 数")
    ax.set_ylabel("decode 步时 (ms)")
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24, 32])
    ax.set_xlim(0, 33)
    ax.set_ylim(8, 28.5)
    osdi.subcaption(ax, "(a) 均衡布局：步时随档位跳变", y=-0.36)

    ax = axes[1]
    groups = [("32 个 Conv", ["bal8_c4k", "skew8d2_c4k", "skew8d4_c4k", "skew8d8_c4k"]),
              ("64 个 Conv", ["bal16_c4k", "skew16d2_c4k", "skew16d4_c4k", "skew16d8_c4k"])]
    width = 0.2
    hatches = ["", "//", "xx", "\\\\\\\\"]
    colors = [COLOR["cost"], COLOR["light"], COLOR["light"], COLOR["light"]]
    for g, (label, names) in enumerate(groups):
        base = 1e3 * grid[names[0]]["step_s"]
        for i, name in enumerate(names):
            ms = 1e3 * grid[name]["step_s"]
            x = g + (i - 1.5) * width
            ax.bar(x, ms, width, color=colors[i], hatch=hatches[i], edgecolor="black", linewidth=0.5)
            layout = "[" + ",".join(str(n) for n in grid[name]["layout"]) + "]"
            ax.text(x, 1.2, layout, rotation=90, ha="center", va="bottom", fontsize=5.5,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
            if i:
                ax.text(x, ms + 0.3, f"+{100 * (ms / base - 1):.0f}%", ha="center", va="bottom", fontsize=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label for label, _ in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("decode 步时 (ms)")
    ax.set_ylim(0, 30)
    ax.set_xlim(-0.5, 1.5)
    osdi.subcaption(ax, "(b) 总数相同，最满 rank 越满越慢", y=-0.36)
    return osdi.save(fig, "fig2_dp_padding")


if __name__ == "__main__":
    print(main())
