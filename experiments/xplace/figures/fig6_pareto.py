"""图 6：各策略的 Code TTFT 与 Conv TPOT（帕累托图；V16 通用版，每个 seed 一轮同作业运行）。

颜色表示调度（并行 / 分时 / 分时 + 按需切片），形状表示 placement（不迁移 / 随机迁移 / 代价感知），标签为策略编号
S2–S10（S7、S8 只有准入或回迁之一是代价感知的，按代价感知画）。虚线为相对受控基线 S2（`no_move_v2_nh_g`）Conv TPOT
退化 20% 的上限。S5–S10 挤在约 6 s × 4 ms 的范围内，放大画在内嵌子图中（主图中的灰框为放大范围）。
kvcached 原生（S1；seed 911 为 V10，seed 914 为 V11，1900 s 截断）远在图外，以箭头标注。
"""
import statistics

import matplotlib.pyplot as plt

import data as D
import osdi
from osdi import COLOR

ROWS = {sid: policy for sid, policy in D.POLICY.items() if sid != "S1"}  # S2 .. S10
BASE = D.POLICY["S2"]
OURS = D.POLICY["S6"]
SCHED = {"corun": ("并行", COLOR["corun"]), "tm": ("分时", COLOR["tm"]), "slice": ("分时 + 按需切片", COLOR["ours"])}
PLACE = {"none": ("不迁移", "o"), "random": ("随机迁移", "^"), "cost": ("代价感知", "s")}
YLIM = (23, 48.5)
ZOOM = (24.0, 33.0, 28.0, 33.6)  # Code TTFT (s) and Conv TPOT (ms) range shown in the inset
INSET = [0.235, 0.115, 0.33, 0.33]  # inset position in main-axes fraction (empty lower-middle region)
# label offsets (points): main panel per seed, inset per seed
OFFSETS = {911: {"S2": (4, 3), "S3": (4, 2), "S4": (4, -7)},
           914: {"S2": (4, 3), "S3": (4, 2), "S4": (-6, -9)}}
INSET_OFFSETS = {911: {"S5": (4, -2), "S6": (5, -4), "S7": (4, 1), "S8": (-11, 3), "S9": (-10, 2), "S10": (-6, -9)},
                 914: {"S5": (4, -2), "S6": (5, 0), "S7": (-11, -2), "S8": (-10, -2), "S9": (4, -1), "S10": (-2, -9)}}


def kind(policy):
    sched = "slice" if "_slo" in policy else "tm" if "_tm" in policy else "corun"
    place = "none" if policy.startswith("no_move") else "random" if policy.startswith("mig_random") else "cost"
    return sched, place


def point(seed, policy):
    runs = D.runs_with(seed, policy)
    if not runs:
        return None
    values = [D.summary_metrics(run, policy) for run in runs]
    return (statistics.mean(v["code_ttft_avg"] for v in values), 1e3 * statistics.mean(v["conv_tpot_avg"] for v in values))


def marker(ax, x, y, policy, sid, offset=None, fontsize=6.5):
    """One strategy point; labeled with its S-id unless offset is None."""
    sched, place = kind(policy)
    ours = policy == OURS
    ax.scatter(x, y, s=34 if ours else 18, marker=PLACE[place][1], facecolor=SCHED[sched][1], edgecolor="black",
               linewidth=0.8 if ours else 0.4, zorder=4 if ours else 3)
    if offset is not None:
        ax.annotate(sid, (x, y), xytext=offset, textcoords="offset points", fontsize=fontsize,
                    fontweight="bold" if ours else "normal", zorder=5)


def panel(ax, seed):
    base = point(seed, BASE)
    ax.axhline(base[1] * 1.2, color=COLOR["gray"], linestyle="--", linewidth=0.8)
    ax.text(21, base[1] * 1.2 + 0.35, "Conv 退化 20% 上限（相对 S2）", fontsize=6.5, color="#555555", va="bottom")
    zoom = ax.inset_axes(INSET)
    for sid, policy in ROWS.items():
        xy = point(seed, policy)
        if xy is None:
            continue
        x, y = xy
        print(f"fig6: seed {seed} {sid} {policy}: Code TTFT {x:.1f} s, Conv TPOT {y:.2f} ms")
        inside = ZOOM[0] <= x <= ZOOM[1] and ZOOM[2] <= y <= ZOOM[3]
        if inside:
            marker(ax, x, y, policy, sid)  # labeled in the inset
            marker(zoom, x, y, policy, sid, INSET_OFFSETS[seed][sid], fontsize=6)
        else:
            marker(ax, x, y, policy, sid, OFFSETS[seed][sid])
    native = D.summary_metrics(D.NATIVE_RUNS[seed], "kvcached_native")
    print(f"fig6: seed {seed} S1 kvcached_native ({D.NATIVE_RUNS[seed]}, cut at 1900 s): Code TTFT >= "
          f"{native['code_ttft_avg']:.0f} s, Conv TPOT {1e3 * native['conv_tpot_avg']:.0f} ms")
    ax.annotate(f"S1 kvcached 原生\n≥{native['code_ttft_avg']:.0f} s，{1e3 * native['conv_tpot_avg']:.0f} ms",
                xy=(690, 47.6), xytext=(235, 48.2), fontsize=6.5, ha="center", va="top",
                arrowprops={"arrowstyle": "->", "linewidth": 0.6, "color": COLOR["native"]})
    ax.set_xscale("log")
    ax.set_xlim(20, 720)
    ax.set_ylim(*YLIM)
    ax.set_xticks([25, 50, 100, 200, 400])
    ax.set_xticklabels(["25", "50", "100", "200", "400"])
    ax.set_xlabel("Code TTFT 均值 (s，对数轴)")
    ax.set_ylabel("Conv TPOT 均值 (ms)")
    ax.grid(axis="x", linestyle="--", linewidth=0.4, color="#c8c8c8")

    zoom.set_xlim(ZOOM[0], ZOOM[1])
    zoom.set_ylim(ZOOM[2], ZOOM[3])
    zoom.set_xticks([25, 28, 31])
    zoom.set_yticks([29, 31, 33])
    zoom.tick_params(labelsize=5.5, pad=1.5, length=1.5)
    zoom.grid(axis="both", linestyle="--", linewidth=0.3, color="#d8d8d8")
    zoom.set_facecolor("white")
    ax.indicate_inset_zoom(zoom, edgecolor="#777777", linewidth=0.5, alpha=0.9)


def main():
    osdi.apply()
    fig, axes = plt.subplots(1, 2, figsize=(osdi.FULL, 2.45), gridspec_kw={"wspace": 0.25})
    panel(axes[0], 911)
    panel(axes[1], 914)
    osdi.subcaption(axes[0], "(a) 踪迹1", y=-0.25)
    osdi.subcaption(axes[1], "(b) 踪迹2", y=-0.25)
    handles = [plt.Line2D([], [], marker="s", linestyle="none", markerfacecolor=c, markeredgecolor="black",
                          markeredgewidth=0.4, markersize=4.5) for _, c in SCHED.values()]
    handles += [plt.Line2D([], [], marker=m, linestyle="none", markerfacecolor="white", markeredgecolor="black",
                           markeredgewidth=0.6, markersize=4.5) for _, m in PLACE.values()]
    labels = [name for name, _ in SCHED.values()] + [name for name, _ in PLACE.values()]
    osdi.legend_top(fig, handles, labels, ncol=6, y=0.97)
    return osdi.save(fig, "fig6_pareto")


if __name__ == "__main__":
    print(main())
