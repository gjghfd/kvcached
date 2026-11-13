"""图 5：(a) 跨 DP rank 的在线 KV 迁移事务；(b) 跨模型分时与按需切片的时序（示意）。

(a) 控制器选好迁移集合后，Conv 实例的所有 rank 在 32 步边界暂停；源 rank 导出块表、目的 rank 预分配块，所有 rank
一起参与 all_gatherv（EP 要求同步推进），目的 rank 回传并逐字节校验后提交，再释放源页并恢复（或在分时下转为持有）。
(b) 通用版的分时：长 prompt（这里是 Code）的独占收益 TM_GAIN > 0 时，它的实例独占，其他实例（这里是 Conv）先停在
边界并被持有，再准入；被持有的实例有请求等首 token 满 T_wait=2 s，或持有满 S_max=15 s，就暂停独占实例、放行被持有的
实例 R=1 s；触发请求进入 decode 后不再切片，出首 token 满 D=2 s 或完成后释放。事件为示意，时长按这些参数与约 25 s 的
prefill 画。(a) 中的迁移统计取 V16 全部 20 轮。
"""
import json
import statistics

import matplotlib.pyplot as plt

import data as D
import osdi
import schematic as S
from osdi import COLOR

CODE, CONV, DECODE = COLOR["tm"], COLOR["corun"], COLOR["ours"]
CONV_FILL = "#cfe3f3"


def frameless(fig, rect, xmax):
    ax = fig.add_axes(rect)
    width, height = fig.get_size_inches()
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, xmax * rect[3] * height / (rect[2] * width))
    ax.axis("off")
    return ax


def migration_stats(runs=("V16-s911", "V16-s914", "V16-s919-l14")):
    """(moves, bitwise-verified moves, median per-move copy time) over all policies of one job (V16: its three
    per-seed directories together hold all 20 trials)."""
    moves = [m for run in runs for policy in sorted(D.RUNS[run].iterdir()) if (policy / "transactions.json").exists()
             for t in json.loads((policy / "transactions.json").read_text()) for m in t.get("moves", [])]
    return len(moves), sum(bool(m.get("bitwise_verified")) for m in moves), statistics.median(m["elapsed_s"] for m in moves)


def transaction(ax):
    lanes = [("控制器", 20.0), ("源 rank", 14.5), ("目的 rank", 9.0), ("其他 rank", 3.5)]
    for name, y in lanes:
        S.rect(ax, 7.6, y - 0.3, 42.4, 4.6, fc="#f4f4f4", ec="none", zorder=1)
        ax.text(7.2, y + 2.0, name, ha="right", va="center", fontsize=6.5)
    y = {name: y for name, y in lanes}
    h, mid = 4.0, 2.0
    ctrl, src, dst, other = (y[name] for name, _ in lanes)
    S.box(ax, 8.0, ctrl, 7.8, h, title="选迁移集合", fc="white", ec="#333333", title_size=5.8)
    S.rect(ax, 16.4, other, 1.4, src + h - other, fc="white", ec="#555555", hatch="////", lw=0.5)
    S.label(ax, 17.1, 1.6, "32 步边界暂停", size=5.8, box=False)
    S.box(ax, 18.6, src, 6.2, h, title="导出块表", fc=CONV_FILL, ec=CONV, title_size=5.8)
    S.box(ax, 18.6, dst, 6.2, h, title="预分配块", fc=CONV_FILL, ec=CONV, title_size=5.8)
    S.box(ax, 25.6, other, 7.0, src + h - other, title="all_gatherv\n复制 KV", fc=CONV_FILL, ec=CONV, title_size=5.5)
    S.box(ax, 33.4, dst, 6.0, h, title="回传校验", fc=CONV_FILL, ec=CONV, title_size=5.8)
    S.box(ax, 37.0, ctrl, 4.6, h, title="提交", fc="white", ec="#333333", title_size=5.8)
    S.box(ax, 42.6, ctrl, 7.4, h, title="恢复／持有", fc="white", ec="#333333", title_size=5.6)
    S.box(ax, 42.6, src, 7.4, h, title="释放源页", fc=CONV_FILL, ec=CONV, title_size=5.8)
    S.arrow(ax, (15.8, ctrl + 1.0), (17.1, src + h + 0.05))
    for lane in (src, dst):
        S.arrow(ax, (17.8, lane + mid), (18.6, lane + mid))
        S.arrow(ax, (24.8, lane + mid), (25.6, lane + mid))
    S.arrow(ax, (32.6, dst + mid), (33.4, dst + mid))
    S.arrow(ax, (37.6, dst + h), (39.3, ctrl))
    S.arrow(ax, (41.6, ctrl + mid), (42.6, ctrl + mid))
    S.arrow(ax, (41.6, ctrl + 0.5), (44.0, src + h))
    moves, verified, median = migration_stats()
    print(f"fig5(a): V16 {moves} migrations, {verified} bitwise verified, median copy {median:.3f} s")
    S.label(ax, 40.5, 1.4, f"V16：{moves} 次迁移，{verified} 次校验通过\n每次复制中位 {median:.2f} s", size=5.8,
            box=False)


def timeline(ax):
    y_code, y_conv, h = 15.0, 8.0, 3.6
    x0, scale = 7.6, 1.2  # 1 s of the sketch = 1.2 units
    t = lambda s: x0 + scale * s  # noqa: E731
    ax.text(x0 - 0.4, y_code + h / 2, "Code 实例", ha="right", va="center", fontsize=6.5)
    ax.text(x0 - 0.4, y_conv + h / 2, "Conv 实例", ha="right", va="center", fontsize=6.5)
    for a, b, kind in [(0, 2, "run"), (2, 10, "hold"), (10, 11, "run"), (11, 26, "hold"), (26, 27, "run"),
                       (27, 31, "hold"), (31, 35, "run")]:
        if kind == "run":
            S.rect(ax, t(a), y_conv, scale * (b - a), h, fc=CONV, ec=CONV, lw=0.4)
        else:
            S.rect(ax, t(a), y_conv, scale * (b - a), h, fc="white", ec="#555555", hatch="////", lw=0.4)
    for a, b, color in [(2.2, 10, CODE), (11, 26, CODE), (27, 29.5, CODE), (29.5, 31, DECODE)]:
        S.rect(ax, t(a), y_code, scale * (b - a), h, fc=color, ec=color, lw=0.4)
    for a, b in [(10, 11), (26, 27)]:
        S.rect(ax, t(a), y_code, scale * (b - a), h, fc="white", ec="#555555", hatch="////", lw=0.4)
        ax.plot(t(a + 0.5), y_code + h + 1.0, marker="D", color=COLOR["cost"], markersize=3, zorder=6)
    S.arrow(ax, (x0, 2.4), (t(36), 2.4), lw=0.6)
    ax.text(t(36), 1.2, "时间", ha="right", va="center", fontsize=6)
    ax.plot(t(1), y_code + h + 1.0, marker="v", color="black", markersize=3.2)
    S.label(ax, t(1), y_code + h + 2.6, "Code 到达", size=5.8, box=False)
    ax.plot(t(8), y_conv - 1.0, marker="^", color="black", markersize=3.2)
    S.label(ax, t(7.4), y_conv - 1.0, "新 Conv 到达", size=5.8, box=False, ha="right")
    S.arrow(ax, (t(8.3), y_conv - 1.0), (t(10), y_conv - 1.0), both=True, lw=0.5, scale=4)
    S.label(ax, t(9.6), y_conv - 2.5, "等首 token 满 2 s", size=5.8, box=False)
    S.arrow(ax, (t(11), y_code + h + 1.0), (t(26), y_code + h + 1.0), both=True, lw=0.5, scale=4)
    S.label(ax, t(18.5), y_code + h + 2.3, "或持有满 15 s", size=5.8, box=False)
    S.label(ax, t(10.5), y_code - 1.3, "放行被持有的实例 1 s", size=5.8, box=False)
    S.label(ax, t(30.2), y_code + h + 2.6, "进入 decode\n不再切片", size=5.8, box=False)
    S.label(ax, t(35.2), y_conv - 1.9, "出首 token 满 2 s\n或完成：释放", size=5.8, box=False, ha="right")


def main():
    osdi.apply()
    fig = plt.figure(figsize=(osdi.FULL, 1.95))
    ax_a = frameless(fig, [0.0, 0.13, 0.48, 0.87], 50.0)
    ax_b = frameless(fig, [0.51, 0.13, 0.49, 0.87], 50.0)
    transaction(ax_a)
    timeline(ax_b)
    fig.text(0.24, 0.03, "(a) 跨 rank 的在线 KV 迁移事务", ha="center", fontsize=8)
    fig.text(0.755, 0.03, "(b) 分时与按需切片（示意）", ha="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=CODE, ec=CODE), plt.Rectangle((0, 0), 1, 1, fc=DECODE, ec=DECODE),
               plt.Rectangle((0, 0), 1, 1, fc=CONV, ec=CONV),
               plt.Rectangle((0, 0), 1, 1, fc="white", ec="#555555", hatch="////", lw=0.4),
               plt.Line2D([], [], marker="D", color=COLOR["cost"], linestyle="none", markersize=3)]
    ax_b.legend(handles, ["Code prefill", "Code decode", "Conv 运行", "暂停／持有", "切片"], loc="upper center",
                bbox_to_anchor=(0.55, 1.07), ncol=5, fontsize=6, handlelength=1.1, columnspacing=0.7)
    return osdi.save(fig, "fig5_design")


if __name__ == "__main__":
    print(main())
