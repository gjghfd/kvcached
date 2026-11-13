"""图 10：影响因素（模拟为分布，真机为大点；受控策略均为通用版）。

(a) 切片参数：周期切片 S ∈ {3,4,5,6,8} s 与按需切片 T_wait ∈ {1,2,4,8} s 的 Code TTFT 与 Conv TTFT p99
    （模拟：通用版 `ca_full_tm_v2_g` 加切片，seed 911、914 各 3 个副本取均值，`rq4-slicing-general-20261107.json`；
    真机：V16 两个 seed 的 S6（按需 T=2）与 S5（不切片））；
(b)(c) Conv 负载 λ（16 GiB/卡）下的 Code TTFT 与相对受控基线 S2 的 Conv TPOT 变化；
(d) 每卡 KV 预算（λ=0.12）下的 Code TTFT。
(b)–(d) 的模拟为非最终 seed 801–806、每个 2 个副本：线为中位数，阴影为最小–最大。S2、S3、S6 取通用版扫描
（planner v5，`rq4-sweep-general-20261107.json`）；kvcached 原生不经过控制器，取原扫描。真机：S2、S6 为 V16
（seed 911、914 与 λ=0.14 的 seed 919），原生为 V10、V11（1900 s 截断，只是下界）。
"""
import json
import math
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import data as D
import osdi
from osdi import COLOR

BASE, OURS = "no_move_v2_nh_g", "ca_full_tm_slo_g"
POLICIES = [("kvcached_native", "kvcached 原生（S1）", COLOR["native"], ":", "X"),
            (BASE, "受控基线（S2）", COLOR["nomove"], "--", "o"),
            ("ca_full_v2_g", "并行 + 代价感知（S3）", COLOR["cost"], "-.", "s"),
            (OURS, "XPlace（S6）", COLOR["ours"], "-", "D")]
STYLE = {policy: (color, marker) for policy, _, color, _, marker in POLICIES}
# (x, run, policies) of the live points; the two seeds at the same x are dodged. kvcached native is not affected by the
# controller and keeps its V10 (seed 911) / V11 (seed 914) runs
RATE_ANCHORS = [(0.118, "V10", ("kvcached_native",)), (0.122, "V11", ("kvcached_native",)),
                (0.118, "V16-s911", (BASE, OURS)), (0.122, "V16-s914", (BASE, OURS)),
                (0.14, "V16-s919-l14", (BASE, OURS))]
BUDGET_ANCHORS = [(15.6, "V10", ("kvcached_native",)), (16.4, "V11", ("kvcached_native",)),
                  (15.6, "V16-s911", (BASE, OURS)), (16.4, "V16-s914", (BASE, OURS))]
SWEEP_NATIVE = "rq4-sweep-blocks-20261105.json"
SWEEP_GENERAL = "rq4-sweep-general-20261107.json"
SLICING = "rq4-slicing-general-20261107.json"


def sweep():
    """Controlled policies from the general-algorithm sweep; kvcached native (no controller) from the original sweep."""
    old = json.loads((D.SIM / SWEEP_NATIVE).read_text())["runs"]
    new = json.loads((D.SIM / SWEEP_GENERAL).read_text())["runs"]
    grouped = defaultdict(list)
    for r in [r for r in old if r["policy"] == "kvcached_native"] + new:
        grouped[(r["rate"], r["budget"], r["policy"])].append(r)
    return grouped


def has(run, policy):
    return (D.RUNS[run] / policy / "requests.json").exists()


def band(grouped, policy, xs, by, fixed, transform):
    """Median and min-max over seeds x replicas of transform(rows, x) at each x (rate or budget)."""
    med, lo, hi = [], [], []
    for x in xs:
        rows = grouped[(x, fixed, policy)] if by == "rate" else grouped[(fixed, x, policy)]
        values = [v for v in transform(rows, x) if v is not None and not (isinstance(v, float) and math.isnan(v))]
        med.append(statistics.median(values) if values else math.nan)
        lo.append(min(values) if values else math.nan)
        hi.append(max(values) if values else math.nan)
    return np.array(med), np.array(lo), np.array(hi)


def code_values(rows, _):
    return [r["code_ttft_avg"] for r in rows if not r.get("timeout")]


def live_code(ax, x, run, policy, color, marker):
    """Live point; runs cut off at the deadline (native) only give a lower bound, drawn with an up arrow."""
    y = D.summary_metrics(run, policy)["code_ttft_avg"]
    ax.scatter(x, y, s=22, marker=marker, facecolor=color, edgecolor="black", linewidth=0.6, zorder=3)
    requests, _, _, _ = D.run_data(run, policy)
    censored = any(r.get("censored") for r in requests if r["model"] == 0)
    print(f"fig10: live {run} {policy} at x={x}: Code TTFT {'>=' if censored else ''}{y:.1f} s")
    if censored:
        ax.annotate("", xy=(x, y * 2.2), xytext=(x, y * 1.15),
                    arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": color})


def log_ttft_axis(ax):
    ax.set_yscale("log")
    ax.set_ylim(15, 8000)
    ticks = [30, 100, 300, 1000, 3000]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylabel("Code TTFT 均值 (s，对数轴)")


def panel_slicing(ax):
    runs = json.loads((D.SIM / SLICING).read_text())["runs"]
    by = defaultdict(list)
    for r in runs:
        by[r["variant"]].append(r)
    mean = {v: (statistics.mean(r["code_ttft_avg"] for r in rows), statistics.mean(r["conv_ttft_p99"] for r in rows))
            for v, rows in by.items()}
    periodic = [f"periodic_S{s}" for s in (3, 4, 5, 6, 8)]
    ondemand = [f"ondemand_T{t}" for t in (1, 2, 4, 8)]
    for v in ["none"] + periodic + ondemand:
        print(f"fig10(a): sim {v}: Code TTFT {mean[v][0]:.1f} s, Conv TTFT p99 {mean[v][1]:.1f} s")
    ax.plot(*zip(*[mean[v] for v in periodic]), color=COLOR["cost"], marker="s", markersize=3.5, label="周期切片（模拟）")
    ax.plot(*zip(*[mean[v] for v in ondemand]), color=COLOR["ours"], marker="o", markersize=3.5, label="按需切片（模拟）")
    ax.plot(*mean["none"], marker="x", color=COLOR["tm"], markersize=5, linestyle="none", label="不切片（模拟）")
    for v, text in zip(periodic, ("周期 3 s", "4", "5", "6", "8")):
        ax.annotate(text, mean[v], xytext=(2, 3), textcoords="offset points", fontsize=6, color=COLOR["cost"])
    for v, text, offset in zip(ondemand, ("T=1", "2", "4", "8"), ((-4, -9), (-9, -7), (-8, -2), (-7, -2))):
        ax.annotate(text, mean[v], xytext=offset, textcoords="offset points", fontsize=6, color=COLOR["ours"])
    for policy, marker, label in ((OURS, "o", "真机 S6（按需 T=2）"), ("ca_full_tm_v2_g", "^", "真机 S5（不切片）")):
        for i, run in enumerate(("V16-s911", "V16-s914")):
            m = D.summary_metrics(run, policy)
            print(f"fig10(a): live {run} {policy}: Code TTFT {m['code_ttft_avg']:.1f} s, "
                  f"Conv TTFT p99 {m['conv_ttft_p99']:.1f} s")
            ax.scatter(m["code_ttft_avg"], m["conv_ttft_p99"], s=24, marker=marker, facecolor="none",
                       edgecolor="black", linewidth=0.8, zorder=3, label=label if i == 0 else None)
    ax.axhline(5, color=COLOR["gray"], linestyle=":", linewidth=0.8)
    ax.text(24.3, 5.4, "SLO 5 s", fontsize=6.5, color="#555555", ha="left")
    ax.set_xlabel("Code TTFT (s)")
    ax.set_ylabel("Conv TTFT p99 (s)")
    ax.set_xlim(24, 38)
    ax.set_ylim(0, 28)
    ax.legend(loc="upper right", fontsize=6, ncol=1)


def panel_rate_code(ax, grouped):
    rates = [0.08, 0.10, 0.12, 0.14]
    for policy, label, color, style, marker in POLICIES:
        med, lo, hi = band(grouped, policy, rates, "rate", 16.0, code_values)
        ax.plot(rates, med, color=color, linestyle=style, marker=marker, markersize=3, label=label)
        ax.fill_between(rates, lo, hi, color=color, alpha=0.12, linewidth=0)
    for rate, run, policies in RATE_ANCHORS:
        for policy in policies:
            if has(run, policy):
                live_code(ax, rate, run, policy, *STYLE[policy])
    log_ttft_axis(ax)
    ax.set_xlabel("Conv 到达率 λ (RPS)")
    ax.set_xticks(rates)


def panel_rate_tpot(ax, grouped):
    rates = [0.08, 0.10, 0.12, 0.14]

    def relative(policy):
        def transform(rows, rate):
            base = {(r["seed"], r["replica"]): r["conv_tpot_avg"] for r in grouped[(rate, 16.0, BASE)]}
            return [100 * (r["conv_tpot_avg"] / base[(r["seed"], r["replica"])] - 1) for r in rows
                    if (r["seed"], r["replica"]) in base and not r.get("timeout")]
        return transform

    for policy, label, color, style, marker in POLICIES[2:]:
        med, lo, hi = band(grouped, policy, rates, "rate", 16.0, relative(policy))
        ax.plot(rates, med, color=color, linestyle=style, marker=marker, markersize=3, label=label)
        ax.fill_between(rates, lo, hi, color=color, alpha=0.12, linewidth=0)
    for rate, run, policies in RATE_ANCHORS:  # same-job pairs with the no-headroom baseline only
        for policy in policies:
            if policy != BASE and has(run, BASE) and has(run, policy):
                change = D.pair(run, BASE, run, policy, 1, "tpot_s")["change"]
                print(f"fig10(c): live {run} {policy} vs {BASE}: Conv TPOT {100 * change:+.1f}%")
                color, marker = STYLE[policy]
                ax.scatter(rate, 100 * change, s=22, marker=marker, facecolor=color, edgecolor="black",
                           linewidth=0.6, zorder=3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(20, color=COLOR["gray"], linestyle="--", linewidth=0.8)
    ax.text(0.081, 21.5, "退化 20% 上限", fontsize=6.5, color="#555555")
    ax.set_xlabel("Conv 到达率 λ (RPS)")
    ax.set_ylabel("相对受控基线的\nConv TPOT 变化 (%)")
    ax.set_xticks(rates)
    ax.set_ylim(-35, 65)


def panel_budget(ax, grouped):
    budgets = [16.0, 20.0, 24.0]
    for policy, label, color, style, marker in POLICIES:
        med, lo, hi = band(grouped, policy, budgets, "budget", 0.12, code_values)
        ax.plot(budgets, med, color=color, linestyle=style, marker=marker, markersize=3, label=label)
        ax.fill_between(budgets, lo, hi, color=color, alpha=0.12, linewidth=0)
    for budget, run, policies in BUDGET_ANCHORS:
        for policy in policies:
            if has(run, policy):
                live_code(ax, budget, run, policy, *STYLE[policy])
    log_ttft_axis(ax)
    ax.set_xlabel("每卡 KV 预算 (GiB)")
    ax.set_xticks(budgets)


def main():
    osdi.apply()
    grouped = sweep()
    fig, axes = plt.subplots(2, 2, figsize=(osdi.FULL, 3.9), gridspec_kw={"wspace": 0.34, "hspace": 0.62})
    panel_slicing(axes[0, 0])
    osdi.subcaption(axes[0, 0], "(a) 切片参数（踪迹1、2）", y=-0.3)
    panel_rate_code(axes[0, 1], grouped)
    osdi.subcaption(axes[0, 1], "(b) Conv 负载对 Code TTFT 的影响（16 GiB）", y=-0.3)
    panel_rate_tpot(axes[1, 0], grouped)
    osdi.subcaption(axes[1, 0], "(c) Conv 负载对 Conv TPOT 的影响（16 GiB）", y=-0.3)
    panel_budget(axes[1, 1], grouped)
    osdi.subcaption(axes[1, 1], "(d) KV 预算对 Code TTFT 的影响（λ=0.12）", y=-0.3)
    handles = [plt.Line2D([], [], color=c, linestyle=s, marker=m, markersize=3.5) for _, _, c, s, m in POLICIES]
    handles.append(plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="black",
                              markersize=4.5))
    labels = [label for _, label, _, _, _ in POLICIES] + ["真机（大点）"]
    osdi.legend_top(fig, handles, labels, ncol=5, y=0.97)
    return osdi.save(fig, "fig10_factors")


if __name__ == "__main__":
    print(main())
