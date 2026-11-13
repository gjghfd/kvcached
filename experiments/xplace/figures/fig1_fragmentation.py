"""图 1：rank 局部 KV 使长请求"总量够、单卡不够"（V16，seed 914，16 GiB/卡）。

(a) 不留余量的受控基线（S2，`no_move_v2_nh_g`）中 4 个 Code 到达时各 rank 的空闲 KV 与该 Code 所需 KV（短虚线）；
(b) 页共享：第一个被挡的 Code 到达时，已映射与实际占用之差最大的 rank 上 Conv 的已映射 KV 与按当前长度计算的
    实际占用（rank 由程序选出并打印）；
(c) 已完成请求数：kvcached 原生（V11，seed 914）在 KV 不够时 DP 空转，到 1900 s 截断时只完成约 1/3。
"""
import math

import matplotlib.pyplot as plt
import numpy as np

import data as D
import osdi
from osdi import COLOR

RUN, BASE = "V16-s914", "no_move_v2_nh_g"
NATIVE_RUN = D.NATIVE_RUNS[914]
BLOCKED_S = 5.0  # a Code that queued at least this long counts as blocked (tick label "等…s")


def codes_of(run, policy):
    requests, _, _, _ = D.run_data(run, policy)
    return sorted((r for r in requests if r["model"] == 0), key=lambda r: r["arrival_s"])


def sample_before(pop, t):
    return max((x for x in pop if x["t"] <= t), key=lambda x: x["t"])


def free_at_arrivals():
    pop = D.population(RUN, BASE)
    rows = []
    for code in codes_of(RUN, BASE):
        sample = sample_before(pop, code["arrival_s"])
        free = [D.CAPACITY_GIB - a - b for a, b in zip(*sample["mapped_gib"])]
        need = math.ceil((code["prompt_tokens"] + 256) / 16) * 1.5 / 1024
        rows.append((code["arrival_s"], free, need, code["reserved_at"] - code["arrived_at"]))
    return rows


def conv_used_gib(requests, origin, rank, t):
    blocks = 0
    for r in requests:
        if r["model"] != 1 or r["rank_initial"] != rank:
            continue
        if not (r["reserved_at"] - origin <= t < r["completed_at"] - origin):
            continue
        generated = 0
        for stamp, count in r["cadence"]:
            if stamp - origin > t:
                break
            generated = count
        blocks += math.ceil(max(r["prompt_tokens"] + 256, r["prompt_tokens"] + generated) / 16)
    return blocks * 1.5 / 1024


def page_sharing_rank():
    """(rank, sample time, blocked Code): the rank with the largest mapped - actual Conv KV in the population sample
    just before the first blocked Code arrives (the baseline never migrates, so rank_initial is the rank)."""
    requests, origin, _, _ = D.run_data(RUN, BASE)
    pop = D.population(RUN, BASE)
    code = next(c for c in codes_of(RUN, BASE) if c["reserved_at"] - c["arrived_at"] >= BLOCKED_S)
    sample = sample_before(pop, code["arrival_s"])
    gaps = [mapped - conv_used_gib(requests, origin, rank, sample["t"])
            for rank, mapped in enumerate(sample["mapped_gib"][1])]
    rank = max(range(len(gaps)), key=gaps.__getitem__)
    print(f"fig1(b): {RUN}/{BASE} rank {rank} at t = {sample['t']:.1f} s (sample before the first blocked Code, "
          f"arrival {code['arrival_s']:.0f} s, queued {code['reserved_at'] - code['arrived_at']:.0f} s); "
          f"mapped - actual Conv KV per rank = {', '.join(f'{g:.2f}' for g in gaps)} GiB")
    return rank, sample["t"], code


def main():
    osdi.apply()
    fig, axes = plt.subplots(1, 3, figsize=(osdi.FULL, 1.75), gridspec_kw={"wspace": 0.38, "width_ratios": [1.2, 1, 1]})

    ax = axes[0]
    rows = free_at_arrivals()
    width = 0.19
    for index, (arrival, free, need, queue) in enumerate(rows):
        for rank, value in enumerate(free):
            fits = value >= need
            ax.bar(index + (rank - 1.5) * width, value, width, color=COLOR["cost"] if fits else COLOR["light"],
                   edgecolor="black", linewidth=0.5, hatch="" if fits else "//")
        ax.plot([index - 2.2 * width, index + 2.2 * width], [need, need], color=COLOR["ours"], linestyle="--",
                linewidth=0.9)
        ax.text(index, max(max(free), need) + 0.45, f"共 {sum(free):.0f}", ha="center", va="bottom", fontsize=6.5)
        print(f"fig1(a): Code {arrival:.0f} s needs {need:.2f} GiB, queued {queue:.1f} s; free per rank "
              f"{', '.join(f'{v:.2f}' for v in free)} GiB (sum {sum(free):.1f})")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR["cost"], edgecolor="black", linewidth=0.5),
               plt.Rectangle((0, 0), 1, 1, facecolor=COLOR["light"], edgecolor="black", linewidth=0.5, hatch="//"),
               plt.Line2D([], [], color=COLOR["ours"], linestyle="--", linewidth=0.9)]
    osdi.legend_top(ax, handles, ["放得下", "放不下", "Code 所需"], ncol=3, y=1.0, fontsize=6.5)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{a:.0f}\n{'直接' if q < BLOCKED_S else f'等{q:.0f}s'}" for a, _, _, q in rows], fontsize=6.5)
    ax.set_xlabel("Code 到达时间 (s)", labelpad=1)
    ax.set_ylabel("各 rank 空闲 KV (GiB)")
    ax.set_ylim(0, 12.8)
    ax.set_xlim(-0.55, 3.55)
    ax.tick_params(axis="x", length=0)
    osdi.subcaption(ax, "(a) Code 到达时各 rank 的空闲 KV", y=-0.5)

    ax = axes[1]
    rank, _, code = page_sharing_rank()
    requests, origin, _, _ = D.run_data(RUN, BASE)
    t0, t1 = 300, 1300
    pop = [x for x in D.population(RUN, BASE) if t0 <= x["t"] <= t1]
    times = np.array([x["t"] for x in pop])
    mapped = np.array([x["mapped_gib"][1][rank] for x in pop])
    used = np.array([conv_used_gib(requests, origin, rank, x["t"]) for x in pop])
    ax.plot(times, mapped, color=COLOR["corun"], label="已映射")
    ax.plot(times, used, color=COLOR["cost"], linestyle="-.", label="实际占用")
    ax.fill_between(times, used, mapped, color=COLOR["corun"], alpha=0.15, linewidth=0, label="页内空洞")
    top = max(14.0, math.ceil(mapped.max() + 2.0))
    admitted = code["reserved_at"] - origin
    for t, text, ha in ((code["arrival_s"], "Code 到达", "right"), (admitted, "准入", "left")):
        ax.axvline(t, color=COLOR["gray"], linestyle=":", linewidth=0.8)
        ax.text(t + (-12 if ha == "right" else 12), top * 0.04, text, ha=ha, va="bottom", fontsize=6, color="#555555")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel(f"rank {rank} 的 Conv KV (GiB)")
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, top)
    osdi.legend_top(ax, ncol=3, y=1.0, fontsize=6.5, handlelength=1.3, columnspacing=0.7)
    osdi.subcaption(ax, "(b) 页共享：已映射与实际占用", y=-0.5)

    ax = axes[2]
    styles = [(NATIVE_RUN, "kvcached_native", "kvcached 原生", COLOR["native"], ":"),
              (RUN, BASE, "受控基线（S2）", COLOR["nomove"], "--"),
              (RUN, "ca_full_tm_slo_g", "XPlace（S6）", COLOR["ours"], "-")]
    total = 0
    for run, policy, label, color, style in styles:
        requests, origin, _, _ = D.run_data(run, policy)
        done = np.sort([r["completed_at"] - origin for r in requests if not r.get("censored")])
        total = max(total, len(requests))
        print(f"fig1(c): {label} ({run}/{policy}) completed {int((done <= 1900).sum())}/{len(requests)} by 1900 s, "
              f"{int((done <= 900).sum())} by 900 s, last at {done[-1]:.0f} s")
        ax.step(np.concatenate([[0], done, [1900]]), np.concatenate([[0], np.arange(1, len(done) + 1), [len(done)]]),
                where="post", color=color, linestyle=style, label=label)
    ax.axvline(900, color=COLOR["gray"], linewidth=0.6)
    ax.text(910, 6, "trace 结束", fontsize=6.5, color="#555555")
    for arrival in sorted({c["arrival_s"] for c in codes_of(RUN, BASE)}):
        ax.plot(arrival, 2, marker="v", color=COLOR["ours"], markersize=3.5, clip_on=False)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("已完成请求数")
    ax.set_xlim(0, 1900)
    ax.set_ylim(0, 10 * math.ceil(1.27 * total / 10))  # headroom: the legend sits above the curves
    ax.legend(loc="upper left", ncol=1, fontsize=6, handlelength=1.2, handletextpad=0.3, borderaxespad=0.2)
    osdi.subcaption(ax, "(c) 已完成请求数（▼为 Code 到达）", y=-0.5)
    return osdi.save(fig, "fig1_fragmentation")


if __name__ == "__main__":
    print(main())
