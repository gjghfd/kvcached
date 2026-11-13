"""图 4：XPlace 系统架构（通用版：决策只看请求与实例的可观测量，不看请求类别）。

控制器在两个 vLLM 实例之外做准入、迁移规划、回迁、分时与切片；两个实例各 DP=4/EP=4，第 r 个 rank 都在 GPU r 上，
共享该卡的 kvcached 物理页池。编号为一个长请求（评估中为 Code）的路径：①到达后在控制器排队；②单卡放不下时规划迁移；
③在 rank 之间迁移被选中请求（这里是 Conv）的 KV；④长 prefill 的实例独占、持有其他实例，并把请求交给目标 rank
（被持有的实例有请求等首 token 时按需切片）；⑤出首 token 并过宽限后释放其他实例，有请求完成时代价感知回迁。
"""
import osdi
import schematic as S
from osdi import COLOR

CODE, CONV = COLOR["tm"], COLOR["corun"]
CODE_FILL, CONV_FILL = "#fbe7c2", "#cfe3f3"
GPU_X = [14.0 + i * 16.75 for i in range(4)]
GPU_W = 15.5
PAGES = ["VVVVVVVFFFFF", "VVVVVVVVFFFF", "CCCCCCCCCVFF", "VVVVVVVVVFFF"]


def controller(ax):
    S.box(ax, 14.0, 23.6, 85.5, 18.3, fc="#f6f6f6", ec="#555555", rounding=0.8, zorder=1)
    ax.text(15.0, 41.0, "XPlace 控制器（事件驱动，位于两个 vLLM 实例之外）", fontsize=7.5, va="top")
    modules = [("准入支撑", "控制器侧排队\n按当前长度\n＋增长余量"),
               ("受阻时迁移", "腾出单卡空间\n控制迁移后的\n负载失衡代价"),
               ("周期均衡", "每 10 s／完成时\n保持容量可行\n降低 rank 失衡"),
               ("可选分时", "长 prefill 实例独占\n持有其他实例\n按需切片"),
               ("监视", "各 rank KV\n（物理空闲）\n步时与进度")]
    for i, (title, body) in enumerate(modules):
        S.box(ax, 15.5 + i * 16.95, 25.0, 15.2, 12.6, title=title, body=body, fc="white", ec="#333333",
              title_size=7.2, body_size=6.2)


def instances(ax):
    for name, y, color, fill in (("Code 实例", 16.4, CODE, CODE_FILL), ("Conv 实例", 11.8, CONV, CONV_FILL)):
        S.box(ax, 13.4, y - 0.4, 66.9, 4.4, fc="white", ec=color, lw=0.8, rounding=0.4, zorder=1)
        ax.text(12.8, y + 1.8, name, ha="right", va="center", fontsize=7)
        for r, x in enumerate(GPU_X):
            S.box(ax, x, y, GPU_W, 3.6, title=f"rank {r}", fc=fill, ec=color, lw=0.5, rounding=0.3, title_size=6.5)
    S.box(ax, 82.0, 11.4, 17.5, 9.4, title="vLLM（DP=4/EP=4）",
          body="调度器补丁：\n当前长度准入\n32 步边界暂停\n迁移 RPC", fc="white", ec="#555555", title_size=6.5, body_size=6.0)


def pools(ax):
    colors = {"C": CODE, "V": CONV, "F": "white"}
    for r, x in enumerate(GPU_X):
        S.box(ax, x, 4.0, GPU_W, 6.6, fc="#fbfbfb", ec="#555555", rounding=0.3, zorder=1)
        ax.text(x + GPU_W / 2, 9.6, f"GPU {r}", ha="center", va="center", fontsize=6.5)
        for k, kind in enumerate(PAGES[r]):
            S.rect(ax, x + 0.35 + k * 1.24, 5.0, 1.1, 2.9, fc=colors[kind], ec="#555555", lw=0.35)
    S.box(ax, 82.0, 4.0, 17.5, 6.6, title="kvcached", body="共享物理页池\n两个实例按需映射", fc="white", ec="#555555",
          title_size=6.5, body_size=6.0)
    for kind, text, y in (("C", "Code 页", 8.9), ("V", "Conv 页", 6.9), ("F", "空闲", 4.9)):
        S.rect(ax, 4.2, y - 0.8, 1.1, 1.6, fc=colors[kind], ec="#555555", lw=0.35)
        ax.text(5.9, y, text, fontsize=6, va="center")


def paths(ax):
    S.box(ax, 0.4, 32.0, 11.4, 5.2, title="Code", body="≈100K token", fc=CODE_FILL, ec=CODE, title_size=7,
          body_size=6)
    S.box(ax, 0.4, 25.4, 11.4, 5.2, title="Conv", body="长输出推理", fc=CONV_FILL, ec=CONV, title_size=7, body_size=6)
    S.arrow(ax, (11.8, 34.6), (15.5, 33.0))
    S.arrow(ax, (11.8, 28.0), (15.5, 29.5))
    S.badge(ax, 13.1, 31.3, 1)
    # controller -> instances
    S.arrow(ax, (23.1, 25.0), (23.1, 20.4))
    S.label(ax, 23.6, 22.2, "交给指定 rank", ha="left", size=5.8, box=False)
    S.arrow(ax, (40.0, 25.0), (40.0, 20.4))
    S.label(ax, 40.5, 22.2, "迁移 RPC", ha="left", size=5.8, box=False)
    S.badge(ax, 38.3, 22.2, 2)
    S.arrow(ax, (57.0, 25.0), (57.0, 20.4))
    S.label(ax, 57.5, 22.2, "回迁", ha="left", size=5.8, box=False)
    S.badge(ax, 55.3, 22.2, 5)
    S.arrow(ax, (74.0, 25.0), (74.0, 20.4))
    S.label(ax, 74.5, 22.2, "暂停／恢复／切片", ha="left", size=5.8, box=False)
    S.badge(ax, 72.3, 22.2, 4)
    S.arrow(ax, (90.8, 20.8), (90.8, 25.0))
    S.label(ax, 91.3, 22.9, "KV、进度", ha="left", size=5.8, box=False)
    # data plane: Conv KV leaves GPU 2 for GPU 1 and GPU 3
    for target in (1, 3):
        S.arrow(ax, (GPU_X[2] + GPU_W / 2, 4.0), (GPU_X[target] + GPU_W / 2, 4.0), color=CONV, lw=0.8,
                rad=-0.14 if target == 1 else 0.14)
    S.label(ax, 57.5, 0.75, "数据面：被迁请求的 KV 跨 rank 迁移（all_gatherv，逐字节校验）", size=6.0, box=False)
    S.badge(ax, 34.2, 0.8, 3)


def main():
    osdi.apply()
    fig, ax = S.canvas(osdi.FULL, 2.8)
    controller(ax)
    instances(ax)
    pools(ax)
    paths(ax)
    return osdi.save(fig, "fig4_architecture")


if __name__ == "__main__":
    print(main())
