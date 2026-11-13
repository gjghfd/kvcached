"""Editable Chinese XPlace architecture with shared request routing and migration."""
from pptx import Presentation
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt

from fragmentation_pptx import (
    finish_slide, line, rectangle, render_presentation, text,
)

WIDTH, HEIGHT = 650, 412
CJK_FONT = "Heiti SC"
COLOR_A = "D8EAF4"
COLOR_B = "F6E1D3"
OUTLINE = "444444"
CONTROL_FILL = "FFFFFF"
MODULE_FILL = "EDF4F8"
POOL_FILL = "FFFFFF"
NEUTRAL_FILL = "F2F2F2"
STATE_COLOR = "707070"


def group(slide, top, height, title, title_width, name, fill, title_left=None):
    border = rectangle(slide, 12, top, 596, height, fill, name, edge="707070", weight=1.3)
    border.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    title_left = (620 - title_width) / 2 if title_left is None else title_left
    heading = rectangle(slide, title_left, top - 12, title_width, 25,
                        "FFFFFF", f"{name}-heading", edge="FFFFFF", weight=0)
    heading.line.fill.background()
    text(slide, title_left, top - 12, title_width, 25, title, size=20, bold=True)


def module(slide, left, top, width, height, name, fill=MODULE_FILL):
    shape = rectangle(slide, left, top, width, height, fill, name, edge=OUTLINE, weight=1.5,
                      shape_type=MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE)
    shape.adjustments[0] = 0.08
    return shape


def route(slide, points, name, color=OUTLINE, width=1.8, dashed=False, arrow=True):
    for index, (start, end) in enumerate(zip(points, points[1:])):
        segment = line(slide, start, end, color=color, width=width,
                       dashed=dashed, arrow=arrow and index == len(points) - 2)
        segment.name = f"{name}-{index}"


def request_migration(slide, top, model):
    connector = line(slide, (220, top), (400, top), color=OUTLINE, width=2.8, arrow=True)
    connector.name = f"same-model-migration-{model}"
    head = OxmlElement("a:headEnd")
    for key, value in {"type": "triangle", "w": "med", "len": "med"}.items():
        head.set(key, value)
    properties = connector.line._get_or_add_ln()
    properties.insert(len(properties) - 1, head)


def request_inputs(slide):
    for model, left, color in (("A", 78, COLOR_A), ("B", 428, COLOR_B)):
        rectangle(slide, left + 4, 13, 20, 25, "FFFFFF", f"input-{model}-back", edge=OUTLINE, weight=1.2)
        rectangle(slide, left, 9, 20, 25, color, f"input-{model}-document", edge=OUTLINE, weight=1.2)
        for top in (17, 23):
            line(slide, (left + 4, top), (left + 16, top), color=OUTLINE, width=1)
        label = text(slide, left + 30, 11, 116, 26, f"模型 {model} 请求", size=17, align=PP_ALIGN.LEFT)
        label.name = f"input-{model}-label"
        destination = 260 if model == "A" else 396
        route(slide, [(left + 69, 41), (left + 69, 66), (destination, 66)], f"input-{model}-route")
    module(slide, 260, 48, 136, 36, "request-router", fill=NEUTRAL_FILL)
    text(slide, 264, 48, 128, 36, "请求路由", size=18)
    route(slide, [(314, 84), (314, 132)], "blocked-trigger")
    text(slide, 206, 86, 96, 22, "容量受阻", size=15)
    route(slide, [(370, 84), (370, 94), (632, 94), (632, 310), (608, 310)], "direct-dispatch")
    text(slide, 508, 69, 92, 22, "请求分发", size=15)


def controller(slide):
    module(slide, 28, 146, 150, 70, "state-cost-model", fill=NEUTRAL_FILL)
    text(slide, 33, 146, 140, 70, "代价模型", size=18)

    module(slide, 224, 132, 208, 40, "blocked-request-migration")
    text(slide, 228, 132, 200, 40, "受阻迁移", size=18)

    module(slide, 224, 188, 208, 40, "periodic-load-balancing")
    text(slide, 228, 188, 200, 40, "周期均衡", size=18)

    module(slide, 480, 146, 112, 70, "migration-runtime", fill=NEUTRAL_FILL)
    text(slide, 485, 146, 102, 70, "迁移执行", size=18)

    route(slide, [(178, 181), (202, 181)], "state-branch", arrow=False)
    route(slide, [(202, 181), (202, 152), (224, 152)], "state-to-blocked")
    route(slide, [(202, 181), (202, 208), (224, 208)], "state-to-balancing")
    route(slide, [(432, 152), (456, 152), (456, 181)], "blocked-plan", arrow=False)
    route(slide, [(432, 208), (456, 208), (456, 181)], "balancing-plan", arrow=False)
    route(slide, [(456, 181), (480, 181)], "plan-to-runtime")


def gpu_pool(slide):
    for rank, left in enumerate((28, 384)):
        module(slide, left, 295, 208, 92, f"gpu-{rank}", fill="FFFFFF")
        text(slide, left + 8, 298, 192, 22, f"GPU {rank}", size=16, bold=True)
        for model, top, color in (("A", 325, COLOR_A), ("B", 357, COLOR_B)):
            rectangle(slide, left + 16, top, 176, 24, color, f"rank-{model}-{rank}",
                      f"模型 {model}", size=16, edge=OUTLINE, weight=1.5)
    request_migration(slide, 337, "A")
    text(slide, 240, 342, 140, 22, "请求迁移", size=15)
    request_migration(slide, 369, "B")


def chinese_fonts(slide):
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        for paragraph in shape.text_frame.paragraphs:
            for run in paragraph.runs:
                properties = run._r.get_or_add_rPr()
                properties.set("lang", "zh-CN")
                east_asian = OxmlElement("a:ea")
                east_asian.set("typeface", CJK_FONT)
                properties.append(east_asian)


def make_slide():
    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = Pt(WIDTH), Pt(HEIGHT)
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    group(slide, 110, 130, "XPlace 控制器", 164, "controller", CONTROL_FILL, title_left=28)
    group(slide, 270, 130, "GPU 池", 100, "gpu-pool", POOL_FILL)
    controller(slide)
    gpu_pool(slide)
    request_inputs(slide)
    route(slide, [(44, 270), (44, 216)], "runtime-state", color=STATE_COLOR, dashed=True)
    text(slide, 53, 244, 64, 22, "状态", size=15, color=STATE_COLOR, align=PP_ALIGN.LEFT)
    route(slide, [(536, 216), (536, 284), (488, 284)], "execute-trunk", arrow=False)
    route(slide, [(488, 284), (132, 284), (132, 295)], "execute-gpu-0")
    route(slide, [(488, 284), (488, 295)], "execute-gpu-1")
    text(slide, 545, 244, 50, 22, "执行", size=15, align=PP_ALIGN.LEFT)
    chinese_fonts(slide)
    finish_slide(slide)
    slide.notes_slide.notes_text_frame.text = (
        "Visual references: WarmServe figures/overview/overview.pptx; "
        "Prorsa: Unleashing the Potential of Prefill-Decode Disaggregation in Agentic Reinforcement Learning, "
        "arXiv:2610.16592v1, Figure 4 (page 6); "
        "云边协同的深度学习作业调度方法, Figure 1 (PDF page 5), for external inputs and layered scheduling. "
        "References inform layout, not XPlace functionality. "
        "Chinese labels use Heiti SC, with Arial for Latin identifiers. "
        "Paper-scale styling uses white groups, gray outlines, and restrained blue/orange model accents. "
        "The two document icons represent model-specific input streams, not request counts. "
        "Request routing depicts the common admission path shared with the baseline, not a new service or algorithm. "
        "Feasible requests go directly to a rank of the requested model; only capacity-blocked requests trigger migration. "
        "Only two migration policies are shown: space-making migration for a blocked long request, "
        "and periodic balancing that preserves current request-state and block-allocation feasibility. "
        "Migration runtime is an implementation component, not a third scheduling algorithm. "
        "The state model uses observed lengths, KV pages, rank loads and calibrated costs, not future outputs or workload classes. "
        "The two GPUs illustrate a deployment that currently uses four GPUs; A and B are independent model instances. "
        "Request KV and execution state migrate only between attention ranks of the same model. "
        "Shared physical KV pages belong to each GPU, not to one cross-GPU memory pool. "
        "The lower layer shows only same-model request migration; memory layout is omitted because Figure 1 covers it. "
        "Dashed gray arrows are runtime-state feedback; thin solid arrows are control; thick double-headed arrows are request migration."
    )
    return presentation


def render_architecture(name="migration3_design"):
    return render_presentation(make_slide(), name)


if __name__ == "__main__":
    print(render_architecture())
