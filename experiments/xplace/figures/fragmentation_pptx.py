"""Editable Figure 1 following WarmServe's universal-gpus.pptx visual style."""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt

import osdi

COLOR_A = "8FAADC"
COLOR_B = "F4B183"
COLOR_WEIGHTS = "A9D18E"
COLOR_FREE = "BFBFBF"
OUTLINE = "172C51"
WIDTH, HEIGHT = 650, 234


def fragmentation_example():
    return {"capacity_units": 6, "weights_units": 2, "request_b_units": 4,
            "balanced": [3, 3], "unbalanced": [1, 5], "destination_b": 0}


def set_text(shape, text, size=18, bold=False, align=PP_ALIGN.CENTER, color="000000"):
    frame = shape.text_frame
    frame.clear()
    frame.margin_left = frame.margin_right = Pt(0)
    frame.margin_top = frame.margin_bottom = Pt(0)
    frame.word_wrap = False
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_before = paragraph.space_after = Pt(0)
    paragraph.line_spacing = 1.0
    run = paragraph.add_run()
    run.text = text
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def text(slide, left, top, width, height, content, size=18, bold=False, align=PP_ALIGN.CENTER, color="000000"):
    shape = slide.shapes.add_textbox(Pt(left), Pt(top), Pt(width), Pt(height))
    set_text(shape, content, size, bold, align, color)
    return shape


def rectangle(slide, left, top, width, height, fill, name, label="", size=16, edge=OUTLINE, weight=2.5,
              shape_type=MSO_AUTO_SHAPE_TYPE.RECTANGLE):
    shape = slide.shapes.add_shape(shape_type, Pt(left), Pt(top), Pt(width), Pt(height))
    shape.name = name
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    shape.line.color.rgb = RGBColor.from_string(edge)
    shape.line.width = Pt(weight)
    if label:
        set_text(shape, label, size)
    return shape


def line(slide, start, end, color="000000", width=2.5, arrow=False, dashed=False):
    shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, *[Pt(value) for value in (*start, *end)])
    shape.line.color.rgb = RGBColor.from_string(color)
    shape.line.width = Pt(width)
    if dashed:
        shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if arrow:
        tail = OxmlElement("a:tailEnd")
        tail.set("type", "triangle")
        tail.set("w", "med")
        tail.set("len", "med")
        shape.line._get_or_add_ln().append(tail)
    return shape


def finish_slide(slide):
    for shape in slide.shapes:
        properties = shape._element.spPr
        properties.append(OxmlElement("a:effectLst"))
        style = shape._element.find("{http://schemas.openxmlformats.org/presentationml/2006/main}style")
        if style is not None:
            effect = style.find("{http://schemas.openxmlformats.org/drawingml/2006/main}effectRef")
            if effect is not None:
                effect.set("idx", "0")
        shape.line._get_or_add_ln().append(OxmlElement("a:miter"))


def curved_migration(slide):
    shape = rectangle(slide, 280, 187, 92, 32, None, "migration-arrow", edge="000000", weight=2.5)
    properties = shape._element.spPr
    properties.remove(properties.find("{http://schemas.openxmlformats.org/drawingml/2006/main}prstGeom"))
    geometry = OxmlElement("a:custGeom")
    for name in ("avLst", "gdLst", "ahLst", "cxnLst"):
        geometry.append(OxmlElement(f"a:{name}"))
    bounds = OxmlElement("a:rect")
    for key, value in {"l": "0", "t": "0", "r": "r", "b": "b"}.items():
        bounds.set(key, value)
    geometry.append(bounds)
    paths = OxmlElement("a:pathLst")
    path = OxmlElement("a:path")
    for key, value in {"w": "92000", "h": "32000", "fill": "none", "stroke": "1"}.items():
        path.set(key, value)
    for command, points in (("moveTo", ((0, 0),)), ("cubicBezTo", ((32000, 32000), (60000, 32000), (92000, 0)))):
        element = OxmlElement(f"a:{command}")
        for horizontal, vertical in points:
            point = OxmlElement("a:pt")
            point.set("x", str(horizontal))
            point.set("y", str(vertical))
            element.append(point)
        path.append(element)
    paths.append(path)
    geometry.append(paths)
    properties.insert(1, geometry)
    tail = OxmlElement("a:tailEnd")
    tail.set("type", "triangle")
    shape.line._get_or_add_ln().append(tail)


def make_slide():
    example = fragmentation_example()
    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = Pt(WIDTH), Pt(HEIGHT)
    presentation.core_properties.title = "Balanced placement can block a long request"
    presentation.core_properties.subject = "Illustrative request layouts; not experimental measurements"
    presentation.core_properties.author = ""
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
    for left, fill, label, label_width in ((6, COLOR_A, "LLM A", 82), (142, COLOR_B, "LLM B", 82),
                                         (278, COLOR_WEIGHTS, "Weights", 110), (442, COLOR_FREE, "Free KV", 120)):
        rectangle(slide, left, 5, 27, 21, fill, f"legend-{label}", weight=3)
        text(slide, left + 37, 3, label_width, 25, label, size=18, align=PP_ALIGN.LEFT)
    line(slide, (326, 36), (326, 183), width=2.5, dashed=True)
    total_units = example["weights_units"] + example["capacity_units"]
    unit, bar_height = 210 / total_units, 22
    for panel, (offset, layout_name, title, outcome) in enumerate(
            ((0, "balanced", "Balanced LLM A", "B blocked"),
             (350, "unbalanced", "Imbalanced LLM A", "B admitted"))):
        bar_left = offset + 62
        kv_left = bar_left + unit * example["weights_units"]
        text(slide, offset + 2, 37, 286, 26, title, size=18)
        request_left = kv_left + unit * (2 if panel == 0 else example[layout_name][example["destination_b"]])
        rectangle(slide, request_left, 74, example["request_b_units"] * unit, bar_height, COLOR_B,
                  f"incoming-B-{panel}", "New B request", size=14)
        arrow_left = request_left + example["request_b_units"] * unit / 2
        line(slide, (arrow_left, 97), (arrow_left, 110), width=1.8, arrow=True)
        if panel == 0:
            line(slide, (arrow_left - 4, 100), (arrow_left + 4, 108), color="C00000", width=2.2)
            line(slide, (arrow_left - 4, 108), (arrow_left + 4, 100), color="C00000", width=2.2)
        for rank, load in enumerate(example[layout_name]):
            top = 112 + rank * 31
            text(slide, offset + 2, top - 1, 54, 24, f"GPU {rank}", size=15, align=PP_ALIGN.LEFT)
            rectangle(slide, bar_left, top, unit * total_units, bar_height, COLOR_FREE,
                      f"gpu-{panel}-{rank}", edge="000000", weight=3)
            rectangle(slide, bar_left, top, unit * example["weights_units"], bar_height, COLOR_WEIGHTS,
                      f"weights-{panel}-{rank}", label="W", size=16, weight=2.5)
            rectangle(slide, kv_left, top, unit * load, bar_height, COLOR_A,
                      f"load-A-{panel}-{rank}", label="A", size=16, weight=2.5)
            if panel == 1 and rank == example["destination_b"]:
                rectangle(slide, kv_left + unit * load, top, unit * example["request_b_units"], bar_height, COLOR_B,
                          "admitted-B", label="B", size=17, weight=2.5)
        text(slide, offset + 50, 172, 236, 22, outcome, size=16)
    curved_migration(slide)
    text(slide, 218, 211, 216, 18, "Migrate A requests", size=15)
    finish_slide(slide)
    slide.notes_slide.notes_text_frame.text = (
        "Visual reference: WarmServe figures/design/universal-gpus.pptx. "
        "Arial; blue/orange model blocks; green weights; gray free memory; horizontal GPU bars; dashed separator; curved transition. "
        "This is a capacity illustration, not a measurement. Two GPUs each have 2 fixed weight units (W) and 6 KV units; B requires 4 units. "
        "Weights have identical size on both GPUs before and after migration; only request KV placement changes. "
        "One continuous blue rectangle per GPU represents aggregate A load, not an individual request. "
        "The same total A load is redistributed from [3,3] to [1,5], leaving room for B beside A on GPU 0."
    )
    return presentation


def render_presentation(presentation, name):
    osdi.OUT.mkdir(parents=True, exist_ok=True)
    source = osdi.OUT / f"{name}.pptx"
    presentation.save(source)
    office = shutil.which("soffice")
    if office is None:
        raise RuntimeError("LibreOffice is required to export editable paper figures")
    environment = os.environ.copy()
    if "FONTCONFIG_FILE" not in environment:
        for candidate in ("/opt/homebrew/etc/fonts/fonts.conf", "/etc/fonts/fonts.conf"):
            if Path(candidate).is_file():
                environment["FONTCONFIG_FILE"] = candidate
                break
    with tempfile.TemporaryDirectory(prefix="xplace-figure-pptx-") as temporary:
        folder = Path(temporary)
        profile = (folder / "office-profile").as_uri()
        result = subprocess.run([office, f"-env:UserInstallation={profile}", "--headless", "--convert-to", "pdf",
                                 "--outdir", str(folder), str(source)],
                                capture_output=True, text=True, check=True, timeout=180, env=environment)
        exported = folder / f"{name}.pdf"
        if not exported.is_file():
            raise RuntimeError(f"Figure PDF export failed: {result.stdout}\n{result.stderr}")
        target = osdi.OUT / f"{name}.pdf"
        shutil.copyfile(exported, target)
    with fitz.open(target) as document:
        page = document[0]
        page.get_pixmap(dpi=600).save(osdi.OUT / f"{name}.png")
        (osdi.OUT / f"{name}.svg").write_text(page.get_svg_image(text_as_path=True), encoding="utf8")
    return target


def render_fragmentation(name="migration1_fragmentation"):
    return render_presentation(make_slide(), name)


if __name__ == "__main__":
    print(render_fragmentation())
