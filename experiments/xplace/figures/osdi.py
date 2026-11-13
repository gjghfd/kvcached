"""OSDI-style plotting defaults for the paper figures (Chinese labels: Times New Roman + STSong fallback).

Figures are written to Paper/figures as PDF (embedded TrueType), SVG (text as paths, for Word) and 600-dpi PNG.
STSong is face 4 of macOS's Songti.ttc; matplotlib only reads face 0 (Black) of a collection, so the regular face
is extracted once into ~/.cache/xplace-figures (not committed).
"""
from pathlib import Path
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.ERROR)  # Apple AAT tables are dropped when subsetting STSong

OUT = Path(__file__).resolve().parents[4] / "Paper" / "figures"
SONGTI = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
FONT_CACHE = Path.home() / ".cache" / "xplace-figures" / "STSong.ttf"

# widths in inches: the journal text block is ~16.5 cm
FULL, HALF, THIRD = 6.6, 3.2, 2.15

# Okabe-Ito (colorblind safe); one color per scheduling family, ours highlighted in vermillion
COLOR = {"native": "#555555", "corun": "#0072B2", "tm": "#E69F00", "ours": "#D55E00", "random": "#CC79A7",
         "nomove": "#56B4E9", "cost": "#009E73", "fair": "#F0E442", "gray": "#9a9a9a", "light": "#d9d9d9"}
HATCH = ["", "//", "\\\\", "xx", "..", "++", "--", "oo"]


def _register_cjk():
    if not FONT_CACHE.exists():
        from fontTools.ttLib import TTCollection
        FONT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        TTCollection(str(SONGTI)).fonts[4].save(str(FONT_CACHE))
    font_manager.fontManager.addfont(str(FONT_CACHE))


def apply():
    _register_cjk()
    matplotlib.rcParams.update({
        "font.family": ["Times New Roman", "STSong"], "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4,
        "patch.linewidth": 0.6, "hatch.linewidth": 0.5,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.grid": True, "axes.grid.axis": "y", "grid.linestyle": "--", "grid.linewidth": 0.4,
        "grid.color": "#c8c8c8", "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.columnspacing": 1.0,
        "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.25,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 150, "savefig.dpi": 600,
    })


def subcaption(ax, text, y=-0.34):
    """Journal-style panel caption under an axes, e.g. '(a) 各 rank 的空闲 KV'."""
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top", fontsize=8)


def legend_top(target, handles=None, labels=None, ncol=4, y=1.02, **kwargs):
    """One-row legend above an axes or a figure (OSDI style)."""
    args = (handles, labels) if handles is not None else ()
    return target.legend(*args, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, **kwargs)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    return OUT / f"{name}.pdf"
