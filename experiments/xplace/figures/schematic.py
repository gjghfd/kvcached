"""Box-and-arrow helpers for the schematic figures (architecture, migration transaction, scheduling)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


def canvas(width, height, xmax=100.0):
    """Figure with one frameless axes whose data units are equal in x and y."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, xmax * height / width)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title="", body="", fc="white", ec="black", lw=0.6, rounding=0.5, title_size=7.0,
        body_size=6.0, hatch=None, zorder=2, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}", fc=fc, ec=ec, lw=lw,
                                hatch=hatch, zorder=zorder))
    tx = x + w / 2 if align == "center" else x + 0.8
    if title and body:
        ax.text(tx, y + h - 0.9, title, ha=align, va="top", fontsize=title_size, zorder=zorder + 1)
        ax.text(tx, y + h - 0.9 - title_size * 0.30, body, ha=align, va="top", fontsize=body_size, color="#333333",
                linespacing=1.3, zorder=zorder + 1)
    elif title:
        ax.text(tx, y + h / 2, title, ha=align, va="center", fontsize=title_size, linespacing=1.25, zorder=zorder + 1)


def rect(ax, x, y, w, h, fc, ec="black", lw=0.5, hatch=None, zorder=3, alpha=1.0):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, hatch=hatch, zorder=zorder, alpha=alpha))


def arrow(ax, start, end, color="black", lw=0.7, both=False, dashed=False, rad=0.0, zorder=5, scale=6.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="<|-|>" if both else "-|>", mutation_scale=scale, color=color,
                                 lw=lw, linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0, zorder=zorder))


def label(ax, x, y, text, size=6.0, color="#333333", ha="center", va="center", box=True, zorder=6, **kwargs):
    bbox = {"facecolor": "white", "edgecolor": "none", "pad": 0.3} if box else None
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va, bbox=bbox, zorder=zorder, linespacing=1.25, **kwargs)


def badge(ax, x, y, n, color="black", r=1.05):
    ax.add_patch(Circle((x, y), r, fc=color, ec="none", zorder=7))
    ax.text(x, y - 0.05, str(n), ha="center", va="center", fontsize=6, color="white", fontweight="bold", zorder=8)
