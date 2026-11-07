"""The figure style for matplotlib, read from figures/figstyle.tex so the two cannot drift.

Colours are parsed from the \\definecolor lines of figures/figstyle.tex; sizes and line widths
follow the same rules (no text below 7 pt, 0.5 pt outlines). Data figures use TeX Gyre Heros,
the Helvetica that the TikZ figures get from \\sffamily under the times package.
"""
from __future__ import annotations

from pathlib import Path
import re

_TEX = Path(__file__).resolve().parent / "figures" / "figstyle.tex"
COLOR = {m[1]: "#" + m[2].lower()
         for m in re.finditer(r"\\definecolor\{(\w+)\}\{HTML\}\{([0-9A-Fa-f]{6})\}",
                              _TEX.read_text())}


def mix(a: str, b: str, t: float) -> str:
    """t of colour a and 1 - t of colour b, as xcolor's a!t*100!b."""
    pa = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    pb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(t * x + (1 - t) * y):02x}" for x, y in zip(pa, pb))


BLUE, ORANGE, AQUA = COLOR["finblue"], COLOR["finorange"], COLOR["finaqua"]
INK = COLOR["figink"]                         # titles, labels, annotations
INK2 = COLOR["fingray"]                       # secondary marks and tick labels
MUTED = COLOR["figmid"]                       # axes, reference lines
GRID = mix(COLOR["figline"], "#ffffff", 0.5)  # grid lines
UNMOVED = COLOR["figfaint"]                   # empty cells
SHORT = mix(COLOR["figline"], COLOR["figmid"], 0.8)
OK, BAD = COLOR["figok"], COLOR["figbad"]

RC = {
    "font.family": "TeX Gyre Heros", "mathtext.fontset": "custom",
    "mathtext.rm": "TeX Gyre Heros", "mathtext.it": "TeX Gyre Heros:italic", "font.size": 7.5,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.labelcolor": INK, "pdf.fonttype": 42}
