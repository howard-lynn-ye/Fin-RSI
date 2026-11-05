# Figure 1 sources

Figure 1 is built the way the Project Progress Desk teaser was: a CSS 3D scene rendered in Chrome,
then composed in TikZ.

```bash
node scene.cjs       # three perspective layers -> ../overview/raw_p{0,1,2}.png, anchors.json
node panel.cjs       # the flat agent panel -> ../overview/panel.png, panel_anchors.json (counts from numbers.tex)
python3 compose.py   # crops the layers, writes ../overview/{l0,l1,l2,panel_s}.png and ../overview.tex
```

Needs Node with `playwright`, Google Chrome at `/usr/bin/google-chrome`, the Inter and DejaVu Sans
Mono fonts, and Pillow. Rerun `make_numbers.py` first if a count changed. Every name in the scene is
a real adapter, collector, skill, rule card, guard, judge check or API; sparklines are drawn shapes,
not vendor prices. Only the cropped `l*.png` and `panel_s.png` are used by the papers.
