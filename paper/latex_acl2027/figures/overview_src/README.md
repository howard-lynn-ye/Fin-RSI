# Figure 1 sources

Figure 1 draws the library as a dial: a tilted disc rendered in Chrome with CSS 3D, then composed
in TikZ.

```bash
node disc.cjs          # the disc, its rings of pins and the raised pegs -> ../overview/raw_disc.png, disc_anchors.json
python3 compose_disc.py  # crops the render to ../overview/disc.png and writes ../overview.tex
```

Pin counts come from `../../numbers.tex` (rerun `make_numbers.py` first if a count changed): one pin
per self-improvement method, check, skill and investor template. The raised pegs are one example
agent's choices. Needs Node with `playwright`, Google Chrome at `/usr/bin/google-chrome` and
Pillow. Only `../overview/disc.png` is used by the papers.
