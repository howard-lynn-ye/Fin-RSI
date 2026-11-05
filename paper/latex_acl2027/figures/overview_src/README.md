# Figure 1 source

Figure 1 draws Fin-RSI the way library papers draw their overview: one container for the library
with compartments of name chips, the ecosystem on either side, and a tick on what a user picks.

```bash
python3 compose_kit.py     # writes ../overview.tex (pure TikZ)
```

Names are real adapters, collectors, methods, guards, plugins and investor templates. Counts and
the "+N" chips come from `../../numbers.tex`, so rerun `make_numbers.py` first if a count changed.
