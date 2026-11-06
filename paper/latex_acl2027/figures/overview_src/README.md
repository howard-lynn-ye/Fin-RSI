# Figure 2 source

Figure 2 (the library section; an appendix figure in the FinFM short version) draws Fin-RSI
the way library papers draw their overview: one container for the library with compartments
of marks, the ecosystem on either side, and a tick on what a user picks.

```bash
python3 compose_kit.py     # writes ../overview.tex (pure TikZ)
```

Every mark is a real method, guard, skill or rule card. The federated investor templates are
left out: they are off by default and not verified here. Counts and the "+N" chips come from `../../numbers.tex`, so rerun `make_numbers.py` first if a count changed.
