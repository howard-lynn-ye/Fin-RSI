# Figure 1 source

Figure 1 draws what is in Fin-RSI and how an agent uses it: one toolbox with a compartment of
marks per kind of tool, the agent made of the user's model and the picked pieces, and the
write-check-improve loop it runs on each task, coloured like the compartments.

```bash
python3 compose_kit.py     # writes ../overview.tex (pure TikZ)
```

Every mark is a real method, guard, skill or rule card. The federated investor templates are
left out: they are off by default and not verified here. Counts and the "+N" chips come from `../../numbers.tex`, so rerun `make_numbers.py` first if a count changed.
