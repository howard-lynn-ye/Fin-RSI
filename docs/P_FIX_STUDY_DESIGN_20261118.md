# P-fix: the frozen P track on the fixed library interface (exploratory, 2026-11-18)

Written before any P-fix run. Exploratory: it was designed after the frozen study's results were
seen, so it describes what happens on the changed interface; it does not confirm a hypothesis
the frozen study set out to test. Numbers are compared within P-fix only, never pooled with the
frozen study or P-adopt (hardware, software versions and confinement differ; see below).

## Why

In the frozen P study's library arms, 519 programs called the library and 480 of those calls
raised. 167 passed the model's own function as the algorithm, 143 named an id not in the
catalogue, and 126 passed wrong arguments or inputs. Calls that ran often divided an
already-scaled result by the number of assets. For a library built for agents, those are
defects of the interface. PR #77 changed it:
- keyword parameters;
- `run()` on a single input;
- errors that say what to call instead, with did-you-mean ids;
- `signal_positions()` for a model's own rule;
- docstrings stating that the result is final.

P-fix measures whether models now use the library and get the held-out tasks right with it.

## Arms (`fin_skills/rsi/pfix.py`)

| Arm | System prompt | Task prompt |
|---|---|---|
| F0 | frozen system prompt, no library (identical to P-adopt L0) | frozen |
| F1 | frozen system prompt + the current position-task interface text (`library_text(rolling=False, frozen=False)`) | frozen |
| F2 | as F1 | frozen + P-adopt's pointer `The fin_skills library implements this rule as positions("<id>", ...).` |

All arms:
- no rule cards;
- Check feedback: the names of failed checks, and the full error when a program fails to run (as frozen);
- at most 3 attempts, 6 calls and 12,288 generated tokens;
- 4,096-token output cap per reply, temperature 0.1, top-p 1, BF16;
- the 20 protocol models at their frozen revisions;
- 6 held-out tasks x seeds 101, 211, 307: 18 episodes per arm, 54 per model.

## Measures and analysis (declared now)

- **Primary.** Valid episodes per arm and model.
- **Contrasts.** F1 - F0, F2 - F0 and F2 - F1, per model and pooled over models. They are reported by algorithm family and by variant (a: library defaults; b: non-default parameters and ratio factors).
- **Funnel.** Last program calls the library, then it runs, then it passes the visible checks, then it is valid. Library calls that raise are grouped by error, as in jobs 3655794 and 3655802.
- **Statistics.** Three rule families cannot support significance claims, so everything is descriptive.
- **Denominator.** Every refusal, context stop, truncated reply and failed program stays in it.

## Execution and deviations

- **Clusters.** Zaratan (UMD) H100 GPUs, because Beacon could not start new GPU jobs before 2026-11-22. The same jobs are also queued on Beacon. For each model, only the first complete run is kept; the other cluster's job is cancelled before it starts, or its output is not read.
- **Zaratan software.** Python 3.12, torch 2.7.1 + CUDA 12.6 (Zaratan's driver does not run the frozen CUDA 13 build), transformers 4.57.6, accelerate 1.12.0, numpy 1.26.4, pandas 2.2.3, scipy 1.15.2.
- **Confinement.** Zaratan's kernel (4.18) has no Landlock. Model programs keep Beacon's system-call denials (no sockets, exec, fork, signals, namespaces or mounts) and resource limits, but not its filesystem rules (`confine_without_landlock`, accepted only with `FIN_TASKS_DECLARED_DEVIATION=no-landlock` and recorded in each run's protocol). Afterwards, every executed program is scanned for file access outside its inputs. Any such episode is reported.
- **Model weights.** The group's scratch quota (100 GB) cannot hold the 20 models, and Zaratan compute nodes cannot reach the Hugging Face file CDN. Each job therefore receives its model's frozen-revision files from the login node into node-local disk, streamed through its own Slurm allocation. Every file's SHA-256 is compared on both ends, and weight files are also compared with the Hub's recorded SHA-256, before the model loads.
- **Validation first.** One `--dev` job on a small model must finish both units, and every declared first prompt must reach the model. It reads no grade.
