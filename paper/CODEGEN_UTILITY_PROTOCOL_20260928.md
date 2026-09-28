# First-submission code generation and domain-library comparison

Frozen before model inference: four quantitative rules, data/generation seeds 11, 23 and 37,
Qwen2.5-Coder 7B/14B at the revisions in `codegen_utility.py`, and two arms (48 programs).
Each pair receives identical raw split-affected synthetic prices, event-time split factors,
mathematical requirements and output schema. These are author-designed public regression
tasks, not an independently authored holdout. Training contamination is not established.

Tasks: 60-return diagonal risk parity (inverse volatility, not full-covariance ERC),
60-return positive cross-sectional momentum, 5/20 moving-average crossover, and 20-return
volatility-targeted momentum. The shared prompt states adjustment, warmup and one-bar timing
requirements explicitly. No strategy-return or profitability endpoint is measured.

The raw arm generates NumPy/pandas/SciPy code from scratch. The library arm additionally
receives actual catalog discovery output and interface/guard documentation, and generates
code calling the discovered algorithms. Both get one response, 2048 output tokens,
temperature 0.1 and top-p 1. There is no error-feedback repair or selected rerun. Library
calls and agent-initiated guard calls are traced rather than inferred from citations.
This tests documentation plus callable library support, not an isolated guard intervention.

All responses are saved and hashed before scoring. Programs run in the existing Linux
Landlock/seccomp confinement, with a fresh scratch directory and a 90-second worker limit.
The private controller protocol is unreadable to the submitted code. Sandbox qualification
must pass before model inference. Ordinary program behavior is evaluated; this is not an
adversarial proof against a malicious Python program tampering with its own process.

The same independent post-hoc grader applies to both arms. Primary endpoints are execution
without syntax/type/shape/nonfinite errors, numerical correctness at absolute tolerance
1e-8, unchanged positions through t=k when prices at/after k are perturbed (k=32,64), and
invariance to an economically identical split-adjusted representation. A failure to execute
stays in the complete denominator and is distinct from a measured leakage violation.
The library's actual causality guard is also run post hoc, separately from agent guard use.
Passing sampled perturbations does not certify universal causality. All-zero programs can
pass invariance while failing numerical correctness; valid construction requires all checks.

Report all 12 programs per model/arm, exception types, library usage, truncations, prompt
and completion tokens, generation time and worker time. Auditing multiple executions is
separate from first-submission latency. Paired outcomes are descriptive: three seeds on four
fixed task definitions do not constitute twelve independently sampled finance problems.
One shared GPU allocation does not estimate asymptotic scaling. Do not infer O(N^0.26)
from this experiment or describe an empirical fit as guaranteed complexity.

Freeze and audit artifacts live under `benchmarks/agent_study/evidence/20260928-codegen-utility/`.
Any infrastructure interruption is recorded separately; already generated responses are
never replaced. A benchmark result is reported only for a complete, hash-verified batch.
