# Optional RSI module source verification

The module adds an explicitly enabled playbook loop; it is not connected to the frozen 32B
trading experiment. The test log contains 42 passing source-contract tests using deterministic
synthetic evaluators. It is not a model benchmark and contains no financial Return Rate.

The first development test run exposed a JSON tuple/list round-trip comparison bug (26 failures,
12 passes). Canonical JSON comparison fixed it; the original 38 cases then passed, followed by
four additional boundary tests in the final 42-case run. No tests were skipped to obtain this result.

Index generation, package generation and repository validation passed. Source hashes, limits and
the unchanged experiment identity are in `verification.json`. Cross-platform CI and installed
wheel/sdist smoke checks run on the published PR head; their result is recorded by GitHub.

The sole Beacon experiment remains job 2232777 on its original source29 snapshot. Its eventual
net Return Rates, all decisions and complete daily capital/trade ledgers must be published after
completion and the independent audit. This source-verification report does not stand in for them.
