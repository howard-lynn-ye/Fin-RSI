# Repaired v9 full-period run

This wave runs the terminal-return v8 task through v9's optional library RAG route.
It is a new freeze, not a rescore of v7 or an extension of the single-date pilots.
There are no new return results in this directory at preparation time.

The planned first wave has five models, three seeds each, and requires both arms to complete
all 44 decision dates. The manifest includes all three failed v9 pilot models. It is a subset of the
existing twenty-model plan, not completion of that plan. Model identity and revisions
come from the existing pinned matrix; the model decides investments and tool use.

The repaired interaction has a separate final-stage system message with only the two
allowed actions. The latest feedback also delivers this contract before response 8.
The v9 retrieval guide is absent from that final-stage message. Hold explicitly preserves
cash, including an all-cash account. A submit may include a string rationale alongside
valid weights; unknown order fields still fail. These usability changes do not establish
how much of any earlier return gap came from integration errors.

Both arms retain eight responses, 1,024 output tokens each, identical dated information,
next-session-close execution, and five basis points per traded side. The output limit
can still cause failures; there is no automatic investment fallback or silent truncation.
Incomplete infrastructure runs are reported separately from failed model decisions.

Each pair starts at USD 100,000. The report must show ending capital and
`Return Rate (%) = 100 * (ending capital / initial capital - 1)`, followed by library
minus raw in percentage points. Scoring requires a complete inference receipt and a
separate cash/unit accounting audit. A model counts as complete only after all three seeds.

`prepare.sbatch` builds, validates, tests and freezes a fresh source directory on a CPU
node. Submit `full_cycle.sbatch` only with an `afterok` dependency on that preparation
job. Two array tasks divide the five models and run their three seeds sequentially,
so the fifteen pairs consume only two submitted-job slots. Each pair is an entire period,
not a pilot. Every decision is checkpointed for same-source resume. An infrastructure
exception preserves the pair and skips the remaining seeds of that model; other models
in the shard can continue. Raw responses and
evidence stay on Beacon; `public-result.json` contains only verified aggregate results
and provenance hashes for publication after completion.

On October 7 the first preparation submission was rejected with `AssocMaxSubmitJobLimit`;
no preparation or full-cycle job ID was created. The controller reported this user's
`beacon` association at `MaxJobs=6(6)` and `MaxSubmitJobs=11(11)` (limit and current use).
The partition allows `medium` QOS and up to three days; these scripts request the existing
`angliece` account, `medium` QOS and at most two days. The immediate blocker was occupied
job slots. A fifteen-task array would also exceed the eleven-job limit even when idle,
so it was replaced before successful submission. Do not resubmit until a slot is available,
and do not cancel other projects or change accounts to bypass the limit.
`submit.sh prepare`, `submit.sh run0`, and `submit.sh run1` inspect the current controller
quota before submitting one job each. Each command writes its returned job ID and refuses
duplicate submission. The two run commands retain an `afterok` dependency on preparation.
They do not poll or create a heartbeat; a full quota exits without an `sbatch` attempt.

The user authorized local publication checks on October 7 while the queue was full.
That exception permits checking and publishing existing records and code; it does not
authorize local model inference. The new full-cycle batch remains unsubmitted until the
recorded preparation and GPU job IDs exist. No post-repair Return Rate is reported here.
The release archive and launcher names in these scripts replace the unsubmitted candidate
archives; preserve the older archives as preparation history.

The local publication check passed 71 focused runtime, capability, parser and personal-response
tests; three Linux confinement tests were skipped on Windows. This is not GPU qualification.
Preparation on Beacon still runs the Linux tests before any new full-period inference.

Limits: one development market path and retrospective availability assumptions, incomplete
real-world source coverage, and current library reference knowledge. Seeds are not independent
market cycles. This wave does not yet add a downturn window or prove future profitability.
