# Terminal Return Rate protocol (v8)

User instruction, October 6, 2026: keep the task simple; the model chooses how to use
the library, and seeks the highest cumulative return by the deadline.

The task states: "Maximize cumulative Return Rate after costs by [deadline]."
Both arms start with USD 100,000. Report initial capital, ending capital (cash plus marked
holdings, after trading fees), and `100 * (ending_capital / initial_capital - 1)`.
There are no deposits or withdrawals. This is cumulative Return Rate, without annualization
or risk adjustment. The normalized ledger is scaled to those account amounts and independently audited.
Both arms receive the same deadline, dated market/evidence inputs, long-only/no-leverage
constraints, zero cash yield, next-close execution and 5 bps per traded side. No method,
asset allocation, market exposure or minimum tool-use count is prescribed. Only the
library arm receives library access. Detailed contracts are available on demand through
`help_tool(name)` and the existing complete discovery interfaces.

Eight model responses remain the total budget in either arm. Up to seven allow research;
the eighth accepts only a single JSON `submit` or `hold`. Earlier completion is allowed.
No ninth response, automatic portfolio, silent argument correction or profitable-run filter
is introduced. Invalid final responses are retained as failures. `hold` keeps existing units
and cash without trading; it does not resubmit drifted weights. Submission, explicit hold
and unresolved failure are counted separately. The model remains responsible for decisions.

The new version is `multisource-model-v3`, interface `v8`. New freezes default to it.
Protocol hashes bind the objective, deadline, budget, source and inputs; the launch gate
also exercises the new confined worker, help examples and reserved final stage. Existing
capability qualification and independent cash-and-units accounting remain required.
Frozen older runs must use their archived sources. No old result is overwritten, relabeled
or combined with this protocol. There are no measured v8 Return Rates in this change.

This repairs task alignment and interaction contracts. It does not certify every library
parameter combination or expand the historical evidence dataset. Selected monetary news,
surveys, futures positioning and household disclosures retain their coverage/vintage limits;
live collection remains outside the historical replay. Return Rate (%) is the primary
outcome; risk metrics are explanatory. Higher realized returns are an empirical question.

Before a new full batch, execute technical qualification and a bounded model pilot under
this frozen protocol. Judge pilot usability by valid actions and completed decisions, never
by selecting returns. Report all failures. Cross-period validation and the requested twenty
model comparison remain separate outstanding experiment work.
