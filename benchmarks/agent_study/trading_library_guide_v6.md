# fin-skills: read before making this decision

You make the investment decision. fin-skills supplies financial knowledge, data
handling, numerical methods and executable audits. You choose which evidence and
tools to use, how to interpret their output, and the final assets and weights.
An optimizer's output is a candidate allocation, not an instruction to submit it.

## What the library covers

These are starting points, not an exhaustive catalog or a strategy recommendation.
Pass a skill ID below as the `name` argument of `read_skill`.

| Need | Relevant skill IDs | What to check |
|---|---|---|
| Research and backtest validity | `research-integrity-guards`, `backtest-validation` | Availability times, leakage, execution assumptions and transaction costs |
| Public information and news | `public-information-collection` | Implemented collectors, source configuration, provenance and incomplete results |
| Crowd psychology and social discussion | `social-and-influencer-feeds`, `kol-credibility-registry` | Real dated posts, sampling bias, source credibility and independently verified track records |
| Observable group behavior | `institutional-13f`, `signal-reconciler` | Reported holdings and conflicting evidence; a post is not evidence of an executed trade |
| Public officials' disclosures | `congressional-trading-disclosures` | Public disclosure time, amount ranges and delayed reporting; a transaction report is not a complete current portfolio |
| Corporate insiders | `insider-form-4` | Transaction codes and disclosure times; acquisitions need not be open-market purchases |
| Fundamentals and macro | `fundamental-and-macro-data` | Publication dates, revisions and differences between reporting periods and availability |
| Portfolio construction | `portfolio-optimizers` | Input conventions, estimation risk and the objective a method actually optimizes |

Skill documents contain methods, caveats, references and some demonstrations.
They are not themselves observations of today's market. Check verification dates
and distinguish real source records from synthetic examples or secondhand claims.
Do not infer that a method is profitable from a worked example or a skill title.

## Discover and read before applying a method

Search by a relevant phrase, then read a matching skill. For example, these are
separate possible JSON calls; choose according to your task:

```json
{"tool":"list_skills","arguments":{"query":"disclosure","limit":8}}
```

```json
{"tool":"read_skill","arguments":{"name":"congressional-trading-disclosures","offset":0,"limit":3000}}
```

Responses are dictionaries. Text is in `reply["text"]`; a non-null `next_offset`
means only part of the document was returned. Continue from that offset when you
need the rest. `references` names the available reference documents; pass an exact
name as `reference` to `read_skill`. `list_skills(query="")` pages the full catalog.
Prioritize relevant documents within the call budget; no particular strategy or
number of skill calls is required. Do not claim to have read a document you did
not retrieve. Retrieved text still needs your interpretation.

## What this study environment actually supplies

This v6 adapter currently supplies visible raw quotes and corporate actions.
`load_history` prepares full adjusted matrices in library Python; `read_history`
and `read_market` are inspection pages. A snapshot is not return history.
Use `describe_algorithm` before a numerical method for its exact arguments,
defaults, output type and runnable example. Check `reply["ok"]`. Portfolio weights
are `reply["result"]["weights"]`. For guards, execution `ok` and audit `passed`
are different facts. A passed audit does not establish a profitable strategy.

The wider package has public-data collectors, but this study's network is disabled
and those collection tools are not exposed here. Historical news, social posts,
fund-flow records and officials' filings are NOT supplied by these two CSV files.
Reading the corresponding skills does not fill those data gaps. Never invent an
unseen sentiment score, holding, disclosure, fund flow or source track record.
An experiment requiring these inputs must supply a dated evidence bundle before
it can assess the full information workflow; the current price-only study does
not satisfy that requirement. Use only information available by the task cutoff.

## Finish with your own decision

Assess source quality, missing inputs, risk exposure and trading costs before
choosing weights. You may accept, modify or reject a tool's proposed allocation.
Submit only your chosen ticker-to-weight dictionary using `submit(weights)`.
The experiment records the guide in the initial request and records document
retrievals separately; neither proves that you understood or used the material.
