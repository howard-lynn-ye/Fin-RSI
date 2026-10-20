# Installed-library collection acceptance — 2026-10-06

The live acceptance check completed with **`needs_review`**, not an all-sources pass.
This tests installed-library collection, persistence and source health. It contains no
trading decisions or Return Rate measurement.

## Reproduction and provenance

- Library source: `f456ab4103bed3918a0d7de0307c5ab6e19db13b`.
- Harness: `scripts/check_installed_collection.py`; executed file SHA-256
  `cdcdac0221b0099d2bf62b9f6217488c9fd0328fc555f0f8389604ca78f82788`.
- Execution: Beacon Slurm step `1907749.2`, Python 3.12, a freshly built wheel installed
  with the `collect` extra into a new virtual environment. Python ran with `-I` from
  outside the source checkout. The installed CLI's `sources` command also succeeded.
- Collection/news fixtures: **49 passed** before the parser correction below.
- Live observation: **18:39:22–18:41:02 UTC** on 2026-10-06.
- Original aggregate receipt: [live-receipt.json](live-receipt.json). The harness returned
  exit code **2**. The enclosing Slurm step completed successfully because it preserves
  failed acceptance receipts; its exit status is not an acceptance pass.
- Raw responses, PDFs, extracted text and SQLite records remain on Beacon. Only aggregate
  metadata is included here.

## Observed sources

| Source | Result | Interpretation |
|---|---|---|
| Federal Reserve RSS | 20 news records; second scheduled poll succeeded | Latest reported publication: 2026-10-05 20:30 UTC |
| ECB RSS | 15 news records; second scheduled poll succeeded | Latest reported publication: 2026-10-06 13:00 UTC |
| GDELT company-news query | HTTP 429; zero records | Failed availability check; the collector saved the error and a 900-second retry delay |
| House disclosures | Two filings, one extracted transaction, one unparsed filing | Partial; empty layout extraction was subsequently reproduced as a parser defect |
| Schwab pressroom | One web-page record | Transport only; no structured sentiment series verified |
| CFTC report page | One web-page record | Transport only; no structured position series verified |
| SEC Form 4/13F | Skipped | Caller identity was not configured |

All **40 records** survived reopening the database. The next scheduled news polls added
zero duplicate records. This demonstrates persistent state and continued polling, but no
new publication arrived during this short observation, so live new-publication detection
was not observed. Installation alone does not start a background collector.

The digest retained **5 fresh unique news articles** and excluded **30 stale news records**
and **5 non-news records**. Observation time was not substituted for publication time.
Five records had unknown publication times, including the disclosure records; those dates
remain unknown. The receipt's `configure_does_not_fetch` field checks the tool's returned
`started: false` flag, not an independent network trace.

## Disclosure extraction defect

For [House document 20035553](https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2026/20035553.pdf),
the default layout extractor returned zero characters. Plain extraction returned **1,030
characters** and the existing conservative row parser recognized **one transaction row**.
The PDF was therefore not established to be an unreadable scan. Preserving plain text when
layout extraction is empty addresses this specific defect; it does not add OCR or guarantee
correct extraction of every PDF layout.

The correction was built into another wheel and installed in a second fresh environment.
The collection/news fixtures then reported **50 passed**. Replaying the cached PDF through
that installed wheel recovered **921 characters** after removing NUL characters and **one
reviewable transaction row**, including an amount range; see
[parser-replay.json](parser-replay.json) for source, patch and document hashes. This was a
cached-document replay, not a second live collection run. Index/package regeneration and
repository validation also passed, with the existing library-listing budget warning.

The correction and its verification are recorded separately from the original live receipt.
Previously seen filings require an explicit new watch ID to recheck after an upgrade;
historical results and the original collection receipt are preserved.

## Remaining experiment scope

Live source coverage is incomplete. Successful generic page collection is not evidence of
complete sentiment, behavioral or positioning data. This acceptance does not demonstrate
that all registered library tools were exercised, nor that using the library improves
Return Rate.

The separate v8 usability pilot was submitted as Slurm job **1908588** for Qwen3-4B,
Granite-8B and SmolLM3-3B, each with one raw/library decision on the same first date. It uses
the existing frozen historical inputs and protocol; it does not consume this live database.
Submission is not completion, a one-date pilot is not a full-period return experiment,
and it does not satisfy the planned 20-model comparison.
