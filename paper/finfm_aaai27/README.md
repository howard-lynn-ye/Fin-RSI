# FinFM @ AAAI-27 submission (benchmark track)

**Fallback.** On 2026-11-05 the submission switched to the 4-page short paper in
`../finfm_aaai27_short/`. This 7-page version is kept in case the main results are complete for
three or more models by about 18 November.

Workshop: FinFM, Foundation Models and Generative AI for Finance, at AAAI-27 (Montréal,
22 or 23 February 2027). Submission deadline 20 November 2026, 23:59 AoE; notification
2 December 2026. Non-archival; concurrent review elsewhere is allowed (check ARR's own
policy). Long papers: up to 7 pages of technical content, AAAI-27 template, double-blind.
At least one author must present in person; no virtual presentations.
Portal: https://openreview.net/group?id=AAAI.org%2F2027%2FWorkshop%2FFinFM

## One body, two papers

`main.tex` reads the abstract, sections, numbers, figures and tables from
`../latex_acl2027/`, so this version and the ACL 2027 draft cannot drift apart. Only
`sections/results.tex` (what is ready by the deadline) is specific to this submission.

## Template

`aaai2027.sty` and `aaai2027.bst` are the official AAAI-27 author-kit files, unmodified
(do not edit them: AAAI rejects papers that change the style). They were taken from a public
copy whose UPSTREAM.md records the official kit's SHA-256 values; the copies here match:

    aaai2027.sty  01990f8d3716f8e39c5b28a7478f3ac6e28630d1161de6105a7217f5e7f60e43
    aaai2027.bst  5db7765ba99de5c1e4686f9b3940a0add9c5e702f2164514462bec130ccb6e3c

The style forbids hyperref; `main.tex` defines a text-only `\href` because the .bst emits it.

## Build

Needs the `newtx`, `kastrup` (binhex.tex) and TeX Gyre fonts. On Debian/Ubuntu:
`texlive-fonts-extra texlive-plain-generic tex-gyre`.

```bash
cd paper/finfm_aaai27
latexmk -pdf main.tex
```
