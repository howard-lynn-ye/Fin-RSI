# Multi-model execution update, October 5, 2026

The twenty-model requirement is not complete. The manifest declares 22 model IDs;
14 open-weight IDs now have an executable study route. Listing a model, downloading
weights, reviewing code, or passing a tokenizer test does not count as a return experiment.
Each counted model needs all three paired repeats and valid accounting receipts.

The personally authored comparison is complete and separately reported in
[PERSONAL_V7_20261005.md](PERSONAL_V7_20261005.md): raw Return Rate 38.33463727750161%,
library 38.25568611749135%, difference -0.07895116001026281 percentage points. It is
non-blind and excluded from the twenty-model count. Older v6 results remain in
[LEGACY_V6_RESULTS_20261005.md](LEGACY_V6_RESULTS_20261005.md); do not pool versions.

## Repairs needed before expanding the model families

- Preserve `<think>` delimiters, including a delimiter supplied by the chat template.
  Execute only the answer after balanced reasoning. Reject unfinished or malformed
  reasoning, and reject unqualified alternative special-token delimiters before inference.
- Stop decoding at the effective EOS token and record that stop set in each response.
  Reasoning still consumes the declared 1024-token generation budget. No budget increase
  or strategy-generated replacement trade is introduced.
- Bind the model manifest and qualification source to the frozen protocol. Reject model
  alias collisions and disagreement with a legacy revision; support hyphens in model IDs
  when aggregating. Newly declared matrix runs require the v7 interface, whose backend
  infrastructure errors stop inference rather than silently becoming cash holdings.
- Explicitly load publisher-compatible Mistral tokenization. The pinned Nemo 12B regex
  already matched its own `tekken.json`; its warning was not evidence of a mismatch.
  The pinned Small 24B regex differed and matches after `fix_mistral_regex=True`.
  This does not establish that tokenization caused any historical return gap.

The Mistral comparison used the actual pinned `tekken.json` files for revisions
`04d8a90549d23fc6bd7f642064003592df51e9b3` and
`9527884be6e5616bdd54de542f9ae13384489724`. The publisher's explanation is in
[Mistral's tokenizer discussion](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84).

## Verification

Slurm job **1907725** completed with exit code 0:

- 58 focused tests passed; no skips.
- Index generation, package generation and repository validation passed.
- All 13 accessible pinned tokenizers passed synthetic final-action decoding checks.
- Both Mistral variants matched their own pinned primary-source regex after loading.

These are executable regression and tokenizer checks, not GPU competence or return results.
Llama remains gated. Native subscription model execution remains unqualified; reviews by
Opus, Fable or Gemini do not count as experiments. The GPT route still needs an explicit
account of what model identity its native receipt can and cannot attest.

The independent Opus review used verified `claude-opus-5-5`, requested effort `medium`.
Response SHA256: `72242fd557f93a8874692fd21b8d5f0ae62f89f5d113b4d069260e739e0af2ce`.
Its alias/revision and v6-route findings were repaired and tested. Its conditional concern
about `transformers_chat.py` source hashing was checked: `trading_study_v5.sources()` already
includes it. The newly used matrix qualification module was added explicitly.

## Submitted work

All paths below are under `/beacon-projects/radfm/wy891/fin-multisource-models-20261005`.

| Job | Work | Meaning |
|---|---|---|
| 1907701 | Prepare all three Qwen 7B/14B seed pairs | Completed; frozen `v7-models-v1/source` |
| 1907708, tasks 0–1 | Qwen 7B/14B seed 11 | GPU probe then paired inference and scoring; remaining seeds not submitted in this array |
| 1907724 | Phi mini, Granite 8B, SmolLM3 weights | Download completed; no inference implied |
| 1907729 | Prepare Mistral Nemo, Phi mini, Granite, SmolLM3, seeds 11/23/37 | Uses validated `matrix-runtime-v2/source`; creates `matrix-replay-v1` |
| 1907730, tasks 0–1 | Mistral Nemo and Phi mini, three repeats each | Depends on preparation and weights; each pair gets a GPU/tool probe before inference |
| 1907732, tasks 2–3 | Granite and SmolLM3, three repeats each | Same dependency and probe requirements |
| 1907731 | Six more pinned open-weight model downloads | Preparation only; inference not submitted |

The six additional downloads are Qwen general 7B, Qwen3 4B, Mistral Small 24B, OLMo 7B,
Falcon3 10B and DeepSeek R1 Distill 14B. Their GPU qualifications and complete experiments
remain outstanding, as do Qwen 32B and the native subscription track.

The earlier unused Mistral jobs **1907712/1907713** were cancelled before inference while
investigating the tokenizer warning, and replaced by the new matrix preparation/run jobs.
No unrelated job was cancelled. Shared SSH remained usable. Slurm accounting database
access through `sacct` failed once; use scheduler state and completion artifacts until
database recovery is established rather than retrying it repeatedly.

The archived `matrix-runtime-v2` source is the exact execution source. A later explanatory
comment in Git correctly distinguishes Nemo's already-correct regex from Small's mismatch;
score every run with its own archived source, including when the difference is only a comment.
