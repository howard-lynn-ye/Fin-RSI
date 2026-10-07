"""Run a full frozen pair after template checks, then export only audited aggregates.

The CLI freezes and qualifies each pair before calling this runner. Failed model
decisions are retained. Exceptions preserve the partial pair for the same-source
resume; they never produce a completed result or zero-return placeholder.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study.qualify_model_matrix import prompts
from benchmarks.agent_study.transformers_chat import context_limit, load_tokenizer, tokenize_chat

HERE = Path(__file__).resolve().parent


def run(root):
    from transformers import AutoConfig
    p = study.verify(root)
    manifest = study.old.load(HERE / 'manifest.json')
    assert p['interface'] == manifest['interface']
    assert p['interaction_revision'] == manifest['interaction_revision']
    assert p['seed'] in manifest['seeds']
    assert p['model'][0] in {m['family'] for m in manifest['models']}
    assert p['decisions_per_arm'] == manifest['decisions_per_arm']
    study.terminal_runtime(p['interface']).require_qualification(root)
    name, revision = p['model'][1:]
    config = AutoConfig.from_pretrained(name, revision=revision, local_files_only=True)
    tokenizer = load_tokenizer(name, revision=revision, local_files_only=True)
    counts = {'raw': [], 'library': []}
    limit = context_limit(config, tokenizer)
    for arm, messages in prompts(root):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert all(m['content'] in rendered for m in messages), 'template dropped content'
        ids = tokenize_chat(tokenizer, messages)['input_ids']
        assert ids == tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        assert len(ids) + p['max_tokens'] <= limit, 'initial context cannot fit'
        counts[arm].append(len(ids))
    preflight = dict(model=p['model'], protocol_sha256=study.old.sha(root / 'protocol.json'),
        source_sha256=study.old.sha(Path(__file__)), manifest_sha256=study.old.sha(HERE / 'manifest.json'),
        max_initial_tokens={a: max(c) for a, c in counts.items()}, context_limit=limit,
        limitation='Initial template check only; every inference call enforces context without silent truncation.')
    path = root / 'full-cycle-preflight.json'
    if path.exists():
        assert study.old.load(path) == preflight
    else:
        study.old.write(path, preflight)
    print(json.dumps(dict(event='full_period_start', family=p['model'][0], seed=p['seed'],
                          decisions_per_arm=p['decisions_per_arm'], window=p['window'])), flush=True)
    if not (root / 'inference-receipt.json').exists():
        study.run(root)
    scores = study.score(root)  # Refuses incomplete receipts; independently audits cash and units.
    audit = study.old.load(root / 'independent-model-audit.json')
    assert audit['passed']
    public = dict(interface=p['interface'], interaction_revision=p['interaction_revision'],
        model=p['model'], seed=p['seed'], window=p['window'],
        primary_metric=p['primary_report'], protocol_sha256=study.old.sha(root / 'protocol.json'),
        source_sha256=p['source_sha256'], input_hashes=p['input_hashes'],
        inference_receipt_sha256=study.old.sha(root / 'inference-receipt.json'),
        scores_sha256=study.old.sha(root / 'scores.json'),
        independent_audit_sha256=study.old.sha(root / 'independent-model-audit.json'),
        paths=scores['paths'], return_difference_pp=scores['return_difference_pp'],
        limitations=p['limits'])
    destination = root / 'public-result.json'
    if destination.exists():
        assert study.old.load(destination) == public
    else:
        study.old.write(destination, public)
    print(json.dumps(dict(event='full_period_complete', family=p['model'][0], seed=p['seed'],
        time_utc=datetime.now(timezone.utc).isoformat(),
        paths={a: {k: scores['paths'][a][k] for k in
            ('initial_capital', 'ending_capital', 'return_rate_pct', 'failed_decisions')}
            for a in ('raw', 'library')})), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root)
