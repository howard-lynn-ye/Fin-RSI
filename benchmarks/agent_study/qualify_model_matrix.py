"""CPU-only template/context checks; never counts as a completed model experiment.

Uses pinned public model metadata/tokenizers, without loading weights or remote code.
The output contains sizes and failures, never the dated evidence or full prompts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study import trading_capabilities as cap
from benchmarks.agent_study.transformers_chat import (
    context_limit, load_tokenizer, tokenize_chat, validate_reasoning_tokens,
)
from benchmarks.agent_study.model_matrix import matrix


def prompts(root):
    protocol = study.old.load(root / 'protocol.json')
    packets = study.old.load(root / 'evidence-packets.json')
    inputs = study.old.load(root / 'inputs.json')
    holdings = dict.fromkeys(study.md.TICKERS, 1 / len(study.md.TICKERS))
    for arm in ('raw', 'library'):
        if protocol.get('interface') in ('v8', 'v9'):
            terminal = study.terminal_runtime(protocol['interface'])
            for day in inputs['dates']:
                yield arm, [dict(role='system', content=terminal.system_prompt(arm)),
                            dict(role='user', content=terminal.task(day, holdings, protocol['deadline']) +
                                 study.evidence_text(packets[day]))]
            continue
        guide, _ = study.runtime.orientation(arm, study.GUIDE)
        system = cap.COMMON + (cap.LIBRARY + '\n' + guide if arm == 'library' else '')
        system += '\nCurrent call 1 of 8; 8 calls remain including this one. Finish with submit(weights).'
        for day, pick in zip(inputs['dates'], inputs['picks']):
            yield arm, [dict(role='system', content=system), dict(role='user', content=
                study.previous.task(day, pick, holdings) + study.evidence_text(packets[day]))]


def qualify(root, manifest, output, cache):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if output.exists():
        raise FileExistsError('preserve previous qualification; use a fresh output file')
    rows, examples = matrix(manifest), list(prompts(root))
    results = []
    for row in rows:
        result = dict(id=row['id'], model=row['model'], transport=row['transport'])
        if row['transport'] != 'hf':
            result.update(status='separate_native_qualification_required')
        else:
            try:
                kwargs = dict(revision=row['revision'], cache_dir=cache, trust_remote_code=False)
                config = AutoConfig.from_pretrained(row['model'], **kwargs)
                if type(config) not in AutoModelForCausalLM._model_mapping:
                    raise ValueError('installed transformers has no local causal-LM implementation')
                tokenizer = load_tokenizer(row['model'], **kwargs)
                validate_reasoning_tokens(tokenizer)
                limit = context_limit(config, tokenizer)
                counts = dict(raw=[], library=[])
                default_tokenizer_duplicates = False
                for arm, messages in examples:
                    rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                        add_generation_prompt=True)
                    if not all(m['content'] in rendered for m in messages):
                        raise ValueError('official template dropped supplied instructions or evidence')
                    ids = tokenize_chat(tokenizer, messages)['input_ids']
                    expected = tokenizer.apply_chat_template(messages, tokenize=True,
                        add_generation_prompt=True)
                    if ids != expected:
                        raise ValueError('inference tokens differ from the official chat template')
                    default_tokenizer_duplicates |= tokenizer(rendered)['input_ids'] != ids
                    counts[arm].append(len(ids))
                maxima = {arm: max(values) for arm, values in counts.items()}
                if max(maxima.values()) + 1024 > limit:
                    raise ValueError(f'initial context plus output exceeds {limit} tokens: {maxima}')
                result.update(status='initial_template_passed', revision=row['revision'],
                    model_type=config.model_type, context_limit=limit,
                    max_initial_tokens=maxima, dates_per_arm=len(counts['raw']),
                    default_tokenizer_changes_template=default_tokenizer_duplicates,
                    reasoning_template=('think' in (tokenizer.chat_template or '').lower()),
                    remaining_gate='GPU action parsing, tool round-trip and multi-turn context checks')
            except Exception as exc:
                result.update(status='blocked', error=f'{type(exc).__name__}: {str(exc)[:700]}')
        results.append(result)
        print(f"{row['id']}: {result['status']}", flush=True)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=study.old.sha(manifest), source_sha256=study.old.sha(Path(__file__)),
        evidence_packets_sha256=study.old.sha(root / 'evidence-packets.json'),
        environment=cap.environment(), results=results,
        scope='Metadata and initial prompt delivery only; no inference, tool competence or Return Rate.',
        passed_for_formal_launch=False)
    study.old.write(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('MULTISOURCE_MODEL_MATRIX.json'))
    args = parser.parse_args()
    qualify(args.root, args.manifest, args.output, args.cache)
