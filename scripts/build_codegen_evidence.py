"""Generate manuscript macros for the code-generation study and the FinQA format reanalysis.

Every number is read from hash-checked evidence; `--check` fails if the TeX file is stale.
Official (pre-registered) and post-hoc sensitivity results are kept in separate macros.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'benchmarks/agent_study/evidence'
OFFICIAL = EVIDENCE/'20260928-codegen-utility'
POSTHOC = EVIDENCE/'20260928-codegen-fence-reanalysis'
FINQA = EVIDENCE/'20260928-finqa-format-reanalysis'
ENFORCED = EVIDENCE/'20260929-codegen-enforced'
ENFORCED_V2 = EVIDENCE/'20261029-codegen-enforced-v2'
OUTPUT = ROOT/'paper/latex_naacl/codegen_numbers.tex'
FAMILY = {'7b': 'Seven', '14b': 'Fourteen'}
ARM = {'raw': 'Raw', 'library': 'Library'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verified():
    """Re-check the chain response -> receipt -> audits -> post-hoc extraction -> scores."""
    receipt = load(OFFICIAL/'inference-receipt.json')
    assert receipt['protocol_sha256'] == sha(OFFICIAL/'protocol.json')
    assert len(receipt['rows']) == receipt['planned'] == 48
    for row in receipt['rows']:
        assert sha(OFFICIAL/row['path']) == row['sha256'], row['id']
    official = load(OFFICIAL/'scores.json')
    assert official['inference_receipt_sha256'] == sha(OFFICIAL/'inference-receipt.json')
    audits = {}
    for name, digest in official['audits'].items():
        assert sha(OFFICIAL/'audits'/(name+'.json')) == digest, name
        audits[name] = load(OFFICIAL/'audits'/(name+'.json'))
    extraction = load(POSTHOC/'extraction-receipt.json')
    assert extraction['sha256'] == sha(POSTHOC/'extractions.json')
    assert extraction['fetch_receipt_sha256'] == sha(OFFICIAL/'fetch-receipt.json')
    assert extraction['inference_receipt_sha256'] == sha(OFFICIAL/'inference-receipt.json')
    assert extraction['official_scores_sha256'] == sha(OFFICIAL/'scores.json')
    posthoc = load(POSTHOC/'scores.json')
    assert posthoc['extraction_receipt_sha256'] == sha(POSTHOC/'extraction-receipt.json')
    diagnostics = load(POSTHOC/'diagnostics.json')
    assert diagnostics['scores_sha256'] == sha(POSTHOC/'scores.json')
    finqa = load(FINQA/'scores.json')
    assert finqa['extraction_receipt_sha256'] == sha(FINQA/'extraction-receipt.json')
    return official, audits, load(POSTHOC/'extractions.json'), posthoc, diagnostics, finqa


def macros():
    official, audits, extractions, posthoc, diagnostics, finqa = verified()
    out = {}
    agg, post = official['aggregate'], posthoc['aggregate']
    total = lambda key, table: sum(v[key] for v in table.values())
    out['CGPlanned'] = sum(v['planned'] for v in agg.values())
    out['CGOfficialExecuted'] = total('executed', agg)
    out['CGOfficialValid'] = total('financial_valid', agg)
    out['CGOfficialNumerical'] = total('numerical', agg)
    errors = Counter()
    for value in agg.values():
        errors.update({k: n for k, n in value['errors'].items() if k != 'none'})
    assert set(errors) == {'SyntaxError'}
    out['CGFormatFailures'] = errors['SyntaxError']
    out['CGTruncated'] = sum(a['finish_reason'] != 'stop' for a in audits.values())
    prose = Counter(r['family'] for r in extractions if r['trailing_prose_chars'])
    out['CGSevenTrailingProse'], out['CGFourteenTrailingProse'] = prose['7b'], prose['14b']
    library = [r for r in extractions if r['arm'] == 'library']
    out['CGLibraryArm'] = len(library)
    out['CGLibraryMentions'] = sum(r['mentions_fin_skills'] for r in library)
    out['CGAlgorithmUsers'] = total('algorithm_users', agg) + total('algorithm_users', post)
    out['CGAgentGuardUsers'] = total('agent_guard_users', agg) + total('agent_guard_users', post)
    out['CGPostExecuted'] = total('executed', post)
    out['CGPostNumerical'] = total('numerical', post)
    out['CGPostValid'] = total('financial_valid', post)
    out['CGPostSplitFailures'] = total('split_failures', post)
    out['CGPostTimingFailures'] = total('future_failures', post)
    timing = diagnostics['timing_summary']
    out['CGSameSession'] = sum(n for k, n in timing.items() if k.startswith('same_session'))
    out['CGStrictlyFuture'] = sum(n for k, n in timing.items() if k.startswith('strictly_future'))
    out['CGSameSessionGuardPassed'] = timing.get('same_session|guard_passed=True', 0)
    out['CGGuardCaught'] = timing.get('strictly_future|guard_passed=False', 0)
    rows = posthoc['rows']
    clean = [r for r in rows if r['execution']['executed'] and r['audit']['future_invariant']]
    out['CGTimingClean'] = len(clean)
    out['CGGuardFalseAlarms'] = sum(r['execution']['posthoc_causality_guard']['passed'] is False
                                    for r in clean)
    out['CGRecumulated'] = sum(diagnostics['recumulated_split_factors'].values())
    out['CGEnvCompared'] = posthoc['environment_check']['compared']
    out['CGEnvReproduced'] = posthoc['environment_check']['reproduced']
    for key, value in agg.items():
        family, arm = key.split('-')
        name = 'CG' + FAMILY[family] + ARM[arm]
        p = post[key]
        out[name+'OfficialExec'] = value['executed']
        out[name+'OfficialValid'] = value['financial_valid']
        out[name+'PostExec'] = p['executed']
        out[name+'PostNumerical'] = p['numerical']
        out[name+'PostSplitFail'] = p['split_failures']
        out[name+'PostTimingFail'] = p['future_failures']
        out[name+'PostValid'] = p['financial_valid']
        out[name+'InTokens'] = f"{value['mean_input_tokens']:,.0f}"
        out[name+'OutTokens'] = f"{value['mean_output_tokens']:.0f}"
        out[name+'GenSeconds'] = f"{value['mean_generation_seconds']:.1f}"
    fam = {'qwen': 'Qwen', 'mistral': 'Mistral'}
    arm = {'rag_api': 'Rag', 'bge': 'Bge', 'kev': 'Kev'}
    for key, value in finqa['aggregate'].items():
        family, condition = key.split('-')
        name = 'FQ' + fam[family] + arm[condition]
        out[name+'Legacy'] = value['legacy_correct']
        out[name+'Fence'] = value['fence_v2_correct']
        out[name+'Recoverable'] = value['recoverable_final_correct']
        out[name+'Missing'] = value['missing_final']
        out[name+'N'] = value['n']
    out['FQLegacyTotal'] = sum(v['legacy_correct'] for v in finqa['aggregate'].values())
    out['FQFenceTotal'] = sum(v['fence_v2_correct'] for v in finqa['aggregate'].values())
    out['FQRecoverableTotal'] = sum(v['recoverable_final_correct']
                                    for v in finqa['aggregate'].values())
    out['FQAnswers'] = sum(v['n'] for v in finqa['aggregate'].values())
    out['FQRecoveredExtra'] = sum(r['recoverable_final_correct'] and not r['strict_correct']
                                  for r in finqa['rows'])
    out['FQRecoveredExtraQwen'] = sum(r['recoverable_final_correct'] and not r['strict_correct']
                                      for r in finqa['rows'] if r['family'] == 'qwen')
    mistral = [v for k, v in finqa['aggregate'].items() if k.startswith('mistral-')]
    out['FQMistralMissingTotal'] = sum(v['missing_final'] for v in mistral)
    out['FQMistralAnswers'] = sum(v['n'] for v in mistral)
    return out


def enforced_macros():
    """Three-arm enforced study; empty until its complete batch is fetched and verified."""
    if not (ENFORCED/'scores.json').exists():
        return {}
    fetch = load(ENFORCED/'fetch-receipt.json')
    assert fetch['complete'] and fetch['all_match']
    protocol = load(ENFORCED/'protocol.json')
    assert sha(ENFORCED/'inputs.json') == protocol['input_sha256']
    receipt = load(ENFORCED/'inference-receipt.json')
    assert receipt['protocol_sha256'] == sha(ENFORCED/'protocol.json')
    assert len(receipt['rows']) == receipt['planned'] == protocol['planned'] == 72
    for row in receipt['rows']:
        for attempt in row['attempts']:
            assert sha(ENFORCED/attempt['path']) == attempt['sha256'], attempt['path']
    scores = load(ENFORCED/'scores.json')
    assert scores['inference_receipt_sha256'] == sha(ENFORCED/'inference-receipt.json')
    for name, digest in scores['audits'].items():
        assert sha(ENFORCED/'audits'/(name+'.json')) == digest, name
    arm = {'raw_exec': 'RawExec', 'raw_guards': 'RawGuards', 'library_guards': 'LibGuards'}
    fam = {'all': 'All', '7b': 'Seven', '14b': 'Fourteen'}
    pct = lambda n, d: f'{100 * n / d:.1f}'
    out = {}
    for key, value in scores['aggregate'].items():
        family, condition = key.split('-')
        name, n = 'EN' + arm[condition] + fam[family], value['planned']
        out[name+'N'] = n
        for metric, field in (('Valid', 'valid_deliverable'), ('Accepted', 'accepted'),
                              ('FinalValid', 'final_valid'), ('FirstValid', 'first_valid'),
                              ('Executed', 'final_executed'), ('Numerical', 'final_numerical'),
                              ('SplitFail', 'final_split_failures'),
                              ('TimingFail', 'final_timing_failures'),
                              ('LibUsers', 'library_users')):
            out[name+metric] = value[field]
            out[name+metric+'Pct'] = pct(value[field], n)
        out[name+'Attempts'] = f"{value['mean_attempts']:.2f}"
        out[name+'CompTokens'] = f"{value['mean_completion_tokens']:,.0f}"
        out[name+'GenSeconds'] = f"{value['mean_generation_seconds']:.1f}"
        out[name+'CtlSeconds'] = f"{value['mean_controller_seconds']:.1f}"
        out[name+'Truncated'] = value['truncated_responses']
    audits = [load(path) for path in sorted((ENFORCED/'audits').glob('*.json'))]
    for condition, label in arm.items():
        group = [r for r in audits if r['arm'] == condition]
        wrong = sum(r['accepted'] and not r['final_valid'] for r in group)
        out['EN' + label + 'AllAcceptedInvalid'] = wrong
        out['EN' + label + 'AllAcceptedInvalidPct'] = pct(wrong, len(group))
    library = []
    for row in receipt['rows']:
        if row['id'].endswith('-library_guards'):
            library.append([load(ENFORCED/a['path']) for a in row['attempts']])
    attempts = [a for episode in library for a in episode]
    out['ENLibEpisodesCalling'] = sum(any(a['controller'].get('library_calls', 0) > 0
                                          for a in episode) for episode in library)
    out['ENLibAttempts'] = len(attempts)
    out['ENLibAttemptsCalling'] = sum(a['controller'].get('library_calls', 0) > 0
                                      for a in attempts)
    out['ENLibAttemptsMentioning'] = sum('fin_skills' in a['code'] for a in attempts)
    out['ENLibAttemptsFailedRun'] = sum(not a['controller'].get('executed') for a in attempts)
    # The prompt's interface text named run()'s second argument `data_dictionary`; the library
    # names it `data`. Count attempts that followed the text, and whether they also re-cumulate.
    keyword = [a for a in attempts if re.search(r'data_dictionary\s*=', a['code'])]
    out['ENKeywordAttempts'] = len(keyword)
    out['ENKeywordEpisodes'] = len({a['id'] for a in keyword})
    out['ENKeywordRecumulated'] = sum(bool(re.search(r'split_factors\s*\.\s*cumprod', a['code']))
                                      for a in keyword)
    out['ENJob'] = fetch['job_id']
    return out


def v2_macros():
    """Complete v2 batch, verified and independently recounted; no candidate execution."""
    root = ENFORCED_V2
    protocol, receipt = load(root/'protocol.json'), load(root/'inference-receipt.json')
    scores = load(root/'scores.json')
    assert sha(root/'inputs.json') == protocol['input_sha256']
    assert receipt['protocol_sha256'] == sha(root/'protocol.json')
    assert scores['inference_receipt_sha256'] == sha(root/'inference-receipt.json')
    planned = {f'{m[0]}-{task}-{seed}-{arm}' for m in protocol['models']
               for task in protocol['tasks'] for seed in protocol['seeds']
               for arm in protocol['arms']}
    assert len(receipt['rows']) == len(planned) == receipt['planned'] == protocol['planned']
    assert {row['id'] for row in receipt['rows']} == planned == set(scores['audits'])
    for row in receipt['rows']:
        for attempt in row['attempts']:
            assert sha(root/attempt['path']) == attempt['sha256'], attempt['path']
    audits = []
    for name, digest in scores['audits'].items():
        path = root/'audits'/(name+'.json')
        assert sha(path) == digest, name
        audit = load(path)
        assert audit['id'] == name
        audits.append(audit)
    out = {'EVPlanned': len(planned), 'EVJob': load(root/'fetch-receipt.json')['job_id']}
    labels = {'raw_exec': 'RawExec', 'raw_guards': 'RawGuards', 'library_guards': 'LibGuards'}
    for arm, label in labels.items():
        group = [a for a in audits if a['arm'] == arm]
        aggregate = scores['aggregate']['all-'+arm]
        assert aggregate['planned'] == len(group)
        for metric, field in (('Valid', 'valid_deliverable'), ('Accepted', 'accepted'),
                              ('FinalValid', 'final_valid')):
            count = sum(a[field] for a in group)
            assert aggregate[field] == count
            out['EV'+label+metric] = count
        out['EV'+label+'N'] = len(group)
        out['EV'+label+'Wrong'] = sum(a['accepted'] and not a['final_valid'] for a in group)
        users = sum(a['library_calls'] > 0 for a in group)
        any_users = 0
        for row in receipt['rows']:
            if row['id'].endswith('-'+arm):
                any_users += any(load(root/a['path'])['controller'].get('library_calls', 0) > 0
                                 for a in row['attempts'])
        assert aggregate['library_users'] == users
        out['EV'+label+'LibUsers'] = users
        out['EV'+label+'LibAnyUsers'] = any_users
    return out


def render():
    lines = ['% GENERATED by scripts/build_codegen_evidence.py from hash-checked evidence.',
             '% Official = pre-registered parser; Post = post-hoc first-fence sensitivity.']
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in macros().items()]
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in enforced_macros().items()]
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in v2_macros().items()]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = render()
    if args.check:
        current = OUTPUT.read_text(encoding='utf-8') if OUTPUT.exists() else ''
        if current != text:
            sys.exit(f'{OUTPUT.relative_to(ROOT)} is stale; '
                     'run scripts/build_codegen_evidence.py')
        print('codegen evidence macros up to date')
        return
    OUTPUT.write_text(text, encoding='utf-8', newline='\n')
    print(f'wrote {OUTPUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
