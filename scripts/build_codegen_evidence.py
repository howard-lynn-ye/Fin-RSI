"""Generate manuscript macros for the code-generation study and the FinQA format reanalysis.

Every number is read from hash-checked evidence; `--check` fails if the TeX file is stale.
Official (pre-registered) and post-hoc sensitivity results are kept in separate macros.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'benchmarks/agent_study/evidence'
OFFICIAL = EVIDENCE/'20260928-codegen-utility'
POSTHOC = EVIDENCE/'20260928-codegen-fence-reanalysis'
FINQA = EVIDENCE/'20260928-finqa-format-reanalysis'
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


def render():
    lines = ['% GENERATED by scripts/build_codegen_evidence.py from hash-checked evidence.',
             '% Official = pre-registered parser; Post = post-hoc first-fence sensitivity.']
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in macros().items()]
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
