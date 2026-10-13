"""Unscored model access probes, separate from autonomous investment decisions.

These demonstrate one common reader and the library RAG entry point, not every
library function, strategy competence, or spontaneous use during the return study.
"""
from pathlib import Path

from benchmarks.agent_study import trading_runtime_v8 as v8, trading_runtime_v9 as v9


CASES = (('raw', 'read_snapshot'), ('library', 'research_context'))
REVISION = '20261013-real-library-access-v2'


def contract():
    return dict(revision=REVISION, cases=[list(row) for row in CASES],
                max_responses_per_case=v8.old.MAX_TURNS,
                scored=False, required_during_investment=False,
                success_criterion='requested reader executed and delivered; termination is recorded separately')


def prompt(arm, turn_index=0):
    base = v9.system_prompt(arm, turn_index)
    if turn_index >= v8.RESEARCH_RESPONSES:
        return base + ('\nUNSCORED INTERFACE CHECK. Finish with one submit or hold; no trade is executed. '
                       'The recorded research attempts determine whether this check passed.')
    name = dict(CASES)[arm]
    task = ('Read the latest visible market snapshot using read_snapshot.' if arm == 'raw' else
            'Use research_context to retrieve financial knowledge and visible evidence. '
            'Choose a relevant query yourself; use the study shortcut, not the package wrapper.')
    return base + '\nUNSCORED INTERFACE CHECK. ' + task + (
        f' Obtain one successful {name} result, then finish with one submit or hold. '
        'This checks tool access only: no order is executed and there is no Return Rate. '
        'Do not invent a result. Use the displayed signatures; final actions are unscored.')


def task(root, protocol):
    from benchmarks.agent_study import multisource_model_study as study
    day = study.old.load(root / 'inputs.json')['dates'][0]
    packet = study.old.load(root / 'evidence-packets.json')[day]
    return day, (v9.task(day, dict.fromkeys(study.md.TICKERS, 0.), protocol['deadline']) +
                 study.evidence_text(packet))


def inspect(record, arm):
    """A submit/hold or a discovery listing alone cannot qualify tool access."""
    v8.validate_decision(record)
    expected = dict(CASES)[arm]
    delivered = [t for t in record['turns'] if t.get('tool') == expected and
                 v8.execution_status(t['result'])['turn_ok']]
    receipts = [r for r in record['tool_calls'] if r.get('tool') == expected and r.get('ok')]
    return dict(arm=arm, required_tool=expected, successful_deliveries=len(delivered),
                successful_receipts=len(receipts), decision_action=record['decision_action'],
                termination_completed=record['decision_completed'],
                passed=bool(delivered and receipts))


def run(root, backend):
    from benchmarks.agent_study import multisource_model_study as study
    root = Path(root)
    p = study.verify(root)
    if p.get('model_access_probe') != contract():
        raise ValueError('model access probe was not declared before inference')
    v9.require_qualification(root)
    (root / 'tmp').mkdir(exist_ok=True)
    day, request = task(root, p)
    rows = []
    for arm, _ in CASES:
        path = root / 'model-access' / f'{arm}.json'
        if not path.exists():
            # Actual decisions independently reset seed/calls and receive none
            # of these messages, responses, or results.
            backend.seed, backend.calls = p['seed'] * 1000, 0
            backend.request_scope = f'access/{arm}'
            with study.decision_workspace(root, arm, 0, personal=False) as workspace:
                study.md.truncate(root / 'data', workspace, day)
                snapshot = v9.write_evidence_snapshot(study.old.load(root / 'evidence.json'),
                    study.old.load(root / 'evidence-packets.json')[day], workspace)
                controller = v9.Controller(root, workspace, arm, p['seed'] * 1000)
                try:
                    record = v8.decide(backend, controller, request, prompt_factory=prompt,
                                       interface='v9', final_output=p['final_output'])
                except v8.schema.DecoderFailure as exc:
                    v8.schema.persist_failure(root, f'access-{arm}', exc)
                    raise
            record.update(arm=arm, date=day, evidence_snapshot_sha256=snapshot,
                          protocol_sha256=study.old.sha(root / 'protocol.json'))
            study.old.write(path, record)
        record = study.old.load(path)
        rows.append(dict(record=path.relative_to(root).as_posix(), sha256=study.old.sha(path),
                         **inspect(record, arm)))
    report = dict(contract=contract(), protocol_sha256=study.old.sha(root / 'protocol.json'),
                  model=p['model'], cases=rows, passed=all(r['passed'] for r in rows))
    study.old.write(root / 'model-access-preflight.json', report)
    return require(root)


def require(root):
    from benchmarks.agent_study import multisource_model_study as study
    import hashlib
    root = Path(root)
    p = study.verify(root)
    report = study.old.load(root / 'model-access-preflight.json')
    if (p.get('model_access_probe') != contract() or report.get('contract') != contract() or
            report['protocol_sha256'] != study.old.sha(root / 'protocol.json') or
            report['model'] != p['model'] or
            [r['arm'] for r in report['cases']] != [a for a, _ in CASES]):
        raise ValueError('model access qualification absent, stale, or incomplete')
    day, request = task(root, p)
    snapshot = v9.evidence_snapshot(study.old.load(root / 'evidence.json'),
                                   study.old.load(root / 'evidence-packets.json')[day])
    snapshot_sha = hashlib.sha256(v9.encoded(snapshot).encode()).hexdigest()
    for row in report['cases']:
        arm = row['arm']
        path = root / 'model-access' / f'{arm}.json'
        if row['record'] != path.relative_to(root).as_posix() or row['sha256'] != study.old.sha(path):
            raise ValueError('model access original record changed')
        record = study.old.load(path)
        v8.validate_decision(record, expected_final_output=p['final_output'])
        v8.schema.check_backend(record, p)
        if p.get('batch') == 'matrix_native':
            from benchmarks.agent_study.native_chat import validate_record
            validate_record(root, record, p, f'access/{arm}')
        if ((record.get('date'), record.get('arm'), record.get('protocol_sha256')) !=
                (day, arm, report['protocol_sha256']) or
                record.get('evidence_snapshot_sha256') != snapshot_sha or
                record['initial_request'] != [dict(role='system', content=prompt(arm)),
                                              dict(role='user', content=request)]):
            raise ValueError('model access inputs differ from frozen first-date probe')
        observed = inspect(record, arm)
        if any(row.get(k) != v for k, v in observed.items()):
            raise ValueError('model access observations differ from original responses')
        if not observed['passed']:
            raise ValueError('model did not demonstrate tool access; preserve failed probe')
    if not report['passed']:
        raise ValueError('model access gate failed')
    return report
