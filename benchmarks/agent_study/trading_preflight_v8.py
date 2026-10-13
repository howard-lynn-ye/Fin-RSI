"""Real model interface checks before a new full-period pair may start.

Each date is an independent all-cash scenario, not a simulated return path.
Failures and original responses remain available; no model is silently dropped.
"""
from pathlib import Path

from benchmarks.agent_study import market_data as md, trading_study as old


def cases(picks):
    """Fixed coverage rule independent of prices, model answers or future returns."""
    if not picks:
        raise ValueError('preflight requires scheduled decisions')
    indices = sorted({0, len(picks) // 2, len(picks) - 1})
    return [(arm, i) for i in indices for arm in ('raw', 'library')]


def run(root, backend):
    from benchmarks.agent_study import multisource_model_study as study
    import pandas as pd

    root = Path(root)
    protocol = study.verify(root)
    runtime = study.terminal_runtime(protocol['interface'])
    runtime.require_qualification(root)
    inputs, packets = old.load(root / 'inputs.json'), old.load(root / 'evidence-packets.json')
    dates = pd.read_csv(root / 'data' / md.HIDDEN, index_col=0).index
    destination = root / 'interface-preflight'
    destination.mkdir(exist_ok=True)
    planned = cases(inputs['picks'])
    rows = []
    for arm, i in planned:
        path = destination / f'{arm}-{i:02d}.json'
        day = str(dates[inputs['picks'][i]])
        if not path.exists():
            backend.seed, backend.calls = protocol['seed'] * 1000 + i, 0
            with study.decision_workspace(root, arm, i) as workspace:
                md.truncate(root / 'data', workspace, day)
                snapshot = runtime.write_evidence_snapshot(old.load(root / 'evidence.json'), packets[day], workspace)
                controller = runtime.Controller(root, workspace, arm, protocol['seed'] * 1000 + i)
                holdings = dict.fromkeys(md.TICKERS, 0.)
                record = runtime.decide(backend, controller,
                    runtime.task(day, holdings, protocol['deadline']) + study.evidence_text(packets[day]))
                record.update(date=day, arm=arm, decision_index=i,
                              evidence_snapshot_sha256=snapshot,
                              protocol_sha256=old.sha(root / 'protocol.json'))
                old.write(path, record)
        record = old.load(path)
        runtime.validate_decision(record)
        if (record.get('protocol_sha256') != old.sha(root / 'protocol.json') or
                (record.get('date'), record.get('arm'), record.get('decision_index')) != (day, arm, i)):
            raise ValueError('preflight record belongs to a different protocol or case')
        rows.append(dict(arm=arm, decision_index=i, date=day,
                         record=path.relative_to(root).as_posix(), sha256=old.sha(path),
                         decision_action=record['decision_action'],
                         decision_completed=record['decision_completed'],
                         research_errors=sum(not t['result'].get('ok') for t in record['turns']
                                             if t['phase'] == 'research')))
    report = dict(protocol_sha256=old.sha(root / 'protocol.json'), model=protocol['model'],
                  passed=all(row['decision_completed'] for row in rows), cases=rows,
                  scope='First, middle and last scheduled dates, both arms, independent all-cash accounts; '
                        'interface qualification only, no Return Rate and no guarantee about other dates.')
    old.write(root / 'decision-preflight.json', report)
    require(root)
    return report


def require(root):
    from benchmarks.agent_study import multisource_model_study as study

    root = Path(root)
    protocol = study.verify(root)
    runtime = study.terminal_runtime(protocol['interface'])
    report = old.load(root / 'decision-preflight.json')
    expected = cases(old.load(root / 'inputs.json')['picks'])
    actual = [(row['arm'], row['decision_index']) for row in report['cases']]
    if (report['protocol_sha256'] != old.sha(root / 'protocol.json') or
            report['model'] != protocol['model'] or actual != expected):
        raise ValueError('model decision preflight is stale or incomplete')
    for row in report['cases']:
        path = root / 'interface-preflight' / f"{row['arm']}-{row['decision_index']:02d}.json"
        if row['record'] != path.relative_to(root).as_posix() or row['sha256'] != old.sha(path):
            raise ValueError('model preflight record changed')
        record = old.load(path)
        runtime.validate_decision(record)
        if (record.get('protocol_sha256') != report['protocol_sha256'] or
                (record.get('date'), record.get('arm'), record.get('decision_index')) !=
                (row['date'], row['arm'], row['decision_index'])):
            raise ValueError('preflight record belongs to a different protocol or case')
        if (not record['decision_completed'] or row['decision_action'] != record['decision_action'] or
                row['decision_completed'] != record['decision_completed']):
            raise ValueError('model did not qualify: preserve failed preflight; full-period run not started')
    if not report['passed']:
        raise ValueError('model did not qualify: full-period run not started')
    return report
