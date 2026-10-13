"""Real model interface checks before a new full-period pair may start.

Each date is an independent all-cash scenario, not a simulated return path.
Failures and original responses remain available; no model is silently dropped.
"""
from pathlib import Path

from benchmarks.agent_study import market_data as md, trading_study as old
from benchmarks.agent_study.trading_schema_v8 import (
    DecoderFailure, persist_failure, check_backend, check_turn, TRANSPORT,
)


def cases(picks):
    """Fixed coverage rule independent of prices, model answers or future returns."""
    if not picks:
        raise ValueError('preflight requires scheduled decisions')
    indices = sorted({0, len(picks) // 2, len(picks) - 1})
    return [(arm, i) for i in indices for arm in ('raw', 'library')]


def observations(record):
    """Separate valid decisions, research delivery and actual library receipts.

    Discovery and a successful submission are not evidence of library computation.
    A receipt proves execution only, never that the agent understood or used it.
    """
    from benchmarks.agent_study.trading_tools_v5 import execution_status

    research = [t for t in record['turns'] if t['phase'] == 'research'
                and t.get('tool') not in ('submit', 'hold')]
    successful = sum(execution_status(t['result'])['turn_ok'] for t in research)
    substantive = {'read_skill', 'read_history', 'load_history', 'run_algorithm',
                   'run_guard', 'research_context', 'library_execution'}
    receipts = [r for r in record['tool_calls'] if r.get('ok') and r.get('tool') in substantive]
    return dict(decision_action=record['decision_action'],
                decision_completed=record['decision_completed'],
                research_attempts=len(research), research_successes=successful,
                research_errors=len(research) - successful,
                research_status=('not_attempted' if not research else
                                 'all_attempts_failed' if not successful else
                                 'successful_delivery_observed'),
                library_execution_observed=bool(receipts),
                library_receipt_tools=sorted({r['tool'] for r in receipts}),
                library_package_tools=sorted({r['library_tool'] for r in receipts
                                              if r.get('library_tool')}),
                structured_final_responses=sum(bool((t.get('backend_metadata') or {}).get('final_output'))
                                               for t in record['turns']),
                parse_failures=sum(t.get('tool') is None for t in record['turns']),
                truncated_responses=sum(t.get('finish_reason') == 'length' for t in record['turns']),
                library_execution_receipts=[{k: r[k] for k in
                    ('tool', 'library_tool', 'result_sha256', 'arguments_sha256', 'page_sha256') if k in r}
                    for r in receipts],
                failed_response_count=sum(not execution_status(t['result'])['turn_ok']
                                          for t in record['turns']))


def interpretation(rows):
    return dict(
        gate='decision_completion_only',
        research_cases_needing_review=[f"{r['arm']}-{r['decision_index']:02d}" for r in rows
                                      if r['research_status'] == 'all_attempts_failed'],
        library_execution_cases=sum(r['arm'] == 'library' and r['library_execution_observed'] for r in rows),
        library_cases=sum(r['arm'] == 'library' for r in rows),
        library_use_required=False,
        claim='Library access is assigned; usage is autonomous. Receipt evidence proves execution, '
              'not interpretation, profitability, or full-capability coverage.',
        information_mode='retrospective_availability_assumptions',
        historical_availability_verified=False,
        library_reference_vintage='current_installed_knowledge')


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
    (root / 'tmp').mkdir(exist_ok=True)
    planned = cases(inputs['picks'])
    decoder_check = qualify_decoder(root, backend, protocol)
    if protocol.get('model_access_probe') is not None:
        from benchmarks.agent_study import trading_access_v9 as access
        if not (root / 'model-access-preflight.json').exists():
            access.run(root, backend)
        access.require(root)
    rows = []
    for arm, i in planned:
        path = destination / f'{arm}-{i:02d}.json'
        day = str(dates[inputs['picks'][i]])
        if not path.exists():
            backend.seed, backend.calls = protocol['seed'] * 1000 + i, 0
            backend.request_scope = f'preflight/{arm}/{i}'
            with study.decision_workspace(root, arm, i, personal=False) as workspace:
                md.truncate(root / 'data', workspace, day)
                snapshot = runtime.write_evidence_snapshot(old.load(root / 'evidence.json'), packets[day], workspace)
                controller = runtime.Controller(root, workspace, arm, protocol['seed'] * 1000 + i)
                holdings = dict.fromkeys(md.TICKERS, 0.)
                try:
                    record = runtime.decide(backend, controller,
                        runtime.task(day, holdings, protocol['deadline']) + study.evidence_text(packets[day]),
                        final_output=protocol['final_output'])
                except DecoderFailure as exc:
                    persist_failure(root, f'preflight-{arm}-{i:02d}', exc)
                    raise
                record.update(date=day, arm=arm, decision_index=i,
                              evidence_snapshot_sha256=snapshot,
                              protocol_sha256=old.sha(root / 'protocol.json'))
                old.write(path, record)
        record = old.load(path)
        runtime.validate_decision(record, expected_final_output=protocol['final_output'])
        check_backend(record, protocol)
        if protocol.get('batch') == 'matrix_native':
            from benchmarks.agent_study.native_chat import validate_record
            validate_record(root, record, protocol, f'preflight/{arm}/{i}')
        if (record.get('protocol_sha256') != old.sha(root / 'protocol.json') or
                (record.get('date'), record.get('arm'), record.get('decision_index')) != (day, arm, i)):
            raise ValueError('preflight record belongs to a different protocol or case')
        rows.append(dict(arm=arm, decision_index=i, date=day,
                         record=path.relative_to(root).as_posix(), sha256=old.sha(path),
                         **observations(record)))
    report = dict(protocol_sha256=old.sha(root / 'protocol.json'), model=protocol['model'],
                  passed=decoder_check['passed'] and all(row['decision_completed'] for row in rows),
                  decoder_check=decoder_check, cases=rows,
                  interpretation=interpretation(rows),
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
    if protocol.get('model_access_probe') is not None:
        from benchmarks.agent_study import trading_access_v9 as access
        access.require(root)
    if report.get('decoder_check') != require_decoder(root, protocol):
        raise ValueError('real decoder qualification absent, failed or stale')
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
        runtime.validate_decision(record, expected_final_output=protocol['final_output'])
        check_backend(record, protocol)
        if protocol.get('batch') == 'matrix_native':
            from benchmarks.agent_study.native_chat import validate_record
            validate_record(root, record, protocol, f"preflight/{row['arm']}/{row['decision_index']}")
        if (record.get('protocol_sha256') != report['protocol_sha256'] or
                (record.get('date'), record.get('arm'), record.get('decision_index')) !=
                (row['date'], row['arm'], row['decision_index'])):
            raise ValueError('preflight record belongs to a different protocol or case')
        if any(row.get(key) != value for key, value in observations(record).items()):
            raise ValueError('preflight observations differ from original records')
        if (not record['decision_completed'] or row['decision_action'] != record['decision_action'] or
                row['decision_completed'] != record['decision_completed']):
            raise ValueError('model did not qualify: preserve failed preflight; full-period run not started')
    if report.get('interpretation') != interpretation(report['cases']):
        raise ValueError('preflight interpretation differs from original observations')
    if not report['passed']:
        raise ValueError('model did not qualify: full-period run not started')
    return report


def qualify_decoder(root, backend, protocol):
    """One unscored, stateless final response ensures early submit cannot skip this route."""
    if protocol['final_output']['transport'] != TRANSPORT:
        return dict(passed=True, scope='declared text transport; no decoder qualification')
    path = root / 'decoder-preflight.json'
    if not path.exists():
        messages = decoder_messages(root, protocol)
        # The actual cases reset seed/calls again. This response/history is never
        # passed to a case or a return ledger and does not select a strategy for it.
        backend.seed, backend.calls = protocol['seed'] * 1000, 0
        try:
            response = backend.final_decision(messages)
        except DecoderFailure as exc:
            persist_failure(root, 'decoder-preflight', exc)
            raise
        old.write(path, dict(protocol_sha256=old.sha(root / 'protocol.json'),
                             messages=messages, response=response))
    return require_decoder(root, protocol)


def require_decoder(root, protocol):
    if protocol['final_output']['transport'] != TRANSPORT:
        return dict(passed=True, scope='declared text transport; no decoder qualification')
    path = root / 'decoder-preflight.json'
    artifact = old.load(path)
    if artifact['protocol_sha256'] != old.sha(root / 'protocol.json'):
        raise ValueError('decoder preflight protocol changed')
    if artifact['messages'] != decoder_messages(root, protocol):
        raise ValueError('decoder qualification prompt differs from frozen first-date task')
    response = artifact['response']
    choice = response['choices'][0]
    turn = dict(response=choice['message']['content'], finish_reason=choice['finish_reason'],
                backend_metadata=response.get('backend_metadata'))
    check_backend({'turns': [turn]}, protocol)
    if choice['finish_reason'] == 'length' or check_turn(turn, protocol['final_output']):
        raise ValueError('real decoder did not produce strict complete JSON; preserve failed qualification')
    return dict(passed=True, record=path.name, sha256=old.sha(path),
                scope='One extra real final response, raw first-date task; format/backend check only, '
                      'no Return Rate, strategy selection or equal-inference-budget claim with old protocols.')



def decoder_messages(root, protocol):
    from benchmarks.agent_study import multisource_model_study as study
    import pandas as pd
    runtime = study.terminal_runtime(protocol['interface'])
    inputs, packets = old.load(root / 'inputs.json'), old.load(root / 'evidence-packets.json')
    dates = pd.read_csv(root / 'data' / md.HIDDEN, index_col=0).index
    day = str(dates[inputs['picks'][0]])
    return [dict(role='system', content=runtime.system_prompt('raw', 6)),
            dict(role='user', content=runtime.task(day, dict.fromkeys(md.TICKERS, 0.),
                protocol['deadline']) + study.evidence_text(packets[day]))]
