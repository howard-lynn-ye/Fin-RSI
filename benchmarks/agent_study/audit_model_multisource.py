"""Independent accounting adapter for nested model decisions; manual cases stay unchanged."""
import numpy as np
import pandas as pd

from benchmarks.agent_study import market_data as md, trading_study as old
from benchmarks.agent_study.audit_manual_multisource import replay


def audit(root):
    from benchmarks.agent_study.multisource_model_study import validate_receipt, verify
    p, inputs = verify(root), validate_receipt(root)
    scores = old.load(root / 'scores.json')
    prices = pd.read_csv(root / 'data' / md.HIDDEN, index_col=0)[p['universe']]
    executions = [prices.index[i + 1] for i in inputs['picks']]
    out = {}
    for arm in inputs['arms']:
        targets = {}
        for i, (pick, day) in enumerate(zip(inputs['picks'], executions)):
            record = old.load(root / 'decisions' / arm / f'{i:02d}.json')
            if (record['index'], record['date']) != (pick, str(prices.index[pick])):
                raise ValueError('decision index/date mismatch')
            if record['target'] is not None and set(record['target']) != set(prices.columns):
                raise ValueError('model targets must explicitly cover the whole universe')
            targets[day] = record['target']
        nav, detail = replay(prices.loc[executions[0]:], targets, cost_bps=p['cost_bps'])
        # The model ledger additionally stores capital before the first execution.
        reference = pd.Series(old.load(root / 'nav' / f'{arm}.json')['nav'], dtype=float)
        if reference.index[0] != str(prices.index[inputs['picks'][0]]) or reference.iloc[0] != 1.:
            raise ValueError('initial capital missing')
        reference = reference.iloc[1:]
        if list(reference.index) != list(nav.index):
            raise ValueError('valuation calendar mismatch')
        np.testing.assert_allclose(nav, reference, atol=1e-12, rtol=0)
        np.testing.assert_allclose(detail['return_rate_pct'], scores['paths'][arm]['return_rate_pct'],
                                   atol=1e-10, rtol=0)
        if p.get('interface') == 'v8':
            reported = scores['paths'][arm]
            if reported['initial_capital'] != p['initial_capital'] or reported['capital_currency'] != p['capital_currency']:
                raise ValueError('reported initial capital/currency differs from protocol')
            np.testing.assert_allclose(reported['ending_capital'], p['initial_capital'] * float(nav.iloc[-1]),
                                       atol=1e-6, rtol=0)
            np.testing.assert_allclose(reported['return_rate_pct'],
                100 * (reported['ending_capital'] / reported['initial_capital'] - 1), atol=1e-10, rtol=0)
        out[arm] = dict(return_rate_pct=detail['return_rate_pct'], daily_valuations=len(nav),
                        max_daily_nav_difference=float((nav - reference).abs().max()))
    report = dict(passed=True, results=out,
                  protocol_sha256=old.sha(root / 'protocol.json'),
                  inference_receipt_sha256=old.sha(root / 'inference-receipt.json'),
                  scores_sha256=old.sha(root / 'scores.json'),
                  scope='Independent accounting identity, not independent market evidence')
    old.write(root / 'independent-model-audit.json', report)
    return report
