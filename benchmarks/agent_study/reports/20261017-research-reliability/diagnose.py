"""Immutable-record diagnostics. Run on a Slurm CPU node, not the login node."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ap = argparse.ArgumentParser()
ap.add_argument('source', type=Path)
ap.add_argument('root', type=Path)
ap.add_argument('output', type=Path)
a = ap.parse_args()
sys.path.insert(0, str(a.source))
from benchmarks.agent_study import multisource_model_study as s
from benchmarks.agent_study import market_data as md
from benchmarks.agent_study.audit_manual_multisource import replay
import numpy as np
import pandas as pd

p = s.verify(a.root)
inputs = s.validate_receipt(a.root)
prices = pd.read_csv(a.root/'data'/md.HIDDEN, index_col=0)[p['universe']]
executions = [prices.index[i+1] for i in inputs['picks']]
period = prices.loc[executions[0]:]
out = dict(protocol_sha256=s.old.sha(a.root/'protocol.json'), attribution={}, baselines={},
           errors={}, limitations=['Retrospective availability assumptions; no verified vintage.',
           'Current skill knowledge; not a strict historical blind test.',
           'Attribution is descriptive, not a causal explanation of treatment effects.'])
private = []
for arm in inputs['arms']:
    targets, errors, attempts = {}, Counter(), []
    for i, execution in enumerate(executions):
        r = s.old.load(a.root/'decisions'/arm/f'{i:02d}.json')
        targets[execution] = r['target']
        for t in r['turns']:
            if t.get('phase') != 'research':
                continue
            if not t['result'].get('ok'):
                message = str(t['result'].get('error', ''))
                errors[(t.get('tool'), message, t.get('parse'))] += 1
                private.append(dict(arm=arm, date=r['date'], turn_index=t['turn_index'],
                                    response=t['response'], result=t['result']))
            try:
                call = json.loads(t['response'])
            except (ValueError, TypeError):
                continue
            if call.get('tool') == 'read_market':
                args = call.get('arguments', {})
                attempts.append(dict(date=r['date'], args=args, ok=t['result'].get('ok'),
                                     as_of=t['result'].get('as_of'), error=t['result'].get('error')))
    nav, detail = replay(period, targets, cost_bps=p['cost_bps'])
    reference = pd.Series(s.old.load(a.root/'nav'/f'{arm}.json')['nav']).iloc[1:]
    np.testing.assert_allclose(nav, reference, atol=1e-12, rtol=0)
    out['attribution'][arm] = detail
    out['errors'][arm] = [dict(tool=k[0], error=k[1], parse=k[2], count=v) for k,v in errors.most_common()]
    out.setdefault('market_requests', {})[arm] = attempts
for name, target in [('equal_weight_buy_hold', dict.fromkeys(prices.columns, 1/len(prices.columns)))]+[
        (f'{t}_buy_hold', {k: float(k==t) for k in prices.columns}) for t in prices.columns]:
    _, detail = replay(period, {executions[0]: target}, cost_bps=p['cost_bps'])
    out['baselines'][name] = detail['return_rate_pct']
q = pd.read_csv(a.root/'data'/'quotes.csv')
ca = pd.read_csv(a.root/'data'/'corporate_actions.csv')
adjusted = md.adjust(q, ca, through=str(prices.index[-1]))[prices.columns]
out['adjusted_vs_hidden_max_daily_return_difference'] = float((adjusted.pct_change()-prices.pct_change()).abs().max().max())
out['period'] = [str(period.index[0]), str(period.index[-1])]
out['decision_count_per_arm'] = len(executions)
a.output.mkdir(exist_ok=False, parents=True)
(a.output/'diagnostic.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
(a.output/'private-failed-turns.json').write_text(json.dumps(private, indent=2, allow_nan=False)+'\n')
print(json.dumps(out, indent=2, allow_nan=False))
