"""Fixed-default portfolio algorithms via the library arm run_algorithm on each dev row (CPU).

    python benchmarks/agent_study/fin_rsi_algo_baselines.py <devset rows.json> <out.json>
"""
import json, shutil, sys, tempfile
from pathlib import Path
from benchmarks.agent_study import fin_rsi_run as runner, fin_rsi_score as score
rows_path = sys.argv[1]
b = json.loads(Path(rows_path).read_text())
total = score.load_total(b['devset']['total_return_close'])
units = sorted(b['units'], key=lambda u: u['PIT_and_input']['decision_index'])
picks = [u['PIT_and_input']['decision_index'] for u in units]
algos = ['equal_weight', 'inverse_volatility', 'min_variance', 'hrp', 'cross_sectional_momentum']
out = {}
for a in algos:
    periods, fails, weights = [], 0, {}
    for u in units:
        P = u['PIT_and_input']
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td) / 'visible'; shutil.copytree(P['visible_workspace'], ws)
            c = runner.Controller(Path(td), ws, 'library', menu_seed=0)
            r = c.call('run_algorithm', dict(algorithm_id=a, lookback=252))
        w = (r.get('result') or {}).get('weights') if r.get('ok') else None
        if w is None: fails += 1
        weights[u['unit_id']] = w
        periods.append(score.period(total, P['decision_index'], P['period_end_index'], P['holdings_before'], w))
    s = score.summarize(score.chain(total, picks, periods), periods)
    out[a] = dict(s, failed_calls=fails)
    Path(sys.argv[2]).with_name(Path(sys.argv[2]).stem + f'-{a}-weights.json').write_text(json.dumps(weights, indent=1))
    print(f"| algo:{a} | {s['net_return_pct']:.3f} | {s['sharpe']:.3f} | {s['max_drawdown_pct']:.3f} | {s['turnover']:.3f} | {s['submitted']}/22 | fails {fails} |", flush=True)
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
