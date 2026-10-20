"""Fin-RSI financial scorer: holding-period net return per decision and chained NAV.

Accounting is the frozen trading_study ledger (v3/v5/factorial): weights execute at the close
of session pick+1 after that session's return accrues, COST_BPS per side on traded notional,
cash earns zero, a missing submission keeps the drifted holdings at no cost.

A decision's holding period is sessions pick+1 .. next_pick (inclusive; the last decision runs
to the final session). Session pick+1 accrues on holdings_before, so chaining periods from the
path holdings reproduces the path NAV exactly.

    # reproduce a stored 44-decision path (e.g. 7B seed11 factorial)
    python -m benchmarks.agent_study.fin_rsi_score path --total <total_return_close.csv> \
        --decisions <run>/decisions/base [--decisions <run>/decisions/full ...]
    # score fin_rsi_run outputs on the fixed dev set
    python -m benchmarks.agent_study.fin_rsi_score dev --rows <devset rows.json> \
        --run raw=<run_dir> --run library=<run_dir> [--out score.json]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

COST_BPS = 5.0


def load_total(path):
    return pd.read_csv(path, index_col=0)


def _drift(weights, returns, tickers):
    grown = {t: weights.get(t, 0.0) * (1 + returns[t]) for t in tickers}
    cash = 1 - sum(weights.get(t, 0.0) for t in tickers)
    nav = cash + sum(grown.values())
    return {t: grown[t] / nav for t in tickers}, nav


def period(total, pick, end, holdings_before, target, cost_bps=COST_BPS):
    """Daily NAV (start 1.0 at pick) for sessions pick+1..end with one trade at pick+1."""
    tickers = list(total.columns)
    returns = total.pct_change().fillna(0.0)
    holdings = {t: float(holdings_before.get(t, 0.0)) for t in tickers}
    nav, navs, turnover, cost = 1.0, {}, 0.0, 0.0
    for k in range(pick + 1, end + 1):
        holdings, growth = _drift(holdings, returns.iloc[k], tickers)
        nav *= growth
        if k == pick + 1 and target is not None:
            turnover = sum(abs(target.get(t, 0.0) - holdings[t]) for t in tickers)
            cost = turnover * cost_bps / 1e4
            nav *= 1 - cost
            holdings = {t: float(target.get(t, 0.0)) for t in tickers}
        navs[total.index[k]] = nav
    return dict(navs=navs, net_return=nav - 1, turnover=turnover, cost=cost,
                end_holdings=holdings, traded=target is not None)


def metrics(nav):
    """Identical to trading_study.metrics; nav includes capital before the first trade."""
    daily = nav.pct_change().dropna()
    years = len(daily) / 252
    total = nav.iloc[-1] / nav.iloc[0] - 1
    std = daily.std(ddof=1)
    return dict(cumulative_return=float(total),
                annualized_return=float((1 + total) ** (1 / years) - 1) if years > 0 else float('nan'),
                annualized_volatility=float(std * np.sqrt(252)) if len(daily) > 1 else float('nan'),
                sharpe=float(daily.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
                max_drawdown=float((nav / nav.cummax() - 1).min()), sessions=int(len(nav)))


def chain(total, picks, periods):
    """Concatenate per-decision daily NAVs into one series starting at 1.0 on picks[0]."""
    level, series = 1.0, {total.index[picks[0]]: 1.0}
    for p in periods:
        for day, value in p['navs'].items():
            series[day] = level * value
        level *= 1 + p['net_return']
    return pd.Series(series)


def path(total, picks, targets, cost_bps=COST_BPS, start=None):
    """Price a sequential path (holdings evolve); returns nav, per-decision periods."""
    holdings = dict(start or dict.fromkeys(total.columns, 0.0))
    periods = []
    for i, pick in enumerate(picks):
        end = picks[i + 1] if i + 1 < len(picks) else len(total) - 1
        p = period(total, pick, end, holdings, targets[i], cost_bps)
        holdings = p['end_holdings']
        periods.append(p)
    return chain(total, picks, periods), periods


def summarize(nav, periods):
    m = metrics(nav)
    return dict(net_return_pct=100 * m['cumulative_return'], sharpe=m['sharpe'],
                max_drawdown_pct=100 * m['max_drawdown'],
                annualized_volatility_pct=100 * m['annualized_volatility'],
                turnover=sum(p['turnover'] for p in periods),
                submitted=sum(p['traded'] for p in periods), decisions=len(periods),
                sessions=m['sessions'])


# ---------------------------------------------------------------- stored 44-decision paths

def score_decision_dir(total, directory, cost_bps=COST_BPS):
    records = [json.loads(f.read_text()) for f in sorted(Path(directory).glob('[0-9][0-9].json'))]
    picks = [r['index'] for r in records]
    nav, periods = path(total, picks, [r.get('target') for r in records], cost_bps)
    # sanity: the model saw the same holdings this ledger implies
    holdings = dict.fromkeys(total.columns, 0.0)
    worst = 0.0
    for r, p in zip(records, [None] + periods[:-1]):
        if p is not None:
            holdings = p['end_holdings']
        worst = max(worst, max(abs(r['holdings_before'][t] - holdings[t]) for t in total.columns))
    out = summarize(nav, periods)
    out['holdings_before_max_abs_diff'] = worst
    out['cost_sensitivity_net_return_pct'] = {
        str(c): 100 * metrics(path(total, picks, [r.get('target') for r in records], c)[0])['cumulative_return']
        for c in (0, 10, 25)}
    return out


# ---------------------------------------------------------------- fixed dev set

def baseline_targets(tickers):
    ew = {t: 1 / len(tickers) for t in tickers}
    return dict(hold=lambda u: None, equal_weight=lambda u: ew)


def dev_score(rows_path, runs, cost_bps=COST_BPS, fixed=None):
    binding = json.loads(Path(rows_path).read_text())
    devset = binding['devset']
    total = load_total(devset['total_return_close'])
    units = sorted(binding['units'], key=lambda u: u['PIT_and_input']['decision_index'])
    picks = [u['PIT_and_input']['decision_index'] for u in units]

    def score_targets(targets):
        periods = [period(total, u['PIT_and_input']['decision_index'],
                          u['PIT_and_input']['period_end_index'],
                          u['PIT_and_input']['holdings_before'], t, cost_bps)
                   for u, t in zip(units, targets)]
        out = summarize(chain(total, picks, periods), periods)
        out['per_unit'] = [dict(unit_id=u['unit_id'], date=u['PIT_and_input']['date'],
                                submitted=p['traded'], net_return_pct=100 * p['net_return'],
                                turnover=p['turnover']) for u, p in zip(units, periods)]
        return out

    result = dict(rows=str(rows_path), cost_bps=cost_bps, units=len(units),
                  baselines={name: score_targets([f(u) for u in units])
                             for name, f in baseline_targets(list(total.columns)).items()},
                  runs={})
    for name, path_ in (fixed or {}).items():  # deterministic per-unit weights, e.g. always min_variance
        weights = json.loads(Path(path_).read_text())
        result['baselines'][name] = score_targets([weights.get(u['unit_id']) for u in units])
    hold = {d['unit_id']: d['net_return_pct'] for d in result['baselines']['hold']['per_unit']}
    for label, run_dirs in runs.items():
        by_seed = {}
        for f in sorted(f for d in run_dirs for f in Path(d).glob('rows/*/decision.json')):
            d = json.loads(f.read_text())
            seen = by_seed.setdefault(d['metrics']['seed'], {})
            if d['metrics']['unit_id'] in seen:
                raise ValueError(f"duplicate {d['metrics']['unit_id']} seed {d['metrics']['seed']} in {label}")
            seen[d['metrics']['unit_id']] = d
        for seed, decisions in sorted(by_seed.items()):
            missing = [u['unit_id'] for u in units if u['unit_id'] not in decisions]
            targets = [decisions[u['unit_id']]['target'] if u['unit_id'] in decisions else None
                       for u in units]
            s = score_targets(targets)
            s['missing_rows'] = missing
            s['mean_excess_vs_hold_pp'] = float(np.mean(
                [p['net_return_pct'] - hold[p['unit_id']] for p in s['per_unit']]))
            m = [decisions[u]['metrics'] for u in decisions]
            s['process'] = dict(
                rows=len(m), submitted=sum(x['submitted'] for x in m),
                run_algorithm_ok=sum(x['run_algorithm_ok'] for x in m),
                skill_reads_ok=sum(x['skill_reads_ok'] for x in m),
                algorithm_then_submit=sum(bool(x['algorithm_then_submit']) for x in m),
                executed_turns=sum(x['executed_turns'] for x in m),
                turns=sum(x['turns'] for x in m),
                chain_algorithm_ids=dict(Counter(a for x in m if x['algorithm_then_submit']
                                                 for a in x['algorithm_ids'])),
                run_algorithm_ids=dict(Counter(a for x in m for a in x['algorithm_ids'])))
            result['runs'][f'{label}/s{seed}'] = s
        seeds = [result['runs'][f'{label}/s{seed}'] for seed in sorted(by_seed)]
        if len(seeds) > 1:
            mean = {k: float(np.mean([x[k] for x in seeds])) for k in
                    ('net_return_pct', 'sharpe', 'max_drawdown_pct', 'turnover', 'submitted',
                     'mean_excess_vs_hold_pp')}
            mean.update(decisions=seeds[0]['decisions'], seeds=sorted(by_seed),
                        process={k: float(np.mean([x['process'][k] for x in seeds])) for k in
                                 ('run_algorithm_ok', 'algorithm_then_submit', 'skill_reads_ok')})
            result['runs'][f'{label}/mean'] = mean
    return result


def _rows(result, name):
    """{(unit_id, seed): per-row net return pp}; baselines have seed None."""
    if name in result['baselines']:
        return {(u['unit_id'], None): u['net_return_pct'] for u in result['baselines'][name]['per_unit']}
    out = {}
    for key, s in result['runs'].items():
        label, _, tag = key.rpartition('/')
        if label == name and tag.startswith('s'):
            out.update({(u['unit_id'], int(tag[1:])): u['net_return_pct'] for u in s['per_unit']})
    if not out:
        raise KeyError(name)
    return out


def paired_ci(result, a, b, boot=10000, seed=0):
    """Mean per-row difference a - b (pp) paired by date x seed, with a date-clustered
    bootstrap 95% CI (seeds of one date are not independent) and the naive row-level t-CI."""
    ra, rb = _rows(result, a), _rows(result, b)
    keys = [(u, sd) for (u, sd) in ra if sd is not None] or [(u, sd) for (u, sd) in rb if sd is not None]
    diffs = {}
    for u, sd in keys:
        va = ra.get((u, sd), ra.get((u, None)))
        vb = rb.get((u, sd), rb.get((u, None)))
        if va is not None and vb is not None:
            diffs.setdefault(u, []).append(va - vb)
    dates = sorted(diffs)
    flat = np.array([x for u in dates for x in diffs[u]])
    rng = np.random.default_rng(seed)
    sums = np.array([sum(diffs[u]) for u in dates])
    counts = np.array([len(diffs[u]) for u in dates])
    idx = rng.integers(0, len(dates), (boot, len(dates)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    se = flat.std(ddof=1) / np.sqrt(len(flat)) if len(flat) > 1 else float('nan')
    return dict(a=a, b=b, pairs=int(len(flat)), dates=len(dates), mean_pp=float(flat.mean()),
                cluster_boot_ci95=[float(lo), float(hi)],
                naive_t_ci95=[float(flat.mean() - 1.96 * se), float(flat.mean() + 1.96 * se)])


def pair_table(rows):
    lines = ['| a − b | pairs (dates) | mean pp/row | cluster-boot 95% CI (lower bound decides) | naive row CI |',
             '|---|---:|---:|---|---|']
    for r in rows:
        lines.append(f"| {r['a']} − {r['b']} | {r['pairs']} ({r['dates']}) | {r['mean_pp']:+.3f} | "
                     f"[{r['cluster_boot_ci95'][0]:+.3f}, {r['cluster_boot_ci95'][1]:+.3f}] | "
                     f"[{r['naive_t_ci95'][0]:+.3f}, {r['naive_t_ci95'][1]:+.3f}] |")
    return '\n'.join(lines)


def table(result):
    lines = ['| path | net return % | Sharpe | max DD % | turnover | submitted | excess vs hold pp/row '
             '| run_algorithm ok | algo→submit rows | skill reads ok |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, s in list(result['baselines'].items()) + list(result['runs'].items()):
        pr = s.get('process', {})
        lines.append(f"| {name} | {s['net_return_pct']:.3f} | {s['sharpe']:.3f} | "
                     f"{s['max_drawdown_pct']:.3f} | {s['turnover']:.3f} | "
                     f"{s['submitted']:g}/{s['decisions']} | {s.get('mean_excess_vs_hold_pp', float('nan')):.3f} | "
                     f"{pr.get('run_algorithm_ok', '-')} | {pr.get('algorithm_then_submit', '-')} | "
                     f"{pr.get('skill_reads_ok', '-')} |")
    ids = {name: s['process'].get('chain_algorithm_ids') for name, s in result['runs'].items()
           if s.get('process', {}).get('chain_algorithm_ids')}
    if ids:
        lines += ['', 'algorithm IDs on algorithm→submit rows: ' + json.dumps(ids)]
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='mode', required=True)
    a = sub.add_parser('path')
    a.add_argument('--total', required=True)
    a.add_argument('--decisions', action='append', required=True)
    a.add_argument('--cost-bps', type=float, default=COST_BPS)
    b = sub.add_parser('dev')
    b.add_argument('--rows', required=True)
    b.add_argument('--run', action='append', default=[], help='label=run_dir (repeatable; same label merges dirs)')
    b.add_argument('--cost-bps', type=float, default=COST_BPS)
    b.add_argument('--fixed', action='append', default=[],
                   help='name=weights.json ({unit_id: weights}) deterministic baseline')
    b.add_argument('--pair', action='append', default=[],
                   help='a,b: paired CI of mean per-row net-return difference a - b (names or labels)')
    b.add_argument('--out')
    args = p.parse_args(argv)
    if args.mode == 'path':
        total = load_total(args.total)
        out = {d: score_decision_dir(total, d, args.cost_bps) for d in args.decisions}
        print(json.dumps(out, indent=1))
        return
    runs = {}
    for label, run_dir in (r.split('=', 1) for r in args.run):
        runs.setdefault(label, []).append(run_dir)  # one label may span several run dirs
    result = dev_score(args.rows, runs, args.cost_bps,
                       dict(f.split('=', 1) for f in args.fixed))
    text = table(result)
    if args.pair:
        result['paired'] = [paired_ci(result, *pr.split(',')) for pr in args.pair]
        text += '\n\n' + pair_table(result['paired'])
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
        Path(args.out).with_suffix('.md').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
