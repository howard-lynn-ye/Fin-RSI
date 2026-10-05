"""Render paper tables from the complete, hash-bound factorial trading evidence.

This command never runs inference or changes frozen experiment artifacts. Use --check
in validation. The published evidence excludes prices, weights and raw model outputs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / 'benchmarks/agent_study/evidence/20261003-factorial-trading'
PAPER = REPO / 'paper/latex_naacl'
ARMS = ('base', 'knowledge', 'tools', 'full')
EXPECTED = {(b, m, s) for b in ('original', 'transfer') for m in ('7b', '14b')
            for s in (11, 23, 37)}


def read_verified(root=EVIDENCE):
    root = Path(root)
    manifest = json.loads((root / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    for name, digest in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Evidence path escapes root')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Evidence hash mismatch: {name}')
    data = json.loads((root / 'analysis.json').read_text(encoding='utf-8'))
    completed = json.loads((root / 'completed.json').read_text(encoding='utf-8'))
    if completed['analysis_sha256'] != manifest['analysis.json']:
        raise ValueError('Completion receipt does not bind analysis')
    validate_batch(data)
    return data


def validate_batch(data):
    rows = data['results']
    keys = [(r['basket'], r['family'], r['seed']) for r in rows]
    if len(keys) != len(EXPECTED) or set(keys) != EXPECTED:
        raise ValueError('Incomplete or duplicate factorial pairs')
    if data['decisions'] != 2112:
        raise ValueError('Incomplete decision count')
    for row in rows:
        if set(row['paths']) != set(ARMS):
            raise ValueError('Missing factorial cell')
        for arm in ARMS:
            if row['paths'][arm]['decisions'] != 44:
                raise ValueError('Incomplete cell denominator')
        r = {a: row['paths'][a]['return_rate_pct'] for a in ARMS}
        effects = dict(full_minus_base=r['full']-r['base'],
                       knowledge_main=(r['knowledge']-r['base']+r['full']-r['tools'])/2,
                       tools_main=(r['tools']-r['base']+r['full']-r['knowledge'])/2,
                       interaction=r['full']-r['knowledge']-r['tools']+r['base'])
        if any(abs(v-row['effects_pp'][k]) > 1e-9 for k, v in effects.items()):
            raise ValueError('Paired effect disagrees with returns')
    groups = data['groups']
    if len(groups) != 4 or {(g['basket'], g['family']) for g in groups} != {
            (b, m) for b in ('original', 'transfer') for m in ('7b', '14b')}:
        raise ValueError('Incomplete group summary')
    for g in groups:
        subset = [r for r in rows if (r['basket'], r['family']) == (g['basket'], g['family'])]
        for arm in ARMS:
            if abs(mean(r['paths'][arm]['return_rate_pct'] for r in subset)
                   - g['mean_return_rate_pct'][arm]) > 1e-9:
                raise ValueError('Group mean disagrees with complete pairs')


def table(spec, header, rows, caption, label):
    return '\n'.join([r'\begin{table*}[t]', r'\centering\small', r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{'+spec+'}', r'\toprule', header+r' \\', r'\midrule',
        *[' & '.join(str(v) for v in row)+r' \\' for row in rows],
        r'\bottomrule', r'\end{tabular}', r'\caption{'+caption+'}',
        r'\label{'+label+'}', r'\end{table*}', ''])


def render(data):
    validate_batch(data)
    rows = data['results']; groups = data['groups']
    def subset(g, arm):
        return [r['paths'][arm] for r in rows
                if (r['basket'], r['family']) == (g['basket'], g['family'])]
    f = lambda x: f'{x:.2f}'
    name = lambda g: [g['basket'].capitalize(), g['family'].upper()]
    result_rows = [name(g)+[f(g['mean_return_rate_pct'][a]) for a in ARMS]
                   +[f(g['mean_effects_pp'][e]) for e in
                     ('full_minus_base','knowledge_main','tools_main','interaction')]
                   for g in groups]
    main = table('llrrrrrrrr', r'Basket & Model & Base & K & T & K+T & Full$-$Base & K effect & T effect & Interaction',
        result_rows, 'Mean cumulative net return (percent) and paired effects (percentage points), '
        'over three sampling seeds on the same historical calendar. All planned decisions, '
        'including missing submissions, contribute. K denotes skill-document access; T denotes '
        'numerical-tool access with calling contracts. Original assets were used in development; '
        'transfer changes assets within the same period.', 'tab:factorial-main')
    detail = table('llrrrrrrr', r'Basket & Model & Seed & Base & K & T & K+T & Full$-$Base & Interaction',
        [name(r)+[r['seed']]+[f(r['paths'][a]['return_rate_pct']) for a in ARMS]
         +[f(r['effects_pp'][e]) for e in ('full_minus_base','interaction')] for r in rows],
        'All seed-level cumulative net returns (percent) and paired effects (percentage points). '
        'Each cell retains all 44 decisions. Seeds vary sampling, not market history.', 'tab:factorial-seeds')
    use=[]; risk=[]; cost=[]
    for g in groups:
        for arm in ARMS:
            ss=subset(g,arm); total=lambda k:sum(x[k] for x in ss)
            use.append(name(g)+[arm, f"{total('submitted')}/{total('decisions')}",
                total('decisions_with_algorithm'),total('submitted_weights_matching_a_tool_output'),
                total('successful_skill_reads'), f"{total('failed_turns')}/{total('total_turns')}"])
            risk.append(name(g)+[arm,f(mean(x['metrics']['sharpe'] for x in ss)),
                f(100*mean(x['metrics']['max_drawdown'] for x in ss)),
                f(100*mean(x['exposure']['mean_risky_weight'] for x in ss)),
                f(100*mean(x['exposure']['mean_largest_asset_weight'] for x in ss)),
                f(mean(x['turnover'] for x in ss))])
            cost.append(name(g)+[arm]+[f(mean(x['fixed_decision_cost_sensitivity_pct'][str(b)] for x in ss)) for b in (0,10,25)]
                +[f(total('prompt_tokens')/1e6),f(total('completion_tokens')/1e6),f(total('generation_seconds')/3600)])
    detail+=table('lllrrrrr', r'Basket & Model & Cell & Submitted & Alg. decisions & Matched & Skill reads & Failed turns', use,
        'Observed uptake and failure denominators, summed across all three seeds. Algorithm decisions '
        'count decisions with at least one successful numerical call; matched counts submissions '
        'whose weights equal a successful tool output within the frozen tolerance. Skill reads '
        'count successful calls, not unique documents. Failed turns include execution and response '
        'delivery failures; the disaggregation is preserved in analysis.json. No-use is an outcome.', 'tab:factorial-use')
    detail+=table('lllrrrrr',r'Basket & Model & Cell & Sharpe & Max DD (\%) & Invested (\%) & Largest (\%) & Turnover',risk,
        'Means of seed-level risk and exposure measures. Sharpe uses daily returns, annualization '
        'by 252 sessions and zero cash yield. Max DD is signed maximum drawdown; invested and largest '
        'are mean daily weights. Turnover is the sum of absolute risky-asset weight changes across trades '
        '(one-way units, not percent). These are descriptive, unadjusted comparisons.', 'tab:factorial-risk')
    detail+=table('lllrrrrrr',r'Basket & Model & Cell & 0 bps & 10 bps & 25 bps & Input M & Output M & Gen. hours',cost,
        'Fixed-decision cost sensitivity: mean return (percent) repriced at 0, 10 and 25 bps per traded '
        'side; the primary analysis uses 5 bps. Tokens and model generation hours are summed across '
        'seeds (not elapsed wall time or API prices). Cost sensitivity does not simulate changes in '
        'agent behavior. Full precision and separate failure counts are in the aggregate evidence.', 'tab:factorial-cost')
    intervals=[]
    for g in groups:
        for effect in ('full_minus_base','knowledge_main','tools_main','interaction'):
            lengths=g['descriptive_block_sensitivity']['block_lengths']
            intervals.append(name(g)+[effect.replace('_',' ')]+[
                f"[{f(lengths[str(b)][effect]['low'])}, {f(lengths[str(b)][effect]['high'])}]" for b in (10,20,40)])
    detail+=table('lllrrr',r'Basket & Model & Effect & 10 sessions & 20 sessions & 40 sessions',intervals,
        r'Descriptive 95\% paired circular calendar-block intervals (percentage points), 2,000 draws '
        'for each length. Calendar indices are shared across cells and seeds in a group. These '
        'intervals do not represent independent markets or establish general performance gains.', 'tab:factorial-intervals')
    baseline=[]
    for b,methods in data['baselines'].items():
        for m,x in methods.items():
            baseline.append([b.capitalize(),m.replace('_',' '),f(x['return_rate_pct']),f(x['metrics']['sharpe']),f(100*x['metrics']['max_drawdown']),f(x['turnover'])])
    detail+=table('llrrrr',r'Basket & Fixed strategy & Return (\%) & Sharpe & Max DD (\%) & Turnover',baseline,
        'Fixed strategy references use the same next-close timing and 5 bps cost. Portfolio methods '
        'use 252 past returns and frozen defaults. Equal-weight hold trades once; other methods '
        'rebalance every ten sessions. The original-only 60/40 is SPY/IEF. Cash earns zero.', 'tab:factorial-baselines')
    numbers=['% Generated by scripts/build_factorial_evidence.py; do not hand-edit.',
             r'\newcommand{\FTDecisions}{'+str(data['decisions'])+'}']
    for g in groups:
        key=g['basket'].capitalize()+('Seven' if g['family']=='7b' else 'Fourteen')
        numbers.append(r'\newcommand{\FT'+key+r'Difference}{'+f(g['mean_effects_pp']['full_minus_base'])+'}')
    alg=sum(r['paths'][a]['decisions_with_algorithm'] for r in rows for a in ARMS)
    reads=sum(r['paths'][a]['successful_skill_reads'] for r in rows for a in ARMS)
    numbers += [r'\newcommand{\FTAlgorithmDecisions}{'+str(alg)+'}',r'\newcommand{\FTSkillReads}{'+str(reads)+'}']
    return {'factorial_numbers.tex':'\n'.join(numbers)+'\n',
            'factorial_results_table.tex':main,'factorial_detail_tables.tex':detail}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    for name,text in render(read_verified()).items():
        path=PAPER/name
        if args.check:
            if not path.exists() or path.read_text(encoding='utf-8') != text:
                raise SystemExit(f'Stale generated factorial table: {path}')
        else:
            path.write_text(text,encoding='utf-8')
    print('Factorial evidence and paper tables: OK')


if __name__=='__main__':
    main()
