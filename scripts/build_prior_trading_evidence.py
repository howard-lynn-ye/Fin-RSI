"""Regenerate the separately archived v5 table; never pool it with factorial cells."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.build_factorial_evidence import table


def render():
    root = REPO / 'benchmarks/agent_study/evidence/20261002-trading-v5'
    for name, digest in json.loads((root / 'SHA256SUMS.json').read_text(encoding='utf-8')).items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Evidence path escapes root')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Changed v5 evidence: {name}')
    rows = json.loads((root / 'posthoc/analysis.json').read_text(encoding='utf-8'))['rows']
    result = []
    for model in ('7b', '14b'):
        for seed in (11, 23, 37):
            raw = next(x for x in rows if x['family']==model and x['seed']==seed and x['arm']=='raw')
            lib = next(x for x in rows if x['family']==model and x['seed']==seed and x['arm']=='library')
            result.append([model.upper(), seed, f"{raw['return_rate_pct']:.2f}",
                f"{lib['return_rate_pct']:.2f}", f"{lib['return_rate_pct']-raw['return_rate_pct']:.2f}",
                f"{raw['submitted']}/{raw['decisions']}", f"{lib['submitted']}/{lib['decisions']}",
                lib['decisions_with_algorithm']])
    return table('lrrrrrrr',
        r'Model & Seed & Raw (\%) & Bundle (\%) & Difference (pp) & Raw submit & Bundle submit & Alg. decisions',
        result, 'Earlier development-exposed bundled-interface study, retained separately from the factorial '
        'batch. All scheduled decisions are scored, and all seed-level signs are retained. The bundle '
        'changes data adapters as well as tools and documentation. Full risk, failure, token, cost, '
        'and post hoc exposure/adoption results accompany the separate evidence archive.', 'tab:prior-trading')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = render()
    path = REPO / 'paper/latex_naacl/prior_trading_table.tex'
    if args.check:
        if path.read_text(encoding='utf-8') != text:
            raise SystemExit('Stale prior trading table')
    else:
        path.write_text(text, encoding='utf-8')
    print('Prior trading evidence and table: OK')


if __name__ == '__main__':
    main()
