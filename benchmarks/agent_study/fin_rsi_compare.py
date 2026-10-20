"""Compare fin_rsi_run summary.json files condition by condition.

    python -m benchmarks.agent_study.fin_rsi_compare NAME=results/fin_rsi/runs/<dir>[@variant] ...
"""
import collections
import json
import re
import sys
from pathlib import Path

ALGORITHM_ID = re.compile(r'''algorithm_id["']?\s*[:=]\s*["']([A-Za-z0-9_.\-]+)''')

COUNTS = ('submitted', 'algorithm_then_submit', 'read_portfolio_skill', 'reached_passage',
          'method_changed')
SUMS = ('run_algorithm_ok', 'run_algorithm_failed', 'real_mentions', 'skill_reads_ok',
        'list_skills_ok', 'list_algorithms_ok', 'describe_algorithm_ok',
        'executed_turns', 'turns')


# portfolio-optimizers SKILL.md: the out-of-sample min-variance vs 1/N table starts at char 4314.
PASSAGE = ('portfolio-optimizers', 4314)


def skill_use(decision):
    """Did the row read the portfolio-method skill, reach the comparison passage, and change method
    away from min_variance (final algorithm differs, or it ran more than one algorithm)?"""
    calls = decision['tool_calls']
    pages = [c for c in calls if c.get('tool') == 'read_skill' and c.get('ok')
             and c.get('skill_name') == PASSAGE[0] and not c.get('reference')]
    algos = [c.get('algorithm_id') for c in calls if c.get('tool') == 'run_algorithm' and c.get('ok')]
    changed = bool(algos) and (algos[-1] != 'min_variance' or len(set(algos)) > 1)
    return dict(read_portfolio_skill=bool(pages),
                reached_passage=any(c['offset'] <= PASSAGE[1] < c['offset'] + c['returned_chars']
                                    for c in pages),
                list_skills_ok=sum(1 for c in calls if c.get('tool') == 'list_skills' and c.get('ok')),
                method_changed=changed)


def load(path, real_ids):
    """Row metrics plus algorithm_id values written in the model's responses (failed calls do
    not record their id), split into catalog IDs and invented ones."""
    rows = []
    for p in sorted(Path(path).glob('rows/*/decision.json')):
        decision = json.loads(p.read_text())
        written = [m for t in decision['turns'] for m in ALGORITHM_ID.findall(t.get('response') or '')]
        rows.append(dict(decision['metrics'],
                         invented=collections.Counter(i for i in written if i not in real_ids),
                         real_mentions=sum(i in real_ids for i in written),
                         run_algorithm_failed=sum(1 for c in decision['tool_calls']
                                                  if c.get('tool') == 'run_algorithm' and not c.get('ok')),
                         **skill_use(decision)))
    return rows


def main(argv):
    from fin_skills.algorithms import catalog
    real_ids = {c['id'] for c in catalog()}
    header = (['condition', 'rows'] + list(COUNTS) + list(SUMS) +
              ['invented_mentions', 'rows_with_invented', 'algorithm_ids', 'top_invented'])
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '---|' * len(header))
    for spec in argv:
        name, _, path = spec.partition('=')
        path, _, variant = path.partition('@')  # optional: one system_variant of a multi-variant run
        rows = [r for r in load(path, real_ids) if not variant or r.get('system_variant') == variant]
        invented = sum((r['invented'] for r in rows), collections.Counter())
        ids = collections.Counter(a for r in rows for a in r.get('algorithm_ids', []))
        cells = [name, str(len(rows))]
        cells += [str(sum(bool(r[k]) for r in rows)) for k in COUNTS]
        cells += [str(sum(r[k] for r in rows)) for k in SUMS]
        cells += [str(sum(invented.values())), str(sum(bool(r['invented']) for r in rows))]
        cells.append(', '.join(f'{k}:{v}' for k, v in ids.most_common()) or '-')
        cells.append(', '.join(f'{k}:{v}' for k, v in invented.most_common(3)) or '-')
        print('| ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main(sys.argv[1:])
