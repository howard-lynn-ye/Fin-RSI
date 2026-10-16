"""Exercise an installed wheel against public sources; retain only local receipts.

Run outside the checkout with the new environment's Python. Raw records stay in the
specified output directory. The shareable receipt contains counts, dates and health,
not source documents. An inaccessible source is a failed/partial check, never a pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import threading


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 30:
        parser.error('poll-seconds must be at least 30')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    import fin_skills
    from fin_skills.collect import Collector, Store
    from fin_skills.tools import call_tool, list_tools

    package_path = Path(fin_skills.__file__).resolve()
    if not package_path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('acceptance must import the installed package from its environment')
    database = str(out / 'events.sqlite3')
    year = datetime.now(timezone.utc).year
    watches = [
        dict(id='fed-news', source='rss',
             target='https://www.federalreserve.gov/feeds/press_all.xml',
             interval_seconds=args.poll_seconds),
        dict(id='ecb-news', source='rss', target='https://www.ecb.europa.eu/rss/press.html',
             interval_seconds=args.poll_seconds),
        dict(id='survey-michigan', source='umich', target='consumer_sentiment',
             interval_seconds=args.poll_seconds),
        dict(id='positions-cftc', source='cftc', target='13874A',
             interval_seconds=args.poll_seconds, options={'weeks': 4}),
        dict(id='house-disclosures', source='house', target='Pelosi',
             interval_seconds=args.poll_seconds,
             options={'year': 'current', 'first_name': 'Nancy', 'lookback_days': 90,
                      'newest_first': True, 'max_filings': 2}),
    ]
    skipped = []
    if os.environ.get('SEC_IDENTITY'):
        watches.append(dict(id='institution-disclosures', source='sec', target='1067983',
                            interval_seconds=1800,
                            options={'forms': ['13F-HR'], 'max_filings': 1,
                                     'since': f'{year}-01-01', 'value_units': 'USD'}))
    else:
        skipped.append({'capability': 'SEC Form 4/13F live collection',
                        'reason': 'SEC_IDENTITY not configured; no substitute identity supplied'})
    started = utcnow()
    write(out / 'started.json', {'started_at': started, 'commit': args.commit,
          'package_path': str(package_path), 'python': sys.version,
          'version': importlib.metadata.version('fin-skills'),
          'configured_watch_ids': [w['id'] for w in watches]})
    configured = call_tool('collection_configure', {'database': database, 'watches': watches})
    assert configured['started'] is False
    print(json.dumps({'phase': 'configured', 'watches': len(watches)}), flush=True)
    first = call_tool('collect_once', {'database': database})
    # Raw tool receipts stay on the execution host, not in the publication receipt.
    write(out / 'first-private.json', first)
    first_count = first['new_records']
    print(json.dumps({'phase': 'first_poll', 'new_records': first_count,
                      'watches': first['watches']}), flush=True)

    # Reopen SQLite and resume without forcing polls past per-source retry deadlines.
    cycles = []
    repolled_ids = set()
    required_ids = {w['id'] for w in watches if w['source'] != 'sec'}
    stop = threading.Event()
    deadline = threading.Timer(max(150, args.poll_seconds * 2 + 30), stop.set)
    deadline.daemon = True
    deadline.start()
    try:
        with Store(database) as store:
            before = store.db.execute('SELECT count(*) FROM events').fetchone()[0]
            assert before == first_count
            def on_cycle(report):
                cycles.append({'at': utcnow(), 'new_records': report['new_records'],
                               'watches': report['watches']})
                repolled_ids.update(w['watch_id'] for w in report['watches'])
                if required_ids <= repolled_ids:
                    stop.set()
            Collector(store).run(stop=stop, on_cycle=on_cycle)
            after = store.db.execute('SELECT count(*) FROM events').fetchone()[0]
            rows = []
            cursor = 0
            while True:
                page = store.events(after=cursor, limit=1000)
                if not page:
                    break
                rows.extend(page)
                cursor = page[-1]['seq']
    finally:
        deadline.cancel()
    assert after >= before
    health = call_tool('collection_status', {'database': database})['watches']
    coverage = call_tool('collection_coverage', {'database': database})
    current = call_tool('collection_events', {'database': database, 'latest': True, 'limit': 1000})
    digest = call_tool('summarize_news', {'events': rows[:1000], 'as_of': utcnow()})
    summaries = []
    for watch in watches:
        records = [r for r in rows if r['watch_id'] == watch['id']]
        dates = [r['published_at'] for r in records if r.get('published_at')]
        observed = [r['observed_at'] for r in records]
        status = next(s for s in health if s['id'] == watch['id'])
        summaries.append({'watch_id': watch['id'], 'source': watch['source'],
                          'records': len(records), 'kinds': dict(Counter(r['kind'] for r in records)),
                          'latest_publication': max(dates) if dates else None,
                          'latest_observation': max(observed) if observed else None,
                          'unknown_publication_dates': sum(not r.get('published_at') for r in records),
                          **status})
    repolled = required_ids <= repolled_ids
    receipt = {
        'started_at': started, 'finished_at': utcnow(), 'commit': args.commit,
        'package_path': str(package_path), 'isolated_installed_wheel': True,
        'checks': {'configure_does_not_fetch': configured['started'] is False,
                   'restart_preserved_records': before == first_count,
                   'continuous_poll_reached_due_watch': repolled,
                   'four_required_categories_covered': coverage['ready'],
                   'records_preserved_after_poll': after >= before,
                   'events_marked_untrusted': current['content_is_untrusted']},
        'sources': summaries, 'cycles': cycles, 'skipped': skipped,
        'coverage': coverage, 'repolled_watch_ids': sorted(repolled_ids),
        'first_records': first_count, 'final_records': after,
        'news_digest': {k: digest[k] for k in ('status', 'unique_articles', 'excluded')},
        'tool_inventory': [s['name'] for s in list_tools()],
        'exercised_tools': ['collection_configure', 'collect_once', 'collection_status',
                            'collection_events', 'collection_coverage', 'summarize_news'],
        'limits': [
            'Inventory is discovery, not execution of every library capability.',
            'Michigan consumer sentiment and CFTC positions are structured observations, not real-time trades.',
            'Unknown publication times remain unknown; availability begins at actual observation.',
            'A successful fetch does not prove complete news/social/disclosure coverage.',
            'No new source publication during this short test would leave update-detection unobserved.',
            'No Return Rate measured in this library acceptance check.',
        ],
    }
    receipt['status'] = ('passed_with_limits' if all(receipt['checks'].values()) and
                         all(s['status'] == 'ok' for s in summaries) and
                         all(w['status'] == 'ok' for w in first['watches']) and
                         all(w['status'] == 'ok' for c in cycles for w in c['watches'])
                         else 'needs_review')
    write(out / 'receipt.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'tool_inventory'}), flush=True)
    return 0 if receipt['status'] == 'passed_with_limits' else 2


if __name__ == '__main__':
    raise SystemExit(main())
