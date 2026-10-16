"""Declared information coverage, separate from collector liveness and poll success."""
from __future__ import annotations

import math
from datetime import datetime

from .model import utc, utcnow


# These are coverage contracts, not claims that every adapter is implemented.
# Page transport and individual social posts do not establish a measured series.
PROFILES = {
    'news': (('news',), 3 * 86400, False),
    'psychology': (('sentiment_survey', 'sentiment_observation'), 30 * 86400, True),
    'behavior': (('positioning_observation', 'fund_flow', 'market_behavior'), 7 * 86400, True),
    'official_disclosures': (('congressional_transaction',), 90 * 86400, True),
    'institutional_disclosures': (('institutional_holding', 'insider_transaction'), 120 * 86400, True),
}
DEFAULT_CATEGORIES = ('news', 'psychology', 'behavior', 'official_disclosures')


def requirements(specifications=None):
    if specifications is None:
        specifications = [{'category': category} for category in DEFAULT_CATEGORIES]
    if not isinstance(specifications, list) or not 1 <= len(specifications) <= len(PROFILES):
        raise ValueError('requirements must be a nonempty list of at most five categories')
    result, seen = [], set()
    for spec in specifications:
        if not isinstance(spec, dict) or set(spec) - {
                'category', 'watch_ids', 'max_age_seconds', 'min_records'}:
            raise ValueError('unknown coverage requirement field')
        category = spec.get('category')
        if not isinstance(category, str) or category not in PROFILES or category in seen:
            raise ValueError('coverage category must be supported and unique')
        seen.add(category)
        kinds, default_age, structured = PROFILES[category]
        age, count = spec.get('max_age_seconds', default_age), spec.get('min_records', 1)
        if type(age) not in (int, float) or not math.isfinite(age) or age <= 0:
            raise ValueError('max_age_seconds must be finite and positive')
        if type(count) is not int or not 1 <= count <= 10000:
            raise ValueError('min_records must be 1..10000')
        watches = spec.get('watch_ids')
        if watches is not None and (not isinstance(watches, list) or not watches or
                any(not isinstance(w, str) or not w.strip() for w in watches) or
                len(set(watches)) != len(watches)):
            raise ValueError('watch_ids must be a nonempty list of distinct watch identifiers')
        result.append(dict(category=category, kinds=list(kinds), structured=structured,
                           watch_ids=watches, max_age_seconds=age, min_records=count))
    return result


def coverage(store, *, specifications=None, as_of=None, max_records=10000):
    rules = requirements(specifications)
    cutoff = utc(utcnow() if as_of is None else as_of)
    stamp = datetime.fromisoformat(cutoff)
    current = as_of is None
    watches = {w.id: w for w in store.watches()}
    try:
        rows = store.retrieval_events(as_of=cutoff, max_records=max_records)
    except ValueError as exc:
        if 'exceeds max_records' not in str(exc):
            raise
        return dict(ready=False, status='unknown', as_of=cutoff, categories=[],
                    requirements=rules, error=str(exc), records_examined=0,
                    reason='snapshot exceeds bound; no partial coverage claim')
    results = []
    for rule in rules:
        excluded = dict(disabled=0, missing_publication=0, stale=0,
                        unstructured=0, invalid_measurement=0, missing_provenance=0)
        matches, eligible = [], []
        for row in rows:
            if row['kind'] not in rule['kinds'] or (rule['watch_ids'] is not None and
                    row['watch_id'] not in rule['watch_ids']):
                continue
            matches.append(row)
            if current and (row['watch_id'] not in watches or not watches[row['watch_id']].enabled):
                excluded['disabled'] += 1
                continue
            if not row.get('url'):
                excluded['missing_provenance'] += 1
                continue
            if not row.get('published_at'):
                excluded['missing_publication'] += 1
                continue
            published = datetime.fromisoformat(utc(row['published_at']))
            observed = datetime.fromisoformat(utc(row['observed_at']))
            age = (stamp - published).total_seconds()
            if age > rule['max_age_seconds']:
                excluded['stale'] += 1
                continue
            if rule['structured'] and row.get('data', {}).get('parse_status') not in (
                    'structured', 'extracted_reviewable'):
                excluded['unstructured'] += 1
                continue
            if rule['category'] in ('psychology', 'behavior'):
                data = row.get('data', {})
                value = data.get('value')
                if (any(not isinstance(data.get(k), str) or not data[k].strip()
                        for k in ('metric', 'unit')) or type(value) not in (int, float) or
                        not math.isfinite(value)):
                    excluded['invalid_measurement'] += 1
                    continue
            eligible.append(dict(watch_id=row['watch_id'], event_id=row['id'], seq=row['seq'],
                                 kind=row['kind'], source=row['source'], url=row['url'],
                                 published_at=row['published_at'], observed_at=row['observed_at'],
                                 age_seconds=age,
                                 observation_lag_seconds=(observed - published).total_seconds()))
        ready = len(eligible) >= rule['min_records']
        results.append(dict(**rule, ready=ready, status='covered' if ready else (
            'missing' if not matches else 'incomplete'), matched_records=len(matches),
            eligible_records=len(eligible), excluded=excluded,
            examples=sorted(eligible, key=lambda r: r['published_at'], reverse=True)[:3]))
    ready = all(item['ready'] for item in results)
    return dict(ready=ready, status='covered' if ready else 'incomplete', as_of=cutoff,
                categories=results, records_examined=len(rows),
                missing_categories=[r['category'] for r in results if not r['ready']],
                scope='declared category/kind, publication age and provenance checks; '
                      'not exhaustive coverage, semantic validation or investment advice',
                availability='both observed_at and published_at must be at or before as_of; '
                             'unknown publication cannot satisfy a requirement',
                current_watch_filter=current)
