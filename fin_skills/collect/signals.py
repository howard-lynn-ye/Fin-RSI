"""Free public survey and positioning observations; no inferred release timestamps."""
from __future__ import annotations

import re
from datetime import date, datetime
from urllib.parse import urlencode

from .feeds import PageSource
from .model import Batch, Event, Watch, utcnow

AAII_URL = 'https://www.aaii.com/sentimentsurvey'
CFTC_API = 'https://publicreporting.cftc.gov/resource/gpe5-46if.json'


class _AAIIClient:
    # The site's robots.txt contains repeated wildcard groups. Some parsers use only
    # the first group's crawl delay; enforce the observed ten-second floor as well.
    def __init__(self, client):
        self.client = client
        self.user_agent = client.user_agent
        self.sleep = client.sleep

    def get(self, *args, **kwargs):
        self.sleep(10)
        return self.client.get(*args, **kwargs)


def reference_day(value, observed):
    """A reporting/survey day, never a publication timestamp."""
    day = date.fromisoformat(value)
    if day > datetime.fromisoformat(observed).date():
        raise ValueError('source reference date is in the future')
    return day.isoformat()


def aaii_readings(content, observed):
    # Anchor to the latest-results panel, excluding historical averages and extremes.
    panel = re.search(r"This week['\u2019]s results\s+Week ending ([A-Za-z]+ \d{1,2}, \d{4})"
                      r'\s+Bullish\s+([\d.]+)%.*?Neutral\s+([\d.]+)%.*?Bearish\s+([\d.]+)%',
                      content, re.DOTALL | re.IGNORECASE)
    if not panel:
        raise ValueError('AAII current results panel missing or changed; no readings inferred')
    day = reference_day(datetime.strptime(panel[1], '%B %d, %Y').date().isoformat(), observed)
    values = [float(panel[i]) for i in (2, 3, 4)]
    if any(not 0 <= v <= 100 for v in values) or abs(sum(values) - 100) > .3:
        raise ValueError('AAII shares are invalid or do not sum to approximately 100 percent')
    return day, dict(zip(('bullish_share', 'neutral_share', 'bearish_share'), values))


class AAIISource:
    def __init__(self, client):
        self.pages = PageSource(_AAIIClient(client))

    def fetch(self, watch, state):
        if watch.target != 'sentiment' or watch.options:
            raise ValueError('AAII target must be sentiment with no options')
        # Reuse the robots-aware bounded crawler, including the provider crawl delay.
        batch = self.pages.fetch(Watch(watch.id, 'page', AAII_URL,
                                 watch.interval_seconds, {'max_pages': 1}), {})
        page = batch.events[0]
        day, readings = aaii_readings(page.data['text'], page.observed_at)
        events = [Event(f'{day}:{metric}', 'aaii', 'sentiment_survey',
            f'AAII {metric}: {value} percent (week ending {day})', AAII_URL,
            page.observed_at, None, 'AAII survey respondents', '',
            dict(metric=metric, value=value, unit='percent', survey_week_ending=day,
                 parse_status='structured', availability_basis='first_observed',
                 published_time_unknown=True, content_is_untrusted=True,
                 population='AAII individual investor survey respondents; not all investors',
                 expected_cadence='weekly; survey period is not publication time'))
            for metric, value in readings.items()]
        return Batch(events, {'latest_survey_week': day})


class CFTCSource:
    """Bounded recent TFF futures-only snapshots, including revisions in that window."""
    GROUPS = {
        'dealer': ('dealer_positions_long_all', 'dealer_positions_short_all'),
        'asset_manager': ('asset_mgr_positions_long', 'asset_mgr_positions_short'),
        'leveraged_money': ('lev_money_positions_long', 'lev_money_positions_short'),
        'other_reportable': ('other_rept_positions_long', 'other_rept_positions_short'),
        'nonreportable': ('nonrept_positions_long_all', 'nonrept_positions_short_all'),
    }

    def __init__(self, client):
        self.client = client

    def fetch(self, watch, state):
        if not re.fullmatch(r'[0-9A-Z]{6}', watch.target) or set(watch.options) - {'weeks'}:
            raise ValueError('CFTC target must be a six-character contract code; options: weeks')
        weeks = watch.options.get('weeks', 4)
        if type(weeks) is not int or not 1 <= weeks <= 52:
            raise ValueError('CFTC weeks must be 1..52')
        url = CFTC_API + '?' + urlencode({'$where': f"cftc_contract_market_code='{watch.target}'",
            '$order': 'report_date_as_yyyy_mm_dd DESC, id DESC', '$limit': weeks})
        rows = self.client.get(url).json()
        if not isinstance(rows, list) or not rows or len(rows) > weeks:
            raise ValueError('CFTC returned no matching observations or an invalid bounded response')
        events, observed, seen = [], utcnow(), set()
        for row in rows:
            if (row.get('cftc_contract_market_code') != watch.target or
                    row.get('futonly_or_combined') != 'FutOnly'):
                raise ValueError('CFTC contract/report scope mismatch')
            raw_day = row.get('report_date_as_yyyy_mm_dd', '')
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T00:00:00(?:\.000)?', raw_day):
                raise ValueError('CFTC report date is missing or invalid')
            day = reference_day(raw_day[:10], observed)
            if day in seen:
                raise ValueError('CFTC duplicate report date for the selected contract')
            seen.add(day)
            def count(field):
                value = row.get(field)
                if not isinstance(value, str) or not re.fullmatch(r'\d+', value):
                    raise ValueError(f'CFTC missing or invalid position field: {field}')
                return int(value)
            interest = count('open_interest_all')
            for group, (long_field, short_field) in self.GROUPS.items():
                long, short = count(long_field), count(short_field)
                if max(long, short) > interest:
                    raise ValueError('CFTC group positions exceed open interest')
                events.append(Event(f'{watch.target}:{day}:{group}', 'cftc',
                    'positioning_observation', f'CFTC {watch.target} {group}: net {long-short} contracts',
                    url, observed, None, group, watch.target,
                    dict(metric='net_contracts', value=long-short, unit='contracts',
                         long_contracts=long, short_contracts=short, open_interest=interest,
                         market=row.get('market_and_exchange_names', ''),
                         contract_units=row.get('contract_units', ''), report_date=day,
                         source_record_id=row.get('id'), dataset='gpe5-46if', report='TFF futures only',
                         parse_status='structured', availability_basis='first_observed',
                         published_time_unknown=True, content_is_untrusted=True,
                         expected_cadence='weekly; report date is positions-as-of, not release time',
                         snapshot_window_weeks=weeks)))
        return Batch(events, {'latest_report_date': max(seen), 'snapshot_window_weeks': weeks})
