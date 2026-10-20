"""Build the fixed Fin-RSI financial development set (rows + units for fin_rsi_run.py).

Source: the current-v7 raw/library study (original 12-instrument basket, 44 ten-session
decisions 2025-01-02 .. 2026-09-22). Selection is a fixed rule declared before any return was
looked at: every other decision, odd indices 1,3,...,43 (22 dates spread evenly over the
window). Even indices are reserved as a second fold and are not built here.

Each unit is one decision date with
  * holdings_before from a neutral reference path (equal weight at every decision, drifted),
    identical for both arms, so arms differ only in the interface they are given;
  * a point-in-time workspace (quotes/actions truncated at the date + eligible evidence);
  * period_end_index for holding-period scoring by fin_rsi_score.
Rows carry the exact v7 initial messages for arm ``raw`` (no library) and ``library``,
produced by the original trading_capabilities.decide prompt assembly.

    python -m benchmarks.agent_study.fin_rsi_devset --out results/fin_rsi/devsets/<name>
"""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from benchmarks.agent_study import market_data as md
from benchmarks.agent_study import trading_capabilities as caps
from benchmarks.agent_study import trading_study_v5 as previous
from benchmarks.agent_study import fin_rsi_score as score
from benchmarks.agent_study.manual_multisource import digest
from benchmarks.agent_study.multisource_model_study import GUIDE, evidence_text

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / ('results/fin_rsi/live_model_experiments/runs/'
                 'current-v7-raw-library-qwen-general-7b-seeds11-23-37/attempt-01/seed-11')
ARMS = ('raw', 'library')
SELECTION = 'odd decision indices 1,3,...,43 of the 44-decision v7 schedule (fixed rule, no return lookup)'


class Captured(Exception):
    pass


def initial_messages(workspace, arm, task, root):
    """Exact first-turn messages of trading_capabilities.decide (no model call)."""
    seen = {}

    def backend(messages):
        seen['messages'] = [dict(m) for m in messages]
        raise Captured
    controller = caps.Controller(root, workspace, arm, 0)
    try:
        caps.decide(backend, controller, task, orientation_path=GUIDE)
    except Captured:
        return seen['messages']
    raise RuntimeError('decide returned without calling the backend')


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def self_check(source, records, packets, scratch):
    """Regenerate stored initial requests of the source run from their own holdings."""
    checked = []
    for arm, index in (('raw', 1), ('library', 1), ('library', 30)):
        stored = json.loads((source / 'decisions' / arm / f'{index:02d}.json').read_text())
        day, pick = stored['date'], stored['index']
        ws = Path(tempfile.mkdtemp(dir=scratch)) / 'visible'
        md.truncate(source / 'data', ws, day)
        caps.write_evidence_snapshot(records, packets[day], ws)
        task = previous.task(day, pick, stored['holdings_before']) + evidence_text(packets[day])
        messages = initial_messages(ws, arm, task, scratch)
        if messages != stored['initial_request']:
            raise RuntimeError(f'regenerated messages differ from stored {arm}/{index:02d}')
        checked.append(f'{arm}/{index:02d}')
    return checked


def build(out, source=SOURCE, seed=11):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    md.TICKERS = tuple(json.loads((source / 'protocol.json').read_text())['universe'])
    inputs = json.loads((source / 'inputs.json').read_text())
    packets = json.loads((source / 'evidence-packets.json').read_text())
    records = json.loads((source / 'evidence.json').read_text())
    total_path = source / 'data' / md.HIDDEN
    total = pd.read_csv(total_path, index_col=0)
    picks = inputs['picks']
    ew = {t: 1 / len(md.TICKERS) for t in md.TICKERS}
    _, reference = score.path(total, picks, [ew] * len(picks))
    holdings = [dict.fromkeys(md.TICKERS, 0.0)] + [p['end_holdings'] for p in reference[:-1]]
    selected = [i for i in range(len(picks)) if i % 2 == 1]

    scratch = out / 'tmp'
    scratch.mkdir()
    checked = self_check(source, records, packets, scratch)
    units, rows = [], []
    for i in selected:
        pick, day = picks[i], str(total.index[picks[i]])
        end = picks[i + 1] if i + 1 < len(picks) else len(total) - 1
        ws = out / 'pit-workspaces' / day
        md.truncate(source / 'data', ws, day)
        caps.write_evidence_snapshot(records, packets[day], ws)
        if pd.read_csv(ws / 'quotes.csv').date.max() != day:
            raise RuntimeError(f'workspace not point-in-time at {day}')
        unit_id = f'fin_dev_{i:02d}_{day}'
        task = previous.task(day, pick, holdings[i]) + evidence_text(packets[day])
        units.append(dict(unit_id=unit_id, role='financial_dev', sampling_seed=seed,
            PIT_and_input=dict(date=day, decision_index=pick, schedule_position=i,
                               period_end_index=end, period_end_date=str(total.index[end]),
                               holdings_before=holdings[i],
                               evidence_packet_sha256=digest(packets[day]),
                               visible_workspace=str(ws.resolve()),
                               visible_files_sha256={f.name: sha_file(f) for f in sorted(ws.iterdir())})))
        for arm in ARMS:
            messages = initial_messages(ws, arm, task, scratch)
            rows.append(dict(row_id=f'{unit_id}__{arm}', unit_id=unit_id, condition=arm, arm=arm,
                             date=day, decision_index=pick, sampling_seed=seed, messages=messages,
                             messages_sha256=hashlib.sha256(json.dumps(messages, sort_keys=True,
                                 ensure_ascii=False).encode()).hexdigest()))
    for f in sorted(scratch.rglob('*'), reverse=True):
        f.unlink() if f.is_file() else f.rmdir()
    scratch.rmdir()
    binding = dict(schema='fin-rsi-financial-devset/1', condition_order=list(ARMS),
                   devset=dict(source_run=str(source), selection=SELECTION,
                               holdings_rule='equal weight at every decision, drifted to the decision date',
                               selected_positions=selected, reserved_positions=[i for i in range(len(picks)) if i % 2 == 0],
                               total_return_close=str(total_path.resolve()),
                               total_return_close_sha256=sha_file(total_path),
                               cost_bps=score.COST_BPS, guide_sha256=sha_file(GUIDE),
                               message_regeneration_self_check=checked,
                               rows_are_model_agnostic=True),
                   units=units, rows=rows)
    path = out / 'rows-v1.json'
    path.write_text(json.dumps(binding, indent=1, ensure_ascii=False))
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    print(build(args.out))


if __name__ == '__main__':
    main()
