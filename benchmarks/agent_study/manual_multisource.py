"""Evidence preparation and an append-only interface for assistant-made decisions.

No strategy, weight generator, model inference, or automatic submit is implemented.
The operator reads each packet, submits raw, examines library output, then submits
library. Future packets and final scoring are gated on those explicit submissions.
This is a non-blind retrospective case, not a sealed causal experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
import zipfile

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

TICKERS = ("SPY", "QQQ", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "NVDA", "AVGO", "ORLY", "FAST")
# Contract codes, not ambiguous substring matches. Futures are context, not ETF flows.
CONTRACTS = {"13874A": "SPY", "209742": "QQQ", "244041": "EFA", "244042": "EEM",
             "020601": "TLT", "043602": "IEF", "088691": "GLD", "067651": "DBC"}
DELAYED = dict(zip(
    ("2025-09-30", "2025-10-07", "2025-10-14", "2025-10-21", "2025-10-28",
     "2025-11-04", "2025-11-10", "2025-11-18", "2025-11-25", "2025-12-02",
     "2025-12-09", "2025-12-16", "2025-12-23"),
    ("2025-11-19", "2025-11-21", "2025-11-25", "2025-12-02", "2025-12-05",
     "2025-12-09", "2025-12-10", "2025-12-12", "2025-12-15", "2025-12-17",
     "2025-12-19", "2025-12-23", "2025-12-29")))
AGE_LIMITS = {"news": 100, "psychology": 210, "behavior": 100, "officials": 550}
# Retention above is not a claim of current information. These shorter windows
# are declared review heuristics, not verified release calendars or alpha rules.
RECENCY_REVIEW_DAYS = {"news": 60, "psychology": 120, "behavior": 28, "officials": 90}


def dump(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(dump(value).encode()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(dump(value))


def days(day, n):
    return (date.fromisoformat(day) + timedelta(days=n)).isoformat()


def cot_eligible(day):
    # A deliberately conservative availability ASSUMPTION, not a release timestamp.
    # Known shutdown delays override the 14-calendar-day lag. Current vintage unverified.
    return max(days(day, 14), days(DELAYED.get(day, day), 1))


def normalize(root):
    root = Path(root)
    manifest = json.loads((root / "acquisition_manifest.json").read_text())
    records, exclusions = [], []
    for m in manifest:
        kind = m["category"]
        if kind not in AGE_LIMITS or m.get("error"):
            continue
        data = (root / "sources" / (m["id"] + ".bin")).read_bytes()
        if hashlib.sha256(data).hexdigest() != m["sha256"]:
            raise ValueError("source changed after acquisition")
        common = dict(source_id=m["id"], url=m["url"], source_sha256=m["sha256"],
                      observed_at=m["observed_at"], category=kind,
                      vintage="current_archive_historical_version_unverified")
        if kind in ("news", "psychology"):
            soup = BeautifulSoup(data, "html.parser")
            if kind == "psychology":
                stamp = soup.select_one(".module_date-text")
                if stamp is None:
                    raise ValueError("survey publication date absent")
                day = datetime.strptime(stamp.get_text(strip=True), "%m/%d/%Y").date().isoformat()
                body = soup.select_one(".module-news-details") or soup
            else:
                day = datetime.strptime(m["date_hint"], "%Y%m%d").date().isoformat()
                body = soup.select_one("#article") or soup.select_one("main") or soup
            for el in body(["script", "style", "nav", "header", "footer"]):
                el.decompose()
            paragraphs = [p.get_text(" ", strip=True) for p in body.select("p,li")]
            paragraphs = [p for p in paragraphs if len(p) > 90 and not re.search(
                "copyright|contact us|more information|media contact|Source:|More Trader", p, re.I)]
            # Store full body separately; packet excerpts are deterministic, not sentiment labels.
            full = body.get_text("\n", strip=True)
            if kind == "psychology":
                selected = [p for p in paragraphs if re.search(
                    "bullish|bearish|overvalu|fielded|included|recession", p, re.I)]
                excerpt = "\n".join(selected[:5])[:2800]
            else:
                selected = [p for p in paragraphs if re.search(
                    "economic activity|inflation|unemployment|target range|uncertainty|job gains|available indicators", p, re.I)]
                excerpt = "\n".join(selected[:4])[:2300]
            if len(excerpt) < 100:
                raise ValueError(f"no useful body extracted: {m['url']}")
            rid = m["id"][:16]
            records.append(dict(common, id=rid, event_date=day, published_date=day,
                eligible_date=days(day, 1), availability_basis="dated_official_page_plus_one_day",
                series=m["series"], text=excerpt, full_text=full))
        elif kind == "behavior":
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                member = z.infolist()[0]
                if member.file_size > 200_000_000:
                    raise ValueError("oversize COT archive member")
                frame = pd.read_csv(z.open(member), dtype={"CFTC_Contract_Market_Code": str})
            frame.columns = frame.columns.str.strip()
            for row in frame.to_dict("records"):
                code = row["CFTC_Contract_Market_Code"].strip()
                if code not in CONTRACTS:
                    continue
                day = str(row["Report_Date_as_YYYY-MM-DD"])[:10]
                if not "2024-10-01" <= day <= "2026-09-22":
                    continue
                oi = float(row["Open_Interest_All"])
                groups = ("Asset_Mgr", "Lev_Money") if "Asset_Mgr_Positions_Long_All" in row else ("M_Money",)
                positions = {g: dict(long=int(row[g + "_Positions_Long_All"]),
                                    short=int(row[g + "_Positions_Short_All"]),
                                    net_pct_oi=round(100 * (float(row[g + "_Positions_Long_All"])
                                        - float(row[g + "_Positions_Short_All"])) / oi, 2))
                             for g in groups}
                records.append(dict(common, id=m["id"][:12] + ":" + code + ":" + day,
                    event_date=day, published_date=None, scheduled_release_override=DELAYED.get(day),
                    eligible_date=cot_eligible(day),
                    availability_basis="assumed_14_day_lag_with_known_shutdown_overrides",
                    series=code, context_ticker=CONTRACTS[code], market=row["Market_and_Exchange_Names"],
                    open_interest=int(oi), positions=positions,
                    scope="classified futures positions, not ETF/share flows or directional conviction"))
        else:
            index = m["index_record"]
            # Do not substitute current amendments for originals or double-count them.
            if index["FilingType"] == "A":
                exclusions.append(dict(id=m["id"], reason="amendment_not_merged_into_original"))
                continue
            text = (root / "sources" / (m["id"] + ".txt")).read_text().replace("\x00", "")
            if len(re.sub(r"\W", "", text)) < 80:
                exclusions.append(dict(id=m["id"], reason="image_or_insufficient_PDF_text_needs_OCR"))
                continue
            if re.search(r"(?:Amendment|Amended)\s*[:\-]?\s*(?:Yes|True)", text, re.I):
                exclusions.append(dict(id=m["id"], reason="amended_PTR_requires_review"))
                continue
            pattern = r"\b(?:SPY|QQQ|EFA|EEM|TLT|IEF|GLD|DBC|NVDA|AVGO|ORLY|FAST)\b|NVIDIA|Broadcom|O.Reilly|Fastenal"
            spans = [(max(0, hit.start() - 90), min(len(text), hit.end() + 330))
                     for hit in re.finditer(pattern, text, re.I)]
            excerpts = [re.sub(r"\s+", " ", text[a:b]) for a, b in spans[:12]]
            actor = index["First"] + " " + index["Last"]
            day = m["filing_date"]
            records.append(dict(common, id=m["id"][:16], event_date=day, filing_date=day,
                published_date=None, eligible_date=days(day, 3),
                availability_basis="ASSUMPTION_filing_date_plus_3_days_public_post_time_unverified",
                series=actor + ":" + index["FilingType"], actor=actor,
                report_type="transactions" if index["FilingType"] == "P" else "annual_holdings",
                report_year=index["Year"], text="\n".join(excerpts) or "No universe match in extracted text; not evidence of no holdings/trades.",
                full_text=text, review_status="PDF_text_excerpts_not_a_complete_or_reconciled_portfolio",
                scope=m["scope"]))
    records.sort(key=lambda r: (r["eligible_date"], r["id"]))
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate evidence IDs")
    write_new(root / "evidence.json", records)
    coverage = dict(counts=dict(Counter(r["category"] for r in records)),
                    actors=dict(Counter(r.get("actor") for r in records if r["category"] == "officials")),
                    excluded=exclusions, evidence_sha256=digest(records))
    write_new(root / "evidence_manifest.json", coverage)
    return coverage


def asof_records(records, cutoff, *, strict=False):
    """Select last eligible record per series; expose staleness, do not fake freshness.

    Strict mode rejects today's backfilled, unverified historical vintages. The manual
    retrospective case explicitly opts into the documented availability assumptions.
    """
    latest = {}
    for row in records:
        if row["eligible_date"] > cutoff:
            continue
        if strict and (row.get("vintage") != "verified_historical_version"
                       or "ASSUMPTION" in row["availability_basis"]
                       or row["availability_basis"].startswith("assumed")):
            continue
        key = row["category"], row["series"]
        if key not in latest or (row["eligible_date"], row["id"]) > (
                latest[key]["eligible_date"], latest[key]["id"]):
            latest[key] = row
    selected = []
    for row in latest.values():
        age = (date.fromisoformat(cutoff) - date.fromisoformat(row["event_date"])).days
        kind = row["category"]
        notes = {
            "news": "Publication age; the underlying events may be older.",
            "psychology": "Publication age, not survey fieldwork age. Compare the same sampled population; opinions are not trades.",
            "behavior": "Position report age, not publication age. Classified futures positions may hedge other exposures.",
            "officials": "Filing age, not transaction or holdings age. Annual reports and option exercises are not fresh purchases.",
        }
        selected.append(dict(row, age_days=age, stale=age > AGE_LIMITS[kind],
            stale_definition="exceeds category retention limit; false does not establish freshness",
            retention_limit_days=AGE_LIMITS[kind],
            recency_review_days=RECENCY_REVIEW_DAYS[kind],
            recency_status=("aged_context_review_required" if age > RECENCY_REVIEW_DAYS[kind]
                            else "within_review_window_underlying_dates_still_required"),
            age_interpretation=notes[kind]))
    return sorted(selected, key=lambda r: (r["category"], r["series"]))


def validate_weights(weights):
    if weights is None:  # explicitly choose to hold drifted positions
        return None
    if not isinstance(weights, dict) or not weights or set(weights) - set(TICKERS):
        raise ValueError("weights must name only universe assets; null means hold")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           or v < 0 for v in weights.values()):
        raise ValueError("weights must be finite, nonnegative numbers")
    if sum(weights.values()) > 1 + 1e-10:
        raise ValueError("weights exceed capital")
    return {t: float(weights.get(t, 0)) for t in TICKERS}


def common_history(data_dir, cutoff, lookback=252):
    q = pd.read_csv(data_dir / "quotes.csv")
    a = pd.read_csv(data_dir / "corporate_actions.csv")
    q, a = q[q.date <= cutoff], a[a.date <= cutoff]
    p = q.pivot(index="date", columns="ticker", values="close").sort_index()[list(TICKERS)]
    if lookback is not None:
        p = p.tail(lookback + 1)
    gross = p.div(p.shift()).iloc[1:]
    for r in a.itertuples():
        if r.date not in gross.index:
            continue
        if r.kind == "split":
            gross.loc[r.date, r.ticker] *= r.value
        elif r.kind == "dividend":
            gross.loc[r.date, r.ticker] += r.value / p.shift().loc[r.date, r.ticker]
        else:
            raise ValueError("unknown action")
    if not np.isfinite(gross.to_numpy()).all() or (gross <= 0).any().any():
        raise ValueError("incomplete price/action history")
    return q, a, gross - 1


class ManualStudy:
    def __init__(self, root, data_dir):
        self.root, self.data_dir = Path(root), Path(data_dir)
        self.evidence = json.loads((self.root / "evidence.json").read_text())
        self.sessions = sorted(pd.read_csv(self.data_dir / "quotes.csv", usecols=["date"]).date.unique())
        first = self.sessions.index("2025-01-02")
        self.picks = list(range(first, len(self.sessions) - 2, 10))
        self.dates = [self.sessions[i] for i in self.picks]
        self.decisions = self.root / "decisions"
        self.decisions.mkdir(exist_ok=True)

    def prepare(self):
        """Freeze conditions and audit coverage without showing future contents or returns."""
        coverage = []
        for day in self.dates:
            rows = asof_records(self.evidence, day)
            counts = Counter(r["category"] for r in rows if not r["stale"])
            missing = sorted(set(AGE_LIMITS) - counts.keys())
            coverage.append(dict(date=day, counts=dict(counts), missing=missing,
                ages={kind: max(r["age_days"] for r in rows if r["category"] == kind)
                      for kind in counts}))
        if any(r["missing"] for r in coverage):
            raise ValueError(dump(coverage))
        protocol = dict(version="manual-multisource-v1", objective="net risk-adjusted return",
            primary_report="cumulative net Return Rate (%) from initial capital before first fee",
            dates=self.dates, universe=list(TICKERS), valuation_end=self.sessions[-1],
            execution="next session close, after that session return", cost_bps_per_side=5,
            initial_nav=1, cash_interest=0, long_only=True, leverage=False,
            weights="personally authored by assistant in conversation; no generator or inference runner",
            ordering="raw locked, library inspected, library locked, then next date",
            common_inputs="same archived evidence and same split/dividend-adjusted common indicators",
            library_treatment="fin-skills history validation, HRP candidate, risk calculations and skill documents; assistant chooses",
            input_mode="retrospective; historical source versions unverified",
            availability={"dated_news_surveys": "+1 calendar day", "House": "filing date +3 days ASSUMPTION; not public posting verification",
                          "CFTC": "report date +14 days ASSUMPTION with later known shutdown releases"},
            limitations=["not blind; prior library knowledge, aggregate returns and source-discovery exposure",
                         "not an equal-compute/token-budget model comparison", "single historical path, not 44 independent experiments",
                         "no source selection or retries based on returns", "no complete social-media/news/Congress coverage"],
            evidence_sha256=digest(self.evidence),
            market_hashes={n: hashlib.sha256((self.data_dir/n).read_bytes()).hexdigest()
                           for n in ("quotes.csv", "corporate_actions.csv", "total_return_close.csv")},
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            frozen_at=datetime.now(timezone.utc).isoformat(), coverage=coverage)
        write_new(self.root / "protocol.json", protocol)
        return dict(protocol_sha256=digest(protocol), decisions_per_arm=len(self.dates),
                    coverage=coverage)

    def locked(self):
        return sorted(self.decisions.glob("[0-9][0-9]-*.json"),
                      key=lambda p: (int(p.name[:2]), 0 if "-raw." in p.name else 1))

    def current(self):
        return len(self.locked()) // 2

    def holdings(self, arm, cutoff):
        """Show drifted current holdings using only past visible prices; no future NAV."""
        # Identical accounting price series to score(); exclude future rows before returns.
        past = pd.read_csv(self.data_dir / "total_return_close.csv", index_col=0)
        past = past.loc[past.index <= cutoff, list(TICKERS)]
        returns = past.pct_change().dropna()
        trades = {}
        for path in self.locked():
            d = json.loads(path.read_text())
            executed = self.sessions[self.picks[d["index"]] + 1]
            if d["arm"] == arm and executed <= cutoff:
                trades[executed] = d["weights"]
        w = pd.Series(0., index=TICKERS)
        for day, r in returns.iterrows():
            if day < self.dates[0]:
                continue
            cash = 1 - float(w.sum())
            grown = w * (1 + r)
            w = grown / (cash + grown.sum())
            if trades.get(day) is not None:
                w = pd.Series(trades[day])
        return w.to_dict()

    def packet(self):
        self.verify_protocol()
        i = self.current()
        if i >= len(self.dates):
            raise ValueError("all decisions complete")
        day = self.dates[i]
        rows = asof_records(self.evidence, day)
        counts = Counter(r["category"] for r in rows if not r["stale"])
        missing = set(AGE_LIMITS) - counts.keys()
        if missing:
            raise ValueError(f"missing required sources at {day}: {sorted(missing)}")
        _, _, returns = common_history(self.data_dir, day)
        prices = (1 + returns).cumprod()
        market = {t: {"r21_pct": round(100*((1+returns[t].tail(21)).prod()-1), 2),
                      "r63_pct": round(100*((1+returns[t].tail(63)).prod()-1), 2),
                      "r126_pct": round(100*((1+returns[t].tail(126)).prod()-1), 2),
                      "vol63_ann_pct": round(100*returns[t].tail(63).std()*np.sqrt(252), 2),
                      "drawdown63_pct": round(100*(prices[t].iloc[-1]/prices[t].tail(63).max()-1), 2)}
                  for t in TICKERS}
        shown = [{k: v for k, v in r.items() if k not in ("full_text", "source_sha256", "observed_at")}
                 for r in rows]
        packet = dict(index=i, date=day, next_execution=self.sessions[self.picks[i]+1],
                      evidence=shown, common_market=market, information_mode="retrospective_assumptions",
                      holdings_before={arm: self.holdings(arm, day) for arm in ("raw", "library")},
                      previous_targets={arm: json.loads((self.decisions / f"{i-1:02d}-{arm}.json").read_text())["weights"]
                                        for arm in ("raw", "library")} if i else {})
        path = self.root / "packets" / f"{i:02d}.json"
        if path.exists():
            if json.loads(path.read_text()) != packet:
                raise ValueError("packet changed after first display")
        else:
            write_new(path, packet)
        return packet

    def commit(self, arm, weights, rationale, evidence_ids):
        n = len(self.locked())
        if arm != ("raw" if n % 2 == 0 else "library"):
            raise ValueError("raw must be locked before library; no skipping or overwriting")
        packet = self.packet()
        if not isinstance(rationale, str) or len(rationale.strip()) < 30:
            raise ValueError("explain the personal decision")
        if not evidence_ids or set(evidence_ids) - {r["id"] for r in packet["evidence"]}:
            raise ValueError("cite eligible packet evidence IDs")
        if arm == "library" and not (self.root / "library_calls" / f"{packet['index']:02d}.json").exists():
            raise ValueError("library arm must actually inspect a recorded call")
        previous = digest(json.loads(self.locked()[-1].read_text())) if n else None
        row = dict(arm=arm, date=packet["date"], index=packet["index"], weights=validate_weights(weights),
                   rationale=rationale, evidence_ids=evidence_ids, packet_sha256=digest(packet),
                   decided_at=datetime.now(timezone.utc).isoformat(), previous_decision_sha256=previous,
                   decision_maker="assistant_in_conversation", blind=False)
        write_new(self.decisions / f"{packet['index']:02d}-{arm}.json", row)
        return dict(locked=n+1, arm=arm, date=packet["date"], sha256=digest(row))

    def library(self):
        if len(self.locked()) % 2 != 1:
            raise ValueError("lock the raw decision before library inspection")
        packet = self.packet()
        q, a, common = common_history(self.data_dir, packet["date"])
        from fin_skills.data import prepare_history
        from fin_skills.algorithms import run
        result = prepare_history(q, a, as_of=packet["date"], lookback=252,
                                 input_adjustment="raw", actions_complete=True, tickers=list(TICKERS))
        returns = result["returns"]
        np.testing.assert_allclose(returns.values, common.values, atol=1e-12)
        covariance = returns.tail(63).cov()
        weights = json.loads((self.decisions / f"{packet['index']:02d}-raw.json").read_text())["weights"]
        if weights is None:
            weights = packet["holdings_before"]["raw"]
            diagnostic_basis = "raw_drifted_holdings_raw_chose_hold"
        else:
            diagnostic_basis = "raw_proposed_weights"
        w = np.array([weights[t] for t in TICKERS])
        variance = float(w @ covariance.to_numpy() @ w)
        contributions = w * (covariance.to_numpy() @ w)
        diagnostics = dict(history=result["metadata"], diagnostic_basis=diagnostic_basis,
            annual_vol_pct=100*np.sqrt(variance*252),
            risk_contribution_pct={t: round(float(x/variance*100), 2) for t,x in zip(TICKERS, contributions)} if variance else {},
            correlation_pairs={f"{x}/{y}": round(float(returns.tail(63).corr().loc[x,y]),3)
                               for x,y in (("SPY","QQQ"),("QQQ","NVDA"),("NVDA","AVGO"),("SPY","GLD"),("SPY","IEF"))},
            hrp_candidate=run("hrp", {"asset_returns": returns.tail(126)}, linkage="single").to_dict(),
            risk=run("historical_var_es", {"returns": returns.tail(126).to_numpy() @ w}),
            instruction="Interpret the candidate and diagnostics; operator chooses all final weights.")
        path = self.root / "library_calls" / f"{packet['index']:02d}.json"
        if not path.exists():
            write_new(path, diagnostics)
        return diagnostics

    def score(self):
        if len(self.locked()) != 2*len(self.dates):
            raise ValueError("refuse partial-path scoring; finish both arms at every date")
        self.verify_protocol()
        from benchmarks.agent_study.trading_study import ledger
        total = pd.read_csv(self.data_dir / "total_return_close.csv", index_col=0)
        results = {}
        for arm in ("raw", "library"):
            decisions = [json.loads((self.decisions/f"{i:02d}-{arm}.json").read_text()) for i in range(len(self.dates))]
            nav, trades = ledger(total, self.picks, [d["weights"] for d in decisions])
            results[arm] = dict(return_rate_pct=100*(float(nav.iloc[-1])-1),
                end_nav=float(nav.iloc[-1]), initial_nav=1.0, total_turnover=sum(t["turnover"] for t in trades),
                n_decisions=len(decisions), end_date=str(nav.index[-1]),
                max_drawdown_pct=100*float((nav/pd.concat([pd.Series([1.0]),nav]).cummax().iloc[1:].to_numpy()-1).min()))
            nav.to_csv(self.root / f"nav-{arm}.csv")
            write_new(self.root / f"trades-{arm}.json", trades)
        results["difference_percentage_points"] = results["library"]["return_rate_pct"] - results["raw"]["return_rate_pct"]
        results["limitations"] = ["non-blind single assistant case; prior skill/result exposure", "historical archive versions unverified", "COT and House availability assumptions", "not equal token/tool budget to 7B/14B", "no causal or generalization claim"]
        write_new(self.root / "results.json", results)
        return results

    def verify_protocol(self):
        protocol = json.loads((self.root / "protocol.json").read_text())
        if protocol["dates"] != self.dates or protocol["valuation_end"] != self.sessions[-1]:
            raise ValueError("frozen decision calendar changed")
        if protocol["universe"] != list(TICKERS) or protocol["cost_bps_per_side"] != 5:
            raise ValueError("frozen universe or costs changed")
        if protocol["evidence_sha256"] != digest(self.evidence):
            raise ValueError("frozen evidence changed")
        if protocol["code_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError("code changed after protocol freeze")
        for name, expected in protocol["market_hashes"].items():
            if hashlib.sha256((self.data_dir / name).read_bytes()).hexdigest() != expected:
                raise ValueError("frozen market data changed: " + name)
        previous = None
        for n, path in enumerate(self.locked()):
            row = json.loads(path.read_text())
            i, arm = n//2, ("raw", "library")[n % 2]
            if (row["date"], row["index"], row["arm"], path.name) != (
                    self.dates[i], i, arm, f"{i:02d}-{arm}.json"):
                raise ValueError("decision order/date/arm changed")
            packet = json.loads((self.root / "packets" / f"{i:02d}.json").read_text())
            if (row["packet_sha256"] != digest(packet) or packet["date"] != self.dates[i]
                    or packet["index"] != i):
                raise ValueError("frozen packet changed")
            if any(e["eligible_date"] > self.dates[i] for e in packet["evidence"]):
                raise ValueError("future evidence in decision packet")
            if row["previous_decision_sha256"] != previous:
                raise ValueError("decision hash chain changed")
            validate_weights(row["weights"])
            previous = digest(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("normalize", "prepare", "serve"))
    p.add_argument("root", type=Path)
    p.add_argument("--data-dir", type=Path)
    a = p.parse_args()
    if a.command == "normalize":
        print(dump(normalize(a.root)))
        return
    study = ManualStudy(a.root, a.data_dir)
    if a.command == "prepare":
        print(dump(study.prepare()))
        return
    inbox = a.root / "inbox"
    inbox.mkdir(exist_ok=True)
    print("READY: explicit operator requests only", flush=True)
    for _ in range(14400):  # at most two hours, no reconnect or automatic trade
        for path in sorted(inbox.glob("request-*.json")):
            out = path.with_name(path.name.replace("request-", "response-"))
            if out.exists():
                continue
            req = json.loads(path.read_text())
            try:
                if req["action"] == "stop":
                    write_new(out, {"ok": True, "stopped": True})
                    return
                fn = {"packet": study.packet, "commit": study.commit, "library": study.library,
                      "score": study.score}[req["action"]]
                result = fn(**req.get("arguments", {}))
                write_new(out, {"ok": True, "result": result})
            except Exception as exc:
                write_new(out, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        time.sleep(0.5)


if __name__ == "__main__":
    main()
