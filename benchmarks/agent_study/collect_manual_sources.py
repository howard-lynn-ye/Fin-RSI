"""Bounded public-archive acquisition for the manual multi-source case study.

Run on a compute node. This collects evidence; it never generates a trade.
Download timestamps are current, not backdated historical observations.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

from bs4 import BeautifulSoup


SCHWAB = "https://pressroom.aboutschwab.com/press-releases/press-release/"
SURVEYS = [
    ("2024", "Schwab-Survey-Bullishness-Among-Traders-Reaches-Highest-Level-in-Two-Years"),
    ("2025", "Schwab-Survey-Two-Thirds-of-Traders-Feel-the-Market-Is-Overvalued-but-Sentiment-for-the-Quarter-Ahead-Remains-Bullish"),
    ("2025", "Charles-Schwab-Announces-Q2-Trader-Sentiment-Survey-Findings"),
    ("2025", "Bullish-Sentiment-Rebounds-Even-as-More-Than-Half-of-Traders-Believe-the-Market-is-Over-Valued"),
    ("2025", "Traders-Remain-Cautiously-Bullish-with-Concerns-About-the-Political-Landscape-a-Potential-Market-Correction-and-Stagflation"),
    ("2026", "Trader-Bullishness-Ticks-Downward-as-Younger-Traders-Take-a-More-Cautious-Stance"),
    ("2026", "Schwab-Q2-Retail-Client-Sentiment-Report-Investors-Turn-Bearish-on-U-S--Stock-Market-but-Remain-Confident-in-their-Investing-Approach-and-Likelihood-of-Reaching-Goals"),
    ("2026", "Schwab-Q3-Retail-Client-Sentiment-Report-Investors-Turn-Bullish-on-U-S--Stock-Market-as-Confidence-Climbs-Despite-Concerns-the-Market-May-be-Overvalued"),
]
# A declared convenience sample, not all Congress or a performance-selected index.
ACTORS = {("Pelosi", "Nancy"), ("Khanna", "Rohit"), ("McCaul", "Michael T.")}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class Archive:
    def __init__(self, root):
        self.root = Path(root)
        self.raw = self.root / "sources"
        self.raw.mkdir(parents=True, exist_ok=True)

    def fetch(self, url, kind, **details):
        """One attempt, cached immutable success, bounded size, explicit failures."""
        key = hashlib.sha256(url.encode()).hexdigest()
        body_path, meta_path = self.raw / (key + ".bin"), self.raw / (key + ".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("sha256"):
                data = body_path.read_bytes()
                if hashlib.sha256(data).hexdigest() != meta["sha256"]:
                    raise ValueError("cached source hash mismatch")
                return data, meta
            return None, meta
        meta = dict(id=key, url=url, category=kind, observed_at=utcnow(), **details)
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "fin-skills public research archive verification/0.1"})
            with urllib.request.urlopen(req, timeout=25) as response:
                data = response.read(25_000_001)
                meta.update(status=response.status, final_url=response.url,
                            content_type=response.headers.get("Content-Type"))
            if len(data) > 25_000_000:
                raise ValueError("source exceeds 25 MB acquisition limit")
            body_path.write_bytes(data)
            meta.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            if "html" in meta.get("content_type", ""):
                soup = BeautifulSoup(data, "html.parser")
                for el in soup(["script", "style", "nav", "header", "footer"]):
                    el.decompose()
                main = (soup.select_one(".module-news-details") or soup.select_one("#article")
                        or soup.select_one("main") or soup)
                (self.raw / (key + ".txt")).write_text(main.get_text("\n", strip=True))
            elif data.startswith(b"%PDF"):
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(data))
                meta["pages"] = len(reader.pages)
                (self.raw / (key + ".txt")).write_text("\n\n".join(
                    f"PAGE {i+1}\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages)))
        except Exception as exc:
            data = None
            meta["error"] = f"{type(exc).__name__}: {exc}"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(json.dumps({k: meta[k] for k in ("id", "category", "url", "bytes", "error")
                          if k in meta}), flush=True)
        time.sleep(0.5)
        return data, meta


def collect(root):
    archive = Archive(root)
    calendar = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    data, _ = archive.fetch(calendar, "index")
    if data:
        links = BeautifulSoup(data, "html.parser").select("a[href]")
        urls = {urllib.parse.urljoin(calendar, a["href"]) for a in links
                if re.search(r"/pressreleases/monetary202[456]\d{4}a\.htm$", a["href"])}
        for url in sorted(urls):
            day = re.search(r"monetary(\d{8})", url)[1]
            if "20241001" <= day <= "20260922":
                archive.fetch(url, "news", series="fomc", date_hint=day)
    for year, slug in SURVEYS:
        archive.fetch(SCHWAB + year + "/" + slug + "/default.aspx", "psychology",
                      series="schwab-survey", sample="Schwab clients, not all investors")
    # Preserve the initially-undated report without assigning it a fictitious release date.
    archive.fetch("https://www.aboutschwab.com/schwab-trader-client-sentiment-survey-q4-2024",
                  "unverified_publication", series="schwab-survey")
    base = "https://www.cftc.gov/"
    archive.fetch(base + "MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm",
                  "release_calendar")
    archive.fetch(base + "PressRoom/PressReleases/9147-25", "release_calendar")
    for year in (2024, 2025, 2026):
        for stem in ("fut_fin_txt_", "fut_disagg_txt_"):
            archive.fetch(base + f"files/dea/history/{stem}{year}.zip", "behavior",
                          series="cftc", year=year)
    for year in (2023, 2024, 2025, 2026):
        url = f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP"
        data, _ = archive.fetch(url, "house_index", year=year)
        if not data:
            continue
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            info = z.getinfo(f"{year}FD.xml")
            if info.file_size > 20_000_000:
                raise ValueError("oversize House XML")
            body = z.read(info)
        if b"<!ENTITY" in body.upper() or b"<!DOCTYPE" in body.upper():
            raise ValueError("unsafe XML")
        rows = [{e.tag: e.text or "" for e in el} for el in ElementTree.fromstring(body)]
        for row in rows:
            if (row.get("Last"), row.get("First")) not in ACTORS:
                continue
            kind = row.get("FilingType")
            if kind not in ("P", "O", "A"):
                continue
            filed = datetime.strptime(row["FilingDate"], "%m/%d/%Y").date().isoformat()
            if not "2024-01-01" <= filed <= "2026-09-22":
                continue
            doc = row["DocID"]
            if not doc.isdigit():
                raise ValueError("invalid document ID")
            folder = "ptr-pdfs" if kind == "P" else "financial-pdfs"
            url = f"https://disclosures-clerk.house.gov/public_disc/{folder}/{year}/{doc}.pdf"
            archive.fetch(url, "officials", series="house", filing_date=filed,
                          index_record=row, index_url=archive.raw.name + "/" +
                          hashlib.sha256((f"https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP").encode()).hexdigest(),
                          scope="selected three representatives and disclosed household interests")
    metas = [json.loads(p.read_text()) for p in sorted(archive.raw.glob("*.json"))]
    (Path(root) / "acquisition_manifest.json").write_text(json.dumps(metas, indent=2))
    print(json.dumps({"files": len(metas), "errors": [m for m in metas if m.get("error")]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    collect(parser.parse_args().root)
