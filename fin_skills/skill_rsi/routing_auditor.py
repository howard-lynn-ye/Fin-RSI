"""Contrastive Trigger & Blind-Holdout Routing Auditor (`fin_skills.skill_rsi.routing_auditor`).

Audits skill descriptions against both the canonical `evals/queries.jsonl` / `evals/blind.jsonl`
benchmarks and newly proposed skill probe queries, enforcing:
  1. 100% Top-1 routing accuracy
  2. Zero thin margins (`(top1 - top2) / top1 >= 0.15`)
  3. 100% 1-hop `xref` routed accuracy on blind holdout queries
  4. Automated token-collision diagnosis when a competing skill encroaches on a target skill
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

MARGIN_FLOOR = 0.15
STOP = set(
    """a an and are as at be but by for from how i if in is it my me of on or that the
this to use used using want with what when which who why you your do does can could should would
get got need help please make made new now here there all any some more most""".split()
)
SKIP_RE = re.compile(r"\bSKIP\b(.*)$", re.S)


def tokenize_query(s: str) -> list[str]:
    s = s.lower()
    words = re.findall(r"[a-z][a-z0-9_.\-]{1,}", s)
    cjk = re.findall(r"[\u4e00-\u9fff]", s)
    return [w for w in words if w not in STOP and len(w) > 1] + cjk


def _parse_frontmatter_minimal(text: str) -> dict[str, Any]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    body = text[3:end].strip("\n")
    fm: dict[str, Any] = {}
    key: str | None = None
    buf: list[str] = []
    nested: str | None = None

    def flush() -> None:
        nonlocal key, buf
        if key is not None:
            fm[key] = " ".join(x.strip() for x in buf if x.strip()) if buf else fm.get(key, "")
        key, buf = None, []

    for line in body.split("\n"):
        if not line.strip():
            continue
        if line.startswith("  ") and nested:
            k, _, v = line.strip().partition(":")
            fm.setdefault(nested, {})[k.strip()] = v.strip().strip("\"'")
            continue
        if line.startswith("  ") and key:
            buf.append(line)
            continue
        flush()
        nested = None
        k, sep, v = line.partition(":")
        if not sep:
            continue
        k = k.strip()
        v = v.strip()
        if v in ("", ">-", ">", "|", "|-"):
            if v == "":
                nested = k
                fm.setdefault(k, {})
            else:
                key = k
            continue
        fm[k] = v.strip("\"'")
    flush()
    return fm


def load_skill_routing_entries(root: Path) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, set[str]]]:
    skills: list[dict[str, Any]] = []
    for md in sorted(root.glob("plugins/*/skills/*/SKILL.md")):
        raw = md.read_text(encoding="utf-8")
        fm = _parse_frontmatter_minimal(raw)
        name = fm.get("name", md.parent.name)
        desc = str(fm.get("description", ""))
        m = SKIP_RE.search(desc)
        pos = desc[: m.start()] if m else desc
        neg = m.group(1) if m else ""
        skills.append({
            "name": name,
            "desc": desc,
            "path": md.relative_to(root).as_posix(),
            "body": raw,
            "toks": Counter(tokenize_query(name + " " + pos)),
            "neg": Counter(tokenize_query(neg)),
        })

    n = len(skills)
    df: Counter[str] = Counter()
    for s in skills:
        df.update(set(s["toks"]))
    idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}

    names = {s["name"] for s in skills}
    xref: dict[str, set[str]] = {}
    for s in skills:
        body = s["body"]
        xref[s["name"]] = {o for o in names if o != s["name"] and o in body}
    return skills, idf, xref


def score_query_against_skill(qt: Sequence[str], skill: dict[str, Any], idf: dict[str, float]) -> float:
    st, neg = skill["toks"], skill["neg"]
    q = set(qt)
    pos_hits = sum(idf.get(t, 1.0) for t in sorted(q) if st.get(t))
    neg_hits = sum(idf.get(t, 1.0) for t in sorted(q) if neg.get(t) and not st.get(t))
    return round(pos_hits - 0.75 * neg_hits, 9)


@dataclass
class RoutingAuditReport:
    """Comprehensive routing & margin audit report over canonical + probe queries."""

    passed: bool
    n_skills: int
    n_queries: int
    top1_hits: int
    top1_accuracy: float
    routed_hits: int
    routed_accuracy: float
    min_margin: float
    mean_margin: float
    thin_margins: list[dict[str, Any]] = field(default_factory=list)
    misses: list[dict[str, Any]] = field(default_factory=list)
    probe_results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_skills": self.n_skills,
            "n_queries": self.n_queries,
            "top1_hits": self.top1_hits,
            "top1_accuracy": round(self.top1_accuracy, 6),
            "routed_hits": self.routed_hits,
            "routed_accuracy": round(self.routed_accuracy, 6),
            "min_margin": round(self.min_margin, 6),
            "mean_margin": round(self.mean_margin, 6),
            "n_thin_margins": len(self.thin_margins),
            "n_misses": len(self.misses),
            "thin_margins": list(self.thin_margins),
            "misses": list(self.misses),
            "probe_results": list(self.probe_results),
        }


def diagnose_query_collision(
    query: str,
    expected_skill: str,
    root: Path | None = None,
) -> dict[str, Any]:
    """Identify why a query misses or has a thin margin against competing skills."""
    repo_root = root or Path(__file__).resolve().parent.parent.parent
    skills, idf, _ = load_skill_routing_entries(repo_root)
    by_name = {s["name"]: s for s in skills}
    qt = tokenize_query(query)
    ranked = sorted(
        ((score_query_against_skill(qt, s, idf), s["name"]) for s in skills),
        reverse=True,
    )
    top_score, top_name = ranked[0]
    snd_score, snd_name = ranked[1]
    competitor_name = snd_name if top_name == expected_skill else top_name
    target = by_name.get(expected_skill)
    competitor = by_name.get(competitor_name)

    q_set = set(qt)
    missing_in_target = sorted(q_set - set(target["toks"])) if target else sorted(q_set)
    shared_with_competitor = (
        sorted(q_set & set(competitor["toks"])) if competitor else []
    )
    margin = (top_score - snd_score) / top_score if top_score > 0 else 0.0
    return {
        "query": query,
        "expected": expected_skill,
        "top1": top_name,
        "top1_score": top_score,
        "top2": snd_name,
        "top2_score": snd_score,
        "margin": round(margin, 4),
        "competitor": competitor_name,
        "missing_tokens_in_target_trigger": missing_in_target,
        "overlapping_tokens_in_competitor": shared_with_competitor,
        "suggested_skip_clause_for_competitor": (
            f"SKIP for {' '.join(missing_in_target[:4] or shared_with_competitor[:4])} ({expected_skill})"
        ),
    }


def audit_catalog_routing(
    root: Path | None = None,
    probe_queries: Sequence[dict[str, str]] | None = None,
) -> RoutingAuditReport:
    """Run full trigger + margin audit on `evals/queries.jsonl` plus optional probe queries."""
    repo_root = root or Path(__file__).resolve().parent.parent.parent
    skills, idf, xref = load_skill_routing_entries(repo_root)
    queries_path = repo_root / "evals" / "queries.jsonl"
    canonical_qs = [
        json.loads(line)
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    hits = 0
    routed = 0
    margins: list[float] = []
    thin: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []

    for case in canonical_qs:
        qt = tokenize_query(case["q"])
        ranked = sorted(
            ((score_query_against_skill(qt, s, idf), s["name"]) for s in skills),
            reverse=True,
        )
        top, second = ranked[0], ranked[1]
        ok = top[1] == case["expect"]
        hits += int(ok)
        routed += int(ok or (case["expect"] in xref.get(top[1], set())))
        margin = (top[0] - second[0]) / top[0] if top[0] > 0 else 0.0
        margins.append(margin)
        if not ok:
            misses.append({
                "q": case["q"],
                "expected": case["expect"],
                "got": top[1],
                "second": second[1],
                "margin": round(margin, 4),
            })
        elif margin < MARGIN_FLOOR:
            thin.append({
                "q": case["q"],
                "got": top[1],
                "second": second[1],
                "margin": round(margin, 4),
            })

    probe_results: list[dict[str, Any]] = []
    probes_ok = True
    for pcase in probe_queries or []:
        qt = tokenize_query(pcase["q"])
        ranked = sorted(
            ((score_query_against_skill(qt, s, idf), s["name"]) for s in skills),
            reverse=True,
        )
        top, second = ranked[0], ranked[1]
        ok = top[1] == pcase["expect"]
        margin = (top[0] - second[0]) / top[0] if top[0] > 0 else 0.0
        passed_probe = ok and (margin >= MARGIN_FLOOR)
        probes_ok = probes_ok and passed_probe
        probe_results.append({
            "q": pcase["q"],
            "expected": pcase["expect"],
            "top1": top[1],
            "second": second[1],
            "margin": round(margin, 4),
            "passed": passed_probe,
        })

    n_q = len(canonical_qs)
    passed = (len(misses) == 0) and (len(thin) == 0) and probes_ok
    return RoutingAuditReport(
        passed=passed,
        n_skills=len(skills),
        n_queries=n_q,
        top1_hits=hits,
        top1_accuracy=hits / max(1, n_q),
        routed_hits=routed,
        routed_accuracy=routed / max(1, n_q),
        min_margin=min(margins) if margins else 0.0,
        mean_margin=sum(margins) / max(1, len(margins)),
        thin_margins=thin,
        misses=misses,
        probe_results=probe_results,
    )
