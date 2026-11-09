"""Outcome-independent public enrollment rules; unresolved origins retain one dependency block."""
from collections import defaultdict
import hashlib
import json
import re
import unicodedata

SALT = "fin-skills-public-external-20261108-v1"
LEXICAL_UNITS = {
    "percent-or-bp": ("percent", "percentage", "%", "percentage point", "basis point", "bps"),
    "currency-or-scale": ("$", "USD", "dollar", "euro", "GBP", "JPY", "million", "billion", "thousand"),
    "ratio": ("ratio", "proportion", "fraction", "multiple"),
    "count": ("how many", "number of", "shares", "employees", "units", "barrels"),
}


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).replace("\r\n", "\n").split())


def context_key(text, kind):
    if kind == "json":
        value = json.loads(text)
        def cleaned(item):
            if isinstance(item, dict):
                pairs = [(normalize(k), cleaned(v)) for k, v in item.items()]
                if len({k for k, _ in pairs}) != len(pairs):
                    raise ValueError("normalization merges distinct JSON keys")
                return dict(pairs)
            if isinstance(item, list):
                return [cleaned(v) for v in item]
            return normalize(item) if isinstance(item, str) else item
        text = json.dumps(cleaned(value), ensure_ascii=False, sort_keys=True, allow_nan=False)
    else:
        text = normalize(text)
    return hashlib.sha256(text.encode()).hexdigest()


def template_key(question):
    # No entity-name substitutions without verified entity-name metadata.
    value = normalize(question).lower()
    value = re.sub(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", "<NUM>", value)
    return hashlib.sha256(value.encode()).hexdigest()


def unit_tags(question, context):
    def hits(text):
        normalized = unicodedata.normalize("NFKC", text).lower()
        out = {}
        for category, phrases in LEXICAL_UNITS.items():
            spans = []
            for phrase in phrases:
                expression = re.escape(phrase.lower())
                if phrase[0].isalnum():
                    expression = r"(?<!\w)" + expression
                if phrase[-1].isalnum():
                    expression += r"(?!\w)"
                spans.extend(dict(rule=phrase, start=m.start(), end=m.end())
                             for m in re.finditer(expression, normalized))
            if spans:
                out[category] = spans
        return out
    qhits, chits = hits(question), hits(context)
    tags = ["question-explicit-" + k for k in qhits]
    if len(qhits) > 1 or (not qhits and len(chits) > 1):
        tags.append("multiple-or-ambiguous")
    if not qhits and chits:
        tags.append("context-only-unit")
    if not qhits and not chits:
        tags.append("unspecified")
    return dict(tags=sorted(tags), question_hits=qhits, context_hits=chits,
                rule_version="public-lexical-v1", semantic_unit_truth=False)


def rank(stage, source, key):
    return hashlib.sha256("\0".join((SALT, stage, source, key)).encode()).hexdigest()


def select(rows, cap=200):
    """Only pre-qualified rows; labels, programs and model outcomes are never accepted."""
    allowed = {"row_index", "source_task", "sampling_group", "stable_key"}
    if any(set(row) != allowed for row in rows):
        raise ValueError("selection accepts public enrollment keys only")
    if len({row["row_index"] for row in rows}) != len(rows):
        raise ValueError("duplicate source row coordinates")
    if type(cap) is not int or cap < 1:
        raise ValueError("positive fixed group cap required")
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["source_task"]][row["sampling_group"]].append(row)
    result = []
    for source, universe in sorted(groups.items()):
        chosen = sorted(universe, key=lambda k: (rank("group", source, k),
                         min(r["row_index"] for r in universe[k])))[:cap]
        for group in chosen:
            result.append(min(universe[group], key=lambda r: (
                rank("question", source, r["stable_key"]), r["row_index"])))
    return result


def source_variants(record, source):
    """Lossless explicit source representations, not fuzzy matching or answer matching."""
    variants = set()
    if source == "CodeFinQA":
        before, after = record.get("pre_text", []), record.get("post_text", [])
        for key in ("table", "table_ori"):
            table = record.get(key, [])
            if not table or any(not isinstance(row, list) for row in table):
                continue
            for separator in (" ", " | ", "\t"):
                rows = [separator.join(str(cell) for cell in row) for row in table]
                variants.add(context_key("\n".join(before + rows + after), "string"))
    else:
        table = record.get("table", {}).get("table", [])
        if len(table) < 2 or not table[0] or any(len(row) != len(table[0]) for row in table):
            return variants
        columns, labels = table[0][1:], [row[0] for row in table[1:]]
        # Duplicate labels cannot be put into a dict without silently dropping cells.
        if (not all(isinstance(x, str) and x for x in columns + labels)
                or len(set(columns)) != len(columns) or len(set(labels)) != len(labels)):
            return variants
        # Distinct raw labels may collapse under the declared NFKC/whitespace
        # normalization. Such a table is not a lossless mapping representation;
        # leave its origin unresolved rather than drop cells or abort enrollment.
        if (len({normalize(x) for x in columns}) != len(columns)
                or len({normalize(x) for x in labels}) != len(labels)):
            return variants
        by_row = {row[0]: dict(zip(columns, row[1:])) for row in table[1:]}
        by_column = {column: {row[0]: row[i+1] for row in table[1:]}
                     for i, column in enumerate(columns)}
        for value in (by_row, by_column):
            variants.add(context_key(json.dumps(value, ensure_ascii=False), "json"))
    return variants


def origin_index(sources):
    """sources yields (source_task, source_file_id, records); never reads qa answers/programs."""
    index = defaultdict(list)
    for source, file_id, records in sources:
        for position, record in enumerate(records):
            questions = ([record.get("qa", {})] if source == "CodeFinQA" else record.get("questions", []))
            variants = source_variants(record, source)
            for question in questions:
                text = question.get("question")
                if not isinstance(text, str):
                    continue
                index[(source, normalize(text))].append(dict(
                    source_file=file_id, source_record=position, context_keys=variants,
                    document=record.get("filename") if source == "CodeFinQA" else None,
                    table_uid=record.get("table", {}).get("uid") if source != "CodeFinQA" else None))
    return index


def map_origin(source, question, context_hash, index):
    candidates = [item for item in index.get((source, normalize(question)), [])
                  if context_hash in item["context_keys"]]
    # Only an explicit filing path supports a document/issuer label. A table UID does not.
    matched = [re.fullmatch(r"([^/]+)/([12]\d{3})/[^/]+", item["document"] or "")
               for item in candidates]
    if matched and all(matched):
        filings = {(item.group(1), item.group(2)) for item in matched}
        if len(filings) == 1:
            issuer, year = next(iter(filings))
            return dict(status="verified_source_document", sampling_group=f"filing:{issuer}/{year}",
                dependency_block="issuer:" + issuer, issuer=issuer,
                evidence=[{k:v for k,v in item.items() if k != "context_keys"} for item in candidates])
    return dict(status="unresolved_context_group", sampling_group="context:" + context_hash,
        dependency_block="unresolved", issuer=None,
        evidence=[{k:v for k,v in item.items() if k != "context_keys"} for item in candidates])
