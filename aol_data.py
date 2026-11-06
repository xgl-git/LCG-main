"""AOL search-log dataset loading and next-query style helpers."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

QUERY_COLUMNS = [f"Query {i}" for i in range(1, 21)]
GT_SEPARATOR = "|||"


def _normalize_header(name: str) -> str:
    return (name or "").lstrip("\ufeff").strip()


def load_aol_dataset(path: Path) -> list[dict[str, str]]:
    """Load aol.tsv / AOL.tsv (comma-separated despite .tsv extension)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(f, dialect=dialect)
        rows: list[dict[str, str]] = []
        for raw in reader:
            row = {_normalize_header(k): (v or "") for k, v in raw.items() if k is not None}
            rows.append(row)
        return rows


def load_mind_dataset(path: Path) -> list[dict[str, str]]:
    return load_aol_dataset(path)


def parse_query_event(raw: str) -> dict[str, str] | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {"query": text, "content": "", "queryTime": ""}
    if not isinstance(obj, dict):
        return {"query": str(obj), "content": "", "queryTime": ""}
    return {
        "query": str(obj.get("query", "") or "").strip(),
        "content": str(obj.get("content", "") or "").strip(),
        "queryTime": str(obj.get("queryTime", "") or "").strip(),
    }


def parse_ground_truth(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split(GT_SEPARATOR) if p.strip()]
    return parts or [text]


def row_to_behavior_events(row: dict[str, str]) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for col in QUERY_COLUMNS:
        event = parse_query_event(row.get(col, ""))
        if event is None:
            continue
        if not event["query"] and not event["content"]:
            continue
        events.append(event)
    return events


def recent_search_queries(events: list[dict[str, str]], *, limit: int = 8) -> list[str]:
    queries: list[str] = []
    for event in events:
        q = (event.get("query") or "").strip()
        if not q or q == "(empty)":
            continue
        # Prefer the short keyword form when the stored "query" is a long description
        # that actually looks like a paraphrase of a site; keep as-is for scoring style.
        queries.append(q)
    return queries[-limit:]


def prefers_short_queries(queries: list[str]) -> bool:
    if not queries:
        return True
    lengths = [len(q.split()) for q in queries]
    avg = sum(lengths) / len(lengths)
    short_ratio = sum(1 for n in lengths if n <= 6) / len(lengths)
    return avg <= 8.0 or short_ratio >= 0.5


def row_to_behavior_text(row: dict[str, str]) -> str:
    events = row_to_behavior_events(row)
    if not events:
        raise ValueError("Sample has no search/click/browse history.")

    recent = recent_search_queries(events)
    short = prefers_short_queries(recent)
    style = (
        "SHORT keyword / URL / entity search queries (usually 1-6 tokens)"
        if short
        else "longer descriptive search intents / site descriptions"
    )

    lines = [
        "User search / click / browse behavior trajectory:",
        f"Observed query style: {style}",
        f"Recent searches (most recent last): {json.dumps(recent[-5:], ensure_ascii=False)}",
    ]
    for i, event in enumerate(events, start=1):
        lines.append(f"[{i}] queryTime: {event['queryTime'] or 'unknown'}")
        lines.append(f"    User searched: {event['query'] or '(empty)'}")
        if event["content"]:
            lines.append(f"    User browsed content after click: {event['content']}")
        else:
            lines.append("    User browsed content after click: (none)")
    lines.append(
        "Prediction target: output the NEXT content this user wants to search, "
        "matching the Observed query style as closely as possible."
    )
    return "\n".join(lines)


def style_hint_from_behavior(behavior_text: str) -> str:
    if "SHORT keyword" in behavior_text:
        return (
            "STYLE CONSTRAINT: Recent searches are SHORT. "
            "Output a short search query (1-8 tokens), e.g. a keyword, brand, URL stem, "
            "or entity name — NOT a full sentence or news paragraph."
        )
    return (
        "STYLE CONSTRAINT: Recent searches are descriptive. "
        "Output a concise description of the next search intent / website the user wants, "
        "similar in form to prior User searched fields."
    )


_DOMAIN_RE = re.compile(r"\b([a-z0-9-]+(?:\.[a-z]{2,})+)\b", re.I)
_ANSWER_KEY_RE = re.compile(r'"(?:Answer|answer)"\s*:\s*"([^"]+)"')


def _strip_answer_wrappers(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    m = _ANSWER_KEY_RE.search(text)
    if m:
        return m.group(1).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    return text.strip().strip('"').strip()


def extract_short_query_candidates(text: str) -> list[str]:
    """Derive short query-like variants from a possibly verbose prediction."""
    text = _strip_answer_wrappers(text)
    if not text:
        return []
    cands = [text]
    # First line / before em-dash
    first = re.split(r"[\n.]", text, maxsplit=1)[0].strip()
    if first and first not in cands:
        cands.append(first)
    for m in _DOMAIN_RE.findall(text.lower()):
        if m not in cands:
            cands.append(m)
        stem = m.split(".")[0]
        if stem and stem not in cands:
            cands.append(stem)
    # Quoted fragments
    for q in re.findall(r"[\"']([^\"']{2,60})[\"']", text):
        if q not in cands:
            cands.append(q)
    # Keep only reasonably short variants first
    short = [c for c in cands if 0 < len(c.split()) <= 12]
    return short or cands[:1]


def postprocess_next_query(prediction: str, behavior_text: str) -> str:
    """Normalize model output into a next-search query matching history style."""
    text = _strip_answer_wrappers(prediction)
    if not text:
        return ""
    short_pref = "SHORT keyword" in behavior_text
    if not short_pref:
        # Keep descriptive answers, but trim meta preambles.
        text = re.sub(r"^(next search|predicted query|answer)\s*[:\-]\s*", "", text, flags=re.I)
        return text.strip()

    variants = extract_short_query_candidates(text)
    # Prefer variants that overlap recent searches
    recent = re.findall(r'User searched:\s*(.+)', behavior_text)
    recent_norm = {r.strip().lower() for r in recent if r.strip() and r.strip() != "(empty)"}
    for v in variants:
        if v.lower() in recent_norm:
            return v
    # Prefer domain / very short
    for v in variants:
        if _DOMAIN_RE.fullmatch(v.lower()) or len(v.split()) <= 4:
            return v
    return variants[0] if variants else text


def history_query_candidates(behavior_text: str, *, limit: int = 3) -> list[str]:
    recent = [
        line.split("User searched:", 1)[-1].strip()
        for line in behavior_text.splitlines()
        if "User searched:" in line
    ]
    recent = [q for q in recent if q and q != "(empty)"]
    # Unique, most recent first
    seen: set[str] = set()
    out: list[str] = []
    for q in reversed(recent):
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= limit:
            break
    return out


def expand_predictions_for_eval(
    predictions: list[str],
    behavior_text: str,
) -> list[str]:
    """
    Keep a compact, high-precision candidate set:
      1) style-normalized model predictions
      2) at most one recent short historical query (AOL revisit prior)
    Avoid flooding with low-quality variants that dilute average EM.
    """
    expanded: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        key = t.lower()
        if key in seen:
            return
        seen.add(key)
        expanded.append(t)

    for pred in predictions:
        _add(postprocess_next_query(pred, behavior_text) or pred)

    # AOL users often re-issue the latest short query.
    for hq in history_query_candidates(behavior_text, limit=1):
        if "SHORT keyword" in behavior_text and len(hq.split()) <= 8:
            _add(hq)

    return expanded or [p for p in predictions if p.strip()]


def extract_answer_texts(answers: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for a in answers:
        if not isinstance(a, dict):
            continue
        text = str(a.get("answer", "")).strip()
        if text:
            texts.append(text)
    return texts
