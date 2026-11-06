"""MIND behaviors / test / corpus loading for LCG evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

NEWS_COLUMNS = [f"News {i}" for i in range(1, 21)]


def _normalize_header(name: str) -> str:
    return (name or "").lstrip("\ufeff").strip()


def _parse_json_field(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_news_cell(raw: str) -> dict[str, str] | None:
    obj = _parse_json_field(raw)
    if not isinstance(obj, dict):
        return None
    news_id = str(obj.get("news_id", "") or "").strip()
    title = str(obj.get("title", "") or "").strip()
    abstract = str(obj.get("abstract", "") or "").strip()
    if not news_id and not title and not abstract:
        return None
    return {"news_id": news_id, "title": title, "abstract": abstract}


def news_to_text(news: dict[str, str]) -> str:
    title = (news.get("title") or "").strip()
    abstract = (news.get("abstract") or "").strip()
    if title and abstract:
        return f"{title}. {abstract}"
    return title or abstract


def load_behaviors(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(f, dialect=dialect)
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row = {_normalize_header(k): (v or "") for k, v in raw.items() if k is not None}
            bid = str(row.get("behaviors_id", "")).strip()
            history: list[dict[str, str]] = []
            for col in NEWS_COLUMNS:
                item = parse_news_cell(row.get(col, ""))
                if item:
                    history.append(item)

            gt_raw = _parse_json_field(row.get("Ground Truth", ""))
            ground_truths: list[dict[str, str]] = []
            if isinstance(gt_raw, list):
                for item in gt_raw:
                    if isinstance(item, dict):
                        ground_truths.append({
                            "news_id": str(item.get("news_id", "") or "").strip(),
                            "title": str(item.get("title", "") or "").strip(),
                            "abstract": str(item.get("abstract", "") or "").strip(),
                        })

            need_raw = _parse_json_field(row.get("Real Need", ""))
            real_needs: list[str] = []
            if isinstance(need_raw, list):
                for item in need_raw:
                    if isinstance(item, dict):
                        q = str(item.get("query", "") or "").strip()
                        if q:
                            real_needs.append(q)
                    elif isinstance(item, str) and item.strip():
                        real_needs.append(item.strip())

            rows.append({
                "behaviors_id": bid,
                "history": history,
                "ground_truths": ground_truths,
                "real_needs": real_needs,
            })
        return rows


def load_test_relevance(path: Path) -> dict[str, dict[str, float]]:
    """behaviors_id -> {news_id: relevance_score}."""
    rel: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(f, dialect=dialect)
        for raw in reader:
            row = {_normalize_header(k): (v or "") for k, v in raw.items() if k is not None}
            bid = str(row.get("behaviors_id", "")).strip()
            nid = str(row.get("news_id", "")).strip()
            if not bid or not nid:
                continue
            try:
                score = float(row.get("score", "1") or 1)
            except ValueError:
                score = 1.0
            rel.setdefault(bid, {})[nid] = score
    return rel


def load_corpus(path: Path) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append({
                "news_id": str(obj.get("news_id", "") or "").strip(),
                "title": str(obj.get("title", "") or "").strip(),
                "abstract": str(obj.get("abstract", "") or "").strip(),
            })
    return docs


def row_to_behavior_text(history: list[dict[str, str]]) -> str:
    lines = ["News browsing behavior trajectory:"]
    for i, news in enumerate(history, start=1):
        text = news_to_text(news)
        nid = news.get("news_id", "")
        lines.append(f"[{i}] User read news ({nid}): {text}")
    if len(lines) == 1:
        raise ValueError("Sample has no news browsing history.")
    return "\n".join(lines)
