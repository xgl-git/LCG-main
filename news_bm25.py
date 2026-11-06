"""BM25 retriever over MIND corpus.jsonl (title + abstract)."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class NewsBM25Retriever:
    """BM25 over news corpus; indexed text = title + abstract."""

    def __init__(
        self,
        corpus_path: str | Path,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        max_docs: int | None = None,
    ):
        self.k1 = k1
        self.b = b
        self.documents: list[dict[str, Any]] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avg_dl = 0.0
        self.df: Counter[str] = Counter()
        self.inverted: dict[str, list[tuple[int, int]]] = {}
        self._id_to_idx: dict[str, int] = {}
        self._load(corpus_path, max_docs=max_docs)
        self._build_index()

    def _load(self, corpus_path: str | Path, *, max_docs: int | None) -> None:
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                news_id = str(obj.get("news_id", "") or "").strip()
                title = str(obj.get("title", "") or "").strip()
                abstract = str(obj.get("abstract", "") or "").strip()
                body = f"{title} {abstract}".strip()
                self.documents.append({
                    "doc_id": news_id,
                    "news_id": news_id,
                    "title": title,
                    "abstract": abstract,
                    "text": body,
                })
                tokens = tokenize(body)
                self.doc_tokens.append(tokens)
                self.doc_len.append(len(tokens))
                if news_id:
                    self._id_to_idx[news_id] = len(self.documents) - 1
                if max_docs is not None and len(self.documents) >= max_docs:
                    break
        n = len(self.doc_tokens)
        self.avg_dl = sum(self.doc_len) / n if n else 0.0

    def _build_index(self) -> None:
        for doc_idx, tokens in enumerate(self.doc_tokens):
            tf = Counter(tokens)
            for term, freq in tf.items():
                self.df[term] += 1
                self.inverted.setdefault(term, []).append((doc_idx, freq))
        self._n_docs = len(self.doc_tokens)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self._n_docs - df + 0.5) / (df + 0.5))

    def get_by_id(self, news_id: str) -> dict[str, Any] | None:
        idx = self._id_to_idx.get(news_id)
        if idx is None:
            return None
        return self.documents[idx]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.documents:
            return []

        scores: dict[int, float] = {}
        for term in q_tokens:
            if term not in self.inverted:
                continue
            idf = self._idf(term)
            for doc_idx, tf in self.inverted[term]:
                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avg_dl, 1e-9))
                score = idf * (tf * (self.k1 + 1)) / denom
                scores[doc_idx] = scores.get(doc_idx, 0.0) + score

        if not scores:
            return []

        max_score = max(scores.values()) or 1.0
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[dict[str, Any]] = []
        for doc_idx, score in ranked:
            doc = self.documents[doc_idx]
            results.append({
                "doc_id": doc["doc_id"],
                "news_id": doc["news_id"],
                "title": doc["title"],
                "abstract": doc["abstract"],
                "text": doc["text"],
                "bm25_score": round(score, 4),
                "relevance": round(score / max_score, 4),
            })
        return results

    def search_multi(
        self,
        queries: list[str],
        top_k: int = 20,
        per_query_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve with multiple queries; merge by max BM25 score, return top_k."""
        per_query_k = per_query_k or top_k
        best: dict[str, dict[str, Any]] = {}
        for query in queries:
            q = (query or "").strip()
            if not q:
                continue
            for hit in self.search(q, top_k=per_query_k):
                nid = hit["news_id"]
                prev = best.get(nid)
                if prev is None or hit["bm25_score"] > prev["bm25_score"]:
                    best[nid] = hit
        ranked = sorted(best.values(), key=lambda x: x["bm25_score"], reverse=True)[:top_k]
        if not ranked:
            return []
        max_score = ranked[0]["bm25_score"] or 1.0
        for item in ranked:
            item["relevance"] = round(item["bm25_score"] / max_score, 4)
        return ranked
