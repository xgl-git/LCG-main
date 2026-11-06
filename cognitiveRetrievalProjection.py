"""
Cognitive-to-Retrieval Projection (Module 3)

Input:  belief space B_t from Module 2 (belief.json)
Output: Cognitive Focus Distribution + Retrieval Actions + BM25 Evidence + Answers (Answer.json)

Pipeline (LCG.pdf):
  1. Cognitive Focus Selection — Retrieval Projection Agent + belief-aware ranking
  2. Cognitive Retrieval Planning — project each Focus into Retrieval Actions
  3. Evidence Retrieval — BM25 over Wikipedia (wiki.tsv)
  4. Answer Generation — evidence-grounded content from (D_F, A, E*)

Run:
  python cognitiveRetrievalProjection.py
  python cognitiveRetrievalProjection.py --input belief.json --output Answer.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common import (
    LocalLLMClient,
    COGNITIVE_FOCUS_EXTRACTION_SYSTEM,
    DEFAULT_LLM_MODEL,
    EVIDENCE_GROUNDED_ANSWER_GENERATION_SYSTEM,
    RETRIEVAL_PLANNING_SYSTEM,
    RETRIEVAL_PROJECTION_AGENT,
    SAMPLE_BEHAVIOR_RAW,
    behavior_trajectory_text,
    chat_json,
    chat_answer,
    extract_named_list,
    fill_prompt,
    get_llm_client,
    retrieval_projection_agent_system,
    to_score,
)

# Focus importance: S(F) = alpha*B(F) + beta*U(F) + gamma*I(F); S'(F) = S(F) + lambda*H(B)
FOCUS_ALPHA, FOCUS_BETA, FOCUS_GAMMA, FOCUS_LAMBDA = 0.4, 0.3, 0.3, 0.1
# Retrieval action: Score(a) = alpha*Rel(a,F) + beta*IG(a)
ACTION_ALPHA, ACTION_BETA = 0.5, 0.5
# Evidence weight: w(d) = Rel(d, q) + eta * DeltaI
EVIDENCE_ETA = 0.3

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class WikiBM25Retriever:
    """BM25 retriever over wiki.tsv: columns id, text, title."""

    def __init__(
        self,
        wiki_path: str | Path,
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
        self._load(wiki_path, max_docs=max_docs)
        self._build_index()

    def _load(self, wiki_path: str | Path, *, max_docs: int | None = None) -> None:
        with open(wiki_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                doc_id = row.get("id", "")
                title = row.get("title", "")
                text = row.get("text", "")
                body = f"{title} {text}".strip()
                self.documents.append({
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                })
                tokens = tokenize(body)
                self.doc_tokens.append(tokens)
                self.doc_len.append(len(tokens))
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
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
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
                "title": doc["title"],
                "text": doc["text"],
                "bm25_score": round(score, 4),
                "relevance": round(score / max_score, 4),
            })
        return results


@dataclass
class ProjectionResult:
    cognitive_focus_distribution: list[dict[str, Any]]
    retrieval_actions: list[dict[str, Any]]
    selected_retrieval_actions: list[dict[str, Any]]
    retrieved_evidence: list[dict[str, Any]]
    evidence_pool: list[dict[str, Any]]
    top_evidence: list[dict[str, Any]]
    answers: list[dict[str, Any]]
    logs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cognitive_focus_distribution": self.cognitive_focus_distribution,
            "retrieval_actions": self.retrieval_actions,
            "selected_retrieval_actions": self.selected_retrieval_actions,
            "retrieved_evidence": self.retrieved_evidence,
            "evidence_pool": self.evidence_pool,
            "top_evidence": self.top_evidence,
            "answers": self.answers,
            "logs": self.logs,
        }


def _node_text(node: dict[str, Any]) -> str:
    return (
        node.get("content")
        or node.get("name")
        or node.get("reason")
        or ""
    ).strip().lower()


def _focus_signature(focus: dict[str, Any]) -> str:
    parts: list[str] = []
    for node in focus.get("nodes", focus.get("Nodes", [])):
        ntype = (node.get("type") or "").lower()
        if ntype in {"goal", "hypothesis", "knowledgegap"}:
            parts.append(f"{ntype}:{_node_text(node)}")
    return "|".join(sorted(parts))


def _normalize_focus(focus: dict[str, Any], *, source_agent: str) -> dict[str, Any]:
    nodes = focus.get("Nodes", focus.get("nodes", []))
    relations = focus.get("Relations", focus.get("relations", []))
    return {
        "focus_id": focus.get("FocusID", focus.get("focus_id", "")),
        "nodes": nodes,
        "relations": relations,
        "reason": focus.get("Reason", focus.get("reason", "")),
        "source_agent": source_agent,
    }


def _focus_similarity(focus: dict[str, Any], graph: dict[str, Any]) -> float:
    focus_texts = {
        _node_text(n)
        for n in focus.get("nodes", [])
        if (n.get("type") or "").lower() in {"goal", "hypothesis", "knowledgegap"}
        and _node_text(n)
    }
    if not focus_texts:
        return 0.0

    graph_texts = {
        _node_text(n)
        for n in graph.get("nodes", [])
        if n.get("type") in {"Goal", "Hypothesis", "KnowledgeGap"}
        and _node_text(n)
    }
    if not graph_texts:
        return 0.0

    overlap = sum(
        1
        for ft in focus_texts
        if any(ft in gt or gt in ft for gt in graph_texts)
    )
    return overlap / len(focus_texts)


def _belief_support(
    focus: dict[str, Any],
    belief_entries: list[dict[str, Any]],
) -> float:
    total = 0.0
    for entry in belief_entries:
        b_i = float(entry.get("probability", entry.get("belief", 0.5)))
        graph = entry.get("graph", entry)
        total += b_i * _focus_similarity(focus, graph)
    return total


def _focus_uncertainty(focus: dict[str, Any]) -> float:
    gaps = [
        n for n in focus.get("nodes", [])
        if (n.get("type") or "").lower() in {"knowledgegap", "knowledge gap"}
    ]
    if not gaps:
        return 0.5
    values = [float(n.get("uncertainty", n.get("Uncertainty", 1.0))) for n in gaps]
    return sum(values) / len(values)


def _focus_information_gain(focus: dict[str, Any]) -> float:
    gaps = [
        n for n in focus.get("nodes", [])
        if (n.get("type") or "").lower() in {"knowledgegap", "knowledge gap"}
    ]
    if not gaps:
        return 0.5
    values = [
        to_score(
            n.get("expected_information_gain", n.get("ExpectedInformationGain")),
            0.7,
        )
        for n in gaps
    ]
    return sum(values) / len(values)


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    return [e / total for e in exps] if total > 0 else [1.0 / len(scores)] * len(scores)


def _belief_entropy(belief_space: dict[str, Any]) -> float:
    entropy = belief_space.get("entropy")
    if entropy is not None:
        return float(entropy)
    entries = belief_space.get("belief_space", [])
    probs = [float(e.get("probability", e.get("belief", 0.5))) for e in entries]
    total = sum(probs)
    if total <= 0:
        return 0.0
    probs = [p / total for p in probs]
    return -sum(p * math.log(p) for p in probs if p > 0)


def _action_relevance(action: dict[str, Any], focus: dict[str, Any]) -> float:
    action_text = " ".join([
        action.get("Query", action.get("query", "")),
        action.get("TargetEvidence", action.get("target_evidence", "")),
        action.get("Motivation", action.get("motivation", "")),
    ]).lower()

    focus_tokens: set[str] = set()
    for node in focus.get("nodes", []):
        text = _node_text(node)
        focus_tokens.update(re.findall(r"[a-z0-9]{3,}", text))

    if not focus_tokens or not action_text:
        return 0.5

    hits = sum(1 for tok in focus_tokens if tok in action_text)
    return min(1.0, hits / max(1, len(focus_tokens) * 0.3))


def _retrieve_evidence(
    actions: list[dict[str, Any]],
    retriever: WikiBM25Retriever,
    *,
    per_action_k: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """E_i = BM25(q_i, W); E = union; E* = TopK(E, w(d))."""
    retrieved_by_action: list[dict[str, Any]] = []
    weighted_pool: list[dict[str, Any]] = []

    for action in actions:
        query = action.get("query", "")
        delta_i = float(action.get("expected_information_gain", 0.5))
        docs = retriever.search(query, top_k=per_action_k)
        for doc in docs:
            weight = doc["relevance"] + EVIDENCE_ETA * delta_i
            entry = {
                **doc,
                "action_id": action.get("action_id", ""),
                "focus_id": action.get("focus_id", ""),
                "query": query,
                "target_evidence": action.get("target_evidence", ""),
                "expected_information_gain": delta_i,
                "evidence_weight": round(weight, 4),
            }
            weighted_pool.append(entry)

        retrieved_by_action.append({
            "action_id": action.get("action_id", ""),
            "focus_id": action.get("focus_id", ""),
            "query": query,
            "documents": docs,
        })

    # Deduplicate evidence pool by doc_id, keep highest weight
    best_by_doc: dict[str, dict[str, Any]] = {}
    for item in weighted_pool:
        doc_id = str(item["doc_id"])
        if doc_id not in best_by_doc or item["evidence_weight"] > best_by_doc[doc_id]["evidence_weight"]:
            best_by_doc[doc_id] = item

    evidence_pool = sorted(
        best_by_doc.values(),
        key=lambda x: x["evidence_weight"],
        reverse=True,
    )
    top_evidence = evidence_pool[:top_k]
    return retrieved_by_action, evidence_pool, top_evidence


def _news_snippet_from_doc(doc: dict[str, Any]) -> str:
    title = str(doc.get("title", "") or "").strip()
    abstract = str(doc.get("abstract", "") or doc.get("text", "") or "").strip()
    if title and abstract and abstract.lower() != title.lower():
        # Avoid duplicating title if text already starts with it
        body = abstract
        if body.lower().startswith(title.lower()):
            body = body[len(title):].lstrip(" .:-")
        return f"{title}. {body}".strip() if body else title
    return title or abstract


def _fallback_from_evidence(evidence_list: list[dict[str, Any]], query: str = "") -> str:
    if evidence_list:
        return _news_snippet_from_doc(evidence_list[0])
    q = (query or "").strip()
    if q:
        return q if q.endswith(".") else f"{q}."
    return ""


def _evidence_for_prompt(evidence_list: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Strip internal IDs so the model does not cite doc_id in generated text."""
    out: list[dict[str, str]] = []
    for item in evidence_list:
        title = str(item.get("title", "") or "").strip()
        abstract = str(item.get("abstract", "") or "").strip()
        text = str(item.get("text", "") or "").strip()
        if not abstract and text:
            # text is often "title abstract"; prefer residual as abstract
            if title and text.lower().startswith(title.lower()):
                abstract = text[len(title):].strip(" .")
            else:
                abstract = text
        out.append({"title": title, "abstract": abstract})
    return out


def _focus_node_texts(focus_entry: dict[str, Any], *types: str) -> list[str]:
    subgraph = focus_entry.get("focus_subgraph", focus_entry)
    allowed = {t.lower().replace(" ", "") for t in types} if types else None
    texts: list[str] = []
    for node in subgraph.get("nodes", []):
        ntype = (node.get("type") or "").lower().replace(" ", "")
        if allowed and ntype not in allowed:
            continue
        text = (node.get("content") or node.get("name") or "").strip()
        if text:
            texts.append(text)
    return texts


def _fallback_candidate_answer(
    action: dict[str, Any],
    focus_entry: dict[str, Any],
) -> str:
    """Deterministic fallback so every Focus always yields a candidate answer."""
    query = str(action.get("query", "")).strip().rstrip("?")
    goals = _focus_node_texts(focus_entry, "Goal")
    hypotheses = _focus_node_texts(focus_entry, "Hypothesis")
    gaps = _focus_node_texts(focus_entry, "KnowledgeGap")

    if query:
        headline = query[0].upper() + query[1:] if query else query
    elif goals:
        headline = goals[0]
    elif hypotheses:
        headline = hypotheses[0]
    else:
        headline = "Related topics you may want to explore next"

    if not headline.endswith("."):
        headline += "."

    body_parts: list[str] = []
    for part in (*hypotheses[:1], *gaps[:1], action.get("target_evidence", "")):
        part = str(part).strip()
        if part and part not in headline and part not in body_parts:
            body_parts.append(part)

    if not body_parts:
        return headline

    body = " ".join(body_parts[:2])
    if not body.endswith("."):
        body += "."
    return f"{headline} {body}"


class CognitiveRetrievalProjector:
    def __init__(
        self,
        client: LocalLLMClient | None = None,
        *,
        model: str = DEFAULT_LLM_MODEL,
        retriever: WikiBM25Retriever | None = None,
    ):
        self.client = client or get_llm_client()
        self.model = model
        self.retriever = retriever

    def project(
        self,
        belief_space: dict[str, Any],
        behavior_trajectory: str,
        *,
        per_action_k: int = 5,
        top_k: int = 20,
    ) -> ProjectionResult:
        entries = belief_space.get("belief_space", belief_space)
        if isinstance(entries, dict):
            entries = [entries]

        raw_focuses = self._extract_focuses(entries, behavior_trajectory)
        ranked_focuses = self._rank_focuses(raw_focuses, entries, belief_space)
        all_actions: list[dict[str, Any]] = []
        selected_actions: list[dict[str, Any]] = []
        planning_logs: list[dict[str, Any]] = []

        for focus_entry in ranked_focuses:
            focus = focus_entry["focus_subgraph"]
            prob = focus_entry["probability"]
            actions = self._plan_retrieval(behavior_trajectory, focus, prob)
            planning_logs.append({
                "focus_id": focus_entry["focus_id"],
                "agent": RETRIEVAL_PROJECTION_AGENT,
                "raw_actions": actions,
            })
            scored_actions: list[dict[str, Any]] = []
            for action in actions:
                ig = to_score(
                    action.get("ExpectedInformationGain", action.get("expected_information_gain")),
                    0.7,
                )
                rel = _action_relevance(action, focus)
                score = ACTION_ALPHA * rel + ACTION_BETA * ig
                enriched = {
                    "action_id": action.get("ActionID", action.get("action_id", "")),
                    "focus_id": focus_entry["focus_id"],
                    "focus_probability": prob,
                    "query": action.get("Query", action.get("query", "")),
                    "target_evidence": action.get(
                        "TargetEvidence", action.get("target_evidence", "")
                    ),
                    "expected_information_gain": round(ig, 4),
                    "motivation": action.get("Motivation", action.get("motivation", "")),
                    "agent": RETRIEVAL_PROJECTION_AGENT,
                    "relevance_score": round(rel, 4),
                    "selection_score": round(score, 4),
                }
                all_actions.append(enriched)
                if str(enriched.get("query", "")).strip():
                    scored_actions.append(enriched)

            # Keep top-2 concrete actions per focus for broader Real-Need coverage
            scored_actions.sort(key=lambda x: x["selection_score"], reverse=True)
            selected_actions.extend(scored_actions[:2])

        # Deduplicate queries and keep top-5 overall (align with typical Real Need size)
        if selected_actions:
            selected_actions.sort(key=lambda x: x.get("selection_score", 0.0), reverse=True)
            deduped: list[dict[str, Any]] = []
            seen_q: set[str] = set()
            for action in selected_actions:
                key = re.sub(r"\s+", " ", str(action.get("query", "")).lower()).strip()
                if not key or key in seen_q:
                    continue
                seen_q.add(key)
                deduped.append(action)
                if len(deduped) >= 5:
                    break
            selected_actions = deduped

        if not selected_actions:
            if all_actions:
                selected_actions = list(all_actions)
            elif ranked_focuses:
                for idx, focus_entry in enumerate(ranked_focuses):
                    selected_actions.append({
                        "action_id": f"A{idx + 1}",
                        "focus_id": focus_entry["focus_id"],
                        "focus_probability": focus_entry.get("probability", 0.0),
                        "query": "",
                        "target_evidence": "",
                        "expected_information_gain": 0.5,
                        "motivation": "",
                        "agent": RETRIEVAL_PROJECTION_AGENT,
                        "relevance_score": 0.0,
                        "selection_score": 0.0,
                    })

        retrieved_evidence: list[dict[str, Any]] = []
        evidence_pool: list[dict[str, Any]] = []
        top_evidence: list[dict[str, Any]] = []
        if self.retriever and selected_actions:
            retrieved_evidence, evidence_pool, top_evidence = _retrieve_evidence(
                selected_actions,
                self.retriever,
                per_action_k=per_action_k,
                top_k=top_k,
            )

        answers = self._generate_answers(
            behavior_trajectory,
            ranked_focuses,
            selected_actions,
            top_evidence,
        )

        return ProjectionResult(
            cognitive_focus_distribution=ranked_focuses,
            retrieval_actions=all_actions,
            selected_retrieval_actions=selected_actions,
            retrieved_evidence=retrieved_evidence,
            evidence_pool=evidence_pool,
            top_evidence=top_evidence,
            answers=answers,
            logs={
                "focus_extraction_count": len(raw_focuses),
                "planning": planning_logs,
                "retriever": "BM25",
                "answer_count": len(answers),
            },
        )

    def _extract_focuses(
        self,
        entries: list[dict[str, Any]],
        behavior: str,
    ) -> list[dict[str, Any]]:
        system = retrieval_projection_agent_system(COGNITIVE_FOCUS_EXTRACTION_SYSTEM)
        user = (
            f"Behavior Trajectory:\n{behavior}\n\n"
            f"Belief Space (Belief-aware Refined Cognitive Graphs):\n"
            f"{json.dumps(entries, ensure_ascii=False, indent=2)}"
        )
        resp = chat_json(self.client, system, user, model=self.model)
        subgraphs = extract_named_list(resp, "FocusSubgraphs", "focusSubgraphs", "items")

        focuses: list[dict[str, Any]] = []
        seen: set[str] = set()
        for sg in subgraphs:
            focus = _normalize_focus(sg, source_agent=RETRIEVAL_PROJECTION_AGENT)
            sig = _focus_signature(focus)
            if sig and sig in seen:
                continue
            if sig:
                seen.add(sig)
            focuses.append(focus)

        return focuses

    def _rank_focuses(
        self,
        focuses: list[dict[str, Any]],
        entries: list[dict[str, Any]],
        belief_space: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not focuses:
            return []

        entropy = _belief_entropy(belief_space)
        raw_scores: list[float] = []
        scored: list[dict[str, Any]] = []

        for idx, focus in enumerate(focuses):
            b_support = _belief_support(focus, entries)
            uncertainty = _focus_uncertainty(focus)
            info_gain = _focus_information_gain(focus)
            s_f = (
                FOCUS_ALPHA * b_support
                + FOCUS_BETA * uncertainty
                + FOCUS_GAMMA * info_gain
            )
            s_prime = s_f + FOCUS_LAMBDA * entropy
            raw_scores.append(s_prime)
            scored.append({
                "focus_id": focus.get("focus_id") or f"F{idx + 1}",
                "focus_subgraph": focus,
                "scores": {
                    "belief_support": round(b_support, 4),
                    "uncertainty": round(uncertainty, 4),
                    "information_gain": round(info_gain, 4),
                    "raw_score": round(s_f, 4),
                    "entropy_adjusted_score": round(s_prime, 4),
                },
            })

        probs = _softmax(raw_scores)
        for item, prob in zip(scored, probs):
            item["probability"] = round(prob, 4)

        scored.sort(key=lambda x: x["probability"], reverse=True)
        return scored

    def _plan_retrieval(
        self,
        behavior: str,
        focus: dict[str, Any],
        focus_probability: float,
    ) -> list[dict[str, Any]]:
        system = retrieval_projection_agent_system(
            fill_prompt(
                RETRIEVAL_PLANNING_SYSTEM,
                Behavior=behavior,
                Focus=json.dumps(focus, ensure_ascii=False, indent=2),
                FocusProbability=str(focus_probability),
            )
        )
        user = (
            f"User Behavior Trajectory\n{behavior}\n\n"
            f"Cognitive Focus Subgraph\n"
            f"{json.dumps(focus, ensure_ascii=False, indent=2)}\n\n"
            f"Focus Probability\n{focus_probability}\n\n"
            "Generate 1-2 Retrieval Actions.\n"
            "Each Query MUST be a concrete news information need grounded in specific "
            "people/events/topics from the browsing trajectory "
            "(e.g., tattoo removal, a named crime case, Christmas decorating, baby names).\n"
            "Do NOT output meta queries about news feeds, aggregators, algorithms, "
            "or tragic-vs-light content ratios."
        )
        resp = chat_json(self.client, system, user, model=self.model)
        return extract_named_list(resp, "RetrievalActions", "retrievalActions", "items")

    def _generate_answers(
        self,
        behavior: str,
        ranked_focuses: list[dict[str, Any]],
        selected_actions: list[dict[str, Any]],
        top_evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not selected_actions and ranked_focuses:
            selected_actions = [
                {
                    "action_id": f"A{idx + 1}",
                    "focus_id": focus_entry["focus_id"],
                    "query": "",
                    "target_evidence": "",
                }
                for idx, focus_entry in enumerate(ranked_focuses)
            ]

        focus_by_id = {f["focus_id"]: f for f in ranked_focuses}
        answers: list[dict[str, Any]] = []

        for idx, action in enumerate(selected_actions):
            focus_id = action.get("focus_id", f"F{idx + 1}")
            focus_entry = focus_by_id.get(focus_id, {"focus_id": focus_id, "focus_subgraph": {}})

            raw_evidence = [e for e in top_evidence if e.get("focus_id") == focus_id]
            if not raw_evidence:
                # Fall back to evidence retrieved by this action's query match
                raw_evidence = [
                    e for e in top_evidence
                    if e.get("action_id") == action.get("action_id")
                ]
            if not raw_evidence:
                raw_evidence = list(top_evidence[:3])
            focus_evidence = _evidence_for_prompt(raw_evidence[:5])

            user = (
                f"User Behavior Trajectory:\n{behavior}\n\n"
                f"Cognitive Focus:\n"
                f"{json.dumps(focus_entry, ensure_ascii=False, indent=2)}\n\n"
                f"Retrieval Action (Query):\n"
                f"{json.dumps(action, ensure_ascii=False, indent=2)}\n\n"
                f"Retrieved Evidence (news title/abstract):\n"
                f"{json.dumps(focus_evidence, ensure_ascii=False, indent=2)}\n\n"
                "Write ONE next-news snippet as: \"<Headline>. <short abstract>\".\n"
                "If evidence is relevant, closely paraphrase the best matching title+abstract."
            )

            answer_text = ""
            for _ in range(2):
                try:
                    answer_text = chat_answer(
                        self.client,
                        EVIDENCE_GROUNDED_ANSWER_GENERATION_SYSTEM,
                        user,
                        model=self.model,
                        max_tokens=256,
                    ).strip()
                    if answer_text:
                        break
                except Exception:
                    continue

            if not answer_text:
                answer_text = _fallback_from_evidence(raw_evidence, action.get("query", ""))
            if not answer_text:
                answer_text = _fallback_candidate_answer(action, focus_entry)
            if not answer_text:
                answer_text = behavior[:300].strip() or "Related topics you may want to explore next."

            answers.append({
                "answer_id": f"A{idx + 1}",
                "related_focus_id": focus_id,
                "query": action.get("query", ""),
                "target_evidence": action.get("target_evidence", ""),
                "answer": answer_text,
            })

        return answers


def load_input(path: str) -> tuple[dict[str, Any], str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    belief_space = data if "belief_space" in data else {"belief_space": data, "entropy": 0.0}
    behavior = behavior_trajectory_text(
        data.get("behavior_trajectory", SAMPLE_BEHAVIOR_RAW)
    )
    return belief_space, behavior


def main() -> None:
    parser = argparse.ArgumentParser(description="Cognitive-to-Retrieval Projection (Module 3)")
    parser.add_argument("--input", type=str, default="belief.json", help="Belief space JSON from Module 2")
    parser.add_argument("--output", type=str, default="Answer.json", help="Save output JSON")
    parser.add_argument("--wiki", type=str, default="Wiki/wiki.tsv", help="Wikipedia corpus for BM25 retrieval")
    parser.add_argument("--per-action-k", type=int, default=5, help="Top-k docs per retrieval action")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k docs in final evidence pool E*")
    args = parser.parse_args()

    wiki_path = Path(args.wiki)
    if not wiki_path.is_file():
        raise FileNotFoundError(f"Wiki corpus not found: {wiki_path}")

    print(f"Loading BM25 index from {wiki_path} ...")
    retriever = WikiBM25Retriever(wiki_path)
    print(f"Indexed {len(retriever.documents)} Wikipedia documents.")
    belief_space, behavior = load_input(args.input)
    result = CognitiveRetrievalProjector(retriever=retriever).project(
        belief_space,
        behavior,
        per_action_k=args.per_action_k,
        top_k=args.top_k,
    )

    text = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
