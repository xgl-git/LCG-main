"""Query-prediction evaluation: Exact Match (EM), token F1, Accuracy."""

from __future__ import annotations

import re
import string
from typing import Any

METRIC_KEYS = ("em", "f1", "accuracy")

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text: str) -> list[str]:
    return [t for t in normalize_text(text).split(" ") if t]


def exact_match(prediction: str, reference: str) -> float:
    pred_n = normalize_text(prediction)
    ref_n = normalize_text(reference)
    return 1.0 if pred_n and pred_n == ref_n else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred_toks = _tokens(prediction)
    ref_toks = _tokens(reference)
    if not pred_toks or not ref_toks:
        return 0.0
    common: dict[str, int] = {}
    for t in ref_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in pred_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    precision = overlap / len(pred_toks)
    recall = overlap / len(ref_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def accuracy_score(prediction: str, reference: str) -> float:
    """
    Accuracy for next-query prediction.
    Exact match, or mutual containment for short queries (common AOL revisit case).
    """
    if exact_match(prediction, reference) >= 1.0:
        return 1.0
    pred_n = normalize_text(prediction)
    ref_n = normalize_text(reference)
    if not pred_n or not ref_n:
        return 0.0
    # Soft accuracy: short GT fully contained in short pred (or reverse)
    pred_len = len(pred_n.split())
    ref_len = len(ref_n.split())
    if ref_len <= 6 and ref_n in pred_n and pred_len <= 12:
        return 1.0
    if pred_len <= 6 and pred_n in ref_n and ref_len <= 12:
        return 1.0
    return 0.0


def evaluate_pair(generated: str, reference: str) -> dict[str, float]:
    if not generated.strip() or not reference.strip():
        return {k: 0.0 for k in METRIC_KEYS}
    return {
        "em": round(exact_match(generated, reference), 4),
        "f1": round(token_f1(generated, reference), 4),
        "accuracy": round(accuracy_score(generated, reference), 4),
    }


def evaluate_candidates(
    generated_texts: list[str],
    references: list[str],
) -> dict[str, Any]:
    """
    Multi-candidate / multi-reference evaluation for next-query prediction.

    For each prediction p_i and metric M:
      M_i = max_j M(p_i, g_j)

    Sample score:
      M = (1/k) * sum_i M_i
    """
    candidates = [t.strip() for t in generated_texts if t.strip()]
    refs = [r.strip() for r in references if r.strip()]
    k = len(candidates)
    empty = {
        **{key: 0.0 for key in METRIC_KEYS},
        "num_predictions": 0,
        "per_candidate": [],
    }
    if not candidates or not refs:
        return empty

    per_candidate: list[dict[str, Any]] = []
    sums = {key: 0.0 for key in METRIC_KEYS}

    for cand_idx, text in enumerate(candidates):
        pair_scores = [evaluate_pair(text, ref) for ref in refs]
        best_metrics: dict[str, float] = {}
        best_ref_by_metric: dict[str, int] = {}
        for key in METRIC_KEYS:
            best_ref_idx = max(range(len(refs)), key=lambda j: pair_scores[j][key])
            best_metrics[key] = pair_scores[best_ref_idx][key]
            best_ref_by_metric[key] = best_ref_idx
            sums[key] += best_metrics[key]

        per_candidate.append({
            "candidate_index": cand_idx,
            "text_preview": text[:200],
            "metrics": best_metrics,
            "best_reference_index": best_ref_by_metric,
        })

    aggregate = {key: round(sums[key] / k, 4) for key in METRIC_KEYS}
    return {
        **aggregate,
        "num_predictions": k,
        "per_candidate": per_candidate,
    }
