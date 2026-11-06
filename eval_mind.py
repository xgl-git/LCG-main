"""
MIND evaluation for LCG:
  1) INA = ROUGE-L + LLMScore  (predicted retrieval intents vs Real Need)
     NOTE: BGEScore is disabled — no HuggingFace / BGE model download.
  2) RC  = Recall@K / Precision@K / nDCG@K (retrieved news vs test.tsv relevance)
  3) GQ  = BLEU / ROUGE-1 / ROUGE-2 / ROUGE-L (generated answers vs Ground Truth news)
"""

from __future__ import annotations

import math
import re
from typing import Any

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from common import DEFAULT_LLM_MODEL, LocalLLMClient, get_llm_client

INA_KEYS = ("rougeL", "llm_score", "ina")
RC_KEYS = ("recall", "precision", "ndcg")
GQ_KEYS = ("bleu", "rouge1", "rouge2", "rougeL")

LLM_JUDGE_SYSTEM = """You are an expert evaluator of user information-need alignment.

Given:
1) a PREDICTED retrieval intent / search query produced by a model
2) a GOLD Real Need query reflecting the user's true exploration intent

Score how well the predicted intent captures the gold information need.

Consider:
- topical overlap
- whether the prediction would lead to retrieving similar information
- cognitive / exploratory intent consistency

Output ONLY compact JSON:
{"Score": 0.xx, "Reason": "short justification"}
where Score is in [0, 1].
"""


def _tokens(text: str) -> list[str]:
    return word_tokenize((text or "").lower())


def _lcs_length(x: list[str], y: list[str]) -> int:
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_n(hypothesis: str, reference: str, n: int) -> float:
    hyp = _tokens(hypothesis)
    ref = _tokens(reference)
    if len(hyp) < n or len(ref) < n:
        return 0.0
    hyp_ngrams = [tuple(hyp[i : i + n]) for i in range(len(hyp) - n + 1)]
    ref_ngrams = [tuple(ref[i : i + n]) for i in range(len(ref) - n + 1)]
    if not hyp_ngrams or not ref_ngrams:
        return 0.0
    ref_set = set(ref_ngrams)
    overlap = sum(1 for gram in hyp_ngrams if gram in ref_set)
    precision = overlap / len(hyp_ngrams)
    recall = overlap / len(ref_ngrams)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l(hypothesis: str, reference: str) -> float:
    hyp = _tokens(hypothesis)
    ref = _tokens(reference)
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_length(hyp, ref)
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu_score(hypothesis: str, reference: str) -> float:
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    hyp = _tokens(hypothesis)
    ref = [_tokens(reference)]
    smoothing = SmoothingFunction().method1
    return float(sentence_bleu(ref, hyp, smoothing_function=smoothing))


def llm_need_score(
    predicted: str,
    gold: str,
    *,
    client: LocalLLMClient | None = None,
    model: str = DEFAULT_LLM_MODEL,
) -> float:
    if not predicted.strip() or not gold.strip():
        return 0.0
    client = client or get_llm_client()
    user = (
        f"PREDICTED intent:\n{predicted}\n\n"
        f"GOLD Real Need:\n{gold}\n\n"
        "Return JSON with Score in [0,1]."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        content = resp.choices[0].message.content or ""
        match = re.search(r'"Score"\s*:\s*([0-9]*\.?[0-9]+)', content, re.I)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        match = re.search(r"([01](?:\.\d+)?)", content)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
    except Exception:
        return 0.0
    return 0.0


def evaluate_ina(
    predicted_queries: list[str],
    real_needs: list[str],
    *,
    client: LocalLLMClient | None = None,
    model: str = DEFAULT_LLM_MODEL,
    use_llm_judge: bool = True,
) -> dict[str, Any]:
    """
    INA without BGEScore (no HuggingFace download).

    For each predicted query:
      match gold Real Need by max ROUGE-L
      then INA_i = ROUGE-L_i + LLMScore_i
    Sample INA = mean_i INA_i
    """
    preds = [p.strip() for p in predicted_queries if p.strip()]
    golds = [g.strip() for g in real_needs if g.strip()]
    empty = {
        **{k: 0.0 for k in INA_KEYS},
        "bge_score": 0.0,
        "bge_disabled": True,
        "per_prediction": [],
        "num_predictions": 0,
        "num_real_needs": len(golds),
    }
    if not preds or not golds:
        return empty

    per_pred: list[dict[str, Any]] = []
    sum_rl = sum_llm = 0.0

    for pred in preds:
        best_rl = -1.0
        best_gold = golds[0]
        for gold in golds:
            rl = rouge_l(pred, gold)
            if rl > best_rl:
                best_rl = rl
                best_gold = gold
        llm_s = 0.0
        if use_llm_judge:
            llm_s = llm_need_score(pred, best_gold, client=client, model=model)
        ina_i = best_rl + llm_s
        per_pred.append({
            "prediction": pred,
            "matched_real_need": best_gold,
            "rougeL": round(best_rl, 4),
            "llm_score": round(llm_s, 4),
            "ina": round(ina_i, 4),
        })
        sum_rl += best_rl
        sum_llm += llm_s

    k = len(preds)
    mean_rl = sum_rl / k
    mean_llm = sum_llm / k
    return {
        "rougeL": round(mean_rl, 4),
        "llm_score": round(mean_llm, 4),
        "ina": round(mean_rl + mean_llm, 4),
        "bge_score": 0.0,
        "bge_disabled": True,
        "num_predictions": k,
        "num_real_needs": len(golds),
        "per_prediction": per_pred,
    }


def recall_at_k(retrieved_ids: list[str], relevant: dict[str, float], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved_ids[:k]
    hit = sum(1 for nid in top if nid in relevant)
    return hit / len(relevant)


def precision_at_k(retrieved_ids: list[str], relevant: dict[str, float], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hit = sum(1 for nid in top if nid in relevant)
    return hit / len(top)


def ndcg_at_k(retrieved_ids: list[str], relevant: dict[str, float], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    dcg = 0.0
    for i, nid in enumerate(retrieved_ids[:k], start=1):
        rel = float(relevant.get(nid, 0.0))
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        idcg += (2**rel - 1) / math.log2(i + 1)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def evaluate_rc(
    retrieved_ids: list[str],
    relevant: dict[str, float],
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ks": list(ks),
        "num_relevant": len(relevant),
        "num_retrieved": len(retrieved_ids),
    }
    for k in ks:
        out[f"recall@{k}"] = round(recall_at_k(retrieved_ids, relevant, k), 4)
        out[f"precision@{k}"] = round(precision_at_k(retrieved_ids, relevant, k), 4)
        out[f"ndcg@{k}"] = round(ndcg_at_k(retrieved_ids, relevant, k), 4)
    primary_k = ks[0] if ks else 5
    out["recall"] = out.get(f"recall@{primary_k}", 0.0)
    out["precision"] = out.get(f"precision@{primary_k}", 0.0)
    out["ndcg"] = out.get(f"ndcg@{primary_k}", 0.0)
    return out


def evaluate_gq(
    generated_texts: list[str],
    ground_truth_news: list[dict[str, str]],
) -> dict[str, Any]:
    """ACL-style: for each answer, max over GT news texts; then average over answers."""
    from mind_data import news_to_text

    cands = [t.strip() for t in generated_texts if t.strip()]
    refs = [news_to_text(n) for n in ground_truth_news if news_to_text(n).strip()]
    empty = {k: 0.0 for k in GQ_KEYS} | {"num_predictions": 0, "per_candidate": []}
    if not cands or not refs:
        return empty

    sums = {k: 0.0 for k in GQ_KEYS}
    per_candidate: list[dict[str, Any]] = []
    for text in cands:
        best = {k: 0.0 for k in GQ_KEYS}
        best_ref = ""
        for ref in refs:
            pair = {
                "bleu": bleu_score(text, ref),
                "rouge1": rouge_n(text, ref, 1),
                "rouge2": rouge_n(text, ref, 2),
                "rougeL": rouge_l(text, ref),
            }
            if pair["rouge1"] >= best["rouge1"]:
                best = pair
                best_ref = ref
        for k in GQ_KEYS:
            sums[k] += best[k]
        per_candidate.append({
            "text_preview": text[:200],
            "best_reference_preview": best_ref[:200],
            "metrics": {k: round(best[k], 4) for k in GQ_KEYS},
        })

    n = len(cands)
    return {
        **{k: round(sums[k] / n, 4) for k in GQ_KEYS},
        "num_predictions": n,
        "per_candidate": per_candidate,
    }
