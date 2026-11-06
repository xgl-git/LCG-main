"""
Dataset statistics for MIND and OPeRA datasets.

Outputs:
  - Sample count
  - Average behavior length (number of news/actions per sample)
  - Average Ground Truth count
  - Corpus size (MIND only)
  - Real Need / Ground Truth text length statistics
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


def _safe_json_loads(s: str) -> Any:
    """Try to parse a JSON string, return None on failure."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# MIND
# --------------------------------------------------------------------------- #

def mind_stats(dataset_dir: str) -> dict[str, Any]:
    behaviors_path = os.path.join(dataset_dir, "mind", "behaviors.tsv")
    corpus_path = os.path.join(dataset_dir, "mind", "corpus.jsonl")

    # ---- behaviors.tsv ----
    # Columns: behaviors_id | News 1..20 | Ground Truth | Real Need
    with open(behaviors_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)

    # Find column indices
    news_cols = [i for i, h in enumerate(header) if h.startswith("News ")]
    gt_col = header.index("Ground Truth")
    rn_col = header.index("Real Need")

    n_samples = 0
    behavior_lengths: list[int] = []
    gt_counts: list[int] = []
    rn_lengths: list[int] = []    # character length of each Real Need query
    rn_counts: list[int] = []     # number of queries per sample

    with open(behaviors_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            n_samples += 1

            # Behavior length = number of non-empty News columns
            b_len = sum(1 for i in news_cols if i < len(row) and row[i].strip())
            behavior_lengths.append(b_len)

            # Ground Truth
            gt_raw = row[gt_col] if gt_col < len(row) else ""
            gt_list = _safe_json_loads(gt_raw) or []
            gt_counts.append(len(gt_list) if isinstance(gt_list, list) else 0)

            # Real Need
            rn_raw = row[rn_col] if rn_col < len(row) else ""
            rn_list = _safe_json_loads(rn_raw) or []
            if isinstance(rn_list, list):
                rn_counts.append(len(rn_list))
                for item in rn_list:
                    if isinstance(item, dict):
                        q = item.get("query", "")
                    else:
                        q = str(item)
                    rn_lengths.append(len(q))
            else:
                rn_counts.append(0)

    # ---- corpus.jsonl ----
    corpus_size = 0
    if os.path.exists(corpus_path):
        with open(corpus_path, encoding="utf-8") as f:
            for _ in f:
                corpus_size += 1

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

    return {
        "dataset": "MIND",
        "num_samples": n_samples,
        "avg_behavior_length": round(avg(behavior_lengths), 2),
        "min_behavior_length": min(behavior_lengths) if behavior_lengths else 0,
        "max_behavior_length": max(behavior_lengths) if behavior_lengths else 0,
        "avg_ground_truth_count": round(avg(gt_counts), 2),
        "min_ground_truth_count": min(gt_counts) if gt_counts else 0,
        "max_ground_truth_count": max(gt_counts) if gt_counts else 0,
        "corpus_size": corpus_size,
        "avg_real_need_count_per_sample": round(avg(rn_counts), 2),
        "total_real_need_queries": len(rn_lengths),
        "avg_real_need_length_chars": round(avg(rn_lengths), 2),
        "min_real_need_length_chars": min(rn_lengths) if rn_lengths else 0,
        "max_real_need_length_chars": max(rn_lengths) if rn_lengths else 0,
    }


# --------------------------------------------------------------------------- #
# OPeRA
# --------------------------------------------------------------------------- #

def opera_stats(dataset_dir: str) -> dict[str, Any]:
    opera_path = os.path.join(dataset_dir, "opera", "opera_custom_with_profile.tsv")

    # Columns: User Profile | News/Action 1..20 | Ground Truth
    with open(opera_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)

    # News/Action columns are between "User Profile" and "Ground Truth"
    gt_col = header.index("Ground Truth")
    action_cols = list(range(1, gt_col))  # columns between profile and GT

    n_samples = 0
    behavior_lengths: list[int] = []
    gt_counts: list[int] = []
    gt_action_lengths: list[int] = []  # character length of each GT action content

    with open(opera_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            n_samples += 1

            # Behavior length = number of non-empty action columns
            b_len = sum(1 for i in action_cols if i < len(row) and row[i].strip())
            behavior_lengths.append(b_len)

            # Ground Truth
            gt_raw = row[gt_col] if gt_col < len(row) else ""
            gt_list = _safe_json_loads(gt_raw) or []
            if isinstance(gt_list, list):
                gt_counts.append(len(gt_list))
                for item in gt_list:
                    if isinstance(item, dict):
                        content = item.get("content", "")
                    else:
                        content = str(item)
                    gt_action_lengths.append(len(content))
            else:
                gt_counts.append(0)

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

    return {
        "dataset": "OPeRA",
        "num_samples": n_samples,
        "avg_behavior_length": round(avg(behavior_lengths), 2),
        "min_behavior_length": min(behavior_lengths) if behavior_lengths else 0,
        "max_behavior_length": max(behavior_lengths) if behavior_lengths else 0,
        "avg_ground_truth_count": round(avg(gt_counts), 2),
        "min_ground_truth_count": min(gt_counts) if gt_counts else 0,
        "max_ground_truth_count": max(gt_counts) if gt_counts else 0,
        "corpus_size": "N/A (no separate corpus file)",
        "avg_gt_content_length_chars": round(avg(gt_action_lengths), 2),
        "min_gt_content_length_chars": min(gt_action_lengths) if gt_action_lengths else 0,
        "max_gt_content_length_chars": max(gt_action_lengths) if gt_action_lengths else 0,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def print_stats(stats: dict[str, Any]) -> None:
    name = stats["dataset"]
    print(f"\n{'='*60}")
    print(f"  {name} Dataset Statistics")
    print(f"{'='*60}")
    for k, v in stats.items():
        if k == "dataset":
            continue
        label = k.replace("_", " ").title()
        print(f"  {label:<45s}: {v}")
    print()


def main() -> None:
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

    mind = mind_stats(dataset_dir)
    print_stats(mind)

    opera = opera_stats(dataset_dir)
    print_stats(opera)


if __name__ == "__main__":
    main()
