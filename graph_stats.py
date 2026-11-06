"""
Compute Latent Cognitive Graph (LCG) structural statistics on MIND and OPeRA.

Metrics:
  - Average Node Number
  - Average Edge Number
  - Average Degree
  - Graph Density
  - Node Type Distribution
  - Relation Type Distribution

Usage:
  python graph_stats.py --limit 20 --agents 1
  python graph_stats.py --limit 0 --agents 1          # all samples
  python graph_stats.py --from-cache                 # only use saved graphs
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import REASONING_AGENTS, get_llm_client
from latentBeliefConstruction import LatentCognitiveGraphBuilder
from mind_data import load_behaviors, row_to_behavior_text

PROJECT_ROOT = Path(__file__).resolve().parent
MIND_BEHAVIORS = PROJECT_ROOT / "dataset" / "mind" / "behaviors.tsv"
OPERA_TSV = PROJECT_ROOT / "dataset" / "opera" / "opera_custom_with_profile.tsv"
CACHE_DIR = PROJECT_ROOT / "results" / "lcg_graphs"
NODE_TYPES = ("Concept", "Hypothesis", "Goal", "KnowledgeGap")
REL_TYPES = ("Attention", "Transition", "Evidence", "Goal", "Gap")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def load_opera_samples(path: Path) -> list[dict[str, Any]]:
    """Load OPeRA samples as {sample_id, behavior_text}."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        gt_col = header.index("Ground Truth")
        action_cols = list(range(1, gt_col))  # between User Profile and Ground Truth

        samples: list[dict[str, Any]] = []
        for idx, row in enumerate(reader, start=1):
            actions: list[dict[str, Any]] = []
            for c in action_cols:
                if c >= len(row) or not row[c].strip():
                    continue
                obj = _parse_json(row[c])
                if isinstance(obj, dict):
                    actions.append(obj)
            if not actions:
                continue
            lines = ["Web browsing behavior trajectory:"]
            for i, act in enumerate(actions, start=1):
                atype = act.get("action_type", "action")
                content = (act.get("content") or "").strip()
                url = (act.get("url") or "").strip()
                title = (act.get("title") or "").strip()
                piece = f"[{i}] action_type={atype}"
                if title:
                    piece += f", title={title}"
                if url:
                    piece += f", url={url}"
                if content:
                    piece += f", content={content}"
                lines.append(piece)
            samples.append({
                "sample_id": f"opera_{idx}",
                "behavior_text": "\n".join(lines),
            })
        return samples


def load_mind_samples(path: Path) -> list[dict[str, Any]]:
    rows = load_behaviors(path)
    samples: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("history"):
            continue
        samples.append({
            "sample_id": str(row.get("behaviors_id") or ""),
            "behavior_text": row_to_behavior_text(row["history"]),
        })
    return samples


# ---------------------------------------------------------------------------
# Graph metrics
# ---------------------------------------------------------------------------

def graph_metrics(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    n = len(nodes)
    m = len(edges)

    node_type_counts = Counter(
        str(node.get("type", "Unknown")) for node in nodes if isinstance(node, dict)
    )
    rel_counts = Counter(
        str(e.get("relation", e.get("Relation", "Unknown")))
        for e in edges
        if isinstance(e, dict)
    )

    # Directed average degree = 2M / N  (in+out); density = M / (N*(N-1))
    avg_degree = (2.0 * m / n) if n > 0 else 0.0
    density = (m / (n * (n - 1))) if n > 1 else 0.0

    return {
        "num_nodes": n,
        "num_edges": m,
        "avg_degree": avg_degree,
        "density": density,
        "node_types": dict(node_type_counts),
        "relations": dict(rel_counts),
    }


def aggregate_metrics(per_graph: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_graph:
        return {
            "num_graphs": 0,
            "avg_nodes": 0.0,
            "avg_edges": 0.0,
            "avg_degree": 0.0,
            "avg_density": 0.0,
            "node_type_distribution": {},
            "relation_type_distribution": {},
            "avg_node_type_counts": {},
            "avg_relation_type_counts": {},
        }

    n_g = len(per_graph)
    avg_nodes = sum(g["num_nodes"] for g in per_graph) / n_g
    avg_edges = sum(g["num_edges"] for g in per_graph) / n_g
    avg_degree = sum(g["avg_degree"] for g in per_graph) / n_g
    avg_density = sum(g["density"] for g in per_graph) / n_g

    total_nodes = sum(g["num_nodes"] for g in per_graph) or 1
    total_edges = sum(g["num_edges"] for g in per_graph) or 1

    node_sum: Counter[str] = Counter()
    rel_sum: Counter[str] = Counter()
    for g in per_graph:
        node_sum.update(g["node_types"])
        rel_sum.update(g["relations"])

    node_dist = {
        t: round(100.0 * node_sum.get(t, 0) / total_nodes, 2) for t in NODE_TYPES
    }
    rel_dist = {
        r: round(100.0 * rel_sum.get(r, 0) / total_edges, 2) for r in REL_TYPES
    }
    avg_node_counts = {t: round(node_sum.get(t, 0) / n_g, 2) for t in NODE_TYPES}
    avg_rel_counts = {r: round(rel_sum.get(r, 0) / n_g, 2) for r in REL_TYPES}

    return {
        "num_graphs": n_g,
        "avg_nodes": round(avg_nodes, 2),
        "avg_edges": round(avg_edges, 2),
        "avg_degree": round(avg_degree, 2),
        "avg_density": round(avg_density, 4),
        "node_type_distribution_pct": node_dist,
        "relation_type_distribution_pct": rel_dist,
        "avg_node_type_counts": avg_node_counts,
        "avg_relation_type_counts": avg_rel_counts,
    }


# ---------------------------------------------------------------------------
# Build / cache
# ---------------------------------------------------------------------------

def cache_path(dataset: str, sample_id: str, agent: str) -> Path:
    safe_agent = agent.replace(" ", "_")
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in sample_id)
    return CACHE_DIR / dataset / f"{safe_id}__{safe_agent}.json"


def load_cached_graph(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "nodes" in data and "edges" in data:
            return data
    except (OSError, json.JSONDecodeError):
        return None
    return None


def save_cached_graph(path: Path, graph: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False)


def collect_graphs_for_dataset(
    *,
    dataset: str,
    samples: list[dict[str, Any]],
    builder: LatentCognitiveGraphBuilder,
    agents: list[str],
    from_cache_only: bool,
) -> list[dict[str, Any]]:
    metrics_list: list[dict[str, Any]] = []
    total = len(samples)
    for i, sample in enumerate(samples, start=1):
        sid = sample["sample_id"] or f"{dataset}_{i}"
        behavior = sample["behavior_text"]
        print(f"[{dataset}] ({i}/{total}) sample={sid}")

        for agent in agents:
            path = cache_path(dataset, sid, agent)
            graph = load_cached_graph(path)
            if graph is None:
                if from_cache_only:
                    print(f"  skip (no cache): {agent}")
                    continue
                t0 = time.time()
                try:
                    graph = builder.build_one(
                        behavior, agent_role=agent, graph_index=i
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  ERROR {agent}: {exc}")
                    continue
                save_cached_graph(path, graph)
                print(f"  built {agent} in {time.time() - t0:.1f}s "
                      f"(nodes={len(graph.get('nodes', []))}, "
                      f"edges={len(graph.get('edges', []))})")
            else:
                print(f"  cache hit {agent} "
                      f"(nodes={len(graph.get('nodes', []))}, "
                      f"edges={len(graph.get('edges', []))})")
            metrics_list.append(graph_metrics(graph))
    return metrics_list


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------

def format_dist(d: dict[str, float], unit: str = "%") -> str:
    parts = [f"{k}:{v}{unit}" for k, v in d.items()]
    return "; ".join(parts)


def print_table(results: dict[str, dict[str, Any]]) -> None:
    headers = [
        "Metric",
        "MIND",
        "OPeRA",
    ]
    rows: list[list[str]] = []

    def cell(dataset: str, key: str, default: str = "-") -> str:
        stats = results.get(dataset) or {}
        val = stats.get(key, default)
        return str(val)

    rows.append(["# Graphs", cell("MIND", "num_graphs"), cell("OPeRA", "num_graphs")])
    rows.append(["Average Node Number", cell("MIND", "avg_nodes"), cell("OPeRA", "avg_nodes")])
    rows.append(["Average Edge Number", cell("MIND", "avg_edges"), cell("OPeRA", "avg_edges")])
    rows.append(["Average Degree", cell("MIND", "avg_degree"), cell("OPeRA", "avg_degree")])
    rows.append(["Graph Density", cell("MIND", "avg_density"), cell("OPeRA", "avg_density")])

    mind_nt = (results.get("MIND") or {}).get("node_type_distribution_pct", {})
    opera_nt = (results.get("OPeRA") or {}).get("node_type_distribution_pct", {})
    rows.append([
        "Node Type Distribution (%)",
        format_dist(mind_nt) if mind_nt else "-",
        format_dist(opera_nt) if opera_nt else "-",
    ])

    mind_rt = (results.get("MIND") or {}).get("relation_type_distribution_pct", {})
    opera_rt = (results.get("OPeRA") or {}).get("relation_type_distribution_pct", {})
    rows.append([
        "Relation Type Distribution (%)",
        format_dist(mind_rt) if mind_rt else "-",
        format_dist(opera_rt) if opera_rt else "-",
    ])

    # Also show average counts per type (often more interpretable)
    mind_nc = (results.get("MIND") or {}).get("avg_node_type_counts", {})
    opera_nc = (results.get("OPeRA") or {}).get("avg_node_type_counts", {})
    rows.append([
        "Avg Node Counts (C/H/G/K)",
        format_dist(mind_nc, "") if mind_nc else "-",
        format_dist(opera_nc, "") if opera_nc else "-",
    ])
    mind_rc = (results.get("MIND") or {}).get("avg_relation_type_counts", {})
    opera_rc = (results.get("OPeRA") or {}).get("avg_relation_type_counts", {})
    rows.append([
        "Avg Relation Counts",
        format_dist(mind_rc, "") if mind_rc else "-",
        format_dist(opera_rc, "") if opera_rc else "-",
    ])

    # Markdown table
    print("\n## LCG Graph Structure Statistics\n")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="LCG graph structure statistics")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max samples per dataset (0 = all)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--agents", type=int, default=1,
                        help="Number of reasoning agents (1-4)")
    parser.add_argument("--datasets", type=str, default="mind,opera",
                        help="Comma-separated: mind,opera")
    parser.add_argument("--from-cache", action="store_true",
                        help="Only use cached graphs; do not call LLM")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "results" / "graph_structure_stats.json"))
    args = parser.parse_args()

    agents = REASONING_AGENTS[: max(1, min(args.agents, len(REASONING_AGENTS)))]
    builder = LatentCognitiveGraphBuilder(client=get_llm_client())
    wanted = {x.strip().lower() for x in args.datasets.split(",") if x.strip()}

    results: dict[str, dict[str, Any]] = {}

    if "mind" in wanted:
        samples = load_mind_samples(MIND_BEHAVIORS)
        start = max(0, args.offset)
        end = len(samples) if args.limit <= 0 else min(len(samples), start + args.limit)
        selected = samples[start:end]
        print(f"MIND: {len(selected)} samples, agents={agents}")
        metrics = collect_graphs_for_dataset(
            dataset="mind",
            samples=selected,
            builder=builder,
            agents=agents,
            from_cache_only=args.from_cache,
        )
        results["MIND"] = aggregate_metrics(metrics)

    if "opera" in wanted:
        samples = load_opera_samples(OPERA_TSV)
        start = max(0, args.offset)
        end = len(samples) if args.limit <= 0 else min(len(samples), start + args.limit)
        selected = samples[start:end]
        print(f"OPeRA: {len(selected)} samples, agents={agents}")
        metrics = collect_graphs_for_dataset(
            dataset="opera",
            samples=selected,
            builder=builder,
            agents=agents,
            from_cache_only=args.from_cache,
        )
        results["OPeRA"] = aggregate_metrics(metrics)

    print_table(results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "limit": args.limit,
            "offset": args.offset,
            "agents": agents,
            "from_cache": args.from_cache,
        },
        "results": results,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
