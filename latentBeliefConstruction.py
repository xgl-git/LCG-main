"""
LLM-based Latent Cognitive Graph Construction (Module 1)

Pipeline:
  Concept + Hypothesis Extraction (single joint prompt)
  -> Goal Inference
  -> Knowledge Gap Discovery
  -> Global Cognitive Relation Construction

Multiple reasoning-style agents each produce one candidate graph:
  G_j = f_j(B)
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from common import (
    LocalLLMClient,
    CONCEPT_HYPOTHESIS_SYSTEM,
    DEFAULT_LLM_MODEL,
    GLOBAL_RELATION_SYSTEM,
    GOAL_INFERENCE_SYSTEM,
    REASONING_AGENTS,
    SAMPLE_BEHAVIOR_RAW,
    agent_system,
    as_item_list,
    behavior_trajectory_text,
    chat_json,
    get_llm_client,
    is_narrative_text,
    knowledge_gap_system,
    load_cognitive_theories,
    to_score,
)


class LatentCognitiveGraphBuilder:
    def __init__(
        self,
        client: LocalLLMClient | None = None,
        *,
        model: str = DEFAULT_LLM_MODEL,
        max_theories: int | None = None,
    ):
        self.client = client or get_llm_client()
        self.model = model
        self.max_theories = max_theories

    def build(
        self,
        behavior: str | dict[str, Any],
        *,
        agents: list[str] | None = None,
        max_theories: int | None = None,
    ) -> list[dict[str, Any]]:
        behavior_text = behavior_trajectory_text(behavior)
        agents = agents or REASONING_AGENTS
        n_theories = self.max_theories if max_theories is None else max_theories
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = [
                executor.submit(
                    self.build_one,
                    behavior_text,
                    agent_role=agent,
                    graph_index=idx,
                    max_theories=n_theories,
                )
                for idx, agent in enumerate(agents, start=1)
            ]
            return [future.result() for future in futures]

    def build_one(
        self,
        behavior: str,
        *,
        agent_role: str | None = None,
        graph_index: int = 1,
        max_theories: int | None = None,
    ) -> dict[str, Any]:
        n_theories = self.max_theories if max_theories is None else max_theories
        concepts, hypotheses = self.extract_concepts_and_hypotheses(behavior, agent_role)
        goals = self.infer_goals(behavior, concepts, hypotheses, agent_role)
        gaps = self.discover_gaps(
            concepts, hypotheses, goals, behavior, agent_role, max_theories=n_theories
        )
        nodes = self._assemble_nodes(concepts, hypotheses, goals, gaps)
        edges = self.build_relations(behavior, nodes, agent_role)

        return {
            "graph_id": f"LCG_{graph_index:03d}",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "agent": agent_role,
            "nodes": nodes,
            "edges": edges,
        }

    def extract_concepts_and_hypotheses(
        self, behavior: str, agent_role: str | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Jointly extract Concepts and Hypotheses in one LLM call."""
        system = agent_system(CONCEPT_HYPOTHESIS_SYSTEM, agent_role)
        user = (
            f"Behavior Trajectory:\n{behavior}\n\n"
            "Return both Concepts and Hypotheses in one JSON object."
        )
        result = chat_json(self.client, system, user, model=self.model)

        if isinstance(result, list):
            concepts = [
                x for x in result
                if isinstance(x, dict) and ("Concept" in x or "concept" in x)
            ]
            hypotheses = [
                x for x in result
                if isinstance(x, dict) and ("Hypothesis" in x or "hypothesis" in x)
            ]
            return concepts, hypotheses

        if not isinstance(result, dict):
            return [], []

        concepts_raw = result.get("Concepts", result.get("concepts", []))
        hypotheses_raw = result.get("Hypotheses", result.get("hypotheses", []))
        concepts = as_item_list(concepts_raw, "Concept", "Evidence", "Importance") if concepts_raw else []
        hypotheses = (
            as_item_list(hypotheses_raw, "Hypothesis", "Evidence", "Confidence")
            if hypotheses_raw
            else []
        )
        return concepts, hypotheses

    def infer_goals(
        self,
        behavior: str,
        concepts: list[dict[str, Any]],
        hypotheses: list[dict[str, Any]],
        agent_role: str | None = None,
    ) -> list[dict[str, Any]]:
        user = (
            f"Behavior Trajectory:\n{behavior}\n\n"
            f"Concept List:\n{json.dumps(concepts, ensure_ascii=False, indent=2)}\n\n"
            f"Hypothesis List:\n{json.dumps(hypotheses, ensure_ascii=False, indent=2)}"
        )
        result = chat_json(
            self.client, agent_system(GOAL_INFERENCE_SYSTEM, agent_role), user, model=self.model
        )
        return as_item_list(result, "Goal", "Evidence", "Confidence")

    def discover_gaps(
        self,
        concepts: list[dict[str, Any]],
        hypotheses: list[dict[str, Any]],
        goals: list[dict[str, Any]],
        behavior: str,
        agent_role: str | None = None,
        *,
        max_theories: int | None = None,
    ) -> list[dict[str, Any]]:
        n_theories = self.max_theories if max_theories is None else max_theories
        theories = load_cognitive_theories(max_theories=n_theories)
        user = (
            f"Concept:\n{json.dumps(concepts, ensure_ascii=False, indent=2)}\n\n"
            f"Hypothesis:\n{json.dumps(hypotheses, ensure_ascii=False, indent=2)}\n\n"
            f"Goal:\n{json.dumps(goals, ensure_ascii=False, indent=2)}\n\n"
            f"Behavior Trajectory:\n{behavior}\n\n"
            f"Cognitive Theories:\n{theories}"
        )
        result = chat_json(
            self.client,
            agent_system(knowledge_gap_system(theories), agent_role),
            user,
            model=self.model,
        )
        return as_item_list(result, "Gap", "Theory", "Reason", "Expected Information Gain")

    def build_relations(
        self,
        behavior: str,
        nodes: list[dict[str, Any]],
        agent_role: str | None = None,
    ) -> list[dict[str, Any]]:
        by_type = {
            "Concept": [n for n in nodes if n["type"] == "Concept"],
            "Hypothesis": [n for n in nodes if n["type"] == "Hypothesis"],
            "Goal": [n for n in nodes if n["type"] == "Goal"],
            "KnowledgeGap": [n for n in nodes if n["type"] == "KnowledgeGap"],
        }
        user = (
            f"Behavior Trajectory:\n{behavior}\n\n"
            f"Concept Nodes:\n{json.dumps(by_type['Concept'], ensure_ascii=False, indent=2)}\n\n"
            f"Hypothesis Nodes:\n{json.dumps(by_type['Hypothesis'], ensure_ascii=False, indent=2)}\n\n"
            f"Goal Nodes:\n{json.dumps(by_type['Goal'], ensure_ascii=False, indent=2)}\n\n"
            f"Knowledge Gap Nodes:\n{json.dumps(by_type['KnowledgeGap'], ensure_ascii=False, indent=2)}"
        )
        result = chat_json(
            self.client, agent_system(GLOBAL_RELATION_SYSTEM, agent_role), user, model=self.model
        )
        return self._normalize_edges(result.get("Edges", []), nodes)

    def _assemble_nodes(
        self,
        concepts: list[dict[str, Any]],
        hypotheses: list[dict[str, Any]],
        goals: list[dict[str, Any]],
        gaps: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []

        for i, c in enumerate(concepts, start=1):
            imp = to_score(c.get("Importance", c.get("importance")), 0.8)
            nodes.append({
                "id": f"C{i}",
                "type": "Concept",
                "name": c.get("Concept", c.get("concept", "")),
                "belief": round(imp, 2),
                "uncertainty": round(max(0.0, 1.0 - imp), 2),
                "confidence": round(min(1.0, imp + 0.05), 2),
                "priority": round(imp, 2),
                "status": "Observed",
                "evidence": c.get("Evidence", c.get("evidence", "")),
            })

        for i, h in enumerate(hypotheses, start=1):
            conf = to_score(h.get("Confidence", h.get("confidence")), 0.7)
            nodes.append({
                "id": f"H{i}",
                "type": "Hypothesis",
                "name": h.get("Hypothesis", h.get("hypothesis", "")),
                "belief": round(conf, 2),
                "uncertainty": round(max(0.0, 1.0 - conf), 2),
                "confidence": round(conf, 2),
                "priority": round(conf, 2),
                "status": "Active" if i == 1 else "Candidate",
                "evidence": h.get("Evidence", h.get("evidence", "")),
            })

        for i, g in enumerate(goals, start=1):
            conf = to_score(g.get("Confidence", g.get("confidence")), 0.85)
            nodes.append({
                "id": f"G{i}",
                "type": "Goal",
                "name": g.get("Goal", g.get("goal", "")),
                "belief": round(conf, 2),
                "uncertainty": round(max(0.0, 1.0 - conf), 2),
                "confidence": round(conf, 2),
                "priority": 1.0 if i == 1 else round(conf, 2),
                "status": "Active",
                "evidence": g.get("Evidence", g.get("evidence", "")),
            })

        for i, k in enumerate(gaps, start=1):
            eig_raw = k.get(
                "Expected Information Gain",
                k.get("ExpectedInformationGain", k.get("expected_information_gain")),
            )
            reason = k.get("Reason", k.get("reason", "")) or ""
            if is_narrative_text(eig_raw):
                if not reason:
                    reason = str(eig_raw)
                eig = to_score(reason, 0.85)
            else:
                eig = to_score(eig_raw, 0.8)
            node: dict[str, Any] = {
                "id": f"K{i}",
                "type": "KnowledgeGap",
                "name": k.get("Gap", k.get("gap", k.get("KnowledgeGap", ""))),
                "belief": 0.0,
                "uncertainty": 1.0,
                "confidence": round(min(1.0, eig + 0.05), 2),
                "expected_information_gain": round(eig, 2),
                "priority": round(eig, 2),
                "status": "Unexplored",
                "reason": reason,
            }
            theory = k.get("Theory", k.get("theory", ""))
            if theory:
                node["theory"] = theory
            nodes.append(node)

        return nodes

    def _normalize_edges(self, raw_edges: list[dict[str, Any]], nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        name_to_id = {n["name"]: n["id"] for n in nodes}
        edges: list[dict[str, Any]] = []

        for e in raw_edges:
            src = e.get("Source", e.get("source", ""))
            tgt = e.get("Target", e.get("target", ""))
            rel = e.get("Relation", e.get("relation", ""))
            conf = to_score(e.get("Confidence", e.get("confidence", e.get("weight"))), 0.8)

            edge: dict[str, Any] = {
                "source": name_to_id.get(src, src),
                "target": name_to_id.get(tgt, tgt),
                "relation": rel,
                "weight": round(conf, 2),
            }
            if rel == "Attention" and conf >= 0.9:
                edge["strength"] = "Strong"
            if rel == "Evidence":
                edge["type"] = "Supports"
            elif rel == "Goal":
                edge["type"] = "Core" if conf >= 0.9 else "Supporting"
            elif rel == "Gap":
                edge["type"] = "Required" if conf >= 0.85 else "Helpful"
                edge["expected_information_gain"] = round(conf, 2)
            if e.get("Reason"):
                edge["reason"] = e["Reason"]
            edges.append(edge)
        return edges


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-based Latent Cognitive Graph Construction")
    parser.add_argument("--agents", type=int, default=2, help="Number of reasoning agents (max 4)")
    parser.add_argument(
        "--max-theories",
        type=int,
        default=None,
        help="Use first N theories for Knowledge Gap Discovery (ablation; default: all 8)",
    )
    parser.add_argument("--output", type=str, default="graph.json", help="Save graphs JSON to file")
    args = parser.parse_args()

    builder = LatentCognitiveGraphBuilder(max_theories=args.max_theories)
    agents = REASONING_AGENTS[: max(1, min(args.agents, len(REASONING_AGENTS)))]
    graphs = builder.build(SAMPLE_BEHAVIOR_RAW, agents=agents)

    text = json.dumps({"candidate_graphs": graphs}, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
