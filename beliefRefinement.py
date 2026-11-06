"""
Belief-space Cognitive Reconstruction (Module 2)

Input:  candidate cognitive graphs from latentBeliefConstruction.py + behavior trajectory
Output: belief space B_t = {(G_i', b_i)}

Pipeline (LCG.pdf):
  1. Cross-Agent Cognitive Graph Refinement — each reasoning agent refines its own graph
  2. Belief-space Cognitive State Inference — each agent estimates belief for its interpretation

Run:
  python beliefRefinement.py
  python beliefRefinement.py --agents 4 --input graph.json --output belief.json
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from common import (
    LocalLLMClient,
    BELIEF_INFERENCE_SYSTEM,
    DEFAULT_LLM_MODEL,
    GRAPH_REFINEMENT_SYSTEM,
    REASONING_AGENTS,
    SAMPLE_BEHAVIOR_RAW,
    SAMPLE_GRAPH_1,
    SAMPLE_GRAPH_2,
    behavior_trajectory_text,
    chat_json,
    get_llm_client,
    as_dict_response,
    normalize_probabilities,
    reasoning_agent_system,
    to_score,
)


@dataclass
class BeliefSpace:
    entries: list[dict[str, Any]]
    logs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        beliefs = [float(e.get("belief", 0.5)) for e in self.entries]
        weights = normalize_probabilities(beliefs)
        belief_space = []
        for entry, weight in zip(self.entries, weights):
            belief_space.append({**entry, "probability": round(weight, 4)})
        entropy = -sum(p * math.log(p) for p in weights if p > 0)
        return {"belief_space": belief_space, "entropy": round(entropy, 4)}


def index_candidate_graphs(candidate_graphs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map each reasoning agent to its initial cognitive graph."""
    by_agent: dict[str, dict[str, Any]] = {}
    for idx, graph in enumerate(candidate_graphs):
        agent = graph.get("agent") or REASONING_AGENTS[idx % len(REASONING_AGENTS)]
        by_agent[agent] = graph
    return by_agent


def select_agent_graphs(
    candidate_graphs: list[dict[str, Any]],
    agents: list[str],
) -> dict[str, dict[str, Any]]:
    """Keep only graphs for the selected reasoning agents (in agent list order)."""
    by_agent = index_candidate_graphs(candidate_graphs)
    return {agent: by_agent[agent] for agent in agents if agent in by_agent}


def normalize_llm_graph(payload: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """Normalize UpdatedGraph / RefinedCognitiveGraph into pipeline graph schema."""
    graph_body = payload
    for key in ("UpdatedGraph", "updatedGraph", "RefinedCognitiveGraph", "refinedCognitiveGraph"):
        if isinstance(payload.get(key), dict):
            graph_body = payload[key]
            break

    nodes = graph_body.get("Nodes", graph_body.get("nodes", base.get("nodes", [])))
    edges_raw = graph_body.get("Edges", graph_body.get("edges", base.get("edges", [])))

    edges: list[dict[str, Any]] = []
    for edge in edges_raw:
        if not isinstance(edge, dict):
            continue
        normalized = dict(edge)
        normalized["source"] = edge.get("Source", edge.get("source", ""))
        normalized["target"] = edge.get("Target", edge.get("target", ""))
        normalized["relation"] = edge.get("Relation", edge.get("relation", ""))
        for key in ("Source", "Target", "Relation"):
            normalized.pop(key, None)
        edges.append(normalized)

    refined = {
        "graph_id": base.get("graph_id", ""),
        "timestamp": base.get("timestamp", ""),
        "agent": base.get("agent", ""),
        "nodes": nodes,
        "edges": edges,
    }
    if refined["nodes"] and refined["edges"]:
        return refined
    return base


def other_graphs_text(other_graphs: dict[str, dict[str, Any]]) -> str:
    parts = []
    for agent, graph in other_graphs.items():
        parts.append(
            f"Agent: {agent}\n{json.dumps(graph, ensure_ascii=False, indent=2)}"
        )
    return "\n\n".join(parts)


class BeliefSpaceReconstructor:
    def __init__(self, client: LocalLLMClient | None = None, *, model: str = DEFAULT_LLM_MODEL):
        self.client = client or get_llm_client()
        self.model = model

    def reconstruct(
        self,
        candidate_graphs: list[dict[str, Any]],
        behavior_trajectory: str,
        *,
        agents: list[str] | None = None,
    ) -> BeliefSpace:
        agents = agents or REASONING_AGENTS
        graphs_by_agent = select_agent_graphs(candidate_graphs, agents)
        if not graphs_by_agent:
            raise ValueError(
                "No candidate graphs match the selected agents. "
                f"Expected agents: {agents}"
            )

        active_agents = [a for a in agents if a in graphs_by_agent]

        refined_graphs: dict[str, dict[str, Any]] = {}
        refinement_logs: dict[str, dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=len(active_agents)) as executor:
            refine_futures = [
                executor.submit(
                    self._refine_graph,
                    agent,
                    behavior_trajectory,
                    graphs_by_agent[agent],
                    {a: g for a, g in graphs_by_agent.items() if a != agent},
                )
                for agent in active_agents
            ]
            for agent, future in zip(active_agents, refine_futures):
                refined, log = future.result()
                refined_graphs[agent] = refined
                refinement_logs[agent] = log

        entries: list[dict[str, Any]] = []
        belief_logs: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=len(active_agents)) as executor:
            belief_futures = [
                executor.submit(
                    self._infer_belief,
                    agent,
                    behavior_trajectory,
                    refined_graphs[agent],
                    {a: g for a, g in refined_graphs.items() if a != agent},
                )
                for agent in active_agents
            ]
            for agent, future in zip(active_agents, belief_futures):
                belief_resp = future.result()
                own_refined = refined_graphs[agent]
                entries.append({
                    "agent": agent,
                    "belief": round(to_score(belief_resp.get("Belief", belief_resp.get("belief")), 0.5), 2),
                    "reason": belief_resp.get("Reason", belief_resp.get("reason", "")),
                    "graph_edits": refinement_logs[agent].get("graph_edits", []),
                    "graph": own_refined,
                })
                belief_logs.append({"agent": agent, "response": belief_resp})

        return BeliefSpace(entries=entries, logs=[{"refinement": refinement_logs, "belief": belief_logs}])

    def _refine_graph(
        self,
        agent: str,
        behavior: str,
        own_graph: dict[str, Any],
        other_graphs: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        system = reasoning_agent_system(GRAPH_REFINEMENT_SYSTEM, agent)
        user = (
            f"Behavior Trajectory:\n{behavior}\n\n"
            f"Your Initial Cognitive Graph:\n"
            f"{json.dumps(own_graph, ensure_ascii=False, indent=2)}\n\n"
            f"Cognitive Graphs generated by the other reasoning agents:\n"
            f"{other_graphs_text(other_graphs)}"
        )
        resp = as_dict_response(chat_json(self.client, system, user, model=self.model))
        graph_edits = resp.get("GraphEdits", resp.get("graphEdits", []))
        refined = normalize_llm_graph(resp, own_graph)
        refined["graph_id"] = own_graph.get("graph_id", refined.get("graph_id", ""))
        refined["agent"] = agent
        return refined, {"graph_edits": graph_edits, "response": resp}

    def _infer_belief(
        self,
        agent: str,
        behavior: str,
        own_graph: dict[str, Any],
        other_graphs: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        system = reasoning_agent_system(BELIEF_INFERENCE_SYSTEM, agent)
        user = (
            f"Behavior Trajectory:\n{behavior}\n\n"
            f"Your Refined Cognitive Graph:\n"
            f"{json.dumps(own_graph, ensure_ascii=False, indent=2)}\n\n"
            f"Refined Cognitive Graphs generated by the other reasoning agents:\n"
            f"{other_graphs_text(other_graphs)}"
        )
        return as_dict_response(chat_json(self.client, system, user, model=self.model))


def default_candidate_graphs(agents: list[str]) -> list[dict[str, Any]]:
    """Build sample candidate graphs aligned with selected agents."""
    samples = [SAMPLE_GRAPH_1, SAMPLE_GRAPH_2]
    return [
        {**samples[i % len(samples)], "graph_id": f"LCG_{i + 1:03d}", "agent": agent}
        for i, agent in enumerate(agents)
    ]


def load_input(path: str | None, *, agents: list[str]) -> dict[str, Any]:
    if not path:
        return {
            "candidate_graphs": default_candidate_graphs(agents),
            "behavior_trajectory": behavior_trajectory_text(SAMPLE_BEHAVIOR_RAW),
        }
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {
            "candidate_graphs": data,
            "behavior_trajectory": behavior_trajectory_text(SAMPLE_BEHAVIOR_RAW),
        }
    if "candidate_graphs" in data:
        return data
    return {"candidate_graphs": data.get("graphs", []), **data}


def main() -> None:
    parser = argparse.ArgumentParser(description="Belief-space Cognitive Reconstruction")
    parser.add_argument("--input", type=str, default="graph.json", help="Output JSON from latentBeliefConstruction.py")
    parser.add_argument("--output", type=str, default="belief.json")
    parser.add_argument("--agents", type=int, default=2, help="Number of reasoning agents (max 4)")
    args = parser.parse_args()

    agents = REASONING_AGENTS[: max(1, min(args.agents, len(REASONING_AGENTS)))]
    data = load_input(args.input if args.input else None, agents=agents)
    result = BeliefSpaceReconstructor().reconstruct(
        data["candidate_graphs"],
        behavior_trajectory_text(data.get("behavior_trajectory", SAMPLE_BEHAVIOR_RAW)),
        agents=agents,
    )

    text = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
