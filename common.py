"""Shared utilities for LCG pipeline."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any


DEFAULT_LLM_MODEL = "llama-3-8b-instruct"
DEFAULT_LLM_MODEL_PATH = Path(__file__).resolve().parent / "models" / DEFAULT_LLM_MODEL
_LLM_CLIENT: LocalLLMClient | None = None
_LLM_CLIENT_LOCK = Lock()


class LocalLLMClient:
    def __init__(self, model_path: str | Path, *, device: str = "auto"):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"Local Llama model directory not found: {self.model_path}. "
                "Set LLAMA_MODEL_PATH to the directory containing HuggingFace weights and tokenizer files."
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._lock = Lock()
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        load_kwargs: dict[str, Any] = {"torch_dtype": dtype, "local_files_only": True}
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        if device != "cuda":
            self.model.to(device)
        self.model.eval()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(
        self,
        *,
        model: str = DEFAULT_LLM_MODEL,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> SimpleNamespace:
        if stream:
            raise ValueError("Local Llama streaming is not supported.")
        if model not in (DEFAULT_LLM_MODEL, str(self.model_path)):
            raise ValueError(f"This client only serves {DEFAULT_LLM_MODEL}: {self.model_path}")
        max_new_tokens = 2048 if max_tokens is None else max_tokens
        if max_new_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if temperature < 0:
            raise ValueError("temperature must be non-negative.")

        with self._lock:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = inputs.to(self.model.get_input_embeddings().weight.device)
            input_length = inputs["input_ids"].shape[-1]
            context_length = self.model.config.max_position_embeddings
            if input_length >= context_length:
                raise ValueError(
                    f"Prompt has {input_length} tokens, exceeding the local model's "
                    f"{context_length}-token context window. Shorten the input."
                )
            eos_ids = [self.tokenizer.eos_token_id]
            eot_id = self.tokenizer.get_vocab().get("<|eot_id|>")
            if eot_id is not None and eot_id not in eos_ids:
                eos_ids.append(eot_id)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": min(max_new_tokens, context_length - input_length),
                "do_sample": temperature > 0,
                "eos_token_id": eos_ids,
                "pad_token_id": (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                ),
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
            with self._torch.inference_mode():
                output = self.model.generate(**inputs, **generation_kwargs)
            content = self.tokenizer.decode(output[0, input_length:], skip_special_tokens=True)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def get_llm_client() -> LocalLLMClient:
    global _LLM_CLIENT
    with _LLM_CLIENT_LOCK:
        if _LLM_CLIENT is None:
            _LLM_CLIENT = LocalLLMClient(
                os.environ.get("LLAMA_MODEL_PATH", str(DEFAULT_LLM_MODEL_PATH)),
                device=os.environ.get("LLAMA_DEVICE", "auto"),
            )
        return _LLM_CLIENT


def chat_json(
    client: LocalLLMClient,
    system: str,
    user: str,
    *,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> dict[str, Any] | list[Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    return parse_json(content)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _sanitize_llm_json(text: str) -> str:
    """Fix common invalid JSON patterns echoed from prompt templates."""
    text = re.sub(r"\[\s*\.\.\.\s*\]", "[]", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _extract_json_array(text: str) -> str | None:
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json_string_value(text: str, key: str) -> str | None:
    """Extract a JSON string value, tolerating truncated LLM output."""
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"', text, re.IGNORECASE)
    if not match:
        return None

    chars: list[str] = []
    escape = False
    for ch in text[match.end() :]:
        if escape:
            if ch == "n":
                chars.append("\n")
            elif ch == "t":
                chars.append("\t")
            elif ch == "r":
                chars.append("\r")
            elif ch == '"':
                chars.append('"')
            elif ch == "\\":
                chars.append("\\")
            else:
                chars.append(ch)
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            break
        else:
            chars.append(ch)
    value = "".join(chars).strip()
    return value or None


def _fallback_json_fields(text: str) -> dict[str, Any] | None:
    """Extract critical scalar fields when full JSON parsing fails."""
    result: dict[str, Any] = {}
    for key in ("Answer", "answer"):
        answer = _extract_json_string_value(text, key) or _extract_loose_answer(text)
        if answer:
            result["Answer"] = answer
            return result

    for key in ("Belief", "belief", "Confidence", "confidence", "SupportScore", "supportScore"):
        match = re.search(rf'"{re.escape(key)}"\s*:\s*([0-9.]+)', text)
        if match:
            canonical = {
                "belief": "Belief",
                "confidence": "Confidence",
                "supportscore": "SupportScore",
            }.get(key.lower(), key)
            result[canonical] = float(match.group(1))
            break
    reason = re.search(r'"Reason"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if reason:
        result["Reason"] = reason.group(1)
    return result or None


def _extract_loose_answer(text: str) -> str | None:
    """Extract Answer value even when JSON is truncated mid-string."""
    match = re.search(r'"(?:Answer|answer)"\s*:\s*"(.*)', text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    body = match.group(1)
    body = re.sub(r'"\s*}\s*$', "", body)
    body = re.sub(r'"\s*,\s*$', "", body)
    body = body.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
    return body.strip() or None


def parse_json(text: str) -> dict[str, Any] | list[Any]:
    text = _strip_code_fence(text)
    candidates = [text, _sanitize_llm_json(text)]
    extracted_array = _extract_json_array(text)
    if extracted_array:
        candidates.extend([extracted_array, _sanitize_llm_json(extracted_array)])
    extracted = _extract_json_object(text)
    if extracted:
        candidates.extend([extracted, _sanitize_llm_json(extracted)])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            data = json.loads(candidate)
            if isinstance(data, (dict, list)):
                return data
        except json.JSONDecodeError:
            continue

    fallback = _fallback_json_fields(text)
    if fallback:
        return fallback
    raise ValueError(f"Failed to parse JSON from LLM output:\n{text[:800]}")


def extract_answer_from_llm_output(text: str) -> str:
    """Extract Answer from LLM JSON output; works even when JSON is truncated."""
    raw = (text or "").strip()
    if not raw:
        return ""

    content = _strip_code_fence(raw)
    try:
        data = parse_json(content)
        if isinstance(data, dict):
            answer = data.get("Answer", data.get("answer", ""))
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
    except ValueError:
        pass

    for key in ("Answer", "answer"):
        value = _extract_json_string_value(content, key)
        if value:
            return value.strip()

    loose = _extract_loose_answer(content)
    if loose:
        return loose.strip()

    cleaned = content.strip()
    if cleaned and not cleaned.startswith("{"):
        return cleaned

    return _fallback_candidate_answer_from_raw(raw)


def _fallback_candidate_answer_from_raw(text: str) -> str:
    """Last-resort extraction from any non-empty LLM text."""
    cleaned = _strip_code_fence(text).strip()
    if not cleaned:
        return ""

    for key in ("Answer", "answer"):
        match = re.search(rf'"{key}"\s*:\s*"(.+)', cleaned, re.DOTALL | re.IGNORECASE)
        if match:
            body = match.group(1).strip().rstrip('"}')
            if body:
                return body

    if cleaned.startswith("{") and '"Answer"' not in cleaned and '"answer"' not in cleaned:
        return ""

    return cleaned


def chat_answer(
    client: LocalLLMClient,
    system: str,
    user: str,
    *,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """Call LLM and return Answer text; never raises on truncated JSON."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    content = response.choices[0].message.content or ""
    return extract_answer_from_llm_output(content)


def clone_graph(graph: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(graph)


def node_index(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {n["id"]: n for n in graph.get("nodes", [])}


def normalize_probabilities(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        n = len(weights)
        return [1.0 / n] * n if n else []
    return [w / total for w in weights]


def behavior_trajectory_text(behavior: str | dict[str, Any]) -> str:
    if isinstance(behavior, str):
        return behavior
    return json.dumps(behavior, ensure_ascii=False)


def fill_prompt(template: str, **kwargs: str) -> str:
    """Replace {Key} placeholders without interpreting other braces in JSON examples."""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", value)
    return result


def normalize_operations(data: Any) -> list[dict[str, Any]]:
    """Normalize LLM graph edit output to a list of operation dicts."""
    if data is None:
        return []
    if isinstance(data, dict):
        if any(k in data for k in ("Type", "type", "Target", "target")):
            return [data]
        for key in ("Operations", "operations", "RevisedOperations", "revisedOperations"):
            if key in data:
                return normalize_operations(data[key])
        return []
    if isinstance(data, str):
        try:
            return normalize_operations(parse_json(data))
        except (ValueError, json.JSONDecodeError):
            return [{"Type": "Unknown", "Target": "", "Reason": data, "Confidence": 0.5}]
    if isinstance(data, list):
        ops: list[dict[str, Any]] = []
        for item in data:
            ops.extend(normalize_operations(item))
        return ops
    return []


def to_score(value: Any, default: float = 0.8) -> float:
    """Parse LLM score fields; tolerate text like 'High' or long narrative strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1.0:
            v = v / 100.0 if v <= 100 else 1.0
        return max(0.0, min(1.0, v))
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        try:
            v = float(s)
            if v > 1.0:
                v = v / 100.0 if v <= 100 else 1.0
            return max(0.0, min(1.0, v))
        except ValueError:
            pass
        match = re.search(r"(\d+(?:\.\d+)?)", s)
        if match:
            v = float(match.group(1))
            if v > 1.0:
                v = min(1.0, v / 100.0)
            return max(0.0, min(1.0, v))
        low = s.lower()
        if "high" in low or "strong" in low:
            return 0.9
        if "medium" in low or "moderate" in low:
            return 0.6
        if "low" in low:
            return 0.3
    return default


def is_narrative_text(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip()
    if len(s) <= 12:
        return False
    try:
        float(s)
        return False
    except ValueError:
        pass
    if s.lower() in {"high", "medium", "low", "strong", "weak", "moderate"}:
        return False
    return len(s) > 20 or s.count(" ") >= 3


def as_dict_response(data: Any) -> dict[str, Any]:
    """Normalize LLM JSON that may be returned as a list instead of a dict."""
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, list):
        return {"items": data}
    return {}


def extract_named_list(data: Any, *keys: str) -> list[dict[str, Any]]:
    """Extract a list of dict items from varied LLM JSON shapes."""
    if isinstance(data, list):
        source: list[Any] = data
    else:
        mapping = as_dict_response(data)
        source = []
        for key in keys:
            if key not in mapping:
                continue
            val = mapping[key]
            source = val if isinstance(val, list) else [val]
            break
        if not source and mapping:
            if any(
                k in mapping
                for k in (
                    "FocusID", "focus_id", "ActionID", "action_id",
                    "Query", "query", "Nodes", "nodes",
                )
            ):
                source = [mapping]
            elif "items" in mapping and isinstance(mapping["items"], list):
                source = mapping["items"]

    return [item for item in source if isinstance(item, dict)]


def as_item_list(data: dict[str, Any] | list[Any], *field_names: str) -> list[dict[str, Any]]:
    """Normalize LLM JSON into a list of dicts."""
    if isinstance(data, list):
        return [x if isinstance(x, dict) else {"value": x} for x in data]

    for key in ("items", "results", "data", "Concepts", "Hypotheses", "Goals", "Gaps", "KnowledgeGaps"):
        if key in data and isinstance(data[key], list):
            return data[key]

    if any(k in data for k in field_names):
        return [data]

    # e.g. {"Concept": [...], "Evidence": [...]} parallel arrays — zip if possible
    arrays = {k: data[k] for k in field_names if k in data and isinstance(data[k], list)}
    if arrays and len(set(len(v) for v in arrays.values())) == 1:
        keys = list(arrays.keys())
        n = len(arrays[keys[0]])
        return [{k: arrays[k][i] for k in keys if k in arrays} for i in range(n)]

    return [data]


# ---------------------------------------------------------------------------
# Prompts from LCG.pdf
# ---------------------------------------------------------------------------


CONCEPT_HYPOTHESIS_SYSTEM = """
Your task is to jointly infer, from the user's behavior trajectory:
1. persistent cognitive **Concepts**, and
2. latent **Hypotheses** the user is attempting to verify.

The user message will provide Behavior Trajectory.

Produce **both** Concepts and Hypotheses in a **single** JSON response. Do not output only one of them.

## Concept Instructions

1. Identify as many meaningful concepts as possible from the user's behavior trajectory, including entities, events, topics, domains, products, organizations, people, technologies, and recurring interests.
2. Prioritize **high recall** rather than aggressive filtering. Do **not** omit a concept simply because it appears to be less important or has relatively weak evidence.
3. Ignore only clearly transient, accidental, or noise-like browsing behaviors that do not contribute to the user's long-term cognitive interests.
4. Merge semantically equivalent or highly similar concepts into a single concept.
5. Concepts may exist at different levels of abstraction (e.g., "Artificial Intelligence", "Large Language Models", "Retrieval-Augmented Generation"). Preserve meaningful fine-grained concepts instead of merging everything into overly broad categories.
6. Assign each concept an importance score between 0 and 1. Low importance **does not** mean the concept should be excluded. Instead, include the concept and reflect its significance through the importance score.
7. Explain why each concept is important by citing supporting behavioral evidence.
8. Prefer concrete entities, people, events, places, products, and lifestyle topics from the news the user read. Avoid abstract meta-concepts about news feeds, algorithms, or browsing habits.

## Hypothesis Instructions

1. Infer as many plausible hypotheses as possible that could explain the user's exploration process. A hypothesis represents a tentative belief, assumption, prediction, or explanation that the user appears to be verifying through information seeking, rather than a topic of interest.
2. Prioritize **high recall** rather than aggressive filtering. Do **not** omit a plausible hypothesis simply because the available evidence is incomplete. If a hypothesis is weakly supported, include it with a lower confidence score.
3. Each hypothesis should explain a substantial portion of the observed behaviors. Whenever possible, infer hypotheses that connect multiple concepts and multiple behaviors instead of isolated browsing events.
4. Generate multiple alternative hypotheses whenever the behavior trajectory admits different interpretations. Do not assume there is only one correct explanation.
5. A hypothesis should describe **what the user is trying to verify or learn next**, grounded in concrete news topics from the trajectory. Avoid simply restating concepts, and avoid meta-hypotheses about news-feed design or browsing psychology unless the trajectory explicitly supports them.
6. Merge semantically equivalent or nearly identical hypotheses into a single hypothesis.
7. Assign each hypothesis a confidence score between 0 and 1 according to the strength of behavioral evidence. Lower confidence does not mean the hypothesis should be excluded.
8. For every hypothesis, explain the supporting evidence by referring to relevant concepts and behaviors that motivated the inference.
9. Keep concepts and hypotheses distinct: concepts are what the user attends to; hypotheses are what the user is trying to verify.

## Output JSON

Return ONE JSON object containing both lists:

```json
{
  "Concepts": [
    {
      "Concept": "",
      "Evidence": "",
      "Importance": 0.0
    }
  ],
  "Hypotheses": [
    {
      "Hypothesis": "",
      "Evidence": "",
      "Confidence": 0.0
    }
  ]
}
```
"""

# Backward-compatible aliases (prefer CONCEPT_HYPOTHESIS_SYSTEM).
CONCEPT_EXTRACTION_SYSTEM = CONCEPT_HYPOTHESIS_SYSTEM
HYPOTHESIS_GENERATION_SYSTEM = CONCEPT_HYPOTHESIS_SYSTEM

GOAL_INFERENCE_SYSTEM = """
Your task is to infer the user's underlying cognitive goals from behavior trajectories.

The user message will provide:

- Behavior Trajectory
- Concept List
- Hypothesis List

## Instructions

1. Infer as many plausible cognitive goals as possible that explain the user's overall exploration process. A cognitive goal represents the ultimate problem or question the user is attempting to resolve through information seeking.

2. Prefer concrete topical objectives grounded in the articles the user actually read (people, events, places, products, lifestyle topics). For news browsing, a goal may be expressed as the concrete information question the user still wants answered.

3. Prioritize **high recall** rather than aggressive filtering. Do **not** omit a plausible goal simply because the supporting evidence is incomplete. If a goal is weakly supported, include it with a lower confidence score.

4. Each cognitive goal should explain a substantial portion of the observed behaviors and be supported by multiple concepts and hypotheses whenever possible. Prefer several topic-specific goals over one abstract meta-goal.

5. Generate multiple alternative goals whenever the behavior trajectory admits different interpretations. Do not assume there is only one correct goal.

6. Merge semantically equivalent or highly similar goals into a single cognitive goal.

7. Assign each goal a confidence score between 0 and 1 according to the strength of behavioral evidence. Lower confidence does not imply exclusion.

8. Explain why each goal is inferred by citing the relevant concepts, hypotheses, and behavioral evidence.

9. The output should be as comprehensive as possible, covering both primary and secondary cognitive goals supported by the behavior trajectory.

## Important

A cognitive goal should answer questions such as:

- What problem is the user ultimately trying to solve?
- What decision is the user attempting to make?
- What uncertainty is the user trying to eliminate?
- What concrete next news topic does the user still want to understand?

A cognitive goal should **not** describe:

- a browsing action itself,
- a temporary click with no topical interest,
- or a meta-analysis of news feeds / algorithms / browsing habits.

## Output JSON

```json
[
  {
    "Goal": "",
    "Evidence": "",
    "Confidence": 0.0
  }
]
```
"""

KNOWLEDGE_GAP_SYSTEM = """
Your task is to infer the knowledge gaps that currently prevent the user from validating their cognitive hypotheses and achieving their cognitive goals.
Knowledge-gap discovery MUST be constrained by the Cognitive Theories provided below.

The user message will provide:

- Behavior Trajectory
- Concept List
- Hypothesis List
- Goal List
- Cognitive Theories (optional; if omitted, use the theories listed in this system prompt)

## Cognitive Theories

{CognitiveTheories}

## Theory-Guided Constraints

Apply the theories jointly when discovering knowledge gaps. Each gap must be theoretically grounded as missing information that blocks progress toward the goal:

1. **Unresolved desired end state (Goal Systems Theory; Goal-Directed Behavior Theory)**
   A knowledge gap is information still missing for reaching the desired end state that organizes the user's exploration. Prefer gaps whose resolution would unlock subordinate goals/actions, not gaps that merely restate concepts already observed.

2. **Unreduced discrepancy (Control Theory / Self-Regulation; Means–Ends Analysis)**
   Identify gaps as the concrete differences between the current knowledge state and the desired reference/goal state. Prefer gaps that, if filled, would most reduce that discrepancy.

3. **Missing means in the means–end chain (Means–End Chain Theory)**
   Concepts and hypotheses are lower-level means toward higher-level goals. A knowledge gap should be a missing intermediate attribute, evidence link, or means required to climb that chain—not the goal itself and not an already-available concept.

4. **Blocked intention-to-action transition (Theory of Planned Behavior)**
   Prefer gaps whose absence prevents the user from forming or acting on an intention (e.g., missing evaluative evidence, normative comparison, or perceived-control information needed to decide/proceed).

5. **High-utility uncertainty reduction (Expectancy–Value Theory)**
   Prioritize gaps with higher expected value for advancing the goal: high subjective importance × high expected chance that retrieving the missing information will help. Still keep high recall: include weaker but plausible gaps with lower Expected Information Gain.

6. **Mastery vs. performance information needs (Achievement Goal Theory)**
   When supported by evidence, distinguish:
   - mastery-oriented gaps (missing explanations / mechanisms / understanding), and
   - performance/decision-oriented gaps (missing evaluative comparisons / outcomes / success criteria).
   Do not invent a type without behavioral or cognitive support.

When generating each gap, name the primary theory (or theories) that justify why this missing information is a knowledge gap in the `Theory` field.

## Instructions

1. Identify as many meaningful knowledge gaps as possible that explain what information is still missing for validating the user's hypotheses and achieving the inferred cognitive goals, subject to the Theory-Guided Constraints above.

2. A knowledge gap should represent **missing information**, rather than a concept, topic, entity, or search query. It should describe the information the user still needs to acquire in order to reduce cognitive uncertainty.

3. Prioritize **high recall** rather than aggressive filtering. Do **not** omit a plausible knowledge gap simply because the supporting evidence is limited. If a gap is weakly supported, include it with a lower expected information gain.

4. Each knowledge gap should be directly connected to one or more hypotheses and cognitive goals, and explain why the current behavior trajectory is insufficient to validate those hypotheses or accomplish those goals.

5. A knowledge gap should be **actionable**, meaning that it can potentially be resolved through additional information retrieval, evidence collection, or further exploration.

6. Merge semantically equivalent or highly similar knowledge gaps into a single knowledge gap.

7. Assign each knowledge gap an Expected Information Gain score between 0 and 1, indicating how much resolving this gap is expected to reduce the user's cognitive uncertainty and theoretical discrepancy. Lower scores do not imply exclusion.

8. Explain why each knowledge gap is important by citing the relevant concepts, hypotheses, goals, behavioral evidence, **and the theory constraint** that validates it as missing means / unreduced discrepancy rather than an already-observed concept.

9. For each gap, name the primary supporting theory in the `Theory` field (use the short theory name, e.g., "Means–Ends Analysis"). If multiple theories apply, list the most central one first and briefly note others in Reason.

10. The output should be as comprehensive as possible, covering both primary and secondary knowledge gaps supported by the current cognitive state.

## Important

A knowledge gap should answer questions such as:

- What information is still missing?
- What evidence is required to validate the current hypotheses?
- What uncertainty / discrepancy prevents the user from achieving the cognitive goal?
- Which missing means in the means–end chain still blocks progress?

A knowledge gap should **not** describe:

- a concept,
- an entity,
- a search query,
- a browsing action,
- or a cognitive goal itself.

## Output JSON

```json
[
  {
    "Gap": "",
    "Theory": "",
    "Reason": "",
    "Expected Information Gain": 0.0
  }
]
```
"""


def load_cognitive_theories(
    path: str | None = None,
    *,
    max_theories: int | None = None,
) -> str:
    """Load cognitive theories used to constrain Knowledge Gap Discovery.

    ``max_theories`` keeps the first N theories for ablation on theory count.
    """
    theory_path = path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dataset", "theories.txt"
    )
    with open(theory_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"Cognitive theories file is empty: {theory_path}")
    if max_theories is not None:
        lines = lines[: max(0, max_theories)]
    return "\n".join(lines)


def knowledge_gap_system(
    theories: str | None = None,
    *,
    max_theories: int | None = None,
) -> str:
    """Build Knowledge Gap Discovery system prompt with theories injected."""
    theories_text = (
        theories
        if theories is not None
        else load_cognitive_theories(max_theories=max_theories)
    )
    return fill_prompt(KNOWLEDGE_GAP_SYSTEM, CognitiveTheories=theories_text)


GLOBAL_RELATION_SYSTEM = """
Your task is to construct a globally consistent latent cognitive graph that explains the user's underlying reasoning process during exploratory information seeking.

The user message will provide:

- Behavior Trajectory
- Concept List
- Hypothesis List
- Goal List
- Knowledge Gap List

## Instructions

1. Construct a complete cognitive graph that organizes all nodes into a coherent reasoning structure. The graph should explain how the user's behaviors evolve from concepts to hypotheses, goals, and knowledge gaps.

2. Infer relationships by considering the entire cognitive state rather than making independent pairwise decisions. Every relation should be globally consistent with the overall exploration process.

3. Build Attention Relations among Concept nodes to represent concepts that belong to the same cognitive focus.

4. Build Transition Relations among Concept nodes to represent how the user's attention shifts during the exploration process.

5. Build Evidence Relations from Concept nodes to Hypothesis nodes to indicate which concepts provide evidence for each hypothesis.

6. Build Goal Relations from Hypothesis nodes to Goal nodes to indicate which hypotheses contribute to achieving each cognitive goal.

7. Build Gap Relations from Goal nodes to Knowledge Gap nodes to indicate which missing information prevents the user from accomplishing each cognitive goal.

8. Multiple relations may exist between nodes if they are supported by sufficient behavioral evidence.

9. Avoid redundant, contradictory, or logically inconsistent relations. Every relation should contribute to explaining the user's cognitive reasoning process.

10. Assign every relation a confidence score between 0 and 1 according to the strength of supporting evidence.

11. Explain why each relation is inferred by citing the relevant behaviors, concepts, hypotheses, goals, or knowledge gaps.

12. The output should be as complete as possible, covering both dominant and secondary cognitive relations supported by the behavior trajectory.

## Relation Types

- Attention
- Transition
- Evidence
- Goal
- Gap

## Output JSON

```json
{
  "Edges": [
    {
      "Source": "",
      "Target": "",
      "Relation": "",
      "Confidence": 0.0,
      "Reason": ""
    }
  ]
}```
"""

GRAPH_REFINEMENT_SYSTEM = """You are the {AgentRole}.
You have already constructed an initial cognitive graph from the user's
behavior trajectory according to your own reasoning perspective.
The user message will provide:
- Behavior Trajectory
- Your Initial Cognitive Graph
- Cognitive Graphs generated by the other three reasoning agents
## Instructions
1. Refine your cognitive graph by reviewing the cognitive graphs generated by
the other reasoning agents. Treat the other graphs as complementary reasoning
references rather than ground truth.
2. Preserve your own reasoning perspective throughout the refinement process.
Do **not** simply merge all graphs or imitate the reasoning style of other
agents.
3. Incorporate additional concepts, hypotheses, goals, knowledge gaps, or
cognitive relations only when they improve the explanation of the user's
behavior trajectory and remain consistent with your own reasoning strategy.
4. Remove nodes or relations that are weakly supported, redundant, or
inconsistent with the observed behaviors.
5. Revise cognitive relations whenever a more coherent reasoning structure can
better explain the user's exploration process.
6. Refine the graph hierarchy if doing so improves the logical organization of
concepts, hypotheses, goals, and knowledge gaps.
7. Update the confidence of nodes and relations according to both behavioral
evidence and insights obtained from the other reasoning agents.
8. Every modification should be explicitly justified. Do **not** modify the
graph without sufficient supporting evidence.
9. Preserve the global consistency of your cognitive graph after refinement.
Avoid introducing redundant, contradictory, or logically inconsistent
structures.
10. The refined graph should remain faithful to your reasoning perspective
while benefiting from complementary evidence provided by the other reasoning
agents.
## Output JSON
```json
{
"UpdatedGraph": {
"Nodes": [...],
"Edges": [...]
},
"GraphEdits": [
{
"Operation": "Add | Remove | Update | Reorganize",
"Target": "",
"Reason": ""
}
]
}
```"""

BELIEF_INFERENCE_SYSTEM = """You are the {AgentRole}.
You have already refined your cognitive graph through cross-agent cognitive
graph refinement.
Your task is NOT to modify the cognitive graph.
Instead, estimate the belief of your own cognitive interpretation, namely, how
confidently your refined cognitive graph explains the user's current cognitive
state.
The user message will provide:
- Behavior Trajectory
- Your Refined Cognitive Graph
- Refined Cognitive Graphs generated by the other three reasoning agents
## Instructions
1. Evaluate the belief of your own cognitive interpretation rather than the
quality of the cognitive graph itself.
2. Estimate how confidently your refined cognitive graph explains the user's
behavior trajectory and underlying cognitive reasoning process.
3. Consider the following aspects jointly:
**Behavior Consistency**
- Does your cognitive graph consistently explain the observed behavior
trajectory?
- Can most behaviors be explained by the inferred concepts, hypotheses,
goals, and knowledge gaps?
**Cognitive Coherence**
- Are the concepts, hypotheses, goals, knowledge gaps, and cognitive
relations internally consistent?
- Does the graph form a coherent cognitive reasoning process?
**Cross-Agent Support**
- To what extent do the other reasoning agents independently provide
evidence that is compatible with your cognitive interpretation?
- Treat the other cognitive graphs as supporting evidence rather than
ground truth.
4. Preserve your own reasoning perspective. Do **not** modify your graph or
adjust your belief simply to agree with the other reasoning agents.
5. Estimate a belief score between 0 and 1, where a higher score indicates
greater confidence that your cognitive graph accurately represents the user's
current cognitive state.
6. Explain the belief score by referring to behavioral evidence, graph
consistency, and cross-agent support.
## Important
The belief score represents the confidence of your own cognitive
interpretation.
It does **not** represent:
- the quality of the graph,
- the agreement among agents,
- or the probability that your graph is absolutely correct.
Different reasoning agents may assign different belief scores to their own
cognitive graphs.
## Output JSON
```json
{
"RefinedCognitiveGraph": {
"Nodes": [...],
"Edges": [...]
},
"Belief": 0.0,
"Reason": ""
}
```"""

REASONING_AGENTS = ["Causal Agent", "Comparative Agent", "Evidence Agent", "Hierarchical Agent"]

REASONING_AGENT_DEFINITIONS: dict[str, str] = {
    "Causal Agent": """The Causal Agent organizes content according to causal dependencies among concepts. Instead of listing isolated facts, it identifies causal chains that explain why an observed phenomenon occurs and how different events influence one another.
Its objective is to construct a coherent reasoning path from causes to consequences, enabling users to understand the underlying mechanisms behind the current cognitive goal.
Responsibility:
- Identify causal relationships between concepts.
- Discover driving factors and downstream effects.
- Organize content into causal reasoning chains.
- Prioritize explanations that maximize causal interpretability.""",

    "Comparative Agent": """The Comparative Agent organizes information through systematic comparison across alternative entities, hypotheses, or scenarios. Rather than explaining a single reasoning path, it highlights similarities, differences, strengths, weaknesses, and trade-offs among competing candidates.
Its objective is to help users distinguish multiple possible explanations and support informed decision making.
Responsibility:
- Compare competing hypotheses.
- Compare multiple companies/products/entities.
- Analyze advantages and disadvantages.
- Identify discriminative evidence.""",

    "Evidence Agent": """The Evidence Agent organizes content by collecting and evaluating supporting evidence for the current cognitive hypothesis. Rather than focusing on explanation structure, it emphasizes whether sufficient evidence exists to validate or refute a hypothesis.
Its objective is to minimize cognitive uncertainty through evidence accumulation and verification.
Responsibility:
- Retrieve supporting evidence.
- Identify contradictory evidence.
- Evaluate evidence credibility.
- Estimate confidence of hypotheses.""",

    "Hierarchical Agent": """The Hierarchical Agent organizes content according to semantic abstraction levels. Instead of reasoning through causality or comparison, it decomposes a high-level cognitive goal into progressively finer-grained concepts, enabling users to understand complex topics through hierarchical knowledge organization.
Its objective is to transform abstract cognitive goals into structured explanatory hierarchies.
Responsibility:
- Decompose abstract concepts.
- Construct hierarchical topic structures.
- Organize knowledge from general to specific.
- Maintain logical dependencies among concepts.""",
}


def agent_system(base_system: str, agent_name: str | None) -> str:
    """Prepend reasoning-agent role definition to a system prompt."""
    if not agent_name or agent_name not in REASONING_AGENT_DEFINITIONS:
        return base_system
    definition = REASONING_AGENT_DEFINITIONS[agent_name]
    return f"You are the {agent_name}.\n\n{definition}\n\n---\n\n{base_system}"


def reasoning_agent_system(base_system: str, agent_name: str | None) -> str:
    """Fill {AgentRole} and inject REASONING_AGENT_DEFINITIONS after the role line."""
    if not agent_name:
        return fill_prompt(base_system, AgentRole="")
    filled = fill_prompt(base_system, AgentRole=agent_name)
    definition = REASONING_AGENT_DEFINITIONS.get(agent_name)
    if not definition:
        return filled
    role_line, _, rest = filled.partition("\n")
    return f"{role_line}\n\n{definition}\n\n---\n\n{rest.lstrip()}"


STRATEGY_AGENTS = REASONING_AGENTS

RETRIEVAL_PROJECTION_AGENT = "Retrieval Projection Agent"

RETRIEVAL_PROJECTION_AGENT_DEFINITION = """The Retrieval Projection Agent bridges the user's cognitive state and external information retrieval. Its role is distinct from the four reasoning agents (Causal, Comparative, Evidence, Hierarchical) that construct and refine cognitive graphs from different interpretive perspectives.

Instead, this agent:
- Extracts Cognitive Focus Subgraphs from the belief-aware refined cognitive graph(s).
- Projects each Cognitive Focus into executable Retrieval Actions.
- Operates agnostically across all reasoning perspectives, selecting exploration directions based on structural coherence and belief support rather than a single reasoning style."""


def retrieval_projection_agent_system(base_system: str) -> str:
    """Fill {AgentRole} and inject RETRIEVAL_PROJECTION_AGENT_DEFINITION."""
    filled = fill_prompt(base_system, AgentRole=RETRIEVAL_PROJECTION_AGENT)
    role_line, _, rest = filled.partition("\n")
    return f"{role_line}\n\n{RETRIEVAL_PROJECTION_AGENT_DEFINITION}\n\n---\n\n{rest.lstrip()}"


# Cognitive-to-Retrieval Projection (LCG.pdf)
COGNITIVE_FOCUS_EXTRACTION_SYSTEM = """You are the {AgentRole}.
# Task
Your task is to extract Cognitive Focus Subgraphs from the Belief-aware
Refined Cognitive Graph.
The Belief-aware Refined Cognitive Graph has already been generated through
the previous Cross-Agent Cognitive Graph Refinement stage.
# Important Constraint
Your task is NOT to modify, refine, expand, or reconstruct the Cognitive Graph.
Instead, identify multiple coherent cognitive exploration directions
represented by the graph and extract corresponding Focus Subgraphs.
# Cognitive Focus Definition
A Cognitive Focus represents a localized cognitive reasoning structure that
captures one potential exploration direction of the user.
Each Focus Subgraph should preserve the original cognitive reasoning path:
Goal → Hypothesis → Knowledge Gap
A Cognitive Focus should explain:
- What the user wants to understand or solve.
- What assumptions the user is trying to verify.
- What missing information prevents uncertainty reduction.
--------------------------------------------------
# Input
The user message will provide:
- Behavior Trajectory
- Belief Space containing one or more Belief-aware Refined Cognitive Graphs
--------------------------------------------------
# Instructions
## 1. Identify Cognitive Focuses
Analyze the Belief Space holistically, considering all refined cognitive
graphs and their belief weights.
Identify all meaningful Cognitive Focuses contained across the graphs.
A Cognitive Focus should represent a coherent exploration direction rather
than an isolated node or concept.
---
## 2. Focus Extraction Principle
# Do NOT simply divide the graph by node similarity.
Each Cognitive Focus should correspond to a meaningful reasoning path
connecting:
Goal nodes
↓
Hypothesis nodes
↓
Knowledge Gap nodes
Whenever possible, extract a complete cognitive reasoning chain.
---
## 3. Preserve Original Graph Information
For each extracted Focus Subgraph:
Preserve:
- Original node identity.
- Original node type.
- Original node content.
- Existing node belief/confidence information.
- Original relation semantics.
# Do NOT create new nodes or new relations.
---
## 4. Belief-aware Focus Selection
When identifying Cognitive Focuses, consider:
- Node importance.
- Node confidence/belief.
- Relation structure.
- Connectivity among Goal, Hypothesis, and Knowledge Gap nodes.
Focuses with stronger behavioral evidence and higher belief consistency should
be preferred.
---
## 5. Multiple Focuses
A single cognitive graph may contain multiple exploration directions.
Extract multiple Focus Subgraphs when different reasoning paths exist.
# Do NOT merge different Cognitive Focuses into a single graph.
# Do NOT generate redundant Focuses with highly overlapping reasoning
structures.
---
## 6. Forbidden Outputs
Do NOT generate:
- Retrieval queries.
- Retrieval actions.
- Search strategies.
- Final answers.
- Content generation plans.
# Your task is only Focus Subgraph extraction.
---
# Cognitive Focus Subgraph Definition
Each Cognitive Focus Subgraph is defined as:
F = (N_F, R_F)
where:
- N_F represents the cognitive nodes belonging to this exploration direction.
- R_F represents the original relations among these selected nodes.
--------------------------------------------------
# Output Format
Return ONLY JSON:
```json
{
"FocusSubgraphs": [
{
"FocusID": "F1",
"Nodes": [
{
"id": "",
"type": "Goal | Hypothesis | KnowledgeGap | Concept",
"content": "",
"Belief": 0.0
}
],
"Relations": [
{
"source": "",
"target": "",
"type": ""
}
],
"Reason": ""
}
]
}
```"""

RETRIEVAL_PLANNING_SYSTEM = """You are the {AgentRole}.
You are an expert in Cognitive-Driven Retrieval Planning.
Your task is to project the user's current Cognitive Focus into executable
Retrieval Actions.
The Cognitive Focus Subgraph has already been extracted from the user's Belief-
aware Cognitive Graph through the previous Cognitive Focus Selection stage.
Your task is NOT to modify, revise, or regenerate the Cognitive Focus Subgraph.
Your task is NOT to reconstruct or infer the user's Cognitive Graph.
Instead, transform the user's current cognitive exploration direction into
Retrieval Actions through Cognitive-to-Retrieval Projection.
--------------------------------------------------
Input
User Behavior Trajectory
{Behavior}
Cognitive Focus Subgraph
{Focus}
Focus Probability
{FocusProbability}
--------------------------------------------------
Objective
Generate one or multiple Retrieval Actions that can acquire external evidence
for resolving the Knowledge Gaps contained in the given Cognitive Focus.
A Retrieval Action represents one cognitive-driven retrieval behavior for
satisfying the user's current retrieval and generation need.
Each Retrieval Action is defined as
a=(q,e,\\Delta I)
where
• Query (q)
A natural-language retrieval request representing the user's current retrieval
and generation need.
The Query should describe what the user currently wants to understand, verify,
compare, explain, or explore.
The Query should be jointly determined by
- the User Behavior Trajectory,
- the Exploration Goal,
- the current Hypothesis,
- the unresolved Knowledge Gap.
The Query should preserve the user's cognitive reasoning process rather than
simply concatenate keywords.
--------------------------------------------------
• Target Evidence (e)
Target Evidence specifies the external knowledge expected to be retrieved.
It describes the evidence required to
- verify the current hypothesis;
- support or reject the cognitive interpretation;
- resolve the corresponding Knowledge Gap;
- improve the user's current cognitive understanding.
--------------------------------------------------
• Expected Information Gain (\\Delta I)
Expected Information Gain estimates how much uncertainty can be reduced after
obtaining the Target Evidence.
Assign a value between 0 and 1.
A larger value indicates that this Retrieval Action is expected to provide
greater cognitive benefit.
--------------------------------------------------
A Retrieval Action is NOT
- a keyword query,
- a query suggestion,
- a retrieved document,
- a final answer.
Instead, it represents the projection from Cognitive Space into Retrieval
Space:
User Behavior Trajectory
+
Cognitive Focus
↓
Retrieval / Generation Need
↓
Retrieval Action
--------------------------------------------------
Instructions
### 1. Understand the Cognitive Focus
Jointly analyze
- the User Behavior Trajectory;
- the Cognitive Focus Subgraph.
Identify
- the current exploration objective;
- the hypothesis under investigation;
- the unresolved Knowledge Gaps.
Do not modify the Cognitive Focus.
--------------------------------------------------
### 2. Generate Retrieval Actions
Generate one or multiple Retrieval Actions.
Each Retrieval Action should correspond to one meaningful cognitive
exploration direction.
For each Retrieval Action,
generate
• Query
The Query should
- reflect the user's current retrieval/generation need;
- preserve the reasoning encoded in the Cognitive Focus;
- transform the Knowledge Gap into an explicit information-seeking request;
- be executable by a search engine / news retriever;
- support downstream answer generation.
For news browsing trajectories, the Query MUST be a CONCRETE information need
about a specific person, event, story, place, product, or lifestyle topic that
appears (or is clearly implied) in the User Behavior Trajectory.
Good query style examples:
- "What are the most common reasons for tattoo removal?"
- "What happened to Alexis Crawford and the charges against her roommate?"
- "What are some creative Christmas tree decorating ideas?"
Bad queries (FORBIDDEN):
- meta questions about news feeds, aggregators, algorithms, or browsing habits;
- abstract analyses of "tragic vs light news", "content switching", or personalization;
- vague overviews that do not name a concrete topic from the trajectory.
Prefer short natural-language questions that would retrieve related news articles.
--------------------------------------------------
Generate
• Target Evidence
Describe
- what news evidence should be retrieved (title/abstract level);
- why this evidence is needed;
- how it helps verify or refine the hypothesis.
--------------------------------------------------
Generate
• Expected Information Gain
Estimate the expected uncertainty reduction.
Consider
- importance of the Knowledge Gap;
- uncertainty of the Hypothesis;
- usefulness of the expected evidence.
--------------------------------------------------
### 3. Preserve Cognitive Consistency
Do NOT
- introduce new Goals;
- introduce new Hypotheses;
- introduce new Concepts;
- introduce new Knowledge Gaps.
Every Retrieval Action must be fully supported by the provided Cognitive Focus
and grounded in entities/events from the User Behavior Trajectory.
--------------------------------------------------
Important
Different Cognitive Focus Subgraphs may produce different Retrieval Actions
even under the same User Behavior Trajectory.
The generated Retrieval Actions represent the user's current retrieval and
generation needs rather than retrieved evidence or final answers.
When the trajectory is news browsing, each Query should look like a Real Need:
a concrete topical question the user may ask next, not a cognitive meta-analysis.
--------------------------------------------------
Output Format
Return ONLY JSON.
{
"RetrievalActions":[
{
"ActionID":"A1",
"Query":"",
"TargetEvidence":"",
"ExpectedInformationGain":0.0,
"Motivation":"Explain why this Retrieval Action is generated from the
current Cognitive Focus."
}
]
}"""

EVIDENCE_GROUNDED_ANSWER_GENERATION_SYSTEM = """You are an expert in Cognitive Content Generation for exploratory news recommendation.

# Task
Generate one user-facing next-news snippet that the user is likely to want to read next,
based on:
- User Behavior Trajectory (past news browsing)
- Cognitive Focus
- Retrieval Action (Query)
- Retrieved Evidence (news title/abstract used as the primary content source)

# Critical Output Rules
1. Write ONLY the content the user would consume — a news HEADLINE + short ABSTRACT.
   Preferred format: "<Headline>. <1-2 sentence abstract>"
2. If Retrieved Evidence contains relevant news title/abstract, CLOSELY PARAPHRASE or lightly
   rewrite the best-matching evidence. Do NOT invent a long essay about media theory.
3. NEVER mention retrieval, documents, BM25, evidence, doc IDs, cognitive focus, or reasoning.
4. Do NOT apologize or refuse when evidence is weak; still write a plausible next-news snippet
   inferred from browsing history and the retrieval query.
5. Keep total length roughly 20-80 words (news snippet style, not an analysis article).
6. Stay on ONE concrete news topic aligned with the Retrieval Action Query.

# Output Format
Return ONLY compact JSON (no markdown fences):
```json
{
"Answer": ""
}
```"""


# Legacy Cognitive-to-Content Projection prompts (kept for reference)
GAP_EXPANSION_SYSTEM = """You are an expert in exploratory information seeking and domain knowledge
organization.
Your task is NOT to explain the given knowledge gap.
Instead, transform an abstract cognitive gap into a structured retrieval space
that can efficiently reduce the user's cognitive uncertainty.

The user message will provide:
• Behavior Trajectory
• Current Cognitive Goal
• Current Hypothesis
• Knowledge Gap

Your objective is to identify the most informative retrieval directions for
resolving the knowledge gap.
Specifically,
1. Decompose the knowledge gap into multiple complementary retrieval aspects.
2. For each aspect, identify representative entities, companies, products,
technologies, events, datasets, reports, or indicators that should be
retrieved.
3. Ensure the retrieval aspects cover different perspectives rather than
redundant information.
4. Prioritize information that is expected to provide the largest reduction in
cognitive uncertainty.
5. Estimate the importance of each retrieval direction.
Output in JSON format:
{
"RetrievalDirections":[
{
"Aspect":"",
"Description":"",
"RepresentativeItems":[...],
"Importance":0.xx,
"ExpectedInformationGain":0.xx
}
]
}"""

GENERATION_STRATEGY_SYSTEM = """You are an expert in cognitive content planning.

Your task is NOT to generate the final answer.
Instead, design a generation strategy that can best help the user understand
the current cognitive problem.

The user message will provide:
• Behavior Trajectory
• Current Goal
• Current Hypothesis
• Knowledge Gap
• Expanded Content Dimensions

Requirements:
1. Organize the content into a coherent reasoning structure.
2. Select the most informative topics.
3. Specify the logical dependency among topics.
4. Prioritize topics according to their expected cognitive contribution.
5. Maximize the expected reduction of cognitive uncertainty.
Output JSON:
{
"GenerationStrategy":{
"Topics":[...],
"LogicalStructure":[...],
"Priority":[...],
"Reason":"..."
}
}"""

STRATEGY_DEBATE_SYSTEM = """You are participating in a multi-agent generation strategy debate.
Given:
- Current Belief Space
- Cognitive Goal
- Hypothesis
- Knowledge Gap
- Candidate Generation Strategy
Your task is to evaluate whether the candidate strategy is appropriate for the
current cognitive state.
Specifically:
1. Does this strategy support the current cognitive goal?
2. Does it help validate or refute the current hypothesis?
3. Does it sufficiently cover the knowledge gap?
4. Is the reasoning structure coherent and complete?
5. What important content is still missing?
6. Suggest revisions to improve the strategy.
Output JSON:
{
"SupportScore":0-10,
"GoalConsistency":0-1,
"HypothesisSupport":0-1,
"GapCoverage":0-1,
"Coherence":0-1,
"MissingTopics":[...],
"Revision":[...],
"Reason":"..."
}"""

CGPT_SYSTEM = """You are an expert in cognitive planning and structured content generation.

Your task is NOT to generate the final content.
Instead, construct a Cognitive Generation Plan Tree (CGPT) that transforms the
current cognitive state into an executable generation plan.

The user message will provide:
• Behavior Trajectory
• Current Belief Space
• Current Cognitive Goal
• Current Hypotheses
• Current Knowledge Gaps
• Expanded Content Dimensions
• Consensus Generation Strategy

Requirements
1. Use the Cognitive Goal as the root node.
2. Organize Hypotheses under the corresponding Goal.
3. Connect each Knowledge Gap to the Hypothesis it supports.
4. Expand every Knowledge Gap into one or more Content Dimensions.
5. Decompose each Content Dimension into executable Content Sections.
6. For every node, generate:
- Local Cognitive Context
- Generation Action
- Execution Status
7. Preserve logical dependencies among all nodes.
8. Merge redundant branches whenever possible.
9. Ensure every leaf node contributes to reducing the uncertainty of its
parent Knowledge Gap.
10. Output the complete Cognitive Generation Plan Tree in JSON format.

Output
{
"Goal":{
"Name":"",
"Children":[
{
"Hypothesis":"",
"Children":[
{
"KnowledgeGap":"",
"Children":[
{
"ContentDimension":"",
"Children":[
{
"SectionTitle":"",
"LocalContext":{
"Goal":"",
"Hypothesis":"",
"KnowledgeGap":"",
"ContentDimension":""
},
"GenerationAction":{
"Type":"Explain | Compare | Analyze | Evaluate |
Summarize | Predict",
"Instruction":""
},
"ExecutionStatus":"Pending"
}
]
}
]
}
]
}
]
}
}"""


# ---------------------------------------------------------------------------
# Sample inputs
# ---------------------------------------------------------------------------

SAMPLE_BEHAVIOR_RAW = {
    "BM433": (
        "*2026-06-15*12:00 View stock MSTR quotes*12:00 View stock MSII quotes**"
        "2026-07-01*09:24 View stock META quotes*09:25 Switch to all timeframe*09:25 View stock META quotes*09:26 View stock META quotes**"
        "***************2026-06-15*11:33 View stock MU quotes*11:34 View stock MU quotes*11:56 View stock SPCX quotes*11:56 View stock SPCX quotes*"
        "11:57 View stock MSTR quotes*11:57 View stock MSII quotes*11:57 View stock MSTR quotes**"
        "2026-06-14*08:58 View stock TSLA quotes*08:58 View stock TSLA quotes*09:01 View stock SNDK quotes*09:01 View stock SNDK quotes*"
        "09:01 View stock TSLA quotes*09:01 View stock TSLA quotes*09:02 Switch to predictiontab tab*09:02 Switch to ETFs tab*09:02 View etf INTW quotes*"
        "09:03 View etf INTW quotes*09:03 View etf EUVX quotes*09:03 View etf EUVX quotes*09:04 Switch to Stocks tab*09:05 View stock NVDA quotes*"
        "09:05 View stock NVDA quotes*09:05 View stock NVDA quotes*09:05 View stock NVDA quotes*09:06 View stock COOK quotes*09:06 View stock COOK quotes*"
        "09:07 View stock SENEA quotes*09:07 View stock SENEA quotes*09:07 View stock CABO quotes*09:07 View stock CABO quotes*09:15 View stock VECO quotes*"
        "09:18 View stock VECO quotes*09:23 View stock VECO quotes*09:24 View stock VECO quotes*09:24 View stock ALB quotes*09:24 View stock ALB quotes*"
        "09:40 View stock SNDK quotes*09:56 View stock SNDK quotes*10:00 View stock SNDK quotes*13:14 View stock KLAC quotes*13:14 View stock KLAC quotes*"
        "13:15 View stock KLAC quotes*13:15 View stock KLAC quotes*22:11 Switch to Stocks tab*22:15 View stock ARM quotes*22:15 View stock ARM quotes*"
        "22:15 View stock ARM quotes**"
    )
}

SAMPLE_GRAPH_1 = {
    "graph_id": "LCG_001",
    "timestamp": "2026-06-28T10:30:00Z",
    "nodes": [
        {"id": "C1", "type": "Concept", "name": "NVIDIA", "belief": 1.00, "uncertainty": 0.00, "confidence": 0.99, "priority": 0.20, "status": "Observed"},
        {"id": "C2", "type": "Concept", "name": "GPU Demand", "belief": 0.96, "uncertainty": 0.05, "confidence": 0.95, "priority": 0.32, "status": "Observed"},
        {"id": "C3", "type": "Concept", "name": "Cloud CapEx", "belief": 0.82, "uncertainty": 0.18, "confidence": 0.91, "priority": 0.61, "status": "Observed"},
        {"id": "H1", "type": "Hypothesis", "name": "AI demand will continue to grow", "belief": 0.87, "uncertainty": 0.13, "confidence": 0.93, "priority": 0.84, "status": "Active"},
        {"id": "H2", "type": "Hypothesis", "name": "Cloud providers will further increase AI CapEx", "belief": 0.74, "uncertainty": 0.26, "confidence": 0.89, "priority": 0.78, "status": "Candidate"},
        {"id": "H3", "type": "Hypothesis", "name": "Power infrastructure may become the next bottleneck", "belief": 0.43, "uncertainty": 0.57, "confidence": 0.71, "priority": 0.69, "status": "Candidate"},
        {"id": "G1", "type": "Goal", "name": "Identify the next beneficiary of AI infrastructure growth", "belief": 0.91, "uncertainty": 0.09, "confidence": 0.95, "priority": 1.00, "status": "Active"},
        {"id": "K1", "type": "KnowledgeGap", "name": "Microsoft AI CapEx", "belief": 0.00, "uncertainty": 1.00, "confidence": 0.90, "expected_information_gain": 0.93, "priority": 0.98, "status": "Unexplored"},
        {"id": "K2", "type": "KnowledgeGap", "name": "Google TPU Expansion", "belief": 0.00, "uncertainty": 1.00, "confidence": 0.87, "expected_information_gain": 0.88, "priority": 0.91, "status": "Unexplored"},
        {"id": "K3", "type": "KnowledgeGap", "name": "Power Infrastructure Investment", "belief": 0.00, "uncertainty": 1.00, "confidence": 0.83, "expected_information_gain": 0.79, "priority": 0.76, "status": "Unexplored"},
    ],
    "edges": [
        {"source": "C1", "target": "C2", "relation": "Attention", "strength": "Strong", "weight": 0.95},
        {"source": "C2", "target": "C3", "relation": "Transition", "weight": 0.84},
        {"source": "C2", "target": "H1", "relation": "Evidence", "type": "Supports", "weight": 0.96},
        {"source": "C3", "target": "H2", "relation": "Evidence", "type": "Supports", "weight": 0.89},
        {"source": "H1", "target": "G1", "relation": "Goal", "type": "Core", "weight": 0.97},
        {"source": "H2", "target": "G1", "relation": "Goal", "type": "Supporting", "weight": 0.82},
        {"source": "H3", "target": "G1", "relation": "Goal", "type": "Supporting", "weight": 0.64},
        {"source": "G1", "target": "K1", "relation": "Gap", "type": "Required", "expected_information_gain": 0.93},
        {"source": "G1", "target": "K2", "relation": "Gap", "type": "Required", "expected_information_gain": 0.88},
        {"source": "G1", "target": "K3", "relation": "Gap", "type": "Helpful", "expected_information_gain": 0.79},
    ],
}

SAMPLE_GRAPH_2 = {
    "graph_id": "LCG_002",
    "timestamp": "2026-06-28T10:30:00Z",
    "nodes": [
        {"id": "C1", "type": "Concept", "name": "NVIDIA", "belief": 1.00, "uncertainty": 0.00, "confidence": 0.99, "priority": 0.25, "status": "Observed"},
        {"id": "C2", "type": "Concept", "name": "Semiconductor Valuation", "belief": 0.88, "uncertainty": 0.12, "confidence": 0.92, "priority": 0.40, "status": "Observed"},
        {"id": "H1", "type": "Hypothesis", "name": "AI sector valuation may be overstretched", "belief": 0.62, "uncertainty": 0.38, "confidence": 0.85, "priority": 0.88, "status": "Active"},
        {"id": "H2", "type": "Hypothesis", "name": "Cloud CapEx growth will decelerate", "belief": 0.55, "uncertainty": 0.45, "confidence": 0.80, "priority": 0.75, "status": "Candidate"},
        {"id": "G1", "type": "Goal", "name": "Assess whether AI infrastructure stocks face valuation risk", "belief": 0.89, "uncertainty": 0.11, "confidence": 0.94, "priority": 1.00, "status": "Active"},
        {"id": "K1", "type": "KnowledgeGap", "name": "Forward P/E vs historical AI cycle peaks", "belief": 0.00, "uncertainty": 1.00, "confidence": 0.88, "expected_information_gain": 0.91, "priority": 0.96, "status": "Unexplored"},
        {"id": "K2", "type": "KnowledgeGap", "name": "Hyperscaler capex guidance trends", "belief": 0.00, "uncertainty": 1.00, "confidence": 0.85, "expected_information_gain": 0.87, "priority": 0.90, "status": "Unexplored"},
    ],
    "edges": [
        {"source": "C1", "target": "C2", "relation": "Attention", "weight": 0.93},
        {"source": "C2", "target": "H1", "relation": "Evidence", "type": "Supports", "weight": 0.90},
        {"source": "C2", "target": "H2", "relation": "Evidence", "type": "Supports", "weight": 0.84},
        {"source": "H1", "target": "G1", "relation": "Goal", "type": "Core", "weight": 0.95},
        {"source": "H2", "target": "G1", "relation": "Goal", "type": "Supporting", "weight": 0.78},
        {"source": "G1", "target": "K1", "relation": "Gap", "type": "Required", "expected_information_gain": 0.91},
        {"source": "G1", "target": "K2", "relation": "Gap", "type": "Required", "expected_information_gain": 0.87},
    ],
}

# Reference output schema only — do NOT pass to LLM as input
REFERENCE_GRAPH_OUTPUT = {
    "graph_id": "LCG_001",
    "nodes": [
        {"id": "C1", "type": "Concept", "name": "NVIDIA", "belief": 1.0, "uncertainty": 0.0, "confidence": 0.99, "priority": 0.2, "status": "Observed"},
        {"id": "H1", "type": "Hypothesis", "name": "AI demand will continue to grow", "belief": 0.87, "uncertainty": 0.13, "confidence": 0.93, "priority": 0.84, "status": "Active"},
        {"id": "G1", "type": "Goal", "name": "Identify the next beneficiary of AI infrastructure growth", "belief": 0.91, "uncertainty": 0.09, "confidence": 0.95, "priority": 1.0, "status": "Active"},
        {"id": "K1", "type": "KnowledgeGap", "name": "Microsoft AI CapEx", "belief": 0.0, "uncertainty": 1.0, "confidence": 0.9, "expected_information_gain": 0.93, "priority": 0.98, "status": "Unexplored"},
    ],
    "edges": [
        {"source": "C1", "target": "H1", "relation": "Evidence", "type": "Supports", "weight": 0.96},
        {"source": "H1", "target": "G1", "relation": "Goal", "type": "Core", "weight": 0.97},
        {"source": "G1", "target": "K1", "relation": "Gap", "type": "Required", "expected_information_gain": 0.93},
    ],
}

# Module 2 output example (Module 3 input)
SAMPLE_BELIEF_SPACE = {
    "belief_space": [
        {
            "agent": "Causal Agent",
            "belief": 0.82,
            "reason": "Strong behavior consistency; repeated NVDA/ARM/KLAC views support AI infrastructure causal chain.",
            "graph_edits": [],
            "graph": SAMPLE_GRAPH_1,
        },
        {
            "agent": "Comparative Agent",
            "belief": 0.71,
            "reason": "Valuation-risk interpretation fits some behaviors but weaker cross-agent support on CapEx growth.",
            "graph_edits": [],
            "graph": SAMPLE_GRAPH_2,
        },
    ],
    "entropy": 0.673,
}
