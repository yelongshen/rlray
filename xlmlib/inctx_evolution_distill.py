import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


def _tokenize_text(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass
class EvolutionStep:
    model_id: str
    generation: str
    reward: float
    delta_reward: float


@dataclass
class EvolutionTrace:
    trace_id: str
    problem: str
    answer: Optional[str]
    steps: List[EvolutionStep]
    quality_score: float


def build_evolution_traces(
    rollout_events: List[Dict],
    min_steps: int = 2,
    max_steps: int = 5,
    keep_only_improving_steps: bool = True,
) -> List[EvolutionTrace]:
    """Convert flat rollout events into per-problem evolution traces.

    Required event fields:
      - problem_id (or id/unique_id)
      - problem
      - model_id (checkpoint identifier, sortable as string)
      - generation
      - reward
    """
    grouped: Dict[str, List[Dict]] = {}
    for event in rollout_events:
        pid = (
            str(event.get("problem_id"))
            if event.get("problem_id") is not None
            else str(event.get("id", event.get("unique_id", "")))
        )
        if not pid:
            continue
        grouped.setdefault(pid, []).append(event)

    traces: List[EvolutionTrace] = []
    for pid, events in grouped.items():
        events = sorted(events, key=lambda e: str(e.get("model_id", "")))

        steps: List[EvolutionStep] = []
        prev_reward = None
        for ev in events:
            reward = _safe_float(ev.get("reward", 0.0), 0.0)
            delta = 0.0 if prev_reward is None else reward - prev_reward
            prev_reward = reward

            if keep_only_improving_steps and steps and delta < 0:
                continue

            steps.append(
                EvolutionStep(
                    model_id=str(ev.get("model_id", "unknown")),
                    generation=str(ev.get("generation", "")),
                    reward=reward,
                    delta_reward=delta,
                )
            )

        if len(steps) < min_steps:
            continue

        # Keep a compact chain with first + strongest reward jump + final steps.
        if len(steps) > max_steps:
            first = steps[0]
            final = steps[-1]
            internal = steps[1:-1]
            internal = sorted(internal, key=lambda s: s.delta_reward, reverse=True)
            internal = internal[: max(0, max_steps - 2)]
            internal = sorted(internal, key=lambda s: s.model_id)
            steps = [first] + internal + [final]

        positive_deltas = sum(max(0.0, s.delta_reward) for s in steps[1:])
        final_reward = steps[-1].reward
        quality = final_reward + 0.5 * positive_deltas - 0.02 * len(steps)

        problem_text = str(events[-1].get("problem", ""))
        answer = events[-1].get("answer")
        traces.append(
            EvolutionTrace(
                trace_id=pid,
                problem=problem_text,
                answer=str(answer) if answer is not None else None,
                steps=steps,
                quality_score=quality,
            )
        )

    traces.sort(key=lambda t: t.quality_score, reverse=True)
    return traces


def save_traces_jsonl(traces: List[EvolutionTrace], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for trace in traces:
            record = {
                "trace_id": trace.trace_id,
                "problem": trace.problem,
                "answer": trace.answer,
                "quality_score": trace.quality_score,
                "steps": [
                    {
                        "model_id": step.model_id,
                        "generation": step.generation,
                        "reward": step.reward,
                        "delta_reward": step.delta_reward,
                    }
                    for step in trace.steps
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_traces_jsonl(path: str) -> List[EvolutionTrace]:
    traces: List[EvolutionTrace] = []
    for rec in load_jsonl(path):
        steps = [
            EvolutionStep(
                model_id=str(s.get("model_id", "unknown")),
                generation=str(s.get("generation", "")),
                reward=_safe_float(s.get("reward", 0.0), 0.0),
                delta_reward=_safe_float(s.get("delta_reward", 0.0), 0.0),
            )
            for s in rec.get("steps", [])
        ]
        if not steps:
            continue
        traces.append(
            EvolutionTrace(
                trace_id=str(rec.get("trace_id", "")),
                problem=str(rec.get("problem", "")),
                answer=rec.get("answer"),
                steps=steps,
                quality_score=_safe_float(rec.get("quality_score", 0.0), 0.0),
            )
        )
    traces.sort(key=lambda t: t.quality_score, reverse=True)
    return traces


def _problem_similarity(a: str, b: str) -> float:
    ta = set(_tokenize_text(a))
    tb = set(_tokenize_text(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta.intersection(tb))
    union = len(ta.union(tb))
    return inter / max(1, union)


def retrieve_relevant_traces(
    query_problem: str,
    traces: List[EvolutionTrace],
    top_k: int,
) -> List[EvolutionTrace]:
    scored: List[Tuple[float, EvolutionTrace]] = []
    for trace in traces:
        sim = _problem_similarity(query_problem, trace.problem)
        score = sim + 0.1 * trace.quality_score
        scored.append((score, trace))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:top_k]]


def _compress_generation(text: str, max_chars: int = 420) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_evolution_prompt(
    target_problem: str,
    retrieved_traces: List[EvolutionTrace],
    prompt_type: str = "v11",
) -> str:
    # prompt_type is kept for interface compatibility with existing scripts.
    del prompt_type

    lines: List[str] = []
    lines.append("You are solving competition math problems.")
    lines.append("Learn from the evolution traces: each trace shows how model behavior improved.")
    lines.append("Apply the same correction pattern to the target problem.")
    lines.append("")

    for idx, trace in enumerate(retrieved_traces, start=1):
        lines.append(f"[Evolution Trace {idx}]")
        lines.append(f"Problem: {trace.problem}")
        for sidx, step in enumerate(trace.steps, start=1):
            lines.append(
                f"Step {sidx} | model={step.model_id} | reward={step.reward:.3f} | delta={step.delta_reward:+.3f}"
            )
            lines.append(f"Attempt summary: {_compress_generation(step.generation)}")
        lines.append("")

    lines.append("[Target Problem]")
    lines.append(target_problem)
    lines.append("")
    lines.append("Instructions:")
    lines.append("1) Solve step by step.")
    lines.append("2) Self-check your derivation once.")
    lines.append("3) Put the final answer in \\boxed{}.")
    return "\n".join(lines)