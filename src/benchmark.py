from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    answer_lower = answer.lower()
    hits = sum(1 for e in expected if e.lower() in answer_lower)
    if hits == len(expected):
        return 1.0
    if hits > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    if not answer.strip():
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for e in expected if e.lower() in answer_lower)
    coverage = hits / len(expected) if expected else 0.0
    length_score = min(1.0, len(answer) / 50)
    return round(0.6 * coverage + 0.4 * length_score, 2)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> tuple[BenchmarkRow, list[dict[str, Any]]]:
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    total_compactions = 0
    details: list[dict[str, Any]] = []

    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = conv["id"]
        turns: list[str] = conv["turns"]
        recall_questions: list[dict[str, Any]] = conv.get("recall_questions", [])
        conv_detail: dict[str, Any] = {
            "conversation_id": thread_id,
            "user_id": user_id,
            "turn_count": len(turns),
            "turns": [],
            "recall_results": [],
        }

        for turn in turns:
            result = agent.reply(user_id, thread_id, turn)
            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)
            conv_detail["turns"].append({
                "input": turn,
                "output": result.get("response", ""),
                "agent_tokens": result.get("agent_tokens", 0),
                "prompt_tokens": result.get("prompt_tokens", 0),
            })

        compactions = agent.compaction_count(thread_id) if hasattr(agent, "compaction_count") else 0
        total_compactions += compactions

        recall_thread = f"{thread_id}-recall"
        for rq in recall_questions:
            question = rq["question"]
            expected: list[str] = rq["expected_contains"]
            answer_result = agent.reply(user_id, recall_thread, question)
            answer_text = answer_result.get("response", "")

            rp = recall_points(answer_text, expected)
            hq = heuristic_quality(answer_text, expected)
            recall_scores.append(rp)
            quality_scores.append(hq)
            total_agent_tokens += answer_result.get("agent_tokens", 0)
            total_prompt_tokens += answer_result.get("prompt_tokens", 0)

            conv_detail["recall_results"].append({
                "question": question,
                "expected": expected,
                "answer": answer_text,
                "recall_score": rp,
                "quality_score": hq,
            })

        conv_detail["compactions"] = compactions
        details.append(conv_detail)

    avg_recall = round(sum(recall_scores) / len(recall_scores), 2) if recall_scores else 0.0
    avg_quality = round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0.0

    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        memory_growth = agent.memory_file_size(conversations[0]["user_id"]) if conversations else 0

    row = BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )
    return row, details


def format_rows(rows: list[BenchmarkRow]) -> str:
    header = "| Agent | Agent tokens | Prompt tokens | Recall | Quality | Memory (bytes) | Compactions |"
    separator = "|---|---|---|---|---|---|---|"
    lines = [header, separator]
    for r in rows:
        line = (
            f"| {r.agent_name} "
            f"| {r.agent_tokens_only} "
            f"| {r.prompt_tokens_processed} "
            f"| {r.recall_score:.2f} "
            f"| {r.response_quality:.2f} "
            f"| {r.memory_growth_bytes} "
            f"| {r.compactions} |"
        )
        lines.append(line)
    return "\n".join(lines)


def save_results(output_path: Path, report: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    all_results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {},
    }

    print("=" * 60)
    print("STANDARD BENCHMARK")
    print("=" * 60)

    standard_data = load_conversations(config.data_dir / "conversations.json")
    baseline_std = BaselineAgent(config=config, force_offline=True)
    advanced_std = AdvancedAgent(config=config, force_offline=True)

    row_baseline_std, detail_baseline_std = run_agent_benchmark("Baseline", baseline_std, standard_data, config)
    row_advanced_std, detail_advanced_std = run_agent_benchmark("Advanced", advanced_std, standard_data, config)

    print(format_rows([row_baseline_std, row_advanced_std]))

    all_results["benchmarks"]["standard"] = {
        "summary": [asdict(row_baseline_std), asdict(row_advanced_std)],
        "baseline_details": detail_baseline_std,
        "advanced_details": detail_advanced_std,
    }

    print()
    print("=" * 60)
    print("LONG-CONTEXT STRESS BENCHMARK")
    print("=" * 60)

    stress_data = load_conversations(config.data_dir / "advanced_long_context.json")
    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)

    row_baseline_stress, detail_baseline_stress = run_agent_benchmark("Baseline (stress)", baseline_stress, stress_data, config)
    row_advanced_stress, detail_advanced_stress = run_agent_benchmark("Advanced (stress)", advanced_stress, stress_data, config)

    print(format_rows([row_baseline_stress, row_advanced_stress]))

    all_results["benchmarks"]["stress"] = {
        "summary": [asdict(row_baseline_stress), asdict(row_advanced_stress)],
        "baseline_details": detail_baseline_stress,
        "advanced_details": detail_advanced_stress,
    }

    output_path = config.state_dir / "benchmark_results.json"
    save_results(output_path, all_results)


if __name__ == "__main__":
    main()
