from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure local imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from tabulate import tabulate


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
    """Read JSON conversations from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Calculate recall score based on fraction of expected keywords present."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matched = sum(1 for item in expected if item.lower() in ans_lower)
    return matched / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight response quality metric considering recall and conciseness/structure."""
    r_score = recall_points(answer, expected)
    ans_len = len(answer.strip())

    # Penalty for empty or overly verbose responses
    if ans_len == 0:
        return 0.0

    length_score = 1.0
    if ans_len < 20 or ans_len > 600:
        length_score = 0.7

    # Structure bonus (bullets, clean punctuation)
    structure_bonus = 0.1 if ("-" in answer or "\n" in answer) else 0.0

    score = (r_score * 0.75) + (length_score * 0.15) + structure_bonus
    return round(min(1.0, max(0.0, score)), 2)


def run_agent_benchmark(
    agent_name: str, agent: Any, conversations: list[dict[str, Any]], config: LabConfig
) -> BenchmarkRow:
    """Evaluate one agent over multiple conversations and recall inquiries."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    all_recall_scores: list[float] = []
    all_quality_scores: list[float] = []
    total_compactions = 0
    users_seen: set[str] = set()

    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = conv["id"]
        users_seen.add(user_id)

        # 1. Feed regular conversation turns to the agent
        for turn in conv["turns"]:
            reply = agent.reply(user_id, thread_id, turn)
            total_agent_tokens += reply["token_usage"]
            total_prompt_tokens += reply["prompt_tokens"]

        # Track compactions in this conversation thread
        total_compactions += agent.compaction_count(thread_id)

        # 2. Ask recall questions in a brand new fresh thread (testing cross-session memory)
        for idx, rq in enumerate(conv.get("recall_questions", [])):
            fresh_thread_id = f"recall-{conv['id']}-{idx}"
            reply = agent.reply(user_id, fresh_thread_id, rq["question"])
            total_agent_tokens += reply["token_usage"]
            total_prompt_tokens += reply["prompt_tokens"]

            r_score = recall_points(reply["content"], rq["expected_contains"])
            q_score = heuristic_quality(reply["content"], rq["expected_contains"])
            all_recall_scores.append(r_score)
            all_quality_scores.append(q_score)

    avg_recall = sum(all_recall_scores) / len(all_recall_scores) if all_recall_scores else 0.0
    avg_quality = sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0.0

    # Calculate persistent memory file growth
    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        for uid in users_seen:
            memory_growth += agent.memory_file_size(uid)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a clean Markdown table with required columns."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table_data = []
    for r in rows:
        table_data.append(
            [
                r.agent_name,
                f"{r.agent_tokens_only:,}",
                f"{r.prompt_tokens_processed:,}",
                f"{r.recall_score * 100:.1f}%",
                f"{r.response_quality * 100:.1f}%",
                f"{r.memory_growth_bytes:,} B",
                r.compactions,
            ]
        )
    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both standard and long-context stress benchmarks."""
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root)

    standard_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    print("=" * 80)
    print("PHASE 2, TRACK 3, DAY 17: MEMORY SYSTEMS BENCHMARK SUITE")
    print("=" * 80)

    # -------------------------------------------------------------
    # Suite 1: Standard Benchmark
    # -------------------------------------------------------------
    print("\n--- Running Suite 1: Standard Benchmark (10 conversations, user 'dungct') ---")
    std_convs = load_conversations(standard_path)

    # Initialize fresh agents for standard suite
    baseline_std = BaselineAgent(config, force_offline=True)
    advanced_std = AdvancedAgent(config, force_offline=True)

    row_b_std = run_agent_benchmark("Baseline Agent", baseline_std, std_convs, config)
    row_a_std = run_agent_benchmark("Advanced Agent", advanced_std, std_convs, config)

    print("\n### Kết quả Standard Benchmark:")
    print(format_rows([row_b_std, row_a_std]))

    # -------------------------------------------------------------
    # Suite 2: Long-Context Stress Benchmark
    # -------------------------------------------------------------
    print("\n--- Running Suite 2: Long-Context Stress Benchmark (user 'dungct_stress') ---")
    stress_convs = load_conversations(stress_path)

    # Initialize fresh agents for stress suite
    baseline_stress = BaselineAgent(config, force_offline=True)
    advanced_stress = AdvancedAgent(config, force_offline=True)

    row_b_stress = run_agent_benchmark("Baseline Agent", baseline_stress, stress_convs, config)
    row_a_stress = run_agent_benchmark("Advanced Agent", advanced_stress, stress_convs, config)

    print("\n### Kết quả Long-Context Stress Benchmark:")
    print(format_rows([row_b_stress, row_a_stress]))

    # -------------------------------------------------------------
    # Summary & Analysis
    # -------------------------------------------------------------
    prompt_reduction = (
        (row_b_stress.prompt_tokens_processed - row_a_stress.prompt_tokens_processed)
        / row_b_stress.prompt_tokens_processed
        * 100
    )

    print("\n" + "=" * 80)
    print("PHÂN TÍCH TRADE-OFF VÀ KẾT LUẬN:")
    print("=" * 80)
    print(f"1. Khả năng nhớ chéo phiên (Cross-session Recall):")
    print(f"   - Baseline Agent: {row_b_std.recall_score*100:.1f}% (Standard) và {row_b_stress.recall_score*100:.1f}% (Stress).")
    print(f"     -> Baseline chỉ duy trì short-term session; sang thread mới hoàn toàn quên facts.")
    print(f"   - Advanced Agent: {row_a_std.recall_score*100:.1f}% (Standard) và {row_a_stress.recall_score*100:.1f}% (Stress).")
    print(f"     -> Nhờ persistent User.md, Advanced duy trì facts ổn định qua mọi phiên mới.")
    print(f"\n2. Tối ưu chi phí ngữ cảnh (Prompt Tokens Processed):")
    print(f"   - Ở Standard Benchmark (hội thoại ngắn ~10 lượt):")
    print(f"     Prompt tokens Baseline ({row_b_std.prompt_tokens_processed:,}) so với Advanced ({row_a_std.prompt_tokens_processed:,}).")
    print(f"     Advanced có overhead nhẹ do mang theo User.md profile và system prompt.")
    print(f"   - Ở Stress Benchmark (hội thoại dài 16 lượt dày đặc):")
    print(f"     Prompt tokens Baseline: {row_b_stress.prompt_tokens_processed:,} tokens.")
    print(f"     Prompt tokens Advanced: {row_a_stress.prompt_tokens_processed:,} tokens.")
    print(f"     -> Compact memory kích hoạt {row_a_stress.compactions} lần, giúp giảm {prompt_reduction:.1f}% lượng prompt tokens!")
    print(f"\n3. Bộ nhớ bền vững và tốc độ phình to (Memory Growth):")
    print(f"   - File User.md được tổ chức có cấu trúc (khoảng {row_a_std.memory_growth_bytes} B).")
    print(f"   - Cơ chế conflict resolution ngăn chặn việc trùng lặp thông tin mâu thuẫn.")
    print("=" * 80)


if __name__ == "__main__":
    main()
