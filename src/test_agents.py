from __future__ import annotations

import sys
from pathlib import Path

# Ensure src modules can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated LabConfig for testing."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    provider_cfg = ProviderConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
    )

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=80,  # Low threshold to trigger compaction quickly in tests
        compact_keep_messages=2,
        model=provider_cfg,
        judge_model=provider_cfg,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, read, updated, and edited with search/replace."""
    profiles_dir = tmp_path / "profiles"
    store = UserProfileStore(profiles_dir)
    user_id = "test_user_01"

    # 1. Initial read should return default markdown
    default_text = store.read_text(user_id)
    assert user_id in default_text
    assert store.file_size(user_id) == 0

    # 2. Write custom profile
    content = "# Hồ sơ người dùng: test_user_01\n- **Tên**: Nam\n- **Nơi ở**: Hà Nội\n"
    written_path = store.write_text(user_id, content)
    assert written_path.exists()
    assert store.file_size(user_id) > 0
    assert store.read_text(user_id) == content

    # 3. Edit text (search & replace)
    success = store.edit_text(user_id, "Hà Nội", "Đà Nẵng")
    assert success is True
    updated_content = store.read_text(user_id)
    assert "Đà Nẵng" in updated_content
    assert "Hà Nội" not in updated_content

    # 4. Upsert structured facts
    store.upsert_fact(user_id, "Nghề nghiệp", "AI Engineer")
    facts = store.facts(user_id)
    assert facts.get("Tên") == "Nam"
    assert facts.get("Nơi ở") == "Đà Nẵng"
    assert facts.get("Nghề nghiệp") == "AI Engineer"


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long conversation threads trigger memory compaction."""
    manager = CompactMemoryManager(threshold_tokens=60, keep_messages=2)
    thread_id = "test-thread-01"

    # Append short message: no compaction yet
    manager.append(thread_id, "user", "Xin chào bạn.")
    assert manager.compaction_count(thread_id) == 0

    # Append multiple long messages to exceed 60 tokens threshold
    manager.append(
        thread_id,
        "assistant",
        "Chào bạn, mình là trợ lý AI. Mình có thể hỗ trợ bạn về kiến trúc hệ thống và memory đa tầng.",
    )
    manager.append(
        thread_id,
        "user",
        "Hôm nay mình muốn thảo luận chi tiết về trade-off giữa short-term memory, persistent memory và compact memory trong production agent.",
    )
    manager.append(
        thread_id,
        "assistant",
        "Vấn đề cốt lõi là kiểm soát chi phí token prompt khi hội thoại kéo dài mà vẫn không làm mất fact quan trọng của người dùng.",
    )

    # Verify compaction was triggered
    count = manager.compaction_count(thread_id)
    assert count >= 1

    ctx = manager.context(thread_id)
    # Check that older content was summarized
    assert len(ctx["summary"]) > 0
    # Check that kept messages are within limit
    assert len(ctx["messages"]) <= 3


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify Advanced Agent remembers across threads while Baseline Agent forgets."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "dungct_test"

    # Thread 1: User introduces facts
    t1_msg = "Chào bạn, mình tên là DũngCT, ở Đà Nẵng và đồ uống yêu thích là cà phê sữa đá."
    baseline.reply(user_id, "thread-1", t1_msg)
    advanced.reply(user_id, "thread-1", t1_msg)

    # Thread 2: Fresh session / brand new thread
    recall_question = "Mình tên gì và đồ uống yêu thích là gì?"
    base_reply = baseline.reply(user_id, "thread-2-fresh", recall_question)
    adv_reply = advanced.reply(user_id, "thread-2-fresh", recall_question)

    # Baseline should NOT know the user's name or favorite drink in thread-2
    assert "dũngct" not in base_reply["content"].lower()
    assert "cà phê sữa đá" not in base_reply["content"].lower()

    # Advanced MUST recall both facts via persistent User.md
    assert "dũngct" in adv_reply["content"].lower()
    assert "cà phê sữa đá" in adv_reply["content"].lower()


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long conversation thread."""
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "stress_user"
    thread_id = "long-thread"

    long_turns = [
        "Tin tức số một là chương trình không gian Artemis III và kế hoạch bay thử nghiệm năm 2027.",
        "Tin thứ hai là máy bay X-59 với mục tiêu giảm độ ồn sonic boom khi bay siêu thanh Mach 1.1.",
        "Tin thứ ba là báo cáo thời tiết của WMO cảnh báo hiện tượng El Nino với xác suất rất cao.",
        "Tin thứ tư là chính sách năng lượng sạch của British Columbia với chương trình Power Smart 2.0.",
        "Kết nối bốn tin này cho thấy bài toán quản trị phụ thuộc, giảm ngoại tác tiêu cực và truyền thông rủi ro.",
        "Mỗi lượt kéo theo đoạn văn dài như thế này sẽ làm ngữ cảnh phình to rất nhanh.",
        "Baseline sẽ phải mang theo toàn bộ lịch sử thô qua từng lượt.",
        "Advanced kích hoạt compact memory để nén các lượt cũ thành bản tóm tắt súc tích.",
    ]

    for turn in long_turns:
        baseline.reply(user_id, thread_id, turn)
        advanced.reply(user_id, thread_id, turn)

    # Baseline accumulates all previous messages every turn
    base_prompt_tokens = baseline.prompt_token_usage(thread_id)
    # Advanced compacts older messages, bounding prompt size
    adv_prompt_tokens = advanced.prompt_token_usage(thread_id)

    assert advanced.compaction_count(thread_id) > 0
    assert adv_prompt_tokens < base_prompt_tokens


def test_conflict_resolution_and_noise_filtering(tmp_path: Path) -> None:
    """Bonus Test: verify conflict handling (updates supersede old facts) and noise filtering."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    user_id = "test_conflict_user"

    # Step 1: Initial statements
    agent.reply(user_id, "t1", "Chào bạn, mình tên là DũngCT, ở Đà Nẵng và làm backend engineer.")
    facts1 = agent.profile_store.facts(user_id)
    assert facts1.get("Nơi ở") == "Đà Nẵng"
    assert facts1.get("Nghề nghiệp") == "backend engineer"

    # Step 2: Noise insertion (temporary meeting trip & joke about product manager)
    agent.reply(
        user_id,
        "t1",
        "Hà Nội chỉ là nơi mình vừa bay ra họp 2 ngày thôi. Có lúc mình đùa chuyển sang product manager, nhưng chỉ là câu đùa thôi nhé.",
    )
    facts2 = agent.profile_store.facts(user_id)
    # Ensure noise was filtered out
    assert facts2.get("Nơi ở") != "Hà Nội"
    assert facts2.get("Nghề nghiệp") != "product manager"

    # Step 3: Correction statement
    agent.reply(
        user_id,
        "t1",
        "Đính chính nhé: giờ mình đang ở Huế chứ không còn ở Đà Nẵng, và mình chuyển sang làm MLOps engineer.",
    )
    facts3 = agent.profile_store.facts(user_id)
    # Ensure new facts superseded older conflicting ones
    assert facts3.get("Nơi ở") == "Huế"
    assert facts3.get("Nghề nghiệp") == "MLOps engineer"


def test_confidence_threshold_ignores_inquiries(tmp_path: Path) -> None:
    """Bonus Test: verify confidence threshold skips question turns so recall questions don't corrupt profile."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    user_id = "test_inquiry_user"

    agent.reply(user_id, "t1", "Chào bạn, mình tên là DũngCT và thích cà phê sữa đá.")
    facts_before = agent.profile_store.facts(user_id)

    # Ask questions (inquiries)
    agent.reply(user_id, "t2", "Bạn có biết DũngCT là ai không? Mình tên gì vậy?")
    facts_after = agent.profile_store.facts(user_id)

    assert facts_before == facts_after
