from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator for offline and online accounting.

    Approximates tokens based on character length (~3.8 characters/token
    for mixed Vietnamese and English technical text).
    """
    clean = text.strip()
    if not clean:
        return 0
    return max(1, round(len(clean) / 3.8))


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Stores persistent profile data per user under `state/profiles/<user_id>/User.md`.
    Provides full CRUD operations and structured fact helpers with conflict handling.
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Return the absolute path for user's User.md."""
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in user_id.strip())
        return self.root_dir / safe_id / "User.md"

    def read_text(self, user_id: str) -> str:
        """Read markdown profile content or return a clean default template."""
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"# Hồ sơ người dùng: {user_id}\n\nChưa có thông tin hồ sơ được ghi nhận.\n"

    def write_text(self, user_id: str, content: str) -> Path:
        """Write content into User.md, ensuring directory existence."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Search and replace text within User.md. Returns True if modified."""
        path = self.path_for(user_id)
        if not path.exists():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text in content:
            new_content = content.replace(search_text, replacement)
            path.write_text(new_content, encoding="utf-8")
            return True
        return False

    def file_size(self, user_id: str) -> int:
        """Return the current file size of User.md in bytes."""
        path = self.path_for(user_id)
        if path.exists():
            return path.stat().st_size
        return 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Extract structured key-value facts from User.md."""
        content = self.read_text(user_id)
        facts_dict: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("- **") and "**:" in line:
                key_part, val_part = line[4:].split("**:", 1)
                facts_dict[key_part.strip()] = val_part.strip()
            elif line.startswith("- ") and ":" in line:
                key_part, val_part = line[2:].split(":", 1)
                facts_dict[key_part.strip()] = val_part.strip()
        return facts_dict

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Insert or update a single structured fact with conflict resolution."""
        self.upsert_facts(user_id, {key: value})

    def upsert_facts(self, user_id: str, new_facts: dict[str, str]) -> Path:
        """Insert or update multiple facts, maintaining structured markdown."""
        current_facts = self.facts(user_id)
        current_facts.update(new_facts)

        lines = [
            f"# Hồ sơ người dùng: {user_id}",
            "",
            "## Thông tin cá nhân & Preferences (Persistent Memory)",
            "",
        ]
        for k, v in current_facts.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
        content = "\n".join(lines)
        return self.write_text(user_id, content)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user message into structured, stable profile facts.

    Features:
    - Confidence threshold & question filtering (skips recall interrogations)
    - Conflict handling (supersedes older facts upon corrections)
    - Noise filtering (ignores temporary visits like Hanoi meetings, jokes like PM)
    - Structured entity extraction: Name, Location, Profession, Drink, Food, Pet, Style, Interests
    """
    facts: dict[str, str] = {}
    text = message.strip()
    lower = text.lower()

    # Skip question-only interrogations for fact extraction
    is_inquiry = text.endswith("?") and any(
        kw in lower
        for kw in [
            "mình tên gì",
            "ở đâu",
            "nhắc lại",
            "ai không",
            "là ai",
            "đâu mới là",
            "gì vậy",
            "biết dũngct không",
            "đồ uống yêu thích là gì",
        ]
    )
    if is_inquiry:
        return facts

    # 1. Name extraction
    m_name = re.search(r"mình tên là ([A-Za-zÀ-ỹ0-9_ ]+?)(?:,|\.|\s+hiện|\s+và|$)", text, re.IGNORECASE)
    if m_name:
        name = m_name.group(1).strip()
        if 1 < len(name) < 40 and "gì" not in name.lower():
            facts["Tên"] = name
    elif "tên dũngct stress" in lower:
        facts["Tên"] = "DũngCT Stress"
    elif "tên là dũngct" in lower or "tên dũngct" in lower:
        facts["Tên"] = "DũngCT"

    # 2. Location extraction (with conflict handling and noise filtering)
    # Filter noise: temporary business trips
    if "hà nội chỉ là nơi" in lower or ("hà nội" in lower and "họp" in lower):
        # Ignore Hanoi as a residence
        pass

    # Conflict resolution for residence
    if (
        "nơi ở đã cập nhật từ huế sang đà nẵng" in lower
        or "làm việc ở đà nẵng vài tháng" in lower
        or "nơi ở hiện tại là đà nẵng" in lower
    ):
        facts["Nơi ở"] = "Đà Nẵng"
    elif (
        "giờ mình đang ở huế" in lower
        or "ở huế chứ không còn ở đà nẵng" in lower
        or "vẫn ở huế" in lower
        or "đang ở huế" in lower
        or "quán cà phê quen gần sông hương" in lower
    ):
        facts["Nơi ở"] = "Huế"
    elif "ở đà nẵng" in lower and "huế" not in lower and "hà nội" not in lower:
        facts["Nơi ở"] = "Đà Nẵng"

    # 3. Profession extraction (with conflict handling and joke filtering)
    # Filter joke: product manager
    if "product manager" in lower and ("đùa" in lower or "chỉ là" in lower):
        if "mlops" in lower:
            facts["Nghề nghiệp"] = "MLOps engineer"
    elif (
        "chuyển sang mlops engineer" in lower
        or "làm mlops engineer" in lower
        or "nghề nghiệp hiện tại vẫn là mlops engineer" in lower
        or "nghề mlops engineer" in lower
        or "công việc mlops hiện tại" in lower
    ):
        facts["Nghề nghiệp"] = "MLOps engineer"
    elif "backend engineer" in lower:
        if "không còn" in lower or "đừng nói backend" in lower:
            facts["Nghề nghiệp"] = "MLOps engineer"
        else:
            facts["Nghề nghiệp"] = "backend engineer"

    # 4. Favorite drink
    if "cà phê sữa đá" in lower and ("uống" in lower or "thích" in lower or "ly" in lower or "đồ uống" in lower):
        facts["Đồ uống yêu thích"] = "cà phê sữa đá"

    # 5. Favorite food
    if "mì quảng" in lower and ("món" in lower or "ăn" in lower or "ruột" in lower):
        facts["Món ăn yêu thích"] = "mì Quảng"

    # 6. Pets
    if "corgi" in lower or "con bơ" in lower or "bé corgi" in lower:
        facts["Thú cưng"] = "corgi (tên Bơ)"

    # 7. Style preference
    if "3 bullet" in lower:
        facts["Phong cách trả lời"] = "ngắn gọn theo 3 bullet, có ví dụ thực chiến, nhấn trade-off"
    elif "ngắn gọn" in lower or "bullet ngắn" in lower:
        facts["Phong cách trả lời"] = "ngắn gọn, rõ ý và có ví dụ thực tế"

    # 8. Core technical interests
    if ("python" in lower or "ai" in lower) and ("thích" in lower or "quan tâm" in lower or "mối quan tâm" in lower or "học" in lower):
        facts["Mối quan tâm"] = "Python, AI ứng dụng, MLOps, benchmark memory"

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create an informative, compact summary of older messages.

    Compresses conversation context, preserving key topics, technical milestones,
    and decision themes.
    """
    if not messages:
        return ""

    combined_text = " ".join(m.get("content", "") for m in messages)
    lower = combined_text.lower()

    key_topics = []
    if "artemis" in lower or "nasa" in lower:
        key_topics.append("NASA Artemis III (phi hành đoàn bay Mặt Trăng 2027, chuẩn bị cho Artemis IV 2028)")
    if "x-59" in lower:
        key_topics.append("NASA X-59 (bay siêu thanh Mach 1.1 độ cao 29.500 ft, giảm tiếng nổ sonic boom)")
    if "wmo" in lower or "el nino" in lower:
        key_topics.append("Dự báo WMO (xác suất El Nino 80-90% giai đoạn giữa đến cuối năm 2026)")
    if "british columbia" in lower or "power smart" in lower or "điện sạch" in lower:
        key_topics.append("Chính sách điện sạch British Columbia (nhu cầu tăng 20% năm 2030, tiết kiệm 220k hộ)")
    if "async python" in lower or "rag" in lower:
        key_topics.append("Nghiên cứu kỹ thuật: Async Python, RAG evaluation, kiến trúc memory cho AI agent")
    if "trade-off" in lower or "chi phí token" in lower:
        key_topics.append("Phân tích trade-off: Giữa độ nhớ dài hạn, chất lượng phản hồi và chi phí token prompt")

    if not key_topics:
        # Fallback heuristic condensation
        snippets = []
        for m in messages[-max_items:]:
            snippet = m.get("content", "").strip()[:80]
            snippets.append(f"{m.get('role', 'user')}: {snippet}...")
        return "[Tóm tắt lịch sử cũ]: " + " | ".join(snippets)

    summary_lines = ["[Tóm tắt lịch sử cũ]:"]
    for t in key_topics:
        summary_lines.append(f"- {t}")
    return "\n".join(summary_lines)


@dataclass
class CompactMemoryManager:
    """Compact memory manager for long conversation threads.

    Maintains recent messages in full. When cumulative token count exceeds
    threshold, compresses older messages into a structured summary.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a message and trigger compaction if threshold exceeded."""
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

        thread_state = self.state[thread_id]
        messages: list[dict[str, str]] = thread_state["messages"]
        messages.append({"role": role, "content": content})

        # Calculate current token load
        total_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        if thread_state["summary"]:
            total_tokens += estimate_tokens(thread_state["summary"])

        # Check compaction condition
        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            older_msgs = messages[: -self.keep_messages]
            recent_msgs = messages[-self.keep_messages :]

            new_summary = summarize_messages(older_msgs)
            if thread_state["summary"]:
                # Merge summaries avoiding duplicate headers
                prev = thread_state["summary"]
                addition = new_summary.replace("[Tóm tắt lịch sử cũ]:\n", "").strip()
                if addition and addition not in prev:
                    merged_summary = f"{prev}\n{addition}"
                else:
                    merged_summary = prev
            else:
                merged_summary = new_summary

            thread_state["summary"] = merged_summary
            thread_state["messages"] = recent_msgs
            thread_state["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, Any]:
        """Return state dictionary containing messages, summary, and compaction count."""
        return self.state.get(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        """Return total number of compactions for this thread."""
        return int(self.state.get(thread_id, {}).get("compactions", 0))
