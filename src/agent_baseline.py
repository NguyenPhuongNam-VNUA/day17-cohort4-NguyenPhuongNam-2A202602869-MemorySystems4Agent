from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A / Baseline Agent.

    Characteristics:
    - Within-session short-term memory only (scoped strictly to thread_id)
    - No persistent User.md storage
    - Forgets all user facts across new sessions/threads
    - No compact memory; accumulates all previous messages in prompt context
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not self.force_offline:
            try:
                self.langchain_agent = self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Process a turn and return the assistant response with token metrics."""
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(thread_id, message)

        try:
            # Live LangChain invocation
            session = self._get_session(thread_id)
            prompt_context_tokens = self._calculate_prompt_tokens(thread_id, message)
            session.prompt_tokens_processed += prompt_context_tokens

            # Prepare message history for live model
            history = [{"role": m["role"], "content": m["content"]} for m in session.messages]
            history.append({"role": "user", "content": message})

            response = self.langchain_agent.invoke(history)
            response_text = response.content if hasattr(response, "content") else str(response)

            agent_tokens = estimate_tokens(response_text)
            session.token_usage += agent_tokens
            session.messages.append({"role": "user", "content": message})
            session.messages.append({"role": "assistant", "content": response_text})

            return {
                "role": "assistant",
                "content": response_text,
                "token_usage": agent_tokens,
                "prompt_tokens": prompt_context_tokens,
            }
        except Exception:
            return self._reply_offline(thread_id, message)

    def _get_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def _calculate_prompt_tokens(self, thread_id: str, new_message: str) -> int:
        session = self._get_session(thread_id)
        # Baseline keeps all raw past messages in prompt context
        tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        tokens += estimate_tokens(new_message)
        # Add basic system prompt overhead
        tokens += 40
        return tokens

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative response tokens generated in this thread."""
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return cumulative prompt context tokens processed across all turns of this thread."""
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        """Baseline has no compaction mechanism."""
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline mode for baseline agent."""
        session = self._get_session(thread_id)
        prompt_tokens = self._calculate_prompt_tokens(thread_id, message)
        session.prompt_tokens_processed += prompt_tokens

        # Check if this thread has any previous user context
        if not session.messages:
            # Brand new thread: baseline remembers nothing about previous threads
            response_text = (
                "Chào bạn! Rất vui được trò chuyện cùng bạn. Vì đây là phiên làm việc mới, "
                "mình chưa có thông tin trước đó về bạn. Bạn cần hỗ trợ gì hôm nay?"
            )
        else:
            # Existing thread: simple deterministic response acknowledging the turn
            response_text = f"Đã ghi nhận thông tin trong phiên: {message[:60]}... Mình sẵn sàng tiếp tục."

        agent_tokens = estimate_tokens(response_text)
        session.token_usage += agent_tokens
        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": response_text})

        return {
            "role": "assistant",
            "content": response_text,
            "token_usage": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self) -> Any:
        """Instantiate a chat model if API credentials are present."""
        if not self.config.model.api_key and self.config.model.provider not in ("ollama", "custom"):
            return None
        return build_chat_model(self.config.model)
