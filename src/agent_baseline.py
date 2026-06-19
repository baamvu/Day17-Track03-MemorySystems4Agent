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
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

        if not force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()

        session = self.sessions[thread_id]
        session.messages.append({"role": "user", "content": message})

        context_text = " ".join(m["content"] for m in session.messages)
        prompt_tokens = estimate_tokens(context_text)
        session.prompt_tokens_processed += prompt_tokens

        response_text = self._generate_offline_response(session, message)

        agent_tokens = estimate_tokens(response_text)
        session.token_usage += agent_tokens
        session.messages.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _generate_offline_response(self, session: SessionState, message: str) -> str:
        lower = message.lower()

        name_keywords = ["tên gì", "tên mình", "mình tên", "bạn biết mình", "mình là ai"]
        for kw in name_keywords:
            if kw in lower:
                for msg in reversed(session.messages):
                    content = msg["content"].lower()
                    if "tên" in content and msg["role"] == "user":
                        return f"Trong cuộc hội thoại này, bạn có nói về tên. Tôi chỉ nhớ trong cùng session này."
                return "Tôi chưa biết tên bạn trong session này. Bạn có thể cho tôi biết không?"

        recall_keywords = ["nhắc lại", "nhớ không", "có nhớ", "thử nhớ", "biết mình"]
        for kw in recall_keywords:
            if kw in lower:
                return "Tôi chỉ có thể nhớ những gì đã nói trong cuộc hội thoại hiện tại. Nếu đây là session mới, tôi không có thông tin từ session trước."

        return f"Cảm ơn bạn đã chia sẻ. Tôi đã ghi nhận thông tin trong session này."

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()

        session = self.sessions[thread_id]
        session.messages.append({"role": "user", "content": message})

        from langchain_core.messages import HumanMessage

        result = self.langchain_agent.invoke({"messages": [HumanMessage(content=message)]})
        response_text = result["messages"][-1].content

        agent_tokens = estimate_tokens(response_text)
        prompt_tokens = estimate_tokens(message)
        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        session.messages.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        try:
            from langgraph.prebuilt import create_react_agent

            llm = build_chat_model(self.config.model)
            self.langchain_agent = create_react_agent(llm, tools=[])
        except Exception:
            self.langchain_agent = None
