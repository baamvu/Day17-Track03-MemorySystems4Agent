from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    ConfidenceTracker,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    is_correction_message,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.confidence_tracker = ConfidenceTracker(
            min_occurrences=2,
            decay_half_life_hours=24.0,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

        if not force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def get_fact_confidence(self, user_id: str, key: str) -> float:
        return self.confidence_tracker.get_confidence(user_id, key)

    _RECALL_PATTERNS = [
        re.compile(r"\?$"),
        re.compile(r"nhắc lại", re.IGNORECASE),
        re.compile(r"bạn (?:có )?biết", re.IGNORECASE),
        re.compile(r"thử (?:nhớ|gợi|mô tả)", re.IGNORECASE),
        re.compile(r"(?:có )?nhớ không", re.IGNORECASE),
        re.compile(r"bạn biết gì về mình", re.IGNORECASE),
        re.compile(r"tóm tắt", re.IGNORECASE),
        re.compile(r"sang thread mới", re.IGNORECASE),
        re.compile(r"đâu mới là", re.IGNORECASE),
    ]

    def _is_recall_question(self, message: str) -> bool:
        for pat in self._RECALL_PATTERNS:
            if pat.search(message):
                return True
        return False

    def _persist_facts(self, user_id: str, facts: dict[str, str], is_correction: bool) -> None:
        for key, value in facts.items():
            if is_correction:
                self.confidence_tracker.resolve_conflict(user_id, key, value)
                self.profile_store.upsert_fact(user_id, key, value)
            else:
                has_conflict = self.confidence_tracker.detect_conflict(user_id, key, value)
                if has_conflict:
                    self.confidence_tracker.resolve_conflict(user_id, key, value)
                    self.profile_store.upsert_fact(user_id, key, value)
                else:
                    should_persist, _ = self.confidence_tracker.record(user_id, key, value)
                    if should_persist:
                        self.profile_store.upsert_fact(user_id, key, value)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if not self._is_recall_question(message):
            facts = extract_profile_updates(message)
            if facts:
                is_correction = is_correction_message(message)
                self._persist_facts(user_id, facts, is_correction)

        self.compact_memory.append(thread_id, "user", message)

        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        response_text = self._offline_response(user_id, thread_id, message)

        agent_tokens = estimate_tokens(response_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.compact_memory.append(thread_id, "assistant", response_text)

        return {
            "response": response_text,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_text = self.profile_store.read_text(user_id)
        profile_tokens = estimate_tokens(profile_text)

        ctx = self.compact_memory.context(thread_id)
        summary_tokens = estimate_tokens(str(ctx.get("summary", "")))

        recent: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore[assignment]
        recent_text = " ".join(m.get("content", "") for m in recent)
        recent_tokens = estimate_tokens(recent_text)

        return profile_tokens + summary_tokens + recent_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        lower = message.lower()
        profile_facts = self.profile_store.facts(user_id)

        name_q = any(kw in lower for kw in ["tên gì", "tên mình", "mình tên", "bạn biết mình", "mình là ai"])
        profession_q = any(kw in lower for kw in ["nghề", "làm nghề", "công việc", "hiện tại mình làm"])
        location_q = any(kw in lower for kw in ["ở đâu", "nơi ở", "hiện tại mình ở", "đang ở", "đâu mới là"])
        drink_q = any(kw in lower for kw in ["đồ uống", "uống gì", "thức uống"])
        food_q = any(kw in lower for kw in ["món ăn", "thích ăn", "món ruột"])
        style_q = any(kw in lower for kw in ["style", "phong cách", "kiểu trả lời", "trả lời như thế nào", "trả lời mình thích"])
        pet_q = any(kw in lower for kw in ["nuôi con gì", "con corgi", "con gì", "nuôi gì"])
        recall_keywords = [
            "nhắc lại", "bạn biết", "biết mình", "nhớ không", "có nhớ", "thử nhớ",
            "tóm tắt", "mô tả", "bạn biết gì về mình", "mình là ai",
            "hiện tại", "nghề nghiệp và nơi ở",
        ]

        is_recall = any(kw in lower for kw in recall_keywords)
        is_compound = sum([name_q, profession_q, location_q, drink_q, food_q, style_q, pet_q]) >= 2
        asks_any_fact = any([name_q, profession_q, location_q, drink_q, food_q, style_q, pet_q])

        _LABELS = {"name": "Tên", "profession": "Nghề", "location": "Nơi ở",
                   "drink": "Đồ uống", "food": "Món ăn", "style": "Style", "pet": "Pet"}

        if is_recall or is_compound:
            parts: list[str] = []
            if is_recall:
                for key in ["name", "profession", "location", "drink", "food", "style", "pet"]:
                    if key in profile_facts:
                        parts.append(f"{_LABELS[key]}: {profile_facts[key]}")
            else:
                if name_q and "name" in profile_facts:
                    parts.append(f"Tên: {profile_facts['name']}")
                if profession_q and "profession" in profile_facts:
                    parts.append(f"Nghề: {profile_facts['profession']}")
                if location_q and "location" in profile_facts:
                    parts.append(f"Nơi ở: {profile_facts['location']}")
                if drink_q and "drink" in profile_facts:
                    parts.append(f"Đồ uống: {profile_facts['drink']}")
                if food_q and "food" in profile_facts:
                    parts.append(f"Món ăn: {profile_facts['food']}")
                if style_q and "style" in profile_facts:
                    parts.append(f"Style: {profile_facts['style']}")
                if pet_q and "pet" in profile_facts:
                    parts.append(f"Pet: {profile_facts['pet']}")

            if parts:
                return ". ".join(parts) + "."
            return "Tôi chưa có thông tin nào về bạn trong hồ sơ."

        if asks_any_fact:
            parts = []
            if name_q and "name" in profile_facts:
                parts.append(f"Bạn tên là {profile_facts['name']}")
            if profession_q and "profession" in profile_facts:
                parts.append(f"là {profile_facts['profession']}")
            if location_q and "location" in profile_facts:
                parts.append(f"ở {profile_facts['location']}")
            if drink_q and "drink" in profile_facts:
                parts.append(f"Đồ uống yêu thích: {profile_facts['drink']}")
            if food_q and "food" in profile_facts:
                parts.append(f"Món ăn yêu thích: {profile_facts['food']}")
            if style_q and "style" in profile_facts:
                parts.append(f"Style: {profile_facts['style']}")
            if pet_q and "pet" in profile_facts:
                parts.append(f"Pet: {profile_facts['pet']}")
            if parts:
                return ". ".join(parts) + "."

        if is_correction_message(message):
            return "Tôi đã cập nhật thông tin mới nhất vào hồ sơ của bạn."

        if profile_facts:
            name = profile_facts.get("name", "bạn")
            return f"Cảm ơn {name} đã chia sẻ. Tôi đã ghi nhận và lưu vào hồ sơ."

        return "Cảm ơn bạn đã chia sẻ. Tôi đã ghi nhận thông tin."

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if not self._is_recall_question(message):
            facts = extract_profile_updates(message)
            if facts:
                is_correction = is_correction_message(message)
                self._persist_facts(user_id, facts, is_correction)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        from langchain_core.messages import HumanMessage

        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        augmented = f"[User Profile]\n{profile_text}\n\n[Conversation Summary]\n{summary}\n\n[Current Message]\n{message}"

        result = self.langchain_agent.invoke({"messages": [HumanMessage(content=augmented)]})
        response_text = result["messages"][-1].content

        agent_tokens = estimate_tokens(response_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.compact_memory.append(thread_id, "assistant", response_text)

        return {
            "response": response_text,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        try:
            from langgraph.prebuilt import create_react_agent

            llm = build_chat_model(self.config.model)

            from langchain_core.tools import tool

            @tool
            def read_user_profile(user_id: str) -> str:
                """Read the user profile markdown file."""
                return self.profile_store.read_text(user_id)

            @tool
            def write_user_profile(user_id: str, content: str) -> str:
                """Write content to the user profile markdown file."""
                path = self.profile_store.write_text(user_id, content)
                return f"Written to {path}"

            self.langchain_agent = create_react_agent(
                llm, tools=[read_user_profile, write_user_profile]
            )
        except Exception:
            self.langchain_agent = None
