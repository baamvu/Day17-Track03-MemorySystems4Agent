from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


@dataclass
class FactRecord:
    value: str
    occurrences: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    confidence: float = 0.0

    def update_confidence(self, decay_half_life_hours: float = 24.0) -> None:
        age_hours = (time.time() - self.last_seen) / 3600.0
        decay_factor = 0.5 ** (age_hours / decay_half_life_hours)
        frequency_score = min(1.0, self.occurrences / 3.0)
        self.confidence = round(frequency_score * decay_factor, 3)

    def touch(self, value: str) -> None:
        self.value = value
        self.occurrences += 1
        self.last_seen = time.time()
        self.update_confidence()


@dataclass
class ConfidenceTracker:
    min_occurrences: int = 2
    decay_half_life_hours: float = 24.0
    facts: dict[str, dict[str, FactRecord]] = field(default_factory=dict)

    def record(self, user_id: str, key: str, value: str) -> tuple[bool, bool]:
        if user_id not in self.facts:
            self.facts[user_id] = {}

        user_facts = self.facts[user_id]
        is_update = False

        if key in user_facts:
            existing = user_facts[key]
            if existing.value != value:
                is_update = True
                existing.value = value
                existing.occurrences = 1
                existing.last_seen = time.time()
            else:
                existing.touch(value)
        else:
            user_facts[key] = FactRecord(value=value)
            is_update = True

        user_facts[key].update_confidence(self.decay_half_life_hours)
        should_persist = user_facts[key].occurrences >= self.min_occurrences
        return should_persist, is_update

    def get_confidence(self, user_id: str, key: str) -> float:
        if user_id in self.facts and key in self.facts[user_id]:
            self.facts[user_id][key].update_confidence(self.decay_half_life_hours)
            return self.facts[user_id][key].confidence
        return 0.0

    def get_value(self, user_id: str, key: str) -> str | None:
        if user_id in self.facts and key in self.facts[user_id]:
            return self.facts[user_id][key].value
        return None

    def detect_conflict(self, user_id: str, key: str, new_value: str) -> bool:
        if user_id in self.facts and key in self.facts[user_id]:
            existing = self.facts[user_id][key]
            return existing.value.lower() != new_value.lower()
        return False

    def resolve_conflict(self, user_id: str, key: str, new_value: str) -> None:
        if user_id in self.facts and key in self.facts[user_id]:
            self.facts[user_id][key] = FactRecord(value=new_value)

    def facts_with_confidence(self, user_id: str) -> dict[str, tuple[str, float]]:
        result: dict[str, tuple[str, float]] = {}
        if user_id in self.facts:
            for key, rec in self.facts[user_id].items():
                rec.update_confidence(self.decay_half_life_hours)
                result[key] = (rec.value, rec.confidence)
        return result


@dataclass
class UserProfileStore:
    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id).strip("_").lower()
        if not slug:
            slug = "default_user"
        return self.root_dir / f"{slug}.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"# User Profile: {user_id}\n\nNo facts recorded yet.\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if path.exists():
            return path.stat().st_size
        return 0

    def facts(self, user_id: str) -> dict[str, str]:
        text = self.read_text(user_id)
        result: dict[str, str] = {}
        for match in re.finditer(r"^- \*\*(.+?)\*\*:\s*(.+)$", text, re.MULTILINE):
            result[match.group(1).strip()] = match.group(2).strip()
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        text = self.read_text(user_id)
        pattern = re.compile(rf"^(- \*\*{re.escape(key)}\*\*:).*$", re.MULTILINE)
        new_line = f"- **{key}**: {value}"
        if pattern.search(text):
            updated = pattern.sub(new_line, text)
        else:
            if "No facts recorded yet." in text:
                text = text.replace("No facts recorded yet.\n", "")
            updated = text.rstrip() + "\n" + new_line + "\n"
        self.write_text(user_id, updated)

    def upsert_fact_with_meta(self, user_id: str, key: str, value: str, confidence: float) -> None:
        text = self.read_text(user_id)
        pattern = re.compile(rf"^(- \*\*{re.escape(key)}\*\*:).*$", re.MULTILINE)
        new_line = f"- **{key}**: {value} (confidence: {confidence:.2f})"
        if pattern.search(text):
            updated = pattern.sub(new_line, text)
        else:
            if "No facts recorded yet." in text:
                text = text.replace("No facts recorded yet.\n", "")
            updated = text.rstrip() + "\n" + new_line + "\n"
        self.write_text(user_id, updated)


_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("name", re.compile(r"(?:tên mình là|mình tên(?: là)?|gọi mình là)\s+(?:là\s+)?([A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9]{1,20}(?:\s+[A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9]{1,20})?)", re.IGNORECASE)),
    ("location", re.compile(r"(?:mình\s+(?:vẫn\s+)?(?:ở|sống tại|đang (?:ở|làm việc (?:tại|ở))|hiện (?:đang )?ở)|nơi ở(?: hiện tại)?)\s+(?:là\s+|tại\s+)?([A-ZÀ-Ỹa-zà-ỹ][A-Za-zÀ-ỹ\s]{1,30}?)(?:\s*(?:\.|,|;|và|để|cho|nhưng|chứ|$))", re.IGNORECASE)),
    ("profession", re.compile(r"(?:mình\s+(?:vẫn\s+)?(?:là|đang làm|làm(?: nghề)?|chuyển sang)|nghề nghiệp(?: hiện tại)?|giờ(?: mình)?(?: đang)?(?: chuyển sang)?)\s+(?:là\s+)?((?:backend|frontend|mlops|devops|data|fullstack|AI|ML|product)\s*(?:engineer|developer|scientist|manager)?)", re.IGNORECASE)),
    ("drink", re.compile(r"(?:đồ uống yêu thích|(?:mình\s+)?(?:vẫn\s+)?(?:uống|thức uống))\s+(?:là\s+)?(.{3,50}?)(?:\.|,|$)", re.IGNORECASE)),
    ("food", re.compile(r"(?:món (?:ăn )?yêu thích|mình (?:thích ăn|hay ăn))\s+(?:là\s+)?(.{3,50}?)(?:\.|,|$)", re.IGNORECASE)),
    ("pet", re.compile(r"(?:nuôi|mình có)\s+(?:một\s+)?(?:bé\s+)?(\w+)\s+tên\s+(\w+)", re.IGNORECASE)),
    ("style", re.compile(r"(?:trả lời|giải thích)\s+(?:thành\s+)?(?:là\s+)?((?:ngắn gọn|3 bullet|bullet ngắn|có ví dụ|có cấu trúc|thành bullet)[^.]{3,60}?)(?:\.|,|khi|$)", re.IGNORECASE)),
]

_QUESTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\?$"),
    re.compile(r"^(?:bạn|em|anh|chị)\s+(?:có|biết|thấy|nghĩ)", re.IGNORECASE),
]

_CORRECTION_KEYWORDS = ["đính chính", "không còn", "chuyển sang", "không phải", "thực ra", "chứ không"]


def is_correction_message(message: str) -> bool:
    lower = message.lower()
    return any(kw in lower for kw in _CORRECTION_KEYWORDS)


def extract_profile_updates(message: str) -> dict[str, str]:
    for pat in _QUESTION_PATTERNS:
        if pat.search(message.strip()):
            return {}

    facts: dict[str, str] = {}
    for key, pattern in _PATTERNS:
        match = pattern.search(message)
        if match:
            if key == "pet" and match.lastindex and match.lastindex >= 2:
                animal = match.group(1).strip()
                pet_name = match.group(2).strip()
                facts["pet"] = f"{animal} tên {pet_name}"
            else:
                value = match.group(1).strip().rstrip(".,;:")
                if len(value) >= 2:
                    facts[key] = value

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    if not messages:
        return ""
    selected = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for msg in selected:
        role = msg.get("role", "unknown")
        content = msg.get("content", "").strip()
        if content:
            short = content[:200] + ("..." if len(content) > 200 else "")
            lines.append(f"[{role}] {short}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self._ensure_thread(thread_id)
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        messages.append({"role": role, "content": content})

        all_text = " ".join(m["content"] for m in messages)
        total_tokens = estimate_tokens(all_text)

        if total_tokens > self.threshold_tokens and len(messages) > self.keep_messages:
            old_messages = messages[: -self.keep_messages]
            keep_messages = messages[-self.keep_messages :]

            old_summary: str = thread.get("summary", "")  # type: ignore[assignment]
            new_chunk = summarize_messages(old_messages, max_items=len(old_messages))
            if old_summary:
                combined = old_summary + "\n---\n" + new_chunk
            else:
                combined = new_chunk

            thread["summary"] = combined
            thread["messages"] = keep_messages
            thread["compactions"] = thread.get("compactions", 0) + 1  # type: ignore[assignment]

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self._ensure_thread(thread_id)
        return {
            "messages": list(thread.get("messages", [])),  # type: ignore[arg-type]
            "summary": str(thread.get("summary", "")),
            "compactions": int(thread.get("compactions", 0)),
        }

    def compaction_count(self, thread_id: str) -> int:
        thread = self._ensure_thread(thread_id)
        return int(thread.get("compactions", 0))
