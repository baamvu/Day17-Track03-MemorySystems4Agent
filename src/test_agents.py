from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    ConfidenceTracker,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    is_correction_message,
)


def make_config(tmp_path: Path) -> LabConfig:
    config = load_config(Path(__file__).resolve().parent.parent)
    config.state_dir = tmp_path / "state"
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.compact_threshold_tokens = 200
    config.compact_keep_messages = 3
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = UserProfileStore(config.state_dir / "profiles")

    text = store.read_text("test_user")
    assert "No facts recorded" in text

    store.write_text("test_user", "# Profile\n- **name**: TestUser\n")
    assert "TestUser" in store.read_text("test_user")

    changed = store.edit_text("test_user", "TestUser", "UpdatedUser")
    assert changed is True
    assert "UpdatedUser" in store.read_text("test_user")

    assert store.file_size("test_user") > 0

    store.upsert_fact("test_user", "location", "Hanoi")
    facts = store.facts("test_user")
    assert facts["location"] == "Hanoi"

    store.upsert_fact("test_user", "location", "Hue")
    facts = store.facts("test_user")
    assert facts["location"] == "Hue"


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    cm = CompactMemoryManager(threshold_tokens=50, keep_messages=2)

    for i in range(20):
        cm.append("thread-1", "user", f"This is message number {i} with some content to fill tokens.")

    ctx = cm.context("thread-1")
    assert ctx["compactions"] > 0
    assert len(ctx["summary"]) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    user_id = "recall_user"

    for _ in range(3):
        advanced.reply(user_id, "session-1", "Mình tên là DũngCT, mình ở Đà Nẵng.")

    baseline.reply(user_id, "session-1", "Mình tên là DũngCT, mình ở Đà Nẵng.")

    b_result = baseline.reply(user_id, "session-2", "Nhắc lại tên mình?")
    a_result = advanced.reply(user_id, "session-2", "Nhắc lại tên mình?")

    assert "dũngct" in a_result["response"].lower() or "DũngCT" in a_result["response"]
    assert "session" in b_result["response"].lower() or "chưa" in b_result["response"].lower() or "không" in b_result["response"].lower()


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    user_id = "load_user"
    thread_id = "long-thread"

    for i in range(15):
        msg = f"Đây là tin nhắn số {i}. Mình đang chia sẻ nhiều thông tin về công việc MLOps và Python."
        baseline.reply(user_id, thread_id, msg)
        advanced.reply(user_id, thread_id, msg)

    assert advanced.compaction_count(thread_id) > 0
    assert baseline.compaction_count(thread_id) == 0


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0
    assert estimate_tokens("hello") >= 1
    assert estimate_tokens("a" * 100) == 25


def test_extract_profile_updates() -> None:
    facts = extract_profile_updates("Mình tên là DũngCT và mình ở Đà Nẵng")
    assert "name" in facts or "location" in facts

    facts = extract_profile_updates("Bạn có biết mình tên gì không?")
    assert len(facts) == 0


def test_confidence_threshold(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "conf_user"

    agent.reply(user_id, "t1", "Mình tên là Nam")
    facts_after_1 = agent.profile_store.facts(user_id)
    assert "name" not in facts_after_1

    agent.reply(user_id, "t1", "Mình tên là Nam")
    facts_after_2 = agent.profile_store.facts(user_id)
    assert "name" in facts_after_2
    assert facts_after_2["name"] == "Nam"


def test_confidence_tracker_basic() -> None:
    tracker = ConfidenceTracker(min_occurrences=2)

    should_persist, _ = tracker.record("u1", "name", "Nam")
    assert should_persist is False

    should_persist, _ = tracker.record("u1", "name", "Nam")
    assert should_persist is True

    assert tracker.get_confidence("u1", "name") > 0
    assert tracker.get_value("u1", "name") == "Nam"


def test_memory_decay() -> None:
    tracker = ConfidenceTracker(min_occurrences=1, decay_half_life_hours=0.0001)

    tracker.record("u1", "name", "Nam")
    conf_before = tracker.get_confidence("u1", "name")
    assert conf_before > 0

    time.sleep(0.05)
    conf_after = tracker.get_confidence("u1", "name")
    assert conf_after < conf_before


def test_conflict_handling(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "conflict_user"

    for _ in range(3):
        agent.reply(user_id, "t1", "Mình đang ở Huế")

    facts = agent.profile_store.facts(user_id)
    assert facts.get("location") == "Huế"

    agent.reply(user_id, "t1", "Mình đính chính, mình đang ở Đà Nẵng chứ không phải Huế nữa")

    facts = agent.profile_store.facts(user_id)
    assert facts.get("location") == "Đà Nẵng"


def test_correction_detection() -> None:
    assert is_correction_message("Mình đính chính, mình ở Đà Nẵng") is True
    assert is_correction_message("Mình không còn làm backend nữa") is True
    assert is_correction_message("Thực ra mình chuyển sang MLOps") is True
    assert is_correction_message("Mình tên là Nam") is False


def test_conflict_detection() -> None:
    tracker = ConfidenceTracker(min_occurrences=1)

    tracker.record("u1", "location", "Huế")
    assert tracker.detect_conflict("u1", "location", "Đà Nẵng") is True
    assert tracker.detect_conflict("u1", "location", "Huế") is False

    tracker.resolve_conflict("u1", "location", "Đà Nẵng")
    assert tracker.get_value("u1", "location") == "Đà Nẵng"
