import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import time
from memory_store import ConfidenceTracker, extract_profile_updates, is_correction_message

print("=" * 60)
print("QUANTIFY: Confidence Threshold - False Fact Prevention")
print("=" * 60)

messages_false_positive = [
    "Bạn có biết mình tên là gì không?",
    "Mình đang đọc về topic tên là RAG",
    "Hãy giải thích cho mình tên gọi MLOps",
    "Mình muốn hỏi về đồ uống yêu thích của người nổi tiếng",
    "Nếu mình tên là AI thì bạn trả lời sao?",
]

messages_true_positive = [
    "Mình tên là Dũng, mình ở Đà Nẵng",
    "Mình làm MLOps engineer",
    "Đồ uống yêu thích là cà phê sữa đá",
    "Mình nuôi một bé corgi tên Bơ",
    "Mình muốn trả lời ngắn gọn có ví dụ thực chiến",
    "Mình tên là Dũng, mình ở Đà Nẵng",
    "Mình làm MLOps engineer",
    "Đồ uống yêu thích là cà phê sữa đá",
    "Mình nuôi một bé corgi tên Bơ",
    "Mình muốn trả lời ngắn gọn có ví dụ thực chiến",
]

print("\n--- Without confidence threshold (min_occurrences=1) ---")
no_threshold = ConfidenceTracker(min_occurrences=1)
false_persisted_no_thresh = 0
for msg in messages_false_positive:
    facts = extract_profile_updates(msg)
    for key, value in facts.items():
        should_persist, _ = no_threshold.record("u_false", key, value)
        if should_persist:
            false_persisted_no_thresh += 1
            print(f"  [FALSE FACT] '{key}={value}' from: {msg[:60]}")

true_persisted_no_thresh = 0
for msg in messages_true_positive:
    facts = extract_profile_updates(msg)
    for key, value in facts.items():
        should_persist, _ = no_threshold.record("u_true", key, value)
        if should_persist:
            true_persisted_no_thresh += 1

print(f"  False facts persisted: {false_persisted_no_thresh}")
print(f"  True facts persisted: {true_persisted_no_thresh}")

print("\n--- With confidence threshold (min_occurrences=2) ---")
with_threshold = ConfidenceTracker(min_occurrences=2)
false_persisted_thresh = 0
for msg in messages_false_positive:
    facts = extract_profile_updates(msg)
    for key, value in facts.items():
        should_persist, _ = with_threshold.record("u_false2", key, value)
        if should_persist:
            false_persisted_thresh += 1
            print(f"  [FALSE FACT] '{key}={value}' from: {msg[:60]}")

true_persisted_thresh = 0
for msg in messages_true_positive:
    facts = extract_profile_updates(msg)
    for key, value in facts.items():
        should_persist, _ = with_threshold.record("u_true2", key, value)
        if should_persist:
            true_persisted_thresh += 1

print(f"  False facts persisted: {false_persisted_thresh}")
print(f"  True facts persisted: {true_persisted_thresh}")

prevented = false_persisted_no_thresh - false_persisted_thresh
print(f"\n  False facts prevented by threshold: {prevented}")
print(f"  True facts retained: {true_persisted_thresh}/{true_persisted_no_thresh}")

print()
print("=" * 60)
print("QUANTIFY: Memory Decay - Confidence Over Time")
print("=" * 60)

for half_life_hours, sleep_sec, label in [
    (0.001, 0.05, "seconds"),
    (0.01, 0.05, "seconds"),
    (0.1, 0.05, "seconds"),
    (1.0, 0.05, "hours"),
    (24.0, 0.05, "hours"),
]:
    tracker = ConfidenceTracker(min_occurrences=3, decay_half_life_hours=half_life_hours)
    for _ in range(3):
        tracker.record("u1", "name", "Dũng")
    conf_start = tracker.get_confidence("u1", "name")
    time.sleep(sleep_sec)
    conf_end = tracker.get_confidence("u1", "name")
    drop = ((conf_start - conf_end) / conf_start) * 100 if conf_start > 0 else 0
    age_min = sleep_sec / 60
    print(f"  half_life={half_life_hours:6.3f}h | age={age_min:.1f}min: conf {conf_start:.4f} -> {conf_end:.4f} (drop {drop:.2f}%)")

print(f"\n  Practical example:")
print(f"  - Fact seen 3 times today: confidence = 1.0")
print(f"  - After 24h without reminder: confidence = 0.5")
print(f"  - After 48h without reminder: confidence = 0.25")
print(f"  - After 72h without reminder: confidence = 0.125")

print()
print("=" * 60)
print("QUANTIFY: Correction Detection Accuracy")
print("=" * 60)

corrections = [
    ("Mình đính chính, mình đang ở Đà Nẵng", True),
    ("Mình không còn làm backend nữa", True),
    ("Thực ra mình chuyển sang MLOps", True),
    ("Mình vẫn ở Huế, chưa chuyển đi đâu cả", False),
    ("Mình tên là Dũng", False),
    ("Đồ uống yêu thích là cà phê sữa đá", False),
    ("Mình muốn bạn trả lời ngắn gọn", False),
    ("Mình đính chính thêm một thông tin nghề nghiệp nhé", True),
    ("Chứ không phải Đà Nẵng mỗi ngày nữa", True),
    ("Hôm nay mình làm việc ở quán cà phê", False),
    ("Không phải product manager đâu", True),
    ("Mình chuyển sang làm data scientist", True),
    ("Mình vẫn uống cà phê sữa đá", False),
    ("Đó là câu đùa thôi, nghề mình vẫn là MLOps", True),
]

correct = 0
for msg, expected in corrections:
    result = is_correction_message(msg)
    status = "OK" if result == expected else "FAIL"
    if result == expected:
        correct += 1
    print(f"  [{status}] '{msg[:55]}' -> expected={expected}, got={result}")

print(f"\n  Accuracy: {correct}/{len(corrections)} = {correct/len(corrections)*100:.0f}%")

print()
print("=" * 60)
print("QUANTIFY: Stress Test - Prompt Token Growth Curve")
print("=" * 60)

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
import json

config = load_config()
stress_data = json.load(open(config.data_dir / "advanced_long_context.json", encoding="utf-8"))

baseline = BaselineAgent(config=config, force_offline=True)
advanced = AdvancedAgent(config=config, force_offline=True)

conv = stress_data[0]
user_id = conv["user_id"]
thread_id = conv["id"]

baseline_per_turn = []
advanced_per_turn = []

for turn in conv["turns"]:
    b = baseline.reply(user_id, thread_id, turn)
    a = advanced.reply(user_id, thread_id, turn)
    baseline_per_turn.append(b["prompt_tokens"])
    advanced_per_turn.append(a["prompt_tokens"])

n = len(conv["turns"])
mid = n // 2

first_half_b = sum(baseline_per_turn[:mid])
second_half_b = sum(baseline_per_turn[mid:])
first_half_a = sum(advanced_per_turn[:mid])
second_half_a = sum(advanced_per_turn[mid:])

total_b = sum(baseline_per_turn)
total_a = sum(advanced_per_turn)
savings = total_b - total_a

print(f"\n  Turns: {n}")
print(f"  Compactions: {advanced.compaction_count(thread_id)}")
print(f"\n  Baseline prompt tokens:")
print(f"    First half (turns 0-{mid-1}): {first_half_b}")
print(f"    Second half (turns {mid}-{n-1}): {second_half_b}")
print(f"    Growth: +{second_half_b - first_half_b} ({((second_half_b - first_half_b) / max(1, first_half_b)) * 100:.0f}%)")
print(f"\n  Advanced prompt tokens:")
print(f"    First half (turns 0-{mid-1}): {first_half_a}")
print(f"    Second half (turns {mid}-{n-1}): {second_half_a}")
print(f"    Growth: +{second_half_a - first_half_a} ({((second_half_a - first_half_a) / max(1, first_half_a)) * 100:.0f}%)")
print(f"\n  Total savings: {savings} tokens ({(savings / max(1, total_b)) * 100:.1f}%)")
print(f"\n  Per-turn detail:")
for i in range(n):
    diff = baseline_per_turn[i] - advanced_per_turn[i]
    marker = " <-- compaction" if i == n - 1 and advanced.compaction_count(thread_id) > 0 else ""
    print(f"    Turn {i:2d}: baseline={baseline_per_turn[i]:5d} | advanced={advanced_per_turn[i]:5d} | diff={diff:+5d}{marker}")
