import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from config import load_config
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent

config = load_config()
agent = AdvancedAgent(config=config, force_offline=True)

print("=== TEST 1: Ghi nho user (x2 de vuong confidence threshold) ===")
for turn in [
    "Mình tên là Dũng, mình ở Huế",
    "Mình làm MLOps engineer",
    "Đồ uống yêu thích là cà phê sữa đá",
    "Mình nuôi một bé corgi tên Bơ",
    "Mình muốn bạn trả lời ngắn gọn và có ví dụ thực chiến",
]:
    agent.reply("user1", "thread1", turn)
    agent.reply("user1", "thread1", turn)
    print(f"  Stored: {turn[:50]}")

print()
print("=== TEST 2: Recall o session moi ===")
for q in [
    "Nhắc lại tên mình?",
    "Mình làm nghề gì?",
    "Đồ uống yêu thích của mình là gì?",
    "Mình nuôi con gì?",
    "Style trả lời mình thích là gì?",
    "Tóm tắt về mình",
]:
    r = agent.reply("user1", "thread2", q)
    print(f"  Q: {q}")
    print(f"  A: {r['response']}")
    print()

print("=== TEST 3: Baseline khong nho ===")
baseline = BaselineAgent(config=config, force_offline=True)
baseline.reply("user1", "s1", "Mình tên là Dũng, mình ở Huế")
r = baseline.reply("user1", "s2", "Nhắc lại tên mình?")
print(f"  Baseline: {r['response']}")

print()
print("=== TEST 4: Compact memory ===")
agent2 = AdvancedAgent(config=config, force_offline=True)
for i in range(20):
    agent2.reply("user2", "long-thread", f"Đây là tin nhắn số {i} với nội dung đủ dài để kiểm tra compact memory hoạt động đúng khi vượt ngưỡng token.")
print(f"  Compactions: {agent2.compaction_count('long-thread')}")
print(f"  Token usage: {agent2.token_usage('long-thread')}")
print(f"  Prompt tokens: {agent2.prompt_token_usage('long-thread')}")

print()
print("=== TEST 5: User.md content ===")
print(agent.profile_store.read_text("user1"))
print(f"File: {agent.profile_store.path_for('user1')}")
