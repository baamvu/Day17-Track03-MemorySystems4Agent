import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from config import load_config
from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent

config = load_config()
print(f"Provider: {config.model.provider}")
print(f"Model: {config.model.model_name}")
print(f"API Key: {'set' if config.model.api_key else 'NOT SET'}")
print()

print("=" * 50)
print("ADVANCED AGENT - Live LLM")
print("=" * 50)
agent = AdvancedAgent(config=config, force_offline=False)

print("\n[Turn 1] Giao thieu")
r = agent.reply("bao", "thread1", "Chào bạn, mình tên là Bảo, mình đang làm AI Engineer.")
print(f"User: Chào bạn, mình tên là Bảo, mình đang làm AI Engineer.")
print(f"Agent: {r['response']}")
print(f"Tokens: agent={r['agent_tokens']}, prompt={r['prompt_tokens']}")

print("\n[Turn 2] Them preference")
r = agent.reply("bao", "thread1", "Mình muốn bạn trả lời ngắn gọn, có bullet và ví dụ thực chiến.")
print(f"User: Mình muốn bạn trả lời ngắn gọn, có bullet và ví dụ thực chiến.")
print(f"Agent: {r['response']}")
print(f"Tokens: agent={r['agent_tokens']}, prompt={r['prompt_tokens']}")

print("\n[Turn 3] Hoi ve AI")
r = agent.reply("bao", "thread1", "Giải thích ngắn gọn về RAG (Retrieval Augmented Generation)?")
print(f"User: Giải thích ngắn gọn về RAG?")
print(f"Agent: {r['response']}")
print(f"Tokens: agent={r['agent_tokens']}, prompt={r['prompt_tokens']}")

print("\n" + "=" * 50)
print("RECALL - Thread moi")
print("=" * 50)

print("\n[Recall 1] Nho ten?")
r = agent.reply("bao", "thread2", "Mình tên gì?")
print(f"User: Mình tên gì?")
print(f"Agent: {r['response']}")

print("\n[Recall 2] Nho nghe nghiep?")
r = agent.reply("bao", "thread2", "Mình làm nghề gì?")
print(f"User: Mình làm nghề gì?")
print(f"Agent: {r['response']}")

print("\n[Recall 3] Style tra loi?")
r = agent.reply("bao", "thread2", "Style trả lời mình thích là gì?")
print(f"User: Style trả lời mình thích là gì?")
print(f"Agent: {r['response']}")

print("\n[Recall 4] Tom tat")
r = agent.reply("bao", "thread2", "Tóm tắt về mình")
print(f"User: Tóm tắt về mình")
print(f"Agent: {r['response']}")

print("\n" + "=" * 50)
print("BASELINE AGENT - Live LLM")
print("=" * 50)

baseline = BaselineAgent(config=config, force_offline=False)

print("\n[Turn 1] Gioi thieu")
r = baseline.reply("bao2", "b_thread1", "Chào bạn, mình tên là Bảo, mình đang làm AI Engineer.")
print(f"Agent: {r['response']}")

print("\n[Recall] Thread moi")
r = baseline.reply("bao2", "b_thread2", "Mình tên gì?")
print(f"Agent: {r['response']}")

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Advanced token usage (thread1): {agent.token_usage('thread1')}")
print(f"Advanced prompt tokens (thread1): {agent.prompt_token_usage('thread1')}")
print(f"Advanced User.md size: {agent.memory_file_size('bao')} bytes")
print(f"Advanced compactions (thread1): {agent.compaction_count('thread1')}")
print()
print(f"Baseline token usage (b_thread1): {baseline.token_usage('b_thread1')}")

print("\n--- User.md ---")
print(agent.profile_store.read_text("bao"))
