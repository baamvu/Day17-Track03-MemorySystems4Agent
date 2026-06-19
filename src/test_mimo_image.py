import sys
sys.stdout.reconfigure(encoding="utf-8")

print("=" * 60)
print("TEST: mimo-v2.5 Multimodal (Image Input)")
print("=" * 60)

try:
    from openai import OpenAI

    client = OpenAI(
        api_key="sk-TLgNV1kYMbVTv4hxqfGZcsQjN3i05EWJLjoOFegekikrSrXR",
        base_url="https://api.tokenrouter.com/v1"
    )

    print("\n1. List models containing 'mimo':")
    try:
        models = client.models.list()
        mimo_models = [m.id for m in models.data if 'mimo' in m.id.lower()]
        print(f"   Found: {mimo_models}")
    except Exception as e:
        print(f"   Error listing models: {e}")

    print("\n2. Test text-only with mimo-v2.5:")
    try:
        response = client.chat.completions.create(
            model="mimo/mimo-v2.5",
            messages=[
                {"role": "user", "content": "Hello, reply with one word."}
            ],
            max_tokens=10
        )
        print(f"   Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n3. Test image input with mimo-v2.5:")
    try:
        response = client.chat.completions.create(
            model="mimo/mimo-v2.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image? Reply in one sentence."},
                        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"}}
                    ]
                }
            ],
            max_tokens=50
        )
        print(f"   Response: {response.choices[0].message.content}")
        print("   Image input: SUPPORTED")
    except Exception as e:
        error_str = str(e)
        if "image" in error_str.lower() or "multimodal" in error_str.lower() or "not support" in error_str.lower():
            print(f"   Image input: NOT SUPPORTED via this endpoint")
        print(f"   Error: {error_str[:200]}")

except ImportError:
    print("   openai package not installed")
except Exception as e:
    print(f"   Unexpected error: {e}")
