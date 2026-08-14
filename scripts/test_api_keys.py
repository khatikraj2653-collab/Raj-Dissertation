"""
test_api_keys.py
Quick sanity check that all 4 API keys are valid and working
before running the full 5-model evaluation.
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_openai():
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'OpenAI key works' and nothing else."}],
        max_tokens=30,
    )
    print("OpenAI:", response.choices[0].message.content.strip())


def test_anthropic():
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=30,
        messages=[{"role": "user", "content": "Say 'Anthropic key works' and nothing else."}],
    )
    print("Anthropic:", response.content[0].text.strip())


def test_gemini():
    from google import genai
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Say 'Gemini key works' and nothing else.",
    )
    print("Gemini:", response.text.strip())


def test_groq():
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say 'Groq key works' and nothing else."}],
        max_tokens=30,
    )
    print("Groq (Llama 3.3):", response.choices[0].message.content.strip())


def main():
    tests = [
        ("OpenAI", test_openai),
        ("Anthropic", test_anthropic),
        ("Gemini", test_gemini),
        ("Groq", test_groq),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"{name}: FAILED — {e}")


if __name__ == "__main__":
    main()