"""
run_evaluation.py
Sends every test case to both GPT and Claude, saves raw + parsed responses.
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

from prompts import EXTRACTION_SYSTEM_PROMPT, build_user_prompt

load_dotenv()

TEST_SET_FILE = Path(__file__).parent.parent / "data" / "test_set.json"
RESULTS_FILE = Path(__file__).parent.parent / "results" / "raw_responses.json"

OPENAI_MODEL = "gpt-4o-mini"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def call_gpt(description: str) -> str:
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(description)},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def call_claude(description: str) -> str:
    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": build_user_prompt(description)},
        ],
    )
    return response.content[0].text


def try_parse_json(text: str):
    """Handles cases where the model wraps JSON in markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_text": text}


def main():
    with open(TEST_SET_FILE, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    results = []

    for case in test_set:
        print(f"Processing case {case['id']}...")

        gpt_raw = call_gpt(case["description"])
        gpt_parsed = try_parse_json(gpt_raw)

        claude_raw = call_claude(case["description"])
        claude_parsed = try_parse_json(claude_raw)

        results.append({
            "id": case["id"],
            "activity_uuid": case["activity_uuid"],
            "description": case["description"],
            "ground_truth": case["ground_truth"],
            "gpt_raw": gpt_raw,
            "gpt_parsed": gpt_parsed,
            "claude_raw": claude_raw,
            "claude_parsed": claude_parsed,
        })

        time.sleep(0.5)  # gentle rate limiting

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Saved {len(results)} results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()