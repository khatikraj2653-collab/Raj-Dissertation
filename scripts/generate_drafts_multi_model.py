"""
generate_drafts_multi_model.py
Generates goal-and-scope draft sections from all 5 LLMs for the same
10 sampled activities used in the original Claude/GPT Task 2 run.
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
from anthropic import Anthropic
from google import genai as google_genai
from google.genai import types
from groq import Groq

load_dotenv()

ORIGINAL_DRAFTS_FILE = Path(__file__).parent.parent / "results" / "goal_scope_drafts.json"
OUTPUT_FILE = Path(__file__).parent.parent / "results" / "goal_scope_drafts_multi_model.json"

MODELS = {
    "claude": "claude-haiku-4-5-20251001",
    "gpt": "gpt-4o-mini",
    "gpt56terra": "gpt-5.6-terra",
    "gemini": "gemini-3.5-flash",
    "llama": "llama-3.3-70b-versatile",
    "gptoss": "openai/gpt-oss-120b",
}

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
gemini_client = google_genai.Client(api_key=os.getenv("GOOGLE_API_KEY_1"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DRAFT_PROMPT_TEMPLATE = """You are an LCA (Life Cycle Assessment) practitioner. Write a "Goal and Scope"
section for an LCA study of the following product/activity, following ISO 14044 conventions.

Product/Activity: {activity_name}
Geography: {geography}
Reference product: {reference_product_name}

Write a concise goal and scope section (150-250 words) that covers: the study's goal/purpose,
intended application, target audience, functional unit, system boundaries, and key assumptions/limitations."""


def call_gpt(prompt):
    response = openai_client.chat.completions.create(
        model=MODELS["gpt"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def call_gpt56terra(prompt):
    response = openai_client.chat.completions.create(
        model=MODELS["gpt56terra"],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def call_claude(prompt):
    response = anthropic_client.messages.create(
        model=MODELS["claude"],
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_gemini(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=MODELS["gemini"],
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3),
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                raise


def call_llama(prompt):
    response = groq_client.chat.completions.create(
        model=MODELS["llama"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def call_gptoss(prompt):
    response = groq_client.chat.completions.create(
        model=MODELS["gptoss"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


CALL_FUNCTIONS = {
    "claude": call_claude,
    "gpt": call_gpt,
    "gpt56terra": call_gpt56terra,
    "gemini": call_gemini,
    "llama": call_llama,
    "gptoss": call_gptoss,
}


def main():
    with open(ORIGINAL_DRAFTS_FILE, "r", encoding="utf-8") as f:
        original_cases = json.load(f)

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    for case in original_cases:
        case_id = str(case["id"])
        print(f"\nCase {case_id}: {case['activity_name']}")

        prompt = DRAFT_PROMPT_TEMPLATE.format(
            activity_name=case["activity_name"],
            geography=case["geography"],
            reference_product_name=case["reference_product_name"],
        )

        case_result = results.get(case_id, {
            "id": case["id"],
            "activity_uuid": case["activity_uuid"],
            "activity_name": case["activity_name"],
            "geography": case["geography"],
            "reference_product_name": case["reference_product_name"],
        })

        for model_key, call_fn in CALL_FUNCTIONS.items():
            draft_key = f"{model_key}_draft"
            if case_result.get(draft_key):
                continue
            print(f"  Generating {model_key}...")
            try:
                draft = call_fn(prompt)
                case_result[draft_key] = draft
            except Exception as e:
                print(f"    FAILED: {e}")
                case_result[draft_key] = None
            time.sleep(1)

        results[case_id] = case_result
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()