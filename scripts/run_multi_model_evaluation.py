"""
run_multi_model_evaluation.py
Runs the structured-extraction task (Objective 3, Task 1) across 6 LLMs:
Claude Haiku 4.5, GPT-4o-mini, GPT-5.6 Terra, Gemini 3.5 Flash,
Llama 3.3 70B (Groq), GPT-OSS 120B (Groq).

Reuses existing Claude/GPT results for the original 100 cases (batch="original")
to avoid wasted API calls. Calls all other models fresh. Saves incrementally
after every case so a crash or interruption does not lose progress, and can
resume a partially completed run.
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI
from anthropic import Anthropic
from google import genai as google_genai
from groq import Groq

from prompts import EXTRACTION_SYSTEM_PROMPT, build_user_prompt

load_dotenv()

TEST_SET_FILE = Path(__file__).parent.parent / "data" / "test_set_200.json"
ORIGINAL_RESULTS_FILE = Path(__file__).parent.parent / "results" / "raw_responses.json"
OUTPUT_FILE = Path(__file__).parent.parent / "results" / "multi_model_raw_responses.json"

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


def try_parse_json(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_text": text}


def call_gpt(description):
    response = openai_client.chat.completions.create(
        model=MODELS["gpt"],
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(description)},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def call_gpt56terra(description):
    response = openai_client.chat.completions.create(
        model=MODELS["gpt56terra"],
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(description)},
        ],
    )
    return response.choices[0].message.content


def call_claude(description):
    response = anthropic_client.messages.create(
        model=MODELS["claude"],
        max_tokens=300,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(description)}],
    )
    return response.content[0].text


def call_gemini(description, max_retries=3):
    from google.genai import types
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=MODELS["gemini"],
                contents=build_user_prompt(description),
                config=types.GenerateContentConfig(
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    temperature=0,
                ),
            )
            return response.text
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                print(f"    Gemini rate limited, waiting 15s (attempt {attempt+1}/{max_retries})...")
                time.sleep(15)
            else:
                raise


def call_llama(description):
    response = groq_client.chat.completions.create(
        model=MODELS["llama"],
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(description)},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def call_gptoss(description):
    response = groq_client.chat.completions.create(
        model=MODELS["gptoss"],
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(description)},
        ],
        temperature=0,
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


def load_original_lookup():
    if not ORIGINAL_RESULTS_FILE.exists():
        return {}
    with open(ORIGINAL_RESULTS_FILE, "r", encoding="utf-8") as f:
        original = json.load(f)
    lookup = {}
    for r in original:
        lookup[r["activity_uuid"]] = {
            "claude": r.get("claude_raw"),
            "gpt": r.get("gpt_raw"),
        }
    return lookup


def load_existing_progress():
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(results):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    with open(TEST_SET_FILE, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    original_lookup = load_original_lookup()
    print(f"Loaded {len(original_lookup)} reusable original results (Claude/GPT)")

    results = load_existing_progress()
    print(f"Resuming with {len(results)} cases already completed")

    for case in test_set:
        case_id = str(case["id"])
        if case_id in results and all(
            f"{m}_raw" in results[case_id] and results[case_id][f"{m}_raw"] is not None
            for m in MODELS
        ):
            continue

        print(f"Processing case {case['id']} (batch={case['batch']})...")
        case_result = results.get(case_id, {
            "id": case["id"],
            "activity_uuid": case["activity_uuid"],
            "description": case["description"],
            "ground_truth": case["ground_truth"],
            "batch": case["batch"],
        })

        uuid = case["activity_uuid"]
        reusable = original_lookup.get(uuid, {})

        for model_key, call_fn in CALL_FUNCTIONS.items():
            raw_key = f"{model_key}_raw"
            parsed_key = f"{model_key}_parsed"

            if case_result.get(raw_key) is not None:
                continue

            if model_key in ("claude", "gpt") and reusable.get(model_key):
                raw = reusable[model_key]
            else:
                try:
                    raw = call_fn(case["description"])
                    if model_key == "gemini":
                        time.sleep(1)
                    else:
                        time.sleep(0.5)
                except Exception as e:
                    print(f"  {model_key} FAILED on case {case['id']}: {e}")
                    raw = None

            case_result[raw_key] = raw
            case_result[parsed_key] = try_parse_json(raw) if raw else {"parse_error": True, "raw_text": None}

        results[case_id] = case_result
        save_progress(results)

    print(f"\nDone. {len(results)} cases saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()