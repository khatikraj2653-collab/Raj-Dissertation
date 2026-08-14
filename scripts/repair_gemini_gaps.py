"""
repair_gemini_gaps.py
Finds every case where Gemini's response is missing/failed and re-calls
only those, patching the existing results file in place.
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai as google_genai
from google.genai import types

from prompts import EXTRACTION_SYSTEM_PROMPT, build_user_prompt

load_dotenv()

RESULTS_FILE = Path(__file__).parent.parent / "results" / "multi_model_raw_responses.json"
gemini_client = google_genai.Client(api_key=os.getenv("GOOGLE_API_KEY_1"))


def try_parse_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw_text": text}


def call_gemini(description, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-3.5-flash",
                contents=build_user_prompt(description),
                config=types.GenerateContentConfig(
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    temperature=0,
                ),
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    retry {attempt+1}/{max_retries}: {e}")
                time.sleep(10)
            else:
                raise


def main():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    gaps = [cid for cid, case in results.items() if not case.get("gemini_raw")]
    print(f"Found {len(gaps)} cases needing Gemini repair")

    for i, case_id in enumerate(sorted(gaps, key=int)):
        case = results[case_id]
        print(f"[{i+1}/{len(gaps)}] Repairing case {case_id}...")
        try:
            raw = call_gemini(case["description"])
            results[case_id]["gemini_raw"] = raw
            results[case_id]["gemini_parsed"] = try_parse_json(raw)
        except Exception as e:
            print(f"  FAILED again on case {case_id}: {e}")
        time.sleep(1)

        # save every 10 cases
        if (i + 1) % 10 == 0:
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Done, all repairs saved.")


if __name__ == "__main__":
    main()