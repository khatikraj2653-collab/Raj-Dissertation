"""
generate_drafts.py
Generates goal-and-scope draft sections from GPT and Claude
for 10 sampled activities — Task 2 of Objective 3 (subjective/rubric-based).
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
import pandas as pd

load_dotenv()

SOURCE_FILE = Path(__file__).parent.parent / "data" / "RAJ_DISS.xlsx"
OUTPUT_FILE = Path(__file__).parent.parent / "results" / "goal_scope_drafts.json"

SAMPLE_SIZE = 10
RANDOM_SEED = 7  # different seed from Task 1 = independent sample

OPENAI_MODEL = "gpt-4o-mini"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

DRAFT_PROMPT_TEMPLATE = """You are an LCA (Life Cycle Assessment) practitioner. Write a "Goal and Scope" 
section for an LCA study of the following product/activity, following ISO 14044 conventions.

Product/Activity: {activity_name}
Geography: {geography}
Reference product: {reference_product_name} ({reference_product_unit})

Write a concise goal and scope section (150-250 words) that covers: the study's goal/purpose, 
intended application, target audience, functional unit, system boundaries, and key assumptions/limitations."""


def load_sample():
    df = pd.read_excel(
        SOURCE_FILE, sheet_name="LCIA", header=3,
        usecols="A,C,D,E,F,G",
    )
    df.columns = ["activity_uuid", "activity_name", "geography",
                  "reference_product_name", "reference_product_unit", "reference_product_amount"]
    df = df.dropna(subset=["activity_name", "geography", "reference_product_name"])
    return df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)


def call_gpt(prompt):
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def call_claude(prompt):
    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def main():
    sample = load_sample()
    drafts = []

    for i, row in sample.iterrows():
        print(f"Generating drafts for case {i}: {row['activity_name']}...")

        prompt = DRAFT_PROMPT_TEMPLATE.format(
            activity_name=row["activity_name"],
            geography=row["geography"],
            reference_product_name=row["reference_product_name"],
            reference_product_unit=row["reference_product_unit"],
        )

        gpt_draft = call_gpt(prompt)
        claude_draft = call_claude(prompt)

        drafts.append({
            "id": i,
            "activity_uuid": row["activity_uuid"],
            "activity_name": row["activity_name"],
            "geography": row["geography"],
            "reference_product_name": row["reference_product_name"],
            "gpt_draft": gpt_draft,
            "claude_draft": claude_draft,
        })

        time.sleep(0.5)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(drafts, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Saved {len(drafts)} draft pairs to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()