"""
print_drafts.py
Prints goal-and-scope drafts in a readable format for manual rubric scoring.
"""

import json
from pathlib import Path

DRAFTS_FILE = Path(__file__).parent.parent / "results" / "goal_scope_drafts.json"

with open(DRAFTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

for d in data:
    print(f"\n=== Case {d['id']}: {d['activity_name']} ===")
    print(f"--- GPT ---\n{d['gpt_draft']}\n")
    print(f"--- CLAUDE ---\n{d['claude_draft']}\n")
    print("=" * 80)