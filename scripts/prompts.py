"""
prompts.py
Defines the prompt template used for the extraction task (Objective 3).
Both GPT and Claude receive the exact same prompt for a fair comparison.
"""

EXTRACTION_SYSTEM_PROMPT = """You are an expert in Life Cycle Assessment (LCA) data processing.
You will be given a natural-language description of an industrial activity/process.
Extract exactly four fields from it and respond ONLY with valid JSON, no other text.

Fields to extract:
- activity_name: the name of the activity/process
- geography: the region/location code or name
- reference_product_name: the name of the product this activity produces
- reference_product_unit: the unit of measurement for the reference product

Respond in this exact JSON format, nothing else:
{
  "activity_name": "...",
  "geography": "...",
  "reference_product_name": "...",
  "reference_product_unit": "..."
}
"""

def build_user_prompt(description: str) -> str:
    return f"Extract the fields from this description:\n\n{description}"