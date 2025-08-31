from __future__ import annotations
import os, json
from typing import Dict, Any
from openai import OpenAI

def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    
    base_url = os.getenv("OPENAI_BASE_URL") # None for OpenAI
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)

def _model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

def generate_narrative(dataset_name: str,
                       problem: str,
                       overview: Dict[str, Any],
                       metrics: Dict[str, Any],
                       model_name: str) -> str:
    """
    This function returns a concise markdown narrative.
    """

    context = {
        "dataset_name": dataset_name,
        "problem": problem,
        "n_rows": overview.get("n_rows"),
        "n_cols": overview.get("n_cols"),
        "top_missing": dict(sorted(overview.get("missing_perc", {}).items(),
                                   key=lambda kv: kv[1], reverse=True)[:5]),
        "target_value_counts": overview.get("target_value_counts", {}),
        "selected_model": model_name,
        "metrics": metrics,
    }

    sys_prompt = (
        "You are a data science assistant. Write a factual, concise narrative for a report.\n"
        "Only use the numbers provided in the JSON context. Do not invent values.\n"
        "Explain data shape, notable missingness, target balance (if classification),\n"
        "what model won and how it likely helped (one sentence), and key metric(s).\n"
        "Tone: professional, 10 - 15 sentences. Use plain Markdown, no emojis."
    )

    user_prompt = f"CONTEXT JSON:\n```json\n{json.dumps(context, indent=2)}\n```\nWrite the narrative now."

    client = _client()
    model = _model()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()