from __future__ import annotations
import os, json
from typing import Dict, Any
import google.generativeai as genai

SYSTEM_INSTRUCTIONS = (
    "You are a data science assistant. Write a factual, concise narrative for a report.\n"
    "Only use the numbers provided in the JSON context. Do not invent values.\n"
    "Explain data shape, notable missingness, target balance (if classification),\n"
    "what model won and how it likely helped (one sentence), and key metric(s).\n"
    "Tone: professional, 10-15 sentences. Use plain Markdown, no emojis."
)

def _gemini_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return genai.GenerativeModel(model_name, generation_config={"temperature": 0.2})


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
    user_prompt = f"CONTEXT JSON:\n```json\n{json.dumps(context, indent=2)}\n```\nWrite the narrative now."
    contents = f"{SYSTEM_INSTRUCTIONS}\n\n{user_prompt}"

    model = _gemini_model()
    resp = model.generate_content(contents)
    
    # handle possible SDK shapes
    text = getattr(resp, "text", None)
    if not text:
        try:
            text = resp.candidates[0].content.parts[0].text
        except Exception:
            text = ""
    # safety block handling
    if hasattr(resp, "prompt_feedback") and getattr(resp.prompt_feedback, "block_reason", None):
        raise RuntimeError(f"Gemini blocked: {resp.prompt_feedback.block_reason}")

    return (text or "").strip()