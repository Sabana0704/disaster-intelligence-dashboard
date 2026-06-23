"""
AI Summarizer — calls Claude API to generate situation summaries.
Falls back to rule-based summary if API key is not set.
"""

import os
import json
import requests


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL             = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are a Disaster Intelligence AI assistant.
Given structured disaster data, generate:
1. A clear 2-sentence situation summary for emergency responders.
2. A brief recommended action (1 sentence).

Respond ONLY in this JSON format (no markdown, no extra text):
{
  "summary": "...",
  "recommended_action": "..."
}"""


def summarize_with_claude(extracted: dict, api_key: str) -> dict:
    """Call Claude API to generate summary and recommended action."""
    user_content = f"""Disaster data extracted:
- Type: {extracted.get('disaster_type')}
- Location: {extracted.get('city', 'unknown')}, {extracted.get('country', 'unknown')}
- Severity: {extracted.get('severity')}
- Urgency: {extracted.get('urgency_level')}
- People Affected: {extracted.get('people_affected')}
- Resources Needed: {extracted.get('resources_str')}
- Raw text excerpt: {str(extracted.get('raw_text', ''))[:300]}

Generate a situation summary and recommended action."""

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 300,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_content}],
    }

    try:
        resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"].strip()
        # Strip any accidental markdown fences
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return {
            "summary":              parsed.get("summary", ""),
            "recommended_action":   parsed.get("recommended_action", ""),
        }
    except Exception as e:
        return fallback_summary(extracted)


def fallback_summary(extracted: dict) -> dict:
    """Rule-based fallback when API key is not available."""
    dtype   = extracted.get("disaster_type", "disaster").title()
    city    = extracted.get("city", "an unknown location")
    country = extracted.get("country", "")
    place   = f"{city}, {country}".strip(", ") if country and country != "unknown" else city
    people  = extracted.get("people_affected", "unknown number of")
    sev     = extracted.get("severity", "unknown")
    res     = extracted.get("resources_str", "rescue and relief")

    summary = (
        f"A {sev}-severity {dtype.lower()} has been reported in {place}, "
        f"affecting approximately {people} people. "
        f"Emergency response teams are required on-ground."
    )
    action = (
        f"Deploy {res} immediately to {place} and establish emergency coordination centre."
    )
    return {"summary": summary, "recommended_action": action}


def enrich_dataframe_with_summaries(df, api_key: str = None):
    """Add summary and recommended_action columns to processed dataframe."""
    import pandas as pd

    summaries   = []
    actions     = []

    for _, row in df.iterrows():
        extracted = row.to_dict()
        if api_key:
            result = summarize_with_claude(extracted, api_key)
        else:
            result = fallback_summary(extracted)
        summaries.append(result.get("summary", ""))
        actions.append(result.get("recommended_action", ""))

    df = df.copy()
    df["summary"]            = summaries
    df["recommended_action"] = actions
    return df
