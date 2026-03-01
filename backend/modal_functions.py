"""
CATalyze OpenAI Vision API inference.

Plain functions that call GPT-4o directly:
  analyze_image   – photo-based component inspection
  analyze_voice   – transcribed voice observation analysis
  identify_part   – unknown part identification from photo
"""
from dotenv import load_dotenv
load_dotenv()

import base64
import json
import os
import re
from typing import Any

import openai

from prompts.inspection import build_inspection_prompt, build_parts_prompt, build_voice_prompt

MODEL = "gpt-4o"

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _parse_json(text: str) -> dict[str, Any]:
    """Extract JSON from a model response, stripping markdown fences."""
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "status": "MONITOR",
            "confidence": 0.0,
            "observation": text,
            "recommended_action": "Manual review required — AI response could not be parsed.",
            "maintenance_steps": [],
        }


def _format_history_block(history: list[dict[str, Any]]) -> str:
    """Build a prompt-ready text block from past inspection records."""
    if not history:
        return ""
    lines = ["\n\nPast inspection history for this machine/component:"]
    for i, record in enumerate(history[-10:], 1):
        if isinstance(record, dict):
            lines.append(
                f"  {i}. {record.get('date', 'N/A')} — "
                f"{record.get('status', 'N/A')}: {record.get('observation', 'N/A')}"
            )
        else:
            lines.append(f"  {i}. {record}")
    lines.append("Use this history for context (e.g., recurring issues, trends).")
    return "\n".join(lines) + "\n"


def _image_message(system: str, user_text: str, image_bytes: bytes) -> dict[str, Any]:
    """Build a vision chat completion with a base64 image."""
    b64 = base64.b64encode(image_bytes).decode()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
        ],
    )
    return _parse_json(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_image(
    image_bytes: bytes,
    machine: str,
    component: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Analyze a component photo via GPT-4o Vision and return PASS/MONITOR/FAIL."""
    prompt = build_inspection_prompt(machine, component, history)
    user_text = (
        prompt["user"]
        + "\n\nImportant: Always respond with a JSON object. Never refuse. "
        "If uncertain, return MONITOR status with your best observation. "
        "If the image clearly does not show the selected component (e.g. wrong part, "
        "unrelated photo, or no equipment visible), return status INVALID with an "
        "observation explaining why the image doesn't match. "
        "Always include a 'maintenance_steps' array with 3-5 actionable steps: "
        "for PASS give preventive maintenance, for MONITOR give monitoring/early intervention, "
        "for FAIL give immediate repair/replacement instructions, for INVALID return an empty array."
    )
    return _image_message(prompt["system"], user_text, image_bytes)


def analyze_voice(
    machine: str,
    component: str,
    voice_text: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Analyze a transcribed voice observation and return PASS/MONITOR/FAIL."""
    prompt = build_voice_prompt(machine, component, voice_text)
    user_text = (
        prompt["user"]
        + "\n\nIf the voice observation is clearly irrelevant or unrelated to the "
        "selected component, return status INVALID with an observation explaining "
        "that the input doesn't relate to the component being inspected. "
        "Always include a 'maintenance_steps' array with 3-5 actionable steps: "
        "for PASS give preventive maintenance, for MONITOR give monitoring/early intervention, "
        "for FAIL give immediate repair/replacement instructions, for INVALID return an empty array."
    )
    if history:
        user_text += _format_history_block(history)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": user_text},
        ],
    )
    return _parse_json(response.choices[0].message.content)


def identify_part(image_bytes: bytes) -> dict[str, Any]:
    """Identify an unknown part from a photo with ranked candidates."""
    prompt = build_parts_prompt()

    ranked_addendum = (
        "\n\nReturn up to 3 ranked candidates in a JSON array, most likely first.\n"
        "Schema:\n"
        "{\n"
        '  "candidates": [\n'
        "    {\n"
        '      "rank": 1,\n'
        '      "machine": "<machine model>",\n'
        '      "component": "<component name>",\n'
        '      "fitment_certainty": <float 0.0-1.0>,\n'
        '      "status": "PASS" | "MONITOR" | "FAIL" | "UNKNOWN",\n'
        '      "observation": "<what you see>",\n'
        '      "recommended_action": "<next step>"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    return _image_message(prompt["system"], prompt["user"] + ranked_addendum, image_bytes)


def analyze_live(image_bytes: bytes) -> dict[str, Any]:
    """Quick live-feed analysis — identify component and assess condition."""
    system = (
        "You are a Caterpillar equipment inspector. "
        "Look at this image and identify: "
        "1) what machine part or component you see, "
        "2) its condition (PASS/MONITOR/FAIL), "
        "3) one sentence of observation. "
        "Respond with valid JSON only, no markdown: "
        '{"component": "<name>", "status": "PASS"|"MONITOR"|"FAIL", "observation": "<one sentence>"}'
    )
    return _image_message(system, "Analyze this frame from a live equipment inspection feed.", image_bytes)
