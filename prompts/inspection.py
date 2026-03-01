"""
CATalyze inspection prompts for Gemini Vision API.

Provides machine/component definitions and prompt builders for:
- Image-based inspection (with optional history)
- Voice observation analysis
- Unknown part identification

All analysis responses use: PASS | MONITOR | FAIL with confidence, observation, recommended_action.
"""

from typing import Any

# ---------------------------------------------------------------------------
# Machines and components
# ---------------------------------------------------------------------------

MACHINES: dict[str, list[str]] = {
    "Cat 950 Wheel Loader": [
        "bucket",
        "cutting edge",
        "bucket teeth",
        "lift arms",
        "lift cylinders",
        "tilt cylinders",
        "front axle",
        "rear axle",
        "tires",
        "rims",
        "cab",
        "windows",
        "engine compartment",
        "radiator",
        "hydraulic hoses",
        "hydraulic pump",
        "fuel system",
        "electrical",
        "brakes",
        "steering",
    ],
    "Cat 320 Excavator": [
        "bucket",
        "bucket teeth",
        "boom",
        "stick (arm)",
        "boom cylinders",
        "stick cylinder",
        "bucket cylinder",
        "track (undercarriage)",
        "track pads",
        "sprockets",
        "rollers",
        "idlers",
        "cab",
        "windows",
        "engine compartment",
        "radiator",
        "hydraulic hoses",
        "hydraulic pump",
        "swing bearing",
        "fuel system",
        "electrical",
        "blade (dozer)",
    ],
    "Cat 740 Articulated Truck": [
        "dump body",
        "dump body floor",
        "dump cylinders",
        "tailgate",
        "tailgate hinges",
        "front tires",
        "rear tires",
        "rims",
        "front axle",
        "rear axle",
        "articulation joint",
        "articulation cylinders",
        "cab",
        "windows",
        "engine compartment",
        "radiator",
        "exhaust system",
        "hydraulic hoses",
        "hydraulic pump",
        "fuel system",
        "electrical",
        "brakes",
        "steering",
        "suspension",
        "driveshaft",
    ],
    "Cat D6 Dozer": [
        "blade",
        "blade cutting edge",
        "blade end bits",
        "blade lift cylinders",
        "blade tilt cylinder",
        "push arms",
        "track (undercarriage)",
        "track pads",
        "track links",
        "sprockets",
        "rollers",
        "idlers",
        "track adjuster",
        "ripper",
        "ripper shank",
        "ripper cylinder",
        "cab",
        "windows",
        "engine compartment",
        "radiator",
        "hydraulic hoses",
        "hydraulic pump",
        "fuel system",
        "electrical",
        "final drives",
    ],
    "Cat 308 Mini Excavator": [
        "bucket",
        "bucket teeth",
        "boom",
        "stick (arm)",
        "boom cylinder",
        "stick cylinder",
        "bucket cylinder",
        "track (undercarriage)",
        "track pads",
        "sprockets",
        "rollers",
        "idlers",
        "cab",
        "canopy",
        "windows",
        "engine compartment",
        "radiator",
        "hydraulic hoses",
        "hydraulic pump",
        "swing bearing",
        "blade (dozer)",
        "blade cylinder",
        "fuel system",
        "electrical",
        "thumb attachment",
        "quick coupler",
    ],
    "Cat 420 Backhoe Loader": [
        "front loader bucket",
        "loader bucket cutting edge",
        "loader lift arms",
        "loader lift cylinders",
        "loader tilt cylinders",
        "backhoe boom",
        "backhoe stick",
        "backhoe bucket",
        "backhoe bucket teeth",
        "backhoe boom cylinder",
        "backhoe stick cylinder",
        "backhoe bucket cylinder",
        "stabilizer legs",
        "stabilizer pads",
        "stabilizer cylinders",
        "front tires",
        "rear tires",
        "rims",
        "front axle",
        "rear axle",
        "cab",
        "windows",
        "engine compartment",
        "radiator",
        "hydraulic hoses",
        "hydraulic pump",
        "fuel system",
        "electrical",
        "brakes",
        "steering",
    ],
    "Cat 980 Wheel Loader": [
        "bucket",
        "cutting edge",
        "bucket teeth",
        "lift arms",
        "lift cylinders",
        "tilt cylinders",
        "Z-bar linkage",
        "front axle",
        "rear axle",
        "front tires",
        "rear tires",
        "rims",
        "cab",
        "windows",
        "engine compartment",
        "radiator",
        "aftercooler",
        "hydraulic hoses",
        "hydraulic pump",
        "fuel system",
        "electrical",
        "brakes",
        "steering",
        "driveshaft",
        "transmission",
    ],
    "Cat 745 Articulated Truck": [
        "dump body",
        "dump body floor",
        "dump body liners",
        "dump cylinders",
        "tailgate",
        "tailgate hinges",
        "front tires",
        "rear tires",
        "rims",
        "front axle",
        "rear axle",
        "articulation joint",
        "articulation cylinders",
        "cab",
        "windows",
        "engine compartment",
        "radiator",
        "exhaust system",
        "hydraulic hoses",
        "hydraulic pump",
        "fuel system",
        "electrical",
        "brakes",
        "steering",
        "suspension",
        "driveshaft",
        "differential",
    ],
}

# Valid status values for inspection results
INSPECTION_STATUSES = ("PASS", "MONITOR", "FAIL", "INVALID")

# ---------------------------------------------------------------------------
# Response schema instruction (consistent JSON output)
# ---------------------------------------------------------------------------

RESPONSE_SCHEMA = """
Respond with valid JSON only. No markdown, no extra text.
Schema:
{
  "status": "PASS" | "MONITOR" | "FAIL" | "INVALID",
  "confidence": <float 0.0-1.0>,
  "observation": "<brief description of what you see>",
  "recommended_action": "<concrete next step or \"None\" if PASS>",
  "maintenance_steps": ["<step 1>", "<step 2>", "<step 3>"]
}
maintenance_steps: provide 3-5 specific step-by-step instructions appropriate to the status:
  - PASS: preventive maintenance steps to keep the component in good condition.
  - MONITOR: monitoring and early intervention steps to prevent further degradation.
  - FAIL: immediate repair or replacement instructions.
  - INVALID: leave as an empty array [].
Use INVALID if the image or observation clearly does not show or relate to the selected component.
"""

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_inspection_prompt(
    machine: str,
    component: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """
    Build a Gemini prompt for image-based inspection of a machine component.

    Args:
        machine: Machine type (e.g. "Cat 950 Wheel Loader").
        component: Component name (e.g. "bucket", "tires").
        history: Optional list of past inspection records for this machine/component.
                 Each record can have keys like date, status, observation, etc.

    Returns:
        Dict with "system" and "user" (or "contents") keys suitable for Gemini Vision.
    """
    system = (
        "You are an expert field inspector for heavy equipment. "
        "You analyze photos and classify condition as PASS (good), MONITOR (watch), or FAIL (unsafe/repair needed). "
        "Be concise and actionable. Base your judgment only on visible condition."
    )

    history_block = ""
    if history:
        history_block = "\n\nPast inspection history for this machine/component:\n"
        for i, record in enumerate(history[-10:], 1):  # last 10
            if isinstance(record, dict):
                history_block += (
                    f"  {i}. {record.get('date', 'N/A')} — "
                    f"{record.get('status', 'N/A')}: {record.get('observation', 'N/A')}\n"
                )
            else:
                history_block += f"  {i}. {record}\n"
        history_block += "Use this history for context (e.g., recurring issues, trends).\n"

    user = (
        f"Inspect this photo for: **{machine}** — **{component}**.\n"
        f"{history_block}\n"
        "From the image, determine current condition and output the result as specified.\n\n"
        f"{RESPONSE_SCHEMA}"
    )

    return {"system": system, "user": user}


def build_voice_prompt(
    machine: str,
    component: str,
    voice_text: str,
) -> dict[str, str]:
    """
    Build a Gemini prompt for analyzing a spoken observation (transcribed text).

    Args:
        machine: Machine type.
        component: Component the inspector is describing.
        voice_text: Transcribed speech (what the inspector said).

    Returns:
        Dict with "system" and "user" keys for Gemini.
    """
    system = (
        "You are an expert field inspector for heavy equipment. "
        "You interpret spoken inspection notes and classify condition as PASS, MONITOR, or FAIL. "
        "Extract the inspector's observation and turn it into a structured assessment with a clear recommended action."
    )

    user = (
        f"Machine: **{machine}**\n"
        f"Component: **{component}**\n\n"
        f"Spoken observation (transcribed):\n\"{voice_text}\"\n\n"
        "Based on this observation, output the inspection result as specified.\n\n"
        f"{RESPONSE_SCHEMA}"
    )

    return {"system": system, "user": user}


def build_parts_prompt() -> dict[str, str]:
    """
    Build a Gemini prompt for identifying an unknown part from a photo.

    Use when the user does not know which component they are photographing.
    The model should suggest machine type(s) and component name(s).

    Returns:
        Dict with "system" and "user" keys for Gemini Vision.
    """
    system = (
        "You are an expert in heavy equipment and Caterpillar machinery. "
        "Identify the part or component shown in the image. "
        "Suggest the most likely machine model(s) (e.g., Cat 950 Wheel Loader, Cat 320 Excavator) "
        "and the component name as used in field inspections. "
        "If condition is visible, also provide a PASS/MONITOR/FAIL assessment."
    )

    machine_list = ", ".join(MACHINES.keys())

    user = (
        "Identify this part/component from the photo.\n\n"
        f"Known machines in this system: {machine_list}.\n"
        "Use standard component names (e.g., bucket, boom, stick, track, tires, lift arms, hydraulic hoses).\n\n"
        "Respond with valid JSON only. No markdown.\n"
        "Schema:\n"
        "{\n"
        '  "machine": "<best guess machine model or \"Unknown\">",\n'
        '  "component": "<component name>",\n'
        '  "status": "PASS" | "MONITOR" | "FAIL" | "UNKNOWN",\n'
        '  "confidence": <float 0.0-1.0>,\n'
        '  "observation": "<what you see>",\n'
        '  "recommended_action": "<if status is not PASS, what to do next; else \"None\">"\n'
        "}\n"
    )

    return {"system": system, "user": user}
