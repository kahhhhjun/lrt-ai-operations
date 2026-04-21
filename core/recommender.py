"""Orchestrator: computes three scheduling options, asks GLM to choose one and justify it,
returns a single result dict the UI can consume."""

import json

from core.calculator import compute_options
from core.glm_client import GLM_API_KEY, call_glm

_OPTIONS_SYSTEM = """You are an LRT operations decision-support assistant for Malaysian rail operators.
You will receive operational context and three pre-computed scheduling options (conservative, moderate, aggressive).
Each option already has exact numbers for frequency, cost, revenue, and congestion change.

Your job:
1. Identify the one or two factors driving demand most strongly.
2. Choose one option (conservative, moderate, or aggressive).
3. Explain WHY you chose it and why you rejected the others — cite the numbers.
4. End with a one-sentence action directive for the duty manager.

Respond with ONLY a JSON object, no markdown, no explanation outside the JSON:
{
  "choice": "conservative" | "moderate" | "aggressive",
  "explanation": "your full reasoning here (under 180 words)"
}"""


def _build_options_prompt(inputs: dict, options: list[dict]) -> str:
    return (
        "OPERATIONAL CONTEXT (user inputs):\n"
        f"{json.dumps(inputs, indent=2)}\n\n"
        "THREE PRE-COMPUTED SCHEDULING OPTIONS:\n"
        f"{json.dumps(options, indent=2)}\n\n"
        "Choose one option and explain your decision. Return only the JSON specified."
    )


def _get_glm_decision(inputs: dict, options: list[dict]) -> tuple[str, str]:
    if not GLM_API_KEY:
        return (
            "moderate",
            "[PLACEHOLDER — set GLM_API_KEY in .env to get real GLM reasoning]\n\n"
            "The moderate option balances service quality with cost efficiency. "
            "It meets the mathematically optimal frequency to handle the predicted demand "
            "without over-provisioning. The conservative option risks platform overcrowding "
            "during peak load, while the aggressive option increases cost beyond what the "
            "demand level justifies. Duty manager should action the moderate schedule "
            "immediately.",
        )

    prompt = _build_options_prompt(inputs, options)
    response = call_glm(prompt, system=_OPTIONS_SYSTEM, temperature=0.3)

    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        parsed = json.loads(response[start:end])
        return parsed["choice"], parsed["explanation"]
    except Exception:
        # GLM didn't return clean JSON — treat whole response as explanation
        return "moderate", response


def recommend(inputs: dict) -> dict:
    options = compute_options(inputs)
    choice, explanation = _get_glm_decision(inputs, options)

    chosen = next((o for o in options if o["label"] == choice), options[1])

    delta = chosen["recommended_frequency_per_hr"] - chosen["current_frequency_per_hr"]
    if delta > 0:
        schedule_update = (
            f"Add {delta} train(s)/hr on {inputs.get('line', 'the line')} "
            f"around {inputs['datetime']}."
        )
    elif delta < 0:
        schedule_update = (
            f"Reduce by {abs(delta)} train(s)/hr on {inputs.get('line', 'the line')} "
            f"around {inputs['datetime']}."
        )
    else:
        schedule_update = "Maintain current schedule."

    return {
        # Chosen option metrics at top level (backward compat)
        **{k: v for k, v in chosen.items() if k != "label"},
        # All options for the UI to display
        "options": options,
        "chosen_option": choice,
        "schedule_update": schedule_update,
        "explanation": explanation,
        "confidence": "high" if abs(delta) <= 4 else "medium",
    }
