"""Almanac SN41 IM2 Phase 1 starter miner.

Goal: a low-cost, valid, easy-to-debug baseline:
1) one current web research call through OpenRouter's online model;
2) one synthesis call through gpt-4o-mini;
3) strict probability parsing and a valid beliefPath;
4) fail closed to a valid neutral forecast instead of producing an invalid result.

Designed for the Almanac forecasting sandbox. External network access is only
through ctx.call_provider().
"""

from __future__ import annotations

import json
import re
from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, BeliefStep, ReasoningStepType


SEARCH_MODEL = "openai/gpt-4o-mini:online"
FORECAST_MODEL = "openai/gpt-4o-mini"


def _extract_text(raw: dict) -> str:
    if not isinstance(raw, dict):
        return ""
    output = raw.get("output")
    if isinstance(output, str) and output.strip():
        return output.strip()
    choices = raw.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
    return ""


def _call_chat(
    ctx: ForecastingContext,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float = 0.2,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        raw = ctx.call_provider("openrouter", "chat_completion", params)
    except Exception:
        return ""

    finish = str(raw.get("finishReason") or raw.get("finish_reason") or "").lower()
    if finish == "length":
        retry = dict(params)
        retry["max_tokens"] = max_tokens * 2
        try:
            raw = ctx.call_provider("openrouter", "chat_completion", retry)
        except Exception:
            pass
    return _extract_text(raw)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.I | re.S)
    if fenced:
        stripped = fenced.group(1).strip()
    start = stripped.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(stripped[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _probability(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, str):
            s = value.strip()
            is_percent = s.endswith("%")
            s = s.rstrip("%").strip()
            v = float(s)
            if is_percent:
                v /= 100.0
        else:
            v = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= v <= 1.0:
        return v
    return None


def _forecast_from_text(text: str) -> tuple[float, float | None, str] | None:
    obj = _extract_json(text)
    if obj is None:
        return None
    p = _probability(obj.get("prediction"))
    confidence = _probability(obj.get("confidence"))
    reasoning = str(obj.get("reasoning") or "").strip()
    if p is None or not reasoning:
        return None
    return p, confidence, reasoning


class AlmanacPhase1Starter(BaseAgent):
    """Low-cost search + synthesis baseline for SN41 IM2."""

    agent_id = UUID("6f4c28b2-83d8-4b26-a36f-6f5c6a7e1101")
    agent_version = "1.1.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        title = ctx.event_title
        description = ctx.event_description

        research_prompt = (
            "Research this binary forecasting question using current web information. "
            "Focus on facts that affect whether it resolves YES. Include dates, source "
            "names and URLs where possible. Separate evidence for YES, evidence for NO, "
            "and unresolved uncertainty. Do not fabricate facts.\n\n"
            f"QUESTION: {title}\n"
            f"DESCRIPTION: {description}"
        )
        research = _call_chat(
            ctx,
            SEARCH_MODEL,
            [{"role": "user", "content": research_prompt}],
            max_tokens=800,
            temperature=0.1,
        )

        ctx.record_reasoning_step(
            ReasoningStepType.GAP_QUERY,
            reasoning_text=(
                "Current web research completed. "
                + (research[:1200] if research else "No usable research returned.")
            ),
            provider_id="openrouter",
            inference_model_used=SEARCH_MODEL,
        )

        system = (
            "You are a calibrated binary-event forecaster. Estimate P(YES). "
            "Use base rates, current evidence, time remaining, and the exact resolution "
            "wording. Avoid 0 or 1 except for near-certainty. Return ONLY one JSON object "
            'with keys in this order: "reasoning" (2-5 concise sentences), '
            '"prediction" (decimal 0..1), "confidence" (decimal 0..1).'
        )
        user = (
            f"QUESTION: {title}\n\n"
            f"DESCRIPTION: {description}\n\n"
            f"CURRENT RESEARCH:\n{research or '(research unavailable)'}\n\n"
            "Return the JSON forecast now."
        )
        answer = _call_chat(
            ctx,
            FORECAST_MODEL,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=700,
            temperature=0.15,
        )
        parsed = _forecast_from_text(answer)

        if parsed is None:
            p = 0.50
            confidence = None
            reasoning = (
                "The forecasting model did not return a valid structured probability; "
                "returned a neutral 0.50 forecast to remain valid."
            )
        else:
            p, confidence, reasoning = parsed

        p = round(float(p), 4)
        ctx.record_reasoning_step(
            ReasoningStepType.BELIEF_UPDATE,
            reasoning_text=reasoning,
            intermediate_probability=p,
            inference_model_used=FORECAST_MODEL,
        )

        return AgentResult(
            prediction=p,
            confidence=confidence,
            reasoning=reasoning,
            beliefPath=[
                BeliefStep(
                    step=0,
                    type="prior",
                    probability=0.50,
                    text="Neutral starting prior before current research.",
                ),
                BeliefStep(
                    step=1,
                    type="final",
                    probability=p,
                    text=reasoning[:4000],
                ),
            ],
            metadata={
                "phase": "phase1",
                "search_model": SEARCH_MODEL,
                "forecast_model": FORECAST_MODEL,
            },
        )
