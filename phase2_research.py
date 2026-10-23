"""Almanac SN41 IM2 Phase 2 market-anchored research miner.

Optimized for the current Almanac forecasting scoring design:
- 70% market-relative accuracy / Brier skill
- 30% market-relative edge
- invalid-rate and inactivity gates

The agent starts from the live YES market probability when the assignment exposes
it, researches current evidence and the strongest counter-case, estimates an
independent probability, then shrinks that estimate toward the market. This is
deliberately conservative: it only takes meaningful distance from the market when
the evidence model reports strong evidence.

All network access goes through ctx.call_provider().
"""

from __future__ import annotations

import json
import re
from uuid import UUID

from src.agent.base import BaseAgent
from src.agent.context import ForecastingContext
from src.core.schemas import AgentResult, BeliefStep, ReasoningStepType


SEARCH_MODEL = "openai/gpt-4o-mini:online"
SYNTHESIS_MODEL = "openai/gpt-4o-mini"

MAX_RAW_MARKET_DELTA = 0.20
MIN_FINAL_PROBABILITY = 0.02
MAX_FINAL_PROBABILITY = 0.98


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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
            if isinstance(content, str) and content.strip():
                return content.strip()
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            if isinstance(reasoning, str):
                return reasoning.strip()
    return ""


def _call_chat(
    ctx: ForecastingContext,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float = 0.15,
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
    return v if 0.0 <= v <= 1.0 else None


def _yes_market_probability(ctx: ForecastingContext) -> float | None:
    """Read the live YES price embedded in production assignments when present."""
    event = ctx.event
    prices = getattr(event, "current_outcome_prices", None) or {}
    outcomes = getattr(event, "outcomes", None) or []

    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        name = str(outcome.get("name") or "").strip().lower()
        outcome_id = str(outcome.get("outcomeId") or outcome.get("outcome_id") or "")
        if name == "yes" and outcome_id in prices:
            p = _probability(prices.get(outcome_id))
            if p is not None:
                return p
    return None


def _search_market_probability(ctx: ForecastingContext) -> tuple[float | None, str]:
    """Fallback for local test-agent, where portal events currently omit prices."""
    prompt = (
        "Find the current prediction-market implied probability that this exact event "
        "resolves YES. Prefer the market named in the event; otherwise use Polymarket "
        "or another liquid prediction market only if it clearly matches the same "
        "resolution question. Return ONLY a JSON object with keys: "
        '"market_probability" (0..1 or null), "source" (short string), '
        '"reasoning" (one sentence). Do not invent a market price.\n\n'
        f"QUESTION: {ctx.event_title}\n"
        f"DESCRIPTION: {ctx.event_description}"
    )
    text = _call_chat(
        ctx,
        SEARCH_MODEL,
        [{"role": "user", "content": prompt}],
        max_tokens=450,
        temperature=0.1,
    )
    obj = _extract_json(text)
    if obj is None:
        return None, text
    p = _probability(obj.get("market_probability"))
    return p, str(obj.get("reasoning") or obj.get("source") or text).strip()


def _research_current_state(ctx: ForecastingContext, market_p: float | None) -> str:
    market_note = (
        f"The live YES market prior available to the agent is {market_p:.3f}."
        if market_p is not None
        else "No reliable live market prior is available to the agent."
    )
    prompt = (
        "Act as a prediction-market research analyst. Research the exact binary event "
        "below using current web information. Prioritize primary/official sources and "
        "recent reporting. Identify: (1) resolution wording and deadline traps, "
        "(2) base-rate or historical precedent, (3) strongest evidence for YES, "
        "(4) strongest evidence for NO, (5) what changed most recently, and "
        "(6) the single most decision-relevant unknown. Include dates, source names "
        "and URLs where possible. Do not merely repeat market odds and do not invent "
        "facts.\n\n"
        f"{market_note}\n"
        f"QUESTION: {ctx.event_title}\n"
        f"DESCRIPTION: {ctx.event_description}"
    )
    return _call_chat(
        ctx,
        SEARCH_MODEL,
        [{"role": "user", "content": prompt}],
        max_tokens=1100,
        temperature=0.1,
    )


def _research_countercase(
    ctx: ForecastingContext,
    market_p: float | None,
    research: str,
) -> str:
    prior = f"{market_p:.3f}" if market_p is not None else "unknown"
    prompt = (
        "Stress-test this forecast as an adversarial researcher. Find the strongest "
        "credible case that the obvious/current consensus is wrong. Focus on recent "
        "facts, incentives of key actors, timing, resolution-rule edge cases, and "
        "information the first research pass may have missed. Cite dates, source names "
        "and URLs where possible. Do not be contrarian for its own sake.\n\n"
        f"EVENT: {ctx.event_title}\n"
        f"MARKET YES PRIOR: {prior}\n"
        f"FIRST RESEARCH PASS:\n{research[:7000]}"
    )
    return _call_chat(
        ctx,
        SEARCH_MODEL,
        [{"role": "user", "content": prompt}],
        max_tokens=850,
        temperature=0.1,
    )


def _independent_estimate(
    ctx: ForecastingContext,
    market_p: float | None,
    research: str,
    countercase: str,
) -> tuple[float, float, str]:
    market_note = (
        f"Market prior: {market_p:.4f}. Treat it as a strong prior, but estimate the "
        "event independently from evidence before comparing back to it."
        if market_p is not None
        else "No reliable market prior is available; use base rates and evidence."
    )
    system = (
        "You are a calibrated superforecaster. Return ONLY one JSON object with keys "
        'in this order: "reasoning" (3-6 concise sentences), '
        '"independent_probability" (decimal 0..1), '
        '"evidence_strength" (decimal 0..1). Evidence strength measures how much '
        "the cited information justifies moving away from a reasonable market/base "
        "prior: 0 means weak/noisy, 1 means unusually decisive. Avoid false precision "
        "and avoid extreme probabilities unless resolution is nearly determined."
    )
    user = (
        f"{market_note}\n\n"
        f"QUESTION: {ctx.event_title}\n"
        f"DESCRIPTION: {ctx.event_description}\n\n"
        f"CURRENT-STATE RESEARCH:\n{research or '(none)'}\n\n"
        f"ADVERSARIAL COUNTERCASE:\n{countercase or '(none)'}\n\n"
        "Produce the independent estimate now."
    )
    text = _call_chat(
        ctx,
        SYNTHESIS_MODEL,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=800,
        temperature=0.1,
    )
    obj = _extract_json(text)
    if obj is None:
        fallback = market_p if market_p is not None else 0.50
        return fallback, 0.0, "Synthesis was unparseable; held the prior."

    independent = _probability(obj.get("independent_probability"))
    strength = _probability(obj.get("evidence_strength"))
    reasoning = str(obj.get("reasoning") or "").strip()
    if independent is None:
        independent = market_p if market_p is not None else 0.50
    if strength is None:
        strength = 0.0
    if not reasoning:
        reasoning = "Structured synthesis contained no reasoning; used conservative fallback."
    return independent, strength, reasoning


def _market_anchored_probability(
    market_p: float | None,
    independent_p: float,
    strength: float,
) -> float:
    strength = _clamp(strength, 0.0, 1.0)

    if market_p is not None:
        raw_delta = _clamp(
            independent_p - market_p,
            -MAX_RAW_MARKET_DELTA,
            MAX_RAW_MARKET_DELTA,
        )
        move_fraction = 0.20 + 0.55 * strength
        final = market_p + raw_delta * move_fraction
    else:
        raw_delta = _clamp(independent_p - 0.50, -0.35, 0.35)
        move_fraction = 0.35 + 0.45 * strength
        final = 0.50 + raw_delta * move_fraction

    return round(_clamp(final, MIN_FINAL_PROBABILITY, MAX_FINAL_PROBABILITY), 4)


class AlmanacPhase2Research(BaseAgent):
    """Market-anchored, adversarial-research forecasting agent."""

    agent_id = UUID("8cb6d960-334a-4ea7-a476-8db999c52202")
    agent_version = "2.0.0"

    def predict(self, ctx: ForecastingContext) -> AgentResult:
        market_p = _yes_market_probability(ctx)
        market_source = "assignment.current_outcome_prices"

        if market_p is None:
            market_p, market_note = _search_market_probability(ctx)
            market_source = "online market search"
            ctx.record_reasoning_step(
                ReasoningStepType.GAP_QUERY,
                reasoning_text=(
                    "Production assignment had no directly readable YES price; "
                    f"used online market lookup. {market_note[:900]}"
                ),
                provider_id="openrouter",
                inference_model_used=SEARCH_MODEL,
            )

        prior = market_p if market_p is not None else 0.50
        ctx.record_reasoning_step(
            ReasoningStepType.PRIOR,
            reasoning_text=(
                f"Starting prior={prior:.4f}; source={market_source if market_p is not None else 'neutral fallback'}."
            ),
            intermediate_probability=prior,
        )

        research = _research_current_state(ctx, market_p)
        ctx.record_reasoning_step(
            ReasoningStepType.GAP_QUERY,
            reasoning_text=(
                "Current-state research: "
                + (research[:1300] if research else "no usable research returned")
            ),
            provider_id="openrouter",
            inference_model_used=SEARCH_MODEL,
        )

        countercase = _research_countercase(ctx, market_p, research)
        ctx.record_reasoning_step(
            ReasoningStepType.GAP_QUERY,
            reasoning_text=(
                "Adversarial countercase: "
                + (countercase[:1300] if countercase else "no usable countercase returned")
            ),
            provider_id="openrouter",
            inference_model_used=SEARCH_MODEL,
        )

        independent_p, strength, synthesis_reasoning = _independent_estimate(
            ctx, market_p, research, countercase
        )
        independent_p = round(_clamp(independent_p, 0.0, 1.0), 4)
        ctx.record_reasoning_step(
            ReasoningStepType.BELIEF_UPDATE,
            reasoning_text=(
                f"Independent evidence estimate={independent_p:.4f}; "
                f"evidence_strength={strength:.2f}. {synthesis_reasoning}"
            ),
            intermediate_probability=independent_p,
            inference_model_used=SYNTHESIS_MODEL,
        )

        final_p = _market_anchored_probability(market_p, independent_p, strength)
        market_label = f"{market_p:.4f}" if market_p is not None else "unavailable"
        final_reasoning = (
            f"{synthesis_reasoning} Market prior={market_label}; "
            f"independent estimate={independent_p:.4f}; evidence strength={strength:.2f}. "
            f"Applied conservative market anchoring and submitted P(YES)={final_p:.4f}."
        )

        ctx.record_reasoning_step(
            ReasoningStepType.BELIEF_UPDATE,
            reasoning_text=final_reasoning,
            intermediate_probability=final_p,
            confidence_delta=final_p - prior,
            inference_model_used=SYNTHESIS_MODEL,
        )

        confidence = round(_clamp(0.35 + 0.50 * strength, 0.0, 1.0), 4)
        return AgentResult(
            prediction=final_p,
            confidence=confidence,
            reasoning=final_reasoning[:8000],
            beliefPath=[
                BeliefStep(
                    step=0,
                    type="prior",
                    probability=round(prior, 4),
                    text=(
                        f"Started from live YES market prior {prior:.4f}."
                        if market_p is not None
                        else "No reliable market price available; used neutral 0.50 prior."
                    ),
                ),
                BeliefStep(
                    step=1,
                    type="update",
                    probability=independent_p,
                    text=(
                        f"Independent evidence estimate after current-state and countercase research. "
                        f"Evidence strength={strength:.2f}. {synthesis_reasoning}"
                    )[:4000],
                ),
                BeliefStep(
                    step=2,
                    type="update",
                    probability=final_p,
                    text=(
                        "Risk-controlled shrinkage toward the market/base prior to reduce "
                        "overconfident market-relative Brier errors."
                    ),
                ),
                BeliefStep(
                    step=3,
                    type="final",
                    probability=final_p,
                    text=final_reasoning[:4000],
                ),
            ],
            metadata={
                "phase": "phase2",
                "market_prior": market_p,
                "market_source": market_source if market_p is not None else None,
                "independent_probability": independent_p,
                "evidence_strength": round(strength, 4),
                "search_model": SEARCH_MODEL,
                "synthesis_model": SYNTHESIS_MODEL,
                "max_raw_market_delta": MAX_RAW_MARKET_DELTA,
            },
        )
