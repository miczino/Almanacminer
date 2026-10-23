"""Offline smoke tests for the two Almanac miner agents.

These tests use fake provider responses and spend no portal credits.

Expected layout (recommended):
    parent/
      almanac/          # official corvxai/almanac repo
      Almanacminer/     # this repo

You can also set ALMANAC_REPO to the official repo path.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
official = os.getenv("ALMANAC_REPO")
if official:
    official_path = Path(official).expanduser().resolve()
else:
    official_path = (HERE.parent / "almanac").resolve()

if not (official_path / "src" / "agent").exists():
    raise SystemExit(
        "Official Almanac repo not found. Clone https://github.com/corvxai/almanac "
        "next to this repo or set ALMANAC_REPO to its path."
    )

sys.path.insert(0, str(official_path))

from phase1_starter import AlmanacPhase1Starter
from phase2_research import AlmanacPhase2Research


@dataclass
class FakeEvent:
    title: str = "Will Example Event happen by December 31?"
    description: str = "Resolves YES if Example Event happens by the deadline."
    source: str = "polymarket"
    source_id: str = "example-market"
    outcomes: list | None = None
    current_outcome_prices: dict | None = None

    def __post_init__(self):
        if self.outcomes is None:
            self.outcomes = [
                {"outcomeId": "yes-id", "name": "Yes"},
                {"outcomeId": "no-id", "name": "No"},
            ]
        if self.current_outcome_prices is None:
            self.current_outcome_prices = {"yes-id": 0.60, "no-id": 0.40}


class FakeContext:
    def __init__(self):
        self.event = FakeEvent()
        self.steps = []
        self.calls = []

    @property
    def event_title(self):
        return self.event.title

    @property
    def event_description(self):
        return self.event.description

    def record_reasoning_step(self, *args, **kwargs):
        self.steps.append((args, kwargs))

    def call_provider(self, provider_id, call_type, params):
        self.calls.append((provider_id, call_type, params))
        model = str(params.get("model") or "")
        messages = params.get("messages") or []
        joined = "\n".join(
            str(m.get("content") or "") for m in messages if isinstance(m, dict)
        )

        if model.endswith(":online"):
            if "ADVERSARIAL" in joined.upper() or "STRESS-TEST" in joined.upper():
                output = (
                    "Countercase: the deadline is close and the relevant actor has not "
                    "taken the required action. Source: example official statement."
                )
            else:
                output = (
                    "Current research: one official update supports YES, while the base "
                    "rate remains mixed. Source: example official statement."
                )
            return {"output": output, "finishReason": "stop"}

        if "independent_probability" in joined:
            return {
                "output": (
                    '{"reasoning":"Evidence modestly favors YES but is not decisive.",'
                    '"independent_probability":0.72,"evidence_strength":0.60}'
                ),
                "finishReason": "stop",
            }

        return {
            "output": (
                '{"reasoning":"Current evidence modestly favors YES.",'
                '"prediction":0.64,"confidence":0.62}'
            ),
            "finishReason": "stop",
        }


def _assert_valid_result(result):
    assert 0.0 <= result.prediction <= 1.0
    assert result.reasoning
    assert result.beliefPath
    assert result.beliefPath[-1].type == "final"
    assert abs(result.beliefPath[-1].probability - result.prediction) <= 5e-5


def test_phase1():
    ctx = FakeContext()
    result = AlmanacPhase1Starter().predict(ctx)
    _assert_valid_result(result)
    assert result.prediction == 0.64
    assert len(ctx.calls) == 2


def test_phase2_market_anchor():
    ctx = FakeContext()
    result = AlmanacPhase2Research().predict(ctx)
    _assert_valid_result(result)
    assert 0.60 < result.prediction < 0.72
    assert result.metadata["market_prior"] == 0.60
    assert len(ctx.calls) == 3


def test_phase2_provider_failure_is_still_valid():
    class FailingContext(FakeContext):
        def call_provider(self, provider_id, call_type, params):
            raise RuntimeError("simulated provider failure")

    ctx = FailingContext()
    result = AlmanacPhase2Research().predict(ctx)
    _assert_valid_result(result)
    assert result.prediction == 0.60


if __name__ == "__main__":
    test_phase1()
    test_phase2_market_anchor()
    test_phase2_provider_failure_is_still_valid()
    print("All Almanac miner smoke tests passed.")
