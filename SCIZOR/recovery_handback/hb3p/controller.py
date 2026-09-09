"""Pure HB3-P decision state machine."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from recovery_handback.hb2.metrics import choose_length

TIMES = (20, 80, 160)
LENGTHS = (0, 5, 20, 40, 60, 80)


@dataclass(frozen=True)
class Rule:
    kind: str
    model: Optional[str] = None
    scheduled_t: Optional[int] = None
    scheduled_length: int = 0
    risk_threshold: float = 0.0
    risk_length: int = 80
    penalty: float = 0.25

    def __post_init__(self) -> None:
        if self.kind not in {"none", "scheduled", "model", "risk"}:
            raise ValueError(f"unknown rule kind: {self.kind}")
        if self.kind == "scheduled":
            if self.scheduled_t not in TIMES or self.scheduled_length not in LENGTHS[1:]:
                raise ValueError("scheduled rule is outside the frozen grid")
        if self.kind == "risk":
            if self.risk_length not in LENGTHS or not 0 <= self.risk_threshold <= 1:
                raise ValueError("invalid frozen risk rule")


class SingleIntervention:
    """Choose at fixed times, take over at most once, then permanently hand back."""

    def __init__(self, rule: Rule, horizon: int = 400, *, tolerance: float = 1e-8):
        self.rule = rule
        self.horizon = int(horizon)
        self.tolerance = float(tolerance)
        self.mode = "WAITING"
        self.takeover_t: Optional[int] = None
        self.length = 0
        self.handoff_t: Optional[int] = None
        self.takeovers = 0
        self.query_count = 0
        self.last_t = -1

    def needs_prediction(self, t: int, raw_success_seen: bool = False) -> bool:
        return (
            self.mode == "WAITING"
            and not raw_success_seen
            and t in TIMES
            and self.rule.kind in {"model", "risk"}
        )

    def owner_before_action(
        self,
        t: int,
        probabilities: Optional[dict[str, float]] = None,
        raw_success_seen: bool = False,
    ) -> tuple[str, Optional[str]]:
        if t != self.last_t + 1 or not 0 <= t < self.horizon:
            raise ValueError("call once per environment step, starting at t=0")
        self.last_t = t
        event = None
        if self.mode == "HELPING":
            if self.takeover_t is None:
                raise RuntimeError("helper without takeover")
            if t >= self.takeover_t + self.length:
                self.mode = "HANDED_BACK"
                self.handoff_t = t
                event = "handoff"
            else:
                return "repair", event
        if self.mode in {"HANDED_BACK", "BASE_LOCKED"}:
            return "base", event
        if raw_success_seen:
            self.mode = "BASE_LOCKED"
            return "base", "success_observed_before_takeover"
        if t not in TIMES:
            return "base", event

        length = 0
        if self.rule.kind == "scheduled" and t == self.rule.scheduled_t:
            length = self.rule.scheduled_length
        elif self.rule.kind in {"model", "risk"}:
            if probabilities is None:
                raise ValueError("missing live prediction; refusing silent no-help fallback")
            self.query_count += 1
            if self.rule.kind == "model":
                length = choose_length(
                    probabilities,
                    self.rule.penalty,
                    denominator=float(self.horizon),
                    tolerance=self.tolerance,
                )
            else:
                p0 = float(probabilities["p0"])
                if not math.isfinite(p0) or not 0 <= p0 <= 1:
                    raise ValueError("invalid risk p0")
                if 1.0 - p0 >= self.rule.risk_threshold:
                    length = self.rule.risk_length
        if length:
            if self.takeovers or t + length >= self.horizon:
                raise RuntimeError("invalid takeover")
            self.takeover_t = t
            self.length = int(length)
            self.takeovers = 1
            self.mode = "HELPING"
            return "repair", "takeover"
        if t == TIMES[-1]:
            self.mode = "BASE_LOCKED"
        return "base", event


def select_episode_from_cached_predictions(
    rule: Rule,
    predictions: dict[int, dict[str, float]],
    first_raw_success_state: Optional[int] = None,
    episode_end_state_index: int = 400,
) -> dict:
    controller = SingleIntervention(rule)
    for t in range(min(400, int(episode_end_state_index))):
        seen = first_raw_success_state is not None and int(first_raw_success_state) <= t
        probabilities = predictions.get(t) if controller.needs_prediction(t, seen) else None
        _, event = controller.owner_before_action(t, probabilities, seen)
        if event == "takeover":
            return {
                "takeover_t": t,
                "length": controller.length,
                "queries_before_takeover": controller.query_count,
            }
    return {
        "takeover_t": None,
        "length": 0,
        "queries_before_takeover": controller.query_count,
    }
