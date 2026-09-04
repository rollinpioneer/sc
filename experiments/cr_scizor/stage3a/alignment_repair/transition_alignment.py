"""Frozen mapping between recorded states/images and executed actions."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


LAYOUTS = {
    "pre_action": {"state_before_action_offset": 0, "expected_next_state_offset": 1, "image_before_action_offset": 0, "minimum_query_t": 0},
    "post_action": {"state_before_action_offset": -1, "expected_next_state_offset": 0, "image_before_action_offset": -1, "minimum_query_t": 1},
}
REPLAY_MODES = {"direct_state_reset", "prefix_replay"}


@dataclass(frozen=True)
class TransitionAlignment:
    state_layout: str
    oracle_replay_mode: str
    state_before_action_offset: int
    expected_next_state_offset: int
    image_before_action_offset: int
    minimum_query_t: int

    @classmethod
    def from_layout(cls, state_layout: str, oracle_replay_mode: str) -> "TransitionAlignment":
        if state_layout not in LAYOUTS or oracle_replay_mode not in REPLAY_MODES:
            raise ValueError(f"unsupported alignment: {state_layout}/{oracle_replay_mode}")
        return cls(state_layout=state_layout, oracle_replay_mode=oracle_replay_mode, **LAYOUTS[state_layout])

    @classmethod
    def load(cls, path: str | Path) -> "TransitionAlignment":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("status") != "passed":
            raise ValueError(f"refusing non-passing transition alignment: {data.get('status')}")
        keys = ("state_layout", "oracle_replay_mode", "state_before_action_offset", "expected_next_state_offset", "image_before_action_offset", "minimum_query_t")
        if any(key not in data for key in keys):
            raise ValueError("alignment JSON is incomplete")
        alignment = cls(**{key: data[key] for key in keys})
        if alignment.state_layout not in LAYOUTS or alignment.oracle_replay_mode not in REPLAY_MODES:
            raise ValueError("alignment JSON contains unsupported values")
        if any(getattr(alignment, key) != value for key, value in LAYOUTS[alignment.state_layout].items()):
            raise ValueError("alignment JSON disagrees with frozen layout map")
        return alignment

    def state_before_action_index(self, action_t: int) -> int:
        return int(action_t) + self.state_before_action_offset

    def expected_next_state_index(self, action_t: int) -> int:
        return int(action_t) + self.expected_next_state_offset

    def image_before_action_index(self, action_t: int) -> int:
        return int(action_t) + self.image_before_action_offset

    def is_valid_action_index(self, action_t: int, *, num_states: int, num_actions: int, minimum_future_actions: int = 1) -> bool:
        t = int(action_t)
        return (
            t >= self.minimum_query_t and 0 <= t < int(num_actions)
            and 0 <= self.state_before_action_index(t) < int(num_states)
            and 0 <= self.expected_next_state_index(t) < int(num_states)
            and 0 <= self.image_before_action_index(t) < int(num_states)
            and int(num_actions) - t >= int(minimum_future_actions)
        )
