"""What the review has already done, keyed by (stream, cutoff, horizon).

Progressive resolution is the whole reason this exists: a cutoff's h5 resolves a
week after the run, h10 two weeks, h21 about a month. Keying on the horizon means
a newly resolved h21 is new work while its h5 and h10 stay triaged, and re-running
the weekly job twice does nothing the second time.

States, in order::

    pending -> resolved -> triaged -> deep_dived -> frozen

``frozen`` means every horizon of the cutoff is resolved and triaged; nothing
about the card can change again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal


Status = Literal["pending", "resolved", "triaged", "deep_dived", "frozen"]
_ORDER: tuple[Status, ...] = ("pending", "resolved", "triaged", "deep_dived", "frozen")


@dataclass(frozen=True, order=True)
class ReviewKey:
    stream: str
    cutoff: date
    horizon: int

    def __str__(self) -> str:
        return f"{self.stream}|{self.cutoff.isoformat()}|{self.horizon}"

    @classmethod
    def parse(cls, text: str) -> ReviewKey:
        stream, cutoff, horizon = text.split("|")
        return cls(stream=stream, cutoff=date.fromisoformat(cutoff), horizon=int(horizon))

    @property
    def card_key(self) -> tuple[str, date]:
        return (self.stream, self.cutoff)


@dataclass
class KeyState:
    status: Status = "pending"
    card_hash: str | None = None
    card_version: int | None = None
    annotation_id: str | None = None
    #: Whether the miss/hit verdict at this horizon flipped when it re-scored.
    verdict: str | None = None
    updated_at: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


class ReviewState:
    """A JSON map of key -> `KeyState`. Load, mutate, save; nothing clever."""

    def __init__(self, path: Path):
        self.path = path
        self.keys: dict[ReviewKey, KeyState] = {}
        self.meta: dict[str, Any] = {}
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.meta = dict(payload.get("meta", {}))
            for text, item in payload.get("keys", {}).items():
                self.keys[ReviewKey.parse(text)] = KeyState(**item)

    def get(self, key: ReviewKey) -> KeyState:
        return self.keys.setdefault(key, KeyState())

    def advance(self, key: ReviewKey, status: Status, **notes: Any) -> KeyState:
        """Move a key forward. Moving backward raises: state only ever accrues."""
        state = self.get(key)
        if _ORDER.index(status) < _ORDER.index(state.status):
            raise ValueError(f"{key}: cannot move from {state.status!r} back to {status!r}")
        state.status = status
        state.updated_at = datetime.now().isoformat(timespec="seconds")
        for name, value in notes.items():
            if name in ("card_hash", "card_version", "annotation_id", "verdict"):
                setattr(state, name, value)
            else:
                state.notes[name] = value
        return state

    def keys_in(self, *statuses: Status) -> list[ReviewKey]:
        wanted = set(statuses)
        return sorted(key for key, state in self.keys.items() if state.status in wanted)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self.meta,
            "keys": {str(key): state.__dict__ for key, state in sorted(self.keys.items())},
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["KeyState", "ReviewKey", "ReviewState", "Status"]
