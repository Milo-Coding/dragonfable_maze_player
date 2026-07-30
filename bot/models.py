from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Mode(str, Enum):
    IDLE = "idle"
    TRAINING = "training"
    LIVE = "live"


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    def contains(self, point: Point) -> bool:
        return (
            self.left <= point.x < self.left + self.width
            and self.top <= point.y < self.top + self.height
        )


@dataclass
class Decision:
    state: str
    point: Point | None = None
    reason: str = ""
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def state_name(self) -> str:
        return self.state
