from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .models import Region


class ContextPolicy:
    """Small online softmax policy trained from the user's zone clicks."""

    def __init__(self, path: str | Path = "policy.json", learning_rate: float = 0.12):
        self.path = Path(path)
        self.learning_rate = learning_rate
        self.models: dict[str, dict] = {}
        self.matches = 0
        self.examples = 0
        self.load()

    def load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.models = raw.get("models", {})
            self.matches = int(raw.get("matches", 0))
            self.examples = int(raw.get("examples", 0))

    def save(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "models": self.models,
                    "matches": self.matches,
                    "examples": self.examples,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def features(frame: np.ndarray, regions: dict[str, Region]) -> np.ndarray:
        values: list[float] = [1.0]
        height, width = frame.shape[:2]
        for name in sorted(regions):
            region = regions[name]
            left, top = max(0, region.left), max(0, region.top)
            right = min(width, region.left + region.width)
            bottom = min(height, region.top + region.height)
            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                values.extend([0.0] * 12)
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            for image in (crop, hsv):
                mean, std = cv2.meanStdDev(image)
                values.extend((mean[:, 0] / 255.0).tolist())
                values.extend((std[:, 0] / 255.0).tolist())
        return np.asarray(values, dtype=np.float64)

    def _weights(self, state: str, actions: int, features: int) -> np.ndarray:
        model = self.models.get(state)
        if (
            model is None
            or model.get("actions") != actions
            or model.get("features") != features
        ):
            weights = np.zeros((actions, features), dtype=np.float64)
            self.models[state] = {
                "actions": actions,
                "features": features,
                "weights": weights.tolist(),
            }
            return weights
        return np.asarray(model["weights"], dtype=np.float64)

    def predict(self, state: str, context: np.ndarray, actions: int) -> tuple[int, float]:
        weights = self._weights(state, actions, len(context))
        logits = weights @ context
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        index = int(np.argmax(probabilities))
        return index, float(probabilities[index])

    def train(
        self, state: str, context: np.ndarray, actions: int, predicted: int, actual: int
    ) -> int:
        weights = self._weights(state, actions, len(context))
        logits = weights @ context
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        gradient = -probabilities[:, None] * context[None, :]
        gradient[actual] += context
        weights += self.learning_rate * gradient
        self.models[state]["weights"] = weights.tolist()
        reward = int(predicted == actual)
        self.examples += 1
        self.matches += reward
        self.save()
        return reward

    def reset(self) -> None:
        self.models = {}
        self.matches = 0
        self.examples = 0
        if self.path.exists():
            self.path.unlink()


class TileLayoutDictionary:
    """Persistent nearest-neighbor dictionary of submitted maze tile layouts."""

    def __init__(
        self,
        path: str | Path = "tile_layouts.json",
    ) -> None:
        self.path = Path(path)
        self.last_match_status = "No layout checked"
        self.layouts: list[dict] = []
        if self.path.exists():
            self.layouts = json.loads(self.path.read_text(encoding="utf-8")).get(
                "layouts", []
            )
        changed = False
        for index, layout in enumerate(self.layouts, start=1):
            if "id" not in layout:
                layout["id"] = f"layout_{index:03d}"
                changed = True
        if changed:
            self.save()

    @staticmethod
    def _region_names(regions: dict[str, Region]) -> list[str]:
        names = [
            *(f"{direction}_passage" for direction in ("north", "east", "south", "west")),
        ]
        legacy = [
            f"scene_change_{direction}"
            for direction in ("north", "south", "east", "west")
        ]
        # Preserve hashes created before transition detection was replaced.
        if all(name in regions for name in legacy):
            names.extend(legacy)
        return names

    @classmethod
    def signature(cls, frame: np.ndarray, regions: dict[str, Region]) -> str | None:
        digest = hashlib.sha256()
        for name in cls._region_names(regions):
            region = regions.get(name)
            if region is None:
                return None
            height, width = frame.shape[:2]
            left, top = max(0, region.left), max(0, region.top)
            right = min(width, region.left + region.width)
            bottom = min(height, region.top + region.height)
            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                return None
            digest.update(name.encode("utf-8"))
            digest.update(np.asarray(crop.shape, dtype=np.int32).tobytes())
            digest.update(np.ascontiguousarray(crop).tobytes())
        return digest.hexdigest()

    @classmethod
    def descriptor(
        cls, frame: np.ndarray, regions: dict[str, Region]
    ) -> np.ndarray | None:
        """Return a compact visual descriptor used only for provisional matching."""
        parts: list[np.ndarray] = []
        for name in cls._region_names(regions):
            region = regions.get(name)
            if region is None:
                return None
            height, width = frame.shape[:2]
            left, top = max(0, region.left), max(0, region.top)
            right = min(width, region.left + region.width)
            bottom = min(height, region.top + region.height)
            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                return None
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
            parts.append(resized.astype(np.float32).reshape(-1) / 255.0)
        return np.concatenate(parts)

    def _exact_index(self, signature: str) -> int | None:
        for index, layout in enumerate(self.layouts):
            if layout.get("signature") == signature:
                return index
        return None

    def _new_id(self) -> str:
        used = {layout["id"] for layout in self.layouts}
        number = 1
        while f"layout_{number:03d}" in used:
            number += 1
        return f"layout_{number:03d}"

    def submit(
        self,
        signature: str,
        exits: set[str],
        descriptor: np.ndarray | None = None,
    ) -> tuple[str, bool]:
        index = self._exact_index(signature)
        created = index is None
        if created:
            record = {
                "id": self._new_id(),
                "signature": signature,
                "exits": sorted(exits),
                "submissions": 1,
            }
            if descriptor is not None:
                record["descriptor"] = descriptor.tolist()
            self.layouts.append(record)
            index = len(self.layouts) - 1
        else:
            layout = self.layouts[index]
            layout["exits"] = sorted(exits)
            layout["submissions"] = int(layout.get("submissions", 1)) + 1
            if descriptor is not None:
                layout["descriptor"] = descriptor.tolist()
        self.save()
        return self.layouts[index]["id"], created

    def match(self, signature: str) -> tuple[set[str] | None, float, str | None]:
        index = self._exact_index(signature)
        if index is None:
            self.last_match_status = "No exact layout match"
            return None, 0.0, None
        self.last_match_status = f"Exact match: {self.layouts[index]['id']}"
        return (
            set(self.layouts[index]["exits"]),
            1.0,
            self.layouts[index]["id"],
        )

    def nearest(
        self, descriptor: np.ndarray
    ) -> tuple[set[str] | None, float, str | None]:
        best: tuple[float, dict] | None = None
        for layout in self.layouts:
            stored = layout.get("descriptor")
            if stored is None:
                continue
            candidate = np.asarray(stored, dtype=np.float32)
            if candidate.shape != descriptor.shape:
                continue
            difference = float(np.mean(np.abs(candidate - descriptor)))
            if best is None or difference < best[0]:
                best = (difference, layout)
        if best is None:
            self.last_match_status = "No submitted layout has comparison data"
            return None, 1.0, None
        difference, layout = best
        self.last_match_status = (
            f"Provisional nearest: {layout['id']} "
            f"(difference {difference * 100:.1f}%)"
        )
        return set(layout["exits"]), difference, layout["id"]

    def exits_for(self, layout_id: str) -> set[str]:
        for layout in self.layouts:
            if layout["id"] == layout_id:
                return set(layout["exits"])
        return set()

    def save(self) -> None:
        self.path.write_text(
            json.dumps({"layouts": self.layouts}, indent=2), encoding="utf-8"
        )
