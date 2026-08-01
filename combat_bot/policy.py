from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from bot.models import Region


HUE_BINS = 12
SATURATION_BINS = 4
VALUE_BINS = 4
GRID_SIZE = 8
FEATURES_PER_CELL = HUE_BINS + SATURATION_BINS + VALUE_BINS
FEATURES_PER_ZONE = GRID_SIZE * GRID_SIZE * FEATURES_PER_CELL


def visual_features(frame: np.ndarray, zones: list[Region]) -> np.ndarray:
    """Spatially one-hot encode an 8x8 HSV image for every viewing zone.

    Hue uses 12 bins (30-degree color families), while saturation and value
    each use four bins. Every one of the 64 spatial cells activates three
    categories. Unlike whole-zone averaging, this preserves the shape of
    digits, cooldown glyphs, icons, and colored buttons.
    """
    values: list[float] = []
    height, width = frame.shape[:2]
    for zone in zones:
        left, top = max(0, zone.left), max(0, zone.top)
        right = min(width, zone.left + zone.width)
        bottom = min(height, zone.top + zone.height)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            values.extend([0.0] * FEATURES_PER_ZONE)
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grid = cv2.resize(hsv, (GRID_SIZE, GRID_SIZE), interpolation=cv2.INTER_AREA)
        encoded = np.zeros((GRID_SIZE, GRID_SIZE, FEATURES_PER_CELL), dtype=np.float32)
        for row in range(GRID_SIZE):
            for column in range(GRID_SIZE):
                hue, saturation, value = grid[row, column]
                hue_index = min(HUE_BINS - 1, int(hue) * HUE_BINS // 180)
                saturation_index = min(SATURATION_BINS - 1, int(saturation) * SATURATION_BINS // 256)
                value_index = min(VALUE_BINS - 1, int(value) * VALUE_BINS // 256)
                encoded[row, column, hue_index] = 1.0
                encoded[row, column, HUE_BINS + saturation_index] = 1.0
                encoded[row, column, HUE_BINS + SATURATION_BINS + value_index] = 1.0
        values.extend(encoded.reshape(-1).tolist())
    return np.asarray(values, dtype=np.float32)


class DensePolicy:
    """Additive dense policy with an independent branch for each visual zone."""

    def __init__(self, path: str | Path = "combat_policy_v2.json") -> None:
        self.path = Path(path)
        self.schema: dict = {}
        self.branch_weights: list[list[np.ndarray]] = []
        self.branch_biases: list[list[np.ndarray]] = []
        self.output_bias = np.empty(0, dtype=np.float32)
        self.examples = 0
        self.matches = 0
        self.updates = 0
        self.last_loss = 0.0
        self._rng = np.random.default_rng()
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.schema = raw.get("schema", {})
        self.branch_weights = [[np.asarray(x, dtype=np.float32) for x in branch]
                               for branch in raw.get("branch_weights", [])]
        self.branch_biases = [[np.asarray(x, dtype=np.float32) for x in branch]
                              for branch in raw.get("branch_biases", [])]
        self.output_bias = np.asarray(raw.get("output_bias", []), dtype=np.float32)
        for name in ("examples", "matches", "updates"):
            setattr(self, name, int(raw.get(name, 0)))
        self.last_loss = float(raw.get("last_loss", 0.0))

    def save(self) -> None:
        self.path.write_text(json.dumps({
            "schema": self.schema,
            "branch_weights": [[x.tolist() for x in branch] for branch in self.branch_weights],
            "branch_biases": [[x.tolist() for x in branch] for branch in self.branch_biases],
            "output_bias": self.output_bias.tolist(),
            "examples": self.examples,
            "matches": self.matches,
            "updates": self.updates,
            "last_loss": self.last_loss,
        }), encoding="utf-8")

    def ensure(self, feature_names: list[str], action_names: list[str], hidden: list[int]) -> None:
        schema = {"architecture": "spatial_zone_branches_v1", "features": feature_names,
                  "actions": action_names, "hidden": hidden, "grid": GRID_SIZE}
        if schema == self.schema and len(self.branch_weights) == len(feature_names):
            return
        sizes = [FEATURES_PER_ZONE, *hidden, len(action_names)]
        self.schema = schema
        self.branch_weights = []
        self.branch_biases = []
        for _name in feature_names:
            weights, biases = [], []
            for layer, (fan_in, fan_out) in enumerate(zip(sizes, sizes[1:])):
                scale = np.sqrt(2.0 / max(1, fan_in))
                weights.append(self._rng.normal(0, scale, (fan_out, fan_in)).astype(np.float32))
                # The final action contribution has no branch-specific bias.
                biases.append(np.zeros(fan_out, dtype=np.float32))
            self.branch_weights.append(weights); self.branch_biases.append(biases)
        self.output_bias = np.zeros(len(action_names), dtype=np.float32)
        self.examples = self.matches = self.updates = 0
        self.last_loss = 0.0
        self.save()

    def probabilities(self, inputs: np.ndarray) -> np.ndarray:
        logits = self.output_bias.copy()
        for zone, weights in enumerate(self.branch_weights):
            activation = inputs[zone * FEATURES_PER_ZONE:(zone + 1) * FEATURES_PER_ZONE]
            for layer in range(len(weights) - 1):
                activation = np.maximum(0.0, weights[layer] @ activation + self.branch_biases[zone][layer])
            logits += weights[-1] @ activation
        logits -= logits.max()
        values = np.exp(logits)
        return values / values.sum()

    def zone_importance(self, inputs: np.ndarray, action: int) -> np.ndarray:
        """Return each zone branch's contribution margin for one action.

        A branch is important when it scores the selected action differently
        from its average score for the other actions. Absolute margins are
        used for ranking because strong supporting and opposing evidence can
        both materially determine the final choice.
        """
        importance: list[float] = []
        for zone, weights in enumerate(self.branch_weights):
            activation = inputs[zone * FEATURES_PER_ZONE:(zone + 1) * FEATURES_PER_ZONE]
            for layer in range(len(weights) - 1):
                activation = np.maximum(
                    0.0, weights[layer] @ activation + self.branch_biases[zone][layer]
                )
            contributions = weights[-1] @ activation
            others = np.delete(contributions, action)
            baseline = float(others.mean()) if others.size else 0.0
            importance.append(abs(float(contributions[action]) - baseline))
        return np.asarray(importance, dtype=np.float32)

    def select(self, inputs: np.ndarray) -> tuple[int, float]:
        probabilities = self.probabilities(inputs)
        index = int(np.argmax(probabilities))
        return index, float(probabilities[index])

    def train(self, inputs: np.ndarray, actual: int, predicted: int, learning_rate: float,
              zone_mask: np.ndarray | None = None, count_example: bool = True) -> float:
        caches = []
        logits = self.output_bias.copy()
        for zone, weights in enumerate(self.branch_weights):
            value = inputs[zone * FEATURES_PER_ZONE:(zone + 1) * FEATURES_PER_ZONE]
            activations, preactivations = [value], []
            for layer in range(len(weights) - 1):
                before = weights[layer] @ value + self.branch_biases[zone][layer]
                preactivations.append(before); value = np.maximum(0.0, before); activations.append(value)
            logits += weights[-1] @ value
            caches.append((activations, preactivations))
        logits -= logits.max()
        probabilities = np.exp(logits); probabilities /= probabilities.sum()
        self.last_loss = float(-np.log(max(1e-7, probabilities[actual])))
        delta = probabilities; delta[actual] -= 1.0
        rate = float(learning_rate)
        for zone, weights in enumerate(self.branch_weights):
            if zone_mask is not None and not bool(zone_mask[zone]):
                continue
            activations, preactivations = caches[zone]
            branch_delta = delta.copy()
            for layer in range(len(weights) - 1, -1, -1):
                gradient_w = np.outer(branch_delta, activations[layer])
                if layer:
                    next_delta = (weights[layer].T @ branch_delta) * (preactivations[layer - 1] > 0)
                weights[layer] -= rate * np.clip(gradient_w, -5, 5)
                if layer < len(weights) - 1:
                    self.branch_biases[zone][layer] -= rate * np.clip(branch_delta, -5, 5)
                if layer: branch_delta = next_delta
        # A relevance replay updates only selected zone branches. The global
        # bias is deliberately frozen so no unscoped action preference leaks in.
        if zone_mask is None:
            self.output_bias -= rate * np.clip(delta, -5, 5)
        if count_example:
            self.examples += 1
            self.matches += int(actual == predicted)
        self.updates += 1
        self.save()
        return self.last_loss

    def reset(self) -> None:
        self.schema = {}; self.branch_weights = []; self.branch_biases = []
        self.output_bias = np.empty(0, dtype=np.float32)
        self.examples = self.matches = self.updates = 0
        if self.path.exists():
            self.path.unlink()
