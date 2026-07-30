from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import Region


class SceneTransitionDetector:
    """Detect a completed room change from several small scene regions.

    A transition is accepted only when enough regions differ from the room
    captured before movement *and* the new image has remained stable.  This
    rejects localized character motion and looping animation much more
    reliably than treating a single changing frame as a room transition.
    """

    def __init__(
        self,
        change_threshold: float = 0.02,
        stability_threshold: float = 0.01,
        stable_frames: int = 3,
        minimum_changed_regions: int = 2,
    ) -> None:
        self.change_threshold = max(0.0, float(change_threshold))
        self.stability_threshold = max(0.0, float(stability_threshold))
        self.stable_frames = max(1, int(stable_frames))
        self.minimum_changed_regions = max(1, int(minimum_changed_regions))
        self.reset()

    def reset(self, origin: dict[str, np.ndarray] | None = None) -> None:
        self.origin = {
            name: value.copy() for name, value in (origin or {}).items()
        }
        self.previous = {
            name: value.copy() for name, value in (origin or {}).items()
        }
        self.settled_frames = 0
        self.changed_regions = 0
        self.subtly_changed_regions = 0
        self.maximum_motion = 0.0
        self.maximum_change = 0.0
        self.required_settled_frames = self.stable_frames
        self.change_mode = "waiting"
        self.frames_observed = 0

    @staticmethod
    def _difference(first: np.ndarray, second: np.ndarray) -> float:
        if first.shape != second.shape:
            return 1.0
        return float(np.mean(np.abs(first.astype(np.float32) - second)))

    def update(self, current: dict[str, np.ndarray]) -> bool:
        common = self.origin.keys() & current.keys()
        required = min(self.minimum_changed_regions, len(self.origin))
        if not common or len(common) != len(self.origin):
            self.settled_frames = 0
            return False

        changes = [
            self._difference(self.origin[name], current[name]) for name in common
        ]
        motions = [
            self._difference(self.previous[name], current[name]) for name in common
        ]
        self.changed_regions = sum(
            difference >= self.change_threshold for difference in changes
        )
        subtle_threshold = self.change_threshold * 0.35
        self.subtly_changed_regions = sum(
            difference >= subtle_threshold for difference in changes
        )
        self.maximum_change = max(changes, default=0.0)
        self.maximum_motion = max(motions, default=0.0)

        # Normal confirmation needs several clearly changed regions. Similar
        # rooms also get a coordinated-subtle path, while a single large
        # difference must remain stable twice as long to avoid animations.
        coordinated_required = min(max(3, required), len(self.origin))
        if self.changed_regions >= required:
            scene_changed = True
            self.required_settled_frames = self.stable_frames
            self.change_mode = "multi-region"
        elif self.subtly_changed_regions >= coordinated_required:
            scene_changed = True
            self.required_settled_frames = self.stable_frames
            self.change_mode = "coordinated subtle"
        elif self.maximum_change >= self.change_threshold * 2.0:
            scene_changed = True
            self.required_settled_frames = self.stable_frames * 2
            self.change_mode = "single-region"
        else:
            scene_changed = False
            self.required_settled_frames = self.stable_frames
            self.change_mode = "waiting"
        if (
            self.frames_observed > 0
            and scene_changed
            and self.maximum_motion <= self.stability_threshold
        ):
            self.settled_frames += 1
        else:
            self.settled_frames = 0
        self.previous = {name: value.copy() for name, value in current.items()}
        self.frames_observed += 1
        return (
            scene_changed
            and self.settled_frames >= self.required_settled_frames
        )


class WholeSceneMotionDetector:
    """Recognize transition motion when the destination looks identical."""

    def __init__(
        self,
        activity_threshold: float = 0.012,
        stability_threshold: float = 0.006,
        stable_frames: int = 3,
    ) -> None:
        self.activity_threshold = max(0.0, float(activity_threshold))
        self.stability_threshold = max(0.0, float(stability_threshold))
        self.stable_frames = max(1, int(stable_frames))
        self.reset()

    def reset(self, origin: np.ndarray | None = None) -> None:
        self.previous = origin.copy() if origin is not None else None
        self.activity_seen = False
        self.stable_count = 0
        self.motion = 0.0

    def update(self, current: np.ndarray | None) -> bool:
        if current is None:
            self.stable_count = 0
            return False
        if self.previous is None or self.previous.shape != current.shape:
            self.previous = current.copy()
            return False
        delta = self.previous.astype(np.float32) - current
        self.motion = float(np.sqrt(np.mean(delta * delta)))
        if self.motion >= self.activity_threshold:
            self.activity_seen = True
            self.stable_count = 0
        elif self.activity_seen and self.motion <= self.stability_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0
        self.previous = current.copy()
        return self.activity_seen and self.stable_count >= self.stable_frames


class StateDetector:
    """Template-based baseline; replace individual regions with trained models later."""

    def __init__(self, template_paths: dict[str, str], threshold: float = 0.82) -> None:
        self.threshold = threshold
        self.templates = {
            name: cv2.imread(str(Path(path)), cv2.IMREAD_GRAYSCALE)
            for name, path in template_paths.items()
            if Path(path).exists()
        }

    def detect(
        self, frame_bgr: np.ndarray, regions: dict[str, Region]
    ) -> tuple[str, float]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        best_state, best_score = "unknown", 0.0
        for state, template in self.templates.items():
            if template is None:
                continue
            region = regions.get(f"{state}_anchor")
            view = gray if region is None else gray[
                region.top : region.top + region.height,
                region.left : region.left + region.width,
            ]
            if view.shape[0] < template.shape[0] or view.shape[1] < template.shape[1]:
                continue
            score = float(cv2.matchTemplate(view, template, cv2.TM_CCOEFF_NORMED).max())
            if score > best_score:
                best_state, best_score = state, score
        return (best_state, best_score) if best_score >= self.threshold else (
            "unknown", best_score
        )


class BossDetector:
    def __init__(self, template_path: str | None, threshold: float = 0.82) -> None:
        self.threshold = threshold
        self.template = (
            cv2.imread(str(Path(template_path)), cv2.IMREAD_GRAYSCALE)
            if template_path and Path(template_path).exists()
            else None
        )

    def detect(
        self, frame_bgr: np.ndarray, search_region: Region | None
    ) -> tuple[tuple[int, int] | None, float]:
        if self.template is None or search_region is None:
            return None, 0.0
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        r = search_region
        view = gray[r.top : r.top + r.height, r.left : r.left + r.width]
        if (
            view.shape[0] < self.template.shape[0]
            or view.shape[1] < self.template.shape[1]
        ):
            return None, 0.0
        result = cv2.matchTemplate(view, self.template, cv2.TM_CCOEFF_NORMED)
        _minimum, score, _min_location, location = cv2.minMaxLoc(result)
        if float(score) < self.threshold:
            return None, float(score)
        height, width = self.template.shape
        center = (
            r.left + location[0] + width // 2,
            r.top + location[1] + height // 2,
        )
        return center, float(score)


class PlayerDetector:
    """Match any saved player appearance inside directional edge regions."""

    def __init__(
        self, template_paths: list[str] | None, threshold: float = 0.78
    ) -> None:
        self.threshold = threshold
        self.templates = [
            template
            for path in (template_paths or [])
            if Path(path).exists()
            for template in [cv2.imread(str(Path(path)), cv2.IMREAD_GRAYSCALE)]
            if template is not None
        ]

    def detect_regions(
        self,
        frame_bgr: np.ndarray,
        regions: dict[str, Region],
    ) -> tuple[str | None, float]:
        if not self.templates:
            return None, 0.0
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        best_direction: str | None = None
        best_score = 0.0
        for direction in ("north", "south", "east", "west"):
            region = regions.get(f"scene_change_{direction}")
            if region is None:
                continue
            view = gray[
                region.top : region.top + region.height,
                region.left : region.left + region.width,
            ]
            for template in self.templates:
                if (
                    view.shape[0] < template.shape[0]
                    or view.shape[1] < template.shape[1]
                ):
                    continue
                score = float(
                    cv2.matchTemplate(
                        view, template, cv2.TM_CCOEFF_NORMED
                    ).max()
                )
                if score > best_score:
                    best_direction, best_score = direction, score
        if best_score < self.threshold:
            return None, best_score
        return best_direction, best_score
