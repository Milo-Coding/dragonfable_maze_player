from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .models import Region


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

