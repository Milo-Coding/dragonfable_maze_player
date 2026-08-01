from __future__ import annotations

import ctypes
import threading
import time
import random
from collections.abc import Callable
from pathlib import Path

import cv2
import mss
import numpy as np
from pynput import keyboard, mouse

from bot.models import Decision, Mode, Point, Region
from bot.maze import MazeMemory
from .config import CombatConfig
from .policy import DensePolicy, visual_features


class CombatController:
    def __init__(self, config: CombatConfig, on_update: Callable[[Decision], None],
                 on_mode: Callable[[Mode], None], on_training: Callable[[str], None],
                 on_maze: Callable[[], None] | None = None) -> None:
        self.config = config
        self.policy = DensePolicy()
        self.on_update, self.on_mode, self.on_training = on_update, on_mode, on_training
        self.on_maze = on_maze or (lambda: None)
        self.mode = Mode.IDLE
        self.last_example: dict | None = None
        self.active_task: dict | None = None
        self.maze = MazeMemory()
        self.maze_status = "Waiting for an exploring task"
        self._pending_maze_move: tuple[str, float] | None = None
        self._teleporter_sequence: tuple[str, int] | None = None
        self._latest: dict | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._keyboard = keyboard.Listener(on_press=self._on_key)
        self._mouse = mouse.Listener(on_click=self._on_mouse)

    def start(self) -> None:
        self._keyboard.start(); self._mouse.start()
        self._thread = threading.Thread(target=self._loop, daemon=True); self._thread.start()

    def shutdown(self) -> None:
        self.set_mode(Mode.IDLE); self._stop.set()
        self._keyboard.stop(); self._mouse.stop()

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode
        if mode == Mode.IDLE:
            with self._lock: self._latest = None
            self._teleporter_sequence = None
            self._pending_maze_move = None

    def _on_key(self, key) -> None:
        if key == keyboard.Key.esc:
            self.set_mode(Mode.IDLE); self.on_mode(Mode.IDLE)

    @staticmethod
    def mouse_binding(button) -> str:
        name = getattr(button, "name", None)
        return f"mouse:{name or str(button).removeprefix('Button.')}".lower()

    def _on_mouse(self, x: int, y: int, button, pressed: bool) -> None:
        if not pressed:
            return
        binding = self.mouse_binding(button)
        for mode in (Mode.TRAINING, Mode.LIVE):
            if self.config.data.get("mode_bindings", {}).get(mode.value) == binding:
                self.set_mode(mode)
                self.on_mode(mode)
                return
        if self.mode != Mode.TRAINING:
            return
        with self._lock:
            latest = self._latest
        if latest is None:
            return
        actions = self.config.enabled("action")
        actual = next((i for i, item in enumerate(actions)
                       if self.config.region(item).contains(Point(int(x), int(y)))), None)
        if actual is None or latest["actions"] != [item["name"] for item in actions]:
            return
        if self.active_task is None or self.active_task.get("id") != latest.get("task_id"):
            return
        loss = self.policy.train(latest["features"], actual, latest["predicted"],
                                 self.config.data["learning_rate"])
        self.last_example = {**latest, "actual": actual}
        reward = int(actual == latest["predicted"])
        self.on_training(f"Reward {reward} · demonstrated {actions[actual]['name']} · loss {loss:.3f}")

    def reinforce_last(self, impactful: set[str]) -> bool:
        example = self.last_example
        if not example or not impactful:
            return False
        mask = np.asarray([name in impactful for name in example["views"]], dtype=np.bool_)
        self.policy.train(example["features"], example["actual"], example["predicted"],
                          self.config.data["learning_rate"], zone_mask=mask, count_example=False)
        self.on_training("Relevance replay updated only: " + ", ".join(sorted(impactful)))
        return True

    @staticmethod
    def _template_score(frame_gray: np.ndarray, region: Region, path: str | None) -> float:
        if not path or not Path(path).exists(): return 0.0
        template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        crop = frame_gray[region.top:region.top + region.height, region.left:region.left + region.width]
        if template is None or crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]: return 0.0
        return float(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED).max())

    def _anchor_score(self, frame: np.ndarray, task: dict) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scores = [self._template_score(gray, Region(**anchor["region"]), anchor.get("template"))
                  for anchor in task.get("anchors", [])]
        return max(scores, default=0.0)

    def _active_task(self, frame: np.ndarray) -> tuple[dict | None, float]:
        """Return the first enabled matching task; list order is priority."""
        best_miss = 0.0
        threshold = float(self.config.data["anchor_threshold"])
        for task in self.config.tasks:
            if not task.get("enabled", True):
                continue
            score = self._anchor_score(frame, task)
            best_miss = max(best_miss, score)
            if score >= threshold:
                return task, score
        return None, best_miss

    def _scan_decision(self, frame: np.ndarray, task: dict, score: float) -> Decision:
        target_id = task.get("behavior", "").partition(":")[2]
        target = self.config.scan_target(target_id)
        if target is None:
            return Decision(task["name"], reason="Scan target no longer exists", confidence=score)
        threshold = float(self.config.data.get("scan_match_threshold", 0.82))
        detections: list[tuple[Point, float, int]] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for zone_index, raw_zone in enumerate(target.get("scan_zones", [])):
            zone = Region(**raw_zone)
            view = gray[zone.top:zone.top + zone.height, zone.left:zone.left + zone.width]
            for sprite in target.get("sprites", []):
                path = sprite.get("path") if isinstance(sprite, dict) else sprite
                template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path and Path(path).exists() else None
                if template is None or view.shape[0] < template.shape[0] or view.shape[1] < template.shape[1]:
                    continue
                scores = cv2.matchTemplate(view, template, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(scores >= threshold)
                candidates = sorted(zip(xs.tolist(), ys.tolist()), key=lambda xy: float(scores[xy[1], xy[0]]), reverse=True)
                for x, y in candidates:
                    point = Point(zone.left + x + template.shape[1] // 2,
                                  zone.top + y + template.shape[0] // 2)
                    # Collapse the many adjacent template-match peaks around one sprite.
                    if any((point.x - old.x) ** 2 + (point.y - old.y) ** 2 <
                           (max(template.shape) * 0.5) ** 2 for old, _, _ in detections):
                        continue
                    detections.append((point, float(scores[y, x]), zone_index))
        if not detections:
            return Decision(task["name"], reason=f"Scanning for {target['name']}: no match", confidence=score,
                            details={"task_id": task["id"], "behavior": "scanning"})
        point, match, zone_index = random.choice(detections)
        return Decision(task["name"], point,
                        f"Scanning for {target['name']}: {len(detections)} match(es)", match,
                        {"task_id": task["id"], "behavior": "scanning", "scan_target": target_id,
                         "scan_zone": zone_index})

    def _simple_decision(self, task: dict, score: float) -> Decision:
        value = task.get("simple_point")
        point = Point(**value) if value else None
        return Decision(task["name"], point,
                        "Simple trigger click" if point else "Simple click point not configured",
                        score, {"task_id": task["id"], "behavior": "simple"})

    def reset_maze(self) -> None:
        self.maze.reset(); self._pending_maze_move = None; self._teleporter_sequence = None
        self.maze_status = "Maze reset to (0, 0)"; self.on_maze()

    def set_maze_position(self, x: int, y: int) -> None:
        self.maze.set_position(x, y); self._pending_maze_move = None; self._teleporter_sequence = None
        self.maze_status = f"Maze position set to ({x}, {y})"; self.on_maze()

    def _detected_exits(self, frame: np.ndarray) -> tuple[set[str], dict[str, float]]:
        threshold = float(self.config.data.get("maze_exit_threshold", 0.88))
        exits, scores = set(), {}
        for direction in ("north", "east", "south", "west"):
            scan = self.config.data.get("maze_exit_scans", {}).get(direction)
            if not scan: continue
            region = Region(**scan["region"])
            path = scan.get("template")
            template = cv2.imread(str(path), cv2.IMREAD_COLOR) if path and Path(path).exists() else None
            crop = frame[region.top:region.top + region.height, region.left:region.left + region.width]
            if template is None or crop.shape != template.shape or crop.size == 0:
                score = 0.0
            else:
                difference = float(np.mean(np.abs(crop.astype(np.float32) - template.astype(np.float32))))
                score = max(0.0, 1.0 - difference / 255.0)
            scores[direction] = score
            if score >= threshold: exits.add(direction)
        return exits, scores

    def _exploring_decision(self, frame: np.ndarray, task: dict, score: float) -> Decision:
        teleporter = self.config.data.get("maze_teleporter_points", {})
        if self._teleporter_sequence is not None:
            action, step = self._teleporter_sequence
            points = teleporter.get(action, [])
            value = points[step] if step < len(points) else None
            point = Point(**value) if value else None
            return Decision(task["name"], point,
                            f"{action.title()} teleporter · step {step + 1}/{len(points)}",
                            score, {"task_id": task["id"], "behavior": "exploring",
                                    "teleporter_action": action, "teleporter_step": step})
        if self._pending_maze_move is not None:
            direction, started = self._pending_maze_move
            elapsed = time.monotonic() - started
            if elapsed < float(self.config.data.get("maze_transition_seconds", 1.5)):
                return Decision(task["name"], reason=f"Waiting for {direction} room transition",
                                confidence=score, details={"task_id": task["id"], "behavior": "exploring"})
            moved = self.maze.moved(direction)
            self._pending_maze_move = None
            self.maze_status = f"Moved {direction} to {self.maze.position}" if moved else f"Blocked at maze boundary: {direction}"
        exits, scores = self._detected_exits(frame)
        # The four live comparisons are authoritative for this visit. Replace
        # the tile's prior exit set instead of accumulating stale detections.
        self.maze.confirm_exits(exits)
        can_place = len(teleporter.get("place", [])) == 3 and all(teleporter.get("place", []))
        can_return = len(teleporter.get("return", [])) == 2 and all(teleporter.get("return", []))
        direction = self.maze.recommended_action(can_place, can_return)
        self.maze_status = (f"Current {self.maze.position} · active exits: {', '.join(sorted(exits)) or 'none'} · "
                            f"recommendation: {direction or 'maze exhausted'}")
        self.on_maze()
        if direction in {"place_teleporter", "return_teleporter"}:
            action = "place" if direction == "place_teleporter" else "return"
            self._teleporter_sequence = (action, 0)
            value = teleporter[action][0]
            return Decision(task["name"], Point(**value), f"{action.title()} teleporter · step 1/{len(teleporter[action])}",
                            score, {"task_id": task["id"], "behavior": "exploring",
                                    "teleporter_action": action, "teleporter_step": 0})
        raw_point = self.config.data.get("maze_move_points", {}).get(direction) if direction else None
        point = Point(**raw_point) if raw_point else None
        detail = ", ".join(f"{name[0].upper()} {value:.0%}" for name, value in scores.items())
        reason = f"Explore {direction} ({detail})" if direction and point else (
            f"Set the {direction} movement point" if direction else "No unexplored exit")
        return Decision(task["name"], point, reason, score,
                        {"task_id": task["id"], "behavior": "exploring", "direction": direction})

    def _loop(self) -> None:
        with mss.mss() as capture:
            number = int(self.config.data["monitor"])
            monitor = capture.monitors[number if number < len(capture.monitors) else 1]
            last_click = 0.0
            while not self._stop.is_set():
                if self.mode == Mode.IDLE:
                    time.sleep(0.1); continue
                raw = np.asarray(capture.grab(monitor))
                frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                task, score = self._active_task(frame)
                self.active_task = task
                if task is None:
                    with self._lock: self._latest = None
                    self.on_update(Decision("searching", reason="No enabled task anchor found", confidence=score))
                    time.sleep(float(self.config.data["tick_seconds"])); continue
                behavior = task.get("behavior", "simple")
                if behavior.startswith("scanning:"):
                    with self._lock: self._latest = None
                    decision = self._scan_decision(frame, task, score)
                    self.on_update(decision)
                    now = time.monotonic()
                    if self.mode == Mode.LIVE and decision.point and now - last_click >= float(self.config.data["click_interval_seconds"]):
                        self._safe_click(decision, []); last_click = time.monotonic()
                    time.sleep(float(self.config.data["tick_seconds"])); continue
                if behavior == "simple":
                    with self._lock: self._latest = None
                    decision = self._simple_decision(task, score)
                    self.on_update(decision)
                    now = time.monotonic()
                    if self.mode == Mode.LIVE and decision.point and now - last_click >= float(self.config.data["click_interval_seconds"]):
                        self._safe_click(decision, []); last_click = time.monotonic()
                    time.sleep(float(self.config.data["tick_seconds"])); continue
                if behavior == "exploring":
                    with self._lock: self._latest = None
                    decision = self._exploring_decision(frame, task, score)
                    self.on_update(decision)
                    now = time.monotonic()
                    if self.mode == Mode.LIVE and decision.point and self._pending_maze_move is None and now - last_click >= float(self.config.data["click_interval_seconds"]):
                        if self._safe_click(decision, []):
                            if decision.details.get("teleporter_action"):
                                self._advance_teleporter(decision)
                            else:
                                self._pending_maze_move = (str(decision.details["direction"]), time.monotonic())
                            last_click = time.monotonic()
                    time.sleep(float(self.config.data["tick_seconds"])); continue
                views = self.config.enabled("view"); actions = self.config.enabled("action")
                if not views or not actions:
                    self.on_update(Decision(task["name"], reason="Configure enabled view and action zones", confidence=score))
                    time.sleep(float(self.config.data["tick_seconds"])); continue
                view_names = [item["name"] for item in views]; action_names = [item["name"] for item in actions]
                self.policy.ensure(view_names, action_names, list(self.config.data["hidden_layers"]))
                features = visual_features(frame, [self.config.region(item) for item in views])
                selected, confidence = self.policy.select(features)
                importance = self.policy.zone_importance(features, selected)
                important_indices = np.argsort(-importance)[:min(3, len(views))]
                important_points = []
                for index in important_indices:
                    region = self.config.region(views[int(index)])
                    important_points.append(Point(region.left + region.width // 2,
                                                  region.top + region.height // 2))
                zone = self.config.region(actions[selected])
                point = Point(zone.left + zone.width // 2, zone.top + zone.height // 2)
                with self._lock:
                    self._latest = {"features": features.copy(), "predicted": selected,
                                    "task_id": task["id"],
                                    "views": view_names, "actions": action_names}
                reason = f"Policy: {action_names[selected]}"
                decision = Decision(task["name"], point, reason, confidence,
                                    {"action": selected, "task_id": task["id"], "behavior": "combat",
                                     "important_views": important_points})
                self.on_update(decision)
                now = time.monotonic()
                if self.mode == Mode.LIVE and now - last_click >= float(self.config.data["click_interval_seconds"]):
                    self._safe_click(decision, actions); last_click = time.monotonic()
                time.sleep(float(self.config.data["tick_seconds"]))

    def _safe_click(self, decision: Decision, actions: list[dict]) -> bool:
        if self.mode != Mode.LIVE or decision.point is None:
            return False
        task_id = decision.details.get("task_id")
        active = self.active_task
        if active is None or active.get("id") != task_id:
            return False
        behavior = decision.details.get("behavior")
        if behavior == "combat":
            index = decision.details.get("action")
            if not isinstance(index, int) or index >= len(actions):
                return False
            if not self.config.region(actions[index]).contains(decision.point):
                return False
        elif behavior == "scanning":
            target = self.config.scan_target(str(decision.details.get("scan_target")))
            zones = target.get("scan_zones", []) if target else []
            if not any(Region(**zone).contains(decision.point) for zone in zones):
                return False
        elif behavior == "simple":
            value = active.get("simple_point")
            if not value or Point(**value) != decision.point:
                return False
        elif behavior == "exploring":
            action = decision.details.get("teleporter_action")
            if action in {"place", "return"}:
                step = decision.details.get("teleporter_step")
                points = self.config.data.get("maze_teleporter_points", {}).get(action, [])
                if not isinstance(step, int) or step >= len(points) or not points[step] or Point(**points[step]) != decision.point:
                    return False
            else:
                direction = decision.details.get("direction")
                value = self.config.data.get("maze_move_points", {}).get(direction)
                if not value or Point(**value) != decision.point:
                    return False
        else:
            return False
        ctypes.windll.user32.SetCursorPos(decision.point.x, decision.point.y)
        ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
        return True

    def _advance_teleporter(self, decision: Decision) -> None:
        action = str(decision.details["teleporter_action"])
        step = int(decision.details["teleporter_step"])
        points = self.config.data["maze_teleporter_points"][action]
        if step + 1 < len(points):
            self._teleporter_sequence = (action, step + 1)
            return
        self._teleporter_sequence = None
        if action == "place":
            self.maze.place_teleporter(); self.maze_status = f"Teleporter placed at {self.maze.position}"
        else:
            self.maze.return_to_teleporter(); self._pending_maze_move = None
            self.maze_status = f"Returned to teleporter at {self.maze.position}"
        self.on_maze()
