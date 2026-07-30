from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Callable

import cv2
import mss
import numpy as np
from pynput import keyboard, mouse

from .config import Config
from .maze import MazeMemory
from .models import Decision, Mode, Point
from .policy import ContextPolicy, TileLayoutDictionary
from .vision import (
    BossDetector,
    PlayerDetector,
    SceneTransitionDetector,
    StateDetector,
    WholeSceneMotionDetector,
)


class BotController:
    def __init__(
        self,
        config: Config,
        on_update: Callable[[Decision], None],
        on_emergency_stop: Callable[[], None],
        on_training_event: Callable[[str], None] | None = None,
        on_maze_update: Callable[[], None] | None = None,
        on_mode_hotkey: Callable[[Mode], None] | None = None,
        on_binding_captured: Callable[[Mode, str], None] | None = None,
    ) -> None:
        self.config = config
        self.on_update = on_update
        self.on_emergency_stop = on_emergency_stop
        self.on_training_event = on_training_event or (lambda _message: None)
        self.on_maze_update = on_maze_update or (lambda: None)
        self.on_mode_hotkey = on_mode_hotkey or self.set_mode
        self.on_binding_captured = on_binding_captured or (
            lambda _mode, _binding: None
        )
        self.mode = Mode.IDLE
        self.maze = MazeMemory()
        self.detector = StateDetector(
            config.data["templates"], config.data["confidence_threshold"]
        )
        self.boss_detector = BossDetector(
            config.data.get("boss_template"),
            float(config.data.get("boss_match_threshold", 0.82)),
        )
        self.player_detector = PlayerDetector(
            config.data.get("player_templates", []),
            float(config.data.get("player_match_threshold", 0.78)),
        )
        self.boss_status = (
            "Boss detector ready; waiting for exploration"
            if self.boss_detector.template is not None
            else "Boss detector not configured"
        )
        self.boss_direction: str | None = None
        self.policy = ContextPolicy()
        self.tile_layouts = TileLayoutDictionary()
        self.layout_status = self.tile_layouts.last_match_status
        self.training_enabled = False
        self._latest_learning: tuple[str, np.ndarray, int, int] | None = None
        self._learning_lock = threading.Lock()
        self._binding_lock = threading.Lock()
        self._binding_capture_mode: Mode | None = None
        self._last_live_click = 0.0
        self._latest_edge_signatures: dict[str, np.ndarray] = {}
        self._pending_maze_move: str | None = None
        self._pending_maze_move_started_at: float | None = None
        self._transition_frame = 0
        self._transition_activity_seen = False
        self._edges_stable_count = 0
        self._scene_transition = SceneTransitionDetector(
            config.data.get("transition_edge_activity_difference", 0.02),
            config.data.get("transition_edge_stability_difference", 0.01),
            config.data.get("transition_edge_stable_frames", 3),
            config.data.get("transition_minimum_changed_regions", 2),
        )
        self._whole_scene_transition = WholeSceneMotionDetector(
            config.data.get("transition_whole_scene_activity_difference", 0.012),
            config.data.get("transition_whole_scene_stability_difference", 0.006),
            config.data.get("transition_edge_stable_frames", 3),
        )
        self._latest_whole_scene_signature: np.ndarray | None = None
        self.maze_transition_status = "No pending room transition"
        self._latest_tile_layout_signature: str | None = None
        self._latest_tile_layout_descriptor: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._listener = keyboard.Listener(on_press=self._on_key)
        self._mouse_listener = mouse.Listener(on_click=self._on_mouse_click)

    def start(self) -> None:
        self._listener.start()
        self._mouse_listener.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self.mode = Mode.IDLE
        self._stop.set()
        self._listener.stop()
        self._mouse_listener.stop()

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode

    def set_training_enabled(self, enabled: bool) -> None:
        self.training_enabled = enabled

    def begin_binding_capture(self, mode: Mode) -> None:
        with self._binding_lock:
            self._binding_capture_mode = mode

    def cancel_binding_capture(self) -> None:
        with self._binding_lock:
            self._binding_capture_mode = None

    @staticmethod
    def _mouse_binding(button) -> str:
        name = getattr(button, "name", None)
        return f"mouse:{name or str(button).removeprefix('Button.')}".lower()

    @staticmethod
    def _key_binding(key: keyboard.Key | keyboard.KeyCode | None) -> str | None:
        if key is None:
            return None
        if isinstance(key, keyboard.KeyCode):
            if key.char:
                return f"key:{key.char.lower()}"
            return f"key:vk_{key.vk}" if key.vk is not None else None
        name = getattr(key, "name", None)
        return f"key:{name}".lower() if name else None

    def _capture_or_activate_binding(self, binding: str) -> bool:
        with self._binding_lock:
            capture_mode = self._binding_capture_mode
            if capture_mode is not None:
                self._binding_capture_mode = None
        if capture_mode is not None:
            self.config.set_mode_binding(capture_mode.value, binding)
            self.on_binding_captured(capture_mode, binding)
            return True
        bindings = self.config.data.get("mode_bindings", {})
        for mode in (Mode.TRAINING, Mode.LIVE):
            if bindings.get(mode.value) == binding:
                self.on_mode_hotkey(mode)
                return True
        return False

    def submit_tile_layout(self, exits: set[str]) -> None:
        signature = self._latest_tile_layout_signature
        if signature is None:
            self.on_training_event(
                "Tile layout not submitted: all four passage regions are required"
            )
            return
        layout_id, created = self.tile_layouts.submit(
            signature, exits, self._latest_tile_layout_descriptor
        )
        stored_exits = self.tile_layouts.exits_for(layout_id)
        linked = self.maze.update_linked_layout(layout_id, stored_exits)
        self.maze.link_layout(layout_id, stored_exits, provisional=False)
        action = "Added" if created else "Updated"
        self.on_training_event(
            f"{action} {layout_id}; linked current tile and "
            f"updated {linked} previously linked tile(s)"
        )
        self.on_maze_update()

    def confirm_maze_exits(self, exits: set[str]) -> None:
        if self._latest_tile_layout_signature is not None:
            self.submit_tile_layout(exits)
        else:
            self.maze.confirm_exits(exits)
            self.on_training_event(
                "Saved exits to current tile without a layout link "
                "(no current passage signature)"
            )
            self.on_maze_update()

    def reset_maze(self) -> None:
        self.maze.reset()
        self._clear_pending_move()
        self._latest_edge_signatures = {}
        self.maze_transition_status = "No pending room transition"
        self.on_maze_update()

    def set_maze_position(self, x: int, y: int) -> None:
        self.maze.set_position(x, y)
        self._clear_pending_move()
        self.maze_transition_status = f"Location manually set to ({x}, {y})"
        self.on_maze_update()

    @property
    def movement_pending(self) -> bool:
        return self._pending_maze_move is not None

    def _on_mouse_click(self, x: int, y: int, button, pressed: bool) -> None:
        if not pressed:
            return
        if self._capture_or_activate_binding(self._mouse_binding(button)):
            return
        if self.mode != Mode.TRAINING:
            return
        with self._learning_lock:
            learning = self._latest_learning
        if learning is None:
            return
        state, context, action_count, predicted = learning
        zones = self.config.click_zones.get(state, [])
        point = Point(int(x), int(y))
        actual = next(
            (index for index, zone in enumerate(zones) if zone.contains(point)), None
        )
        if actual is None or action_count != len(zones):
            return
        if state == "exploring":
            direction = self.config.click_zone_name(state, actual).lower()
            if direction in {"north", "east", "south", "west"}:
                self.maze.observe({direction})
                self._start_pending_move(direction)
                self.on_maze_update()
        if not self.training_enabled:
            return
        reward = self.policy.train(state, context, action_count, predicted, actual)
        accuracy = self.policy.matches / max(1, self.policy.examples)
        self.on_training_event(
            f"Reward {reward}: predicted zone {predicted + 1}, "
            f"clicked zone {actual + 1} ({accuracy:.0%} match rate)"
        )

    def _on_key(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key == keyboard.Key.esc:
            self.cancel_binding_capture()
            self.mode = Mode.IDLE
            self.on_emergency_stop()
            return
        binding = self._key_binding(key)
        if binding is not None:
            self._capture_or_activate_binding(binding)

    def _loop(self) -> None:
        with mss.mss() as capture:
            monitor_number = int(self.config.data["monitor"])
            if monitor_number >= len(capture.monitors):
                monitor_number = 1
            monitor = capture.monitors[monitor_number]
            next_decision = 0.0
            while not self._stop.is_set():
                if self.mode == Mode.IDLE:
                    next_decision = 0.0
                    time.sleep(0.1)
                    continue
                raw = np.asarray(capture.grab(monitor))
                frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                state, confidence = self.detector.detect(frame, self.config.regions)
                now = time.monotonic()
                decision_due = now >= next_decision
                self._track_maze_transition(
                    state, frame, observe_layout=decision_due
                )
                if decision_due:
                    decision = self._decide(state, confidence, frame)
                    self.on_update(decision)
                    if (
                        self.mode == Mode.LIVE
                        and decision.point
                        and not (
                            decision.state_name == "exploring"
                            and self.movement_pending
                        )
                        and now - self._last_live_click >= 1.5
                    ):
                        self._safe_click(decision)
                        self._last_live_click = time.monotonic()
                    next_decision = now + float(self.config.data["tick_seconds"])
                time.sleep(
                    max(
                        0.02,
                        float(
                            self.config.data.get(
                                "transition_sample_seconds", 0.1
                            )
                        ),
                    )
                )

    @staticmethod
    def _edge_signature(frame: np.ndarray, region) -> np.ndarray | None:
        height, width = frame.shape[:2]
        left, top = max(0, region.left), max(0, region.top)
        right = min(width, region.left + region.width)
        bottom = min(height, region.top + region.height)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA).astype(
            np.float32
        ) / 255.0

    def _edge_signatures(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        regions = self.config.visual_regions.get("exploring", {})
        signatures: dict[str, np.ndarray] = {}
        for direction in ("north", "south", "east", "west"):
            region = regions.get(f"scene_change_{direction}")
            if region is None:
                continue
            signature = self._edge_signature(frame, region)
            if signature is not None:
                signatures[direction] = signature
        return signatures

    def _whole_scene_signature(self, frame: np.ndarray) -> np.ndarray | None:
        regions = self.config.visual_regions.get("exploring", {})
        edges = [
            regions.get(f"scene_change_{direction}")
            for direction in ("north", "south", "east", "west")
        ]
        edges = [region for region in edges if region is not None]
        if not edges:
            return None
        height, width = frame.shape[:2]
        left = max(0, min(region.left for region in edges))
        top = max(0, min(region.top for region in edges))
        right = min(width, max(region.left + region.width for region in edges))
        bottom = min(height, max(region.top + region.height for region in edges))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.resize(gray, (96, 64), interpolation=cv2.INTER_AREA).astype(
            np.float32
        ) / 255.0

    def _clear_pending_move(self) -> None:
        self._pending_maze_move = None
        self._pending_maze_move_started_at = None
        self._transition_frame = 0
        self._transition_activity_seen = False
        self._edges_stable_count = 0
        self._scene_transition.reset()
        self._whole_scene_transition.reset()

    def _start_pending_move(self, direction: str) -> bool:
        if len(self._latest_edge_signatures) != 4:
            missing = 4 - len(self._latest_edge_signatures)
            self.maze_transition_status = (
                f"Cannot track {direction}: {missing} directional edge region(s) missing"
            )
            return False
        self._pending_maze_move = direction
        self._pending_maze_move_started_at = time.monotonic()
        self._transition_frame = 0
        self._transition_activity_seen = False
        self._edges_stable_count = 0
        self._scene_transition.reset(self._latest_edge_signatures)
        self._whole_scene_transition.reset(self._latest_whole_scene_signature)
        self.maze_transition_status = (
            f"Pending {direction}; waiting for a changed scene to settle"
        )
        return True

    def _track_maze_transition(
        self, state: str, frame: np.ndarray, observe_layout: bool = True
    ) -> None:
        if self._pending_maze_move is not None:
            timeout = max(
                1.0,
                float(
                    self.config.data.get(
                        "transition_pending_timeout_seconds", 10.0
                    )
                ),
            )
            started_at = self._pending_maze_move_started_at
            if started_at is not None and time.monotonic() - started_at >= timeout:
                direction = self._pending_maze_move
                self._clear_pending_move()
                self.maze_transition_status = (
                    f"Pending {direction} expired after {timeout:g}s; "
                    "exploration clicks re-enabled"
                )
                self.on_maze_update()
        if state == "combat":
            # An encounter interrupts movement; combat resolves on the same tile.
            if self._pending_maze_move is not None:
                self._clear_pending_move()
                self.maze_transition_status = "Pending move canceled by combat"
                self.on_maze_update()
            return
        if state != "exploring":
            return
        signatures = self._edge_signatures(frame)
        whole_scene = self._whole_scene_signature(frame)
        self._latest_whole_scene_signature = whole_scene
        if len(signatures) != 4:
            if self._pending_maze_move is not None:
                missing = 4 - len(signatures)
                self.maze_transition_status = (
                    f"Pending move cannot be checked: {missing} directional "
                    "edge region(s) missing"
                )
            return
        self._latest_edge_signatures = signatures
        if self._pending_maze_move is None:
            if observe_layout:
                self._observe_visual_exits(frame)
            return
        selected_direction = self._pending_maze_move
        self._transition_frame += 1
        transition_complete = self._scene_transition.update(signatures)
        identical_transition_complete = self._whole_scene_transition.update(
            whole_scene
        )
        maximum_motion = self._scene_transition.maximum_motion
        self._transition_activity_seen = self._scene_transition.changed_regions > 0
        self._edges_stable_count = self._scene_transition.settled_frames
        self.maze_transition_status = (
            f"Pending {selected_direction}; "
            f"changed regions {self._scene_transition.changed_regions}/"
            f"{self._scene_transition.minimum_changed_regions}, subtle "
            f"{self._scene_transition.subtly_changed_regions}; "
            f"{self._scene_transition.change_mode}; "
            f"change {self._scene_transition.maximum_change:.1%}, "
            f"motion {maximum_motion:.1%}; settled "
            f"{self._edges_stable_count}/"
            f"{self._scene_transition.required_settled_frames}; whole "
            f"{self._whole_scene_transition.motion:.1%}, settled "
            f"{self._whole_scene_transition.stable_count}/"
            f"{self._whole_scene_transition.stable_frames}"
        )
        if self._transition_frame % 3 == 0:
            self.on_maze_update()
        if transition_complete or identical_transition_complete:
            evidence = (
                f"{self._scene_transition.changed_regions} scene regions "
                f"changed and settled"
                if transition_complete
                else "whole-scene transition activity occurred and settled"
            )
            self._commit_maze_move(
                selected_direction,
                evidence,
            )
            self._observe_visual_exits(frame)

    def _commit_maze_move(self, direction: str, evidence: str) -> None:
        moved = self.maze.moved(direction)
        self._clear_pending_move()
        self.maze_transition_status = (
            f"Room transition confirmed: moved {direction}; new tile added "
            f"at {self.maze.position} ({evidence})"
            if moved
            else f"Scene changed, but {direction} would leave the 10x10 maze"
        )
        self.on_maze_update()

    def _observe_visual_exits(self, frame: np.ndarray) -> None:
        regions = self.config.visual_regions.get("exploring", {})
        signature = self.tile_layouts.signature(frame, regions)
        descriptor = self.tile_layouts.descriptor(frame, regions)
        self._latest_tile_layout_signature = signature
        self._latest_tile_layout_descriptor = descriptor
        if signature is None or descriptor is None:
            self.layout_status = "Layout identity needs all four passage regions"
            return
        predicted_exits, _confidence, layout_id = self.tile_layouts.match(signature)
        provisional = False
        if predicted_exits is None:
            predicted_exits, _difference, layout_id = self.tile_layouts.nearest(
                descriptor
            )
            provisional = predicted_exits is not None
        self.layout_status = self.tile_layouts.last_match_status
        if predicted_exits is None:
            return
        x, y = self.maze.position
        in_bounds = {
            "north": y > 0,
            "east": x < 9,
            "south": y < 9,
            "west": x > 0,
        }
        predicted_exits = {
            direction for direction in predicted_exits if in_bounds.get(direction, False)
        }
        if layout_id is not None:
            before = set(self.maze.tiles.get(self.maze.position, ()).exits) if (
                self.maze.position in self.maze.tiles
            ) else set()
            previous_tile = self.maze.tiles.get(self.maze.position)
            previous_link = (
                previous_tile.layout_id,
                previous_tile.layout_provisional,
            ) if previous_tile else (None, False)
            self.maze.link_layout(
                layout_id, predicted_exits, provisional=provisional
            )
            if predicted_exits - before or previous_link != (layout_id, provisional):
                self.on_maze_update()

    def _decide(
        self, state: str, confidence: float, frame: np.ndarray
    ) -> Decision:
        if state == "lobby" and self.config.lobby_start_point:
            self.maze.reset()
            self._clear_pending_move()
            self._latest_edge_signatures = {}
            self.maze_transition_status = "Maze reset in lobby"
            self.on_maze_update()
            return Decision(
                state, self.config.lobby_start_point, "Start quest", confidence
            )
        zones = self.config.click_zones.get(state, [])
        visual_regions = self.config.visual_regions.get(state, {})
        if state != "unknown" and zones:
            context = self.policy.features(frame, visual_regions)
            index, probability = self.policy.predict(state, context, len(zones))
            boss_score = 0.0
            boss_direction: str | None = None
            if state == "exploring":
                previous_boss_status = self.boss_status
                search_area = visual_regions.get("boss_search_area")
                boss_center, boss_score = self.boss_detector.detect(frame, search_area)
                if boss_center is not None and search_area is not None:
                    boss_direction = "mid"
                    self.boss_direction = boss_direction
                    boss_index = self._zone_index_named(state, boss_direction)
                    if boss_index is not None:
                        index = boss_index
                    self.maze.observe(
                        {boss_direction} if boss_direction in {
                            "north", "east", "south", "west"
                        } else set(),
                        tile_type="boss",
                    )
                    self.boss_status = (
                        f"Boss detected ({boss_score:.0%}); prioritize {boss_direction}"
                    )
                else:
                    self.boss_direction = None
                    self.boss_status = (
                        f"Boss not detected (best match {boss_score:.0%})"
                        if search_area is not None and self.boss_detector.template is not None
                        else "Boss detection needs boss sprite and boss_search_area"
                    )
                if self.boss_status != previous_boss_status:
                    self.on_maze_update()
            zone = zones[index]
            point = Point(zone.left + zone.width // 2, zone.top + zone.height // 2)
            with self._learning_lock:
                self._latest_learning = (
                    state,
                    context.copy(),
                    len(zones),
                    index,
                )
            name = self.config.click_zone_name(state, index)
            recommendation = self.maze.recommendation() if state == "exploring" else None
            if state == "exploring":
                selected_name = "mid" if boss_direction else recommendation
                selected_index = self._zone_index_named(state, selected_name)
                if selected_index is not None:
                    index = selected_index
                    zone = zones[index]
                    point = Point(
                        zone.left + zone.width // 2,
                        zone.top + zone.height // 2,
                    )
                    name = self.config.click_zone_name(state, index)
                    with self._learning_lock:
                        self._latest_learning = (
                            state,
                            context.copy(),
                            len(zones),
                            index,
                        )
            reason = f"Policy {name} ({probability:.0%})"
            if boss_direction:
                reason = f"BOSS {boss_score:.0%}: move mid"
            elif recommendation:
                reason = f"Maze recommendation: {recommendation}"
            return Decision(
                state,
                point,
                reason,
                confidence,
                {
                    "zone_index": index,
                    "policy_confidence": probability,
                    "boss_score": boss_score,
                    "boss_direction": boss_direction,
                },
            )
        with self._learning_lock:
            self._latest_learning = None
        return Decision(state, reason="No action configured", confidence=confidence)

    def _zone_index_named(self, state: str, name: str | None) -> int | None:
        if name is None:
            return None
        zones = self.config.click_zones.get(state, [])
        for index in range(len(zones)):
            if self.config.click_zone_name(state, index).lower() == name:
                return index
        return None

    def _safe_click(self, decision: Decision) -> bool:
        point = decision.point
        if (
            point is None
            or self.mode != Mode.LIVE
            or (
                decision.state_name == "exploring"
                and self.movement_pending
            )
        ):
            return False
        allowed = self.config.click_zones.get(decision.state_name, [])
        if not any(zone.contains(point) for zone in allowed):
            self.on_update(
                Decision(
                    decision.state,
                    reason="Blocked: target is outside configured click zones",
                    confidence=decision.confidence,
                )
            )
            return False
        # Check the mode again immediately before the OS-level click.
        if self.mode != Mode.LIVE:
            return False
        if decision.state_name == "exploring":
            index = decision.details.get("zone_index")
            if isinstance(index, int):
                direction = self.config.click_zone_name("exploring", index).lower()
                if direction in {"north", "east", "south", "west"}:
                    if not self._start_pending_move(direction):
                        self.on_maze_update()
                        return False
                    self.maze.observe({direction})
                    self.on_maze_update()
        ctypes.windll.user32.SetCursorPos(point.x, point.y)
        ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)
        ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)
        return True
