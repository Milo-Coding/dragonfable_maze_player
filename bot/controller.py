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
from .maze import MazeMemory, OPPOSITE
from .models import Decision, Mode, Point, Region
from .policy import ContextPolicy, TileLayoutDictionary
from .vision import (
    BossDetector,
    PlayerDetector,
    PlayerMovementTracker,
    StateDetector,
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
        self.clears = max(0, int(config.data.get("clears", 0)))
        self._quest_in_progress = False
        self.detector = StateDetector(
            config.data["templates"], config.data["confidence_threshold"]
        )
        self.boss_detector = BossDetector(
            config.data.get("boss_template"),
            float(config.data.get("boss_match_threshold", 0.82)),
        )
        self.mog_detector = BossDetector(
            config.data.get("mog_template"),
            float(config.data.get("mog_match_threshold", 0.82)),
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
        self.mog_status = (
            "Mog detector ready; waiting for exploration"
            if self.mog_detector.template is not None
            else "Mog detector not configured"
        )
        self._mog_interacted_tiles: set[tuple[int, int]] = set()
        self._mog_menu_pending = False
        self.policy = ContextPolicy()
        self.tile_layouts = TileLayoutDictionary()
        self.layout_status = self.tile_layouts.last_match_status
        self.training_enabled = False
        self._latest_learning: tuple[str, np.ndarray, int, int] | None = None
        self._latest_click_state: str | None = None
        self._learning_lock = threading.Lock()
        self._binding_lock = threading.Lock()
        self._binding_capture_mode: Mode | None = None
        self._last_live_click = 0.0
        self._latest_edge_signatures: dict[str, np.ndarray] = {}
        self._pending_maze_move: str | None = None
        self._teleporter_sequence: tuple[str, int] | None = None
        self._pending_maze_move_started_at: float | None = None
        self._transition_frame = 0
        self._transition_activity_seen = False
        self._edges_stable_count = 0
        self._latest_player_position: tuple[int, int] | None = None
        self._player_movement = PlayerMovementTracker(
            config.data.get("transition_player_movement_pixels", 8.0),
            config.data.get("transition_player_stable_frames", 4),
        )
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
        if mode == Mode.IDLE:
            self._teleporter_sequence = None
            self._mog_menu_pending = False

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
        self._mog_interacted_tiles.clear()
        self._mog_menu_pending = False
        self._teleporter_sequence = None
        self._clear_pending_move()
        self._latest_edge_signatures = {}
        self.maze_transition_status = "No pending room transition"
        self.on_maze_update()

    def set_maze_position(self, x: int, y: int) -> None:
        self.maze.set_position(x, y)
        self._clear_pending_move()
        self.maze_transition_status = f"Location manually set to ({x}, {y})"
        self.on_maze_update()

    def _teleporter_ready(self, action: str) -> bool:
        return all(
            zone is not None
            for zone in self.config.teleporter_click_zones.get(action, [])
        )

    def _teleporter_decision(
        self, state: str, confidence: float, action: str, step: int
    ) -> Decision | None:
        zones = self.config.teleporter_click_zones.get(action, [])
        if step >= len(zones) or zones[step] is None:
            self._teleporter_sequence = None
            return None
        zone = zones[step]
        assert zone is not None
        point = Point(zone.left + zone.width // 2, zone.top + zone.height // 2)
        label = "Place" if action == "place" else "Return to"
        return Decision(
            state,
            point,
            f"{label} teleporter: step {step + 1}/{len(zones)}",
            confidence,
            {
                "teleporter_action": action,
                "teleporter_step": step,
            },
        )

    def _mog_close_decision(
        self, state: str, confidence: float
    ) -> Decision | None:
        zone = self.config.mog_close_click_zone
        if zone is None:
            return None
        point = Point(zone.left + zone.width // 2, zone.top + zone.height // 2)
        return Decision(
            state,
            point,
            "Close Mog menu",
            confidence,
            {"mog_close": True},
        )

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
            state = self._latest_click_state
        if state is None:
            return
        zones = self.config.active_click_zones.get(state, [])
        point = Point(int(x), int(y))
        actual = next(
            (index for index, zone in enumerate(zones) if zone.contains(point)), None
        )
        if actual is None:
            return
        if state == "exploring":
            direction = self.config.active_click_zone_name(state, actual).lower()
            if direction in {"north", "east", "south", "west"}:
                self.maze.observe({direction})
                self._start_pending_move(direction)
                self.on_maze_update()
        if not self.training_enabled or learning is None:
            return
        learned_state, context, action_count, predicted = learning
        if (
            learned_state != "combat"
            or state != learned_state
            or action_count != len(zones)
        ):
            return
        reward = self.policy.train(
            learned_state, context, action_count, predicted, actual
        )
        accuracy = self.policy.matches / max(1, self.policy.examples)
        self.on_training_event(
            f"Reward {reward}: predicted zone {predicted + 1}, "
            f"clicked zone {actual + 1} ({accuracy:.0%} match rate)"
        )

    def _on_key(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key == keyboard.Key.esc:
            self.cancel_binding_capture()
            self.mode = Mode.IDLE
            self._teleporter_sequence = None
            self._mog_menu_pending = False
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
                    click_interval = self._click_interval(decision)
                    if (
                        self.mode == Mode.LIVE
                        and decision.point
                        and not (
                            decision.state_name == "exploring"
                            and self.movement_pending
                        )
                        and now - self._last_live_click >= click_interval
                    ):
                        self._safe_click(decision)
                        self._last_live_click = time.monotonic()
                    next_decision = now + min(
                        float(self.config.data["tick_seconds"]),
                        click_interval,
                    )
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

    def _click_interval(self, decision: Decision) -> float:
        if decision.details.get("teleporter_action") in {"place", "return"}:
            return max(
                0.05,
                float(
                    self.config.data.get(
                        "teleporter_click_interval_seconds", 0.25
                    )
                ),
            )
        return 1.5

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

    def _gameplay_region(self, frame: np.ndarray) -> Region | None:
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
        return Region(left, top, right - left, bottom - top)

    def _clear_pending_move(self) -> None:
        self._pending_maze_move = None
        self._pending_maze_move_started_at = None
        self._transition_frame = 0
        self._transition_activity_seen = False
        self._edges_stable_count = 0
        self._player_movement.reset()

    def _start_pending_move(self, direction: str) -> bool:
        if len(self._latest_edge_signatures) != 4:
            missing = 4 - len(self._latest_edge_signatures)
            self.maze_transition_status = (
                f"Cannot track {direction}: {missing} directional edge region(s) missing"
            )
            return False
        if not self.player_detector.templates:
            self.maze_transition_status = (
                f"Cannot track {direction}: capture at least one player sprite"
            )
            return False
        self._pending_maze_move = direction
        self._pending_maze_move_started_at = time.monotonic()
        self._transition_frame = 0
        self._transition_activity_seen = False
        self._edges_stable_count = 0
        self._player_movement.reset(self._latest_player_position)
        self.maze_transition_status = (
            f"Pending {direction}; waiting for the player to appear stopped "
            "at the destination entryway"
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
        gameplay_region = self._gameplay_region(frame)
        player_position, player_score = (
            self.player_detector.detect(frame, gameplay_region)
            if gameplay_region is not None
            else (None, 0.0)
        )
        if player_position is not None:
            self._latest_player_position = player_position
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
        player_stopped = self._player_movement.update(player_position)
        expected_entry = OPPOSITE[selected_direction]
        entry_region = self.config.visual_regions.get("exploring", {}).get(
            f"scene_change_{expected_entry}"
        )
        at_entryway = (
            player_position is not None
            and entry_region is not None
            and entry_region.contains(Point(*player_position))
        )
        self.maze_transition_status = (
            f"Pending {selected_direction}; "
            f"player {'found' if player_position else 'not found'} "
            f"({player_score:.0%}); "
            f"stopped {self._player_movement.stable_frames}/"
            f"{self._player_movement.stable_frames_required}; "
            f"{'at' if at_entryway else 'not at'} {expected_entry} entryway"
        )
        if self._transition_frame % 3 == 0:
            self.on_maze_update()
        if player_stopped and at_entryway:
            self._commit_maze_move(
                selected_direction,
                f"player appeared stopped at the {expected_entry} entryway",
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

    def _track_clear(self, state: str) -> None:
        if state in {"exploring", "combat"}:
            self._quest_in_progress = True
        elif state == "lobby" and self._quest_in_progress:
            self.clears += 1
            self.config.data["clears"] = self.clears
            self.config.save()
            self._quest_in_progress = False

    def _decide(
        self, state: str, confidence: float, frame: np.ndarray
    ) -> Decision:
        with self._learning_lock:
            self._latest_learning = None
            self._latest_click_state = (
                state if state != "unknown" else None
            )
        if self._mog_menu_pending:
            decision = self._mog_close_decision(state, confidence)
            if decision is not None:
                return decision
        if self._teleporter_sequence is not None and state != "exploring":
            action, step = self._teleporter_sequence
            decision = self._teleporter_decision(
                state, confidence, action, step
            )
            if decision is not None:
                return decision
        self._track_clear(state)
        if state == "lobby" and self.config.lobby_start_point:
            self.maze.reset()
            self._mog_interacted_tiles.clear()
            self._mog_menu_pending = False
            self._clear_pending_move()
            self._latest_edge_signatures = {}
            self.maze_transition_status = "Maze reset in lobby"
            self.on_maze_update()
            return Decision(
                state, self.config.lobby_start_point, "Start quest", confidence
            )
        zones = self.config.active_click_zones.get(state, [])
        visual_regions = self.config.visual_regions.get(state, {})
        if state == "exploring" and zones:
            boss_score = 0.0
            boss_direction: str | None = None
            mog_score = 0.0
            mog_detected = False
            previous_boss_status = self.boss_status
            search_area = visual_regions.get("boss_search_area")
            boss_center, boss_score = self.boss_detector.detect(frame, search_area)
            if boss_center is not None and search_area is not None:
                self._teleporter_sequence = None
                if self._pending_maze_move is not None:
                    self._commit_maze_move(
                        self._pending_maze_move,
                        "boss detected in the destination room",
                    )
                boss_direction = "mid"
                self.boss_direction = boss_direction
                self.maze.observe(set(), tile_type="boss")
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
            if self.maze.boss_position is not None:
                self._teleporter_sequence = None
            previous_mog_status = self.mog_status
            if boss_direction:
                self.mog_status = "Mog check skipped because boss has priority"
            elif self.maze.position in self._mog_interacted_tiles:
                self.mog_status = "Mog already handled on this tile"
            else:
                mog_area = visual_regions.get("mog_search_area")
                mog_center, mog_score = self.mog_detector.detect(frame, mog_area)
                mog_match = mog_center is not None and mog_area is not None
                mog_detected = (
                    mog_match and self.config.mog_close_click_zone is not None
                )
                self.mog_status = (
                    f"Mog detected ({mog_score:.0%}); interact via mid"
                    if mog_detected
                    else (
                        "Mog detected, but its menu close zone is not configured"
                        if mog_match
                        else (
                            f"Mog not detected (best match {mog_score:.0%})"
                            if mog_area is not None
                            and self.mog_detector.template is not None
                            else "Mog detection needs mog sprite and mog_search_area"
                        )
                    )
                )
            if self.mog_status != previous_mog_status:
                self.on_maze_update()
            if boss_direction:
                recommendation = "mid"
            elif mog_detected:
                self._teleporter_sequence = None
                if self._pending_maze_move is not None:
                    self._commit_maze_move(
                        self._pending_maze_move,
                        "Mog detected in the destination room",
                    )
                recommendation = "mid"
            else:
                if self._teleporter_sequence is not None:
                    action, step = self._teleporter_sequence
                    decision = self._teleporter_decision(
                        state, confidence, action, step
                    )
                    if decision is not None:
                        return decision
                recommendation = self.maze.recommended_action(
                    self._teleporter_ready("place"),
                    self._teleporter_ready("return"),
                )
                if recommendation in {"place_teleporter", "return_teleporter"}:
                    action = (
                        "place"
                        if recommendation == "place_teleporter"
                        else "return"
                    )
                    self._teleporter_sequence = (action, 0)
                    decision = self._teleporter_decision(
                        state, confidence, action, 0
                    )
                    if decision is not None:
                        return decision
            selected_name = recommendation
            index = self._zone_index_named(state, selected_name)
            if index is None:
                return Decision(
                    state,
                    reason=f"No click zone named {selected_name!r}",
                    confidence=confidence,
                )
            zone = zones[index]
            point = Point(zone.left + zone.width // 2, zone.top + zone.height // 2)
            reason = f"Maze recommendation: {recommendation}"
            if boss_direction:
                reason = f"BOSS {boss_score:.0%}: move mid"
            elif mog_detected:
                reason = f"MOG {mog_score:.0%}: interact via mid"
            return Decision(
                state,
                point,
                reason,
                confidence,
                {
                    "zone_index": index,
                    "boss_score": boss_score,
                    "boss_direction": boss_direction,
                    "mog_score": mog_score,
                    "mog_detected": mog_detected,
                },
            )
        if state == "combat" and zones:
            context = self.policy.features(frame, visual_regions)
            index, probability = self.policy.predict(state, context, len(zones))
            zone = zones[index]
            point = Point(zone.left + zone.width // 2, zone.top + zone.height // 2)
            with self._learning_lock:
                self._latest_learning = (
                    state, context.copy(), len(zones), index
                )
            name = self.config.active_click_zone_name(state, index)
            return Decision(
                state,
                point,
                f"Combat policy: {name} ({probability:.0%})",
                confidence,
                {"zone_index": index, "policy_confidence": probability},
            )
        if state != "unknown" and len(zones) == 1:
            zone = zones[0]
            point = Point(zone.left + zone.width // 2, zone.top + zone.height // 2)
            return Decision(
                state,
                point,
                f"Only configured action: {self.config.active_click_zone_name(state, 0)}",
                confidence,
                {"zone_index": 0},
            )
        return Decision(state, reason="No action configured", confidence=confidence)

    def _zone_index_named(self, state: str, name: str | None) -> int | None:
        if name is None:
            return None
        zones = self.config.active_click_zones.get(state, [])
        for index in range(len(zones)):
            if self.config.active_click_zone_name(state, index).lower() == name:
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
        allowed = self.config.active_click_zones.get(decision.state_name, [])
        teleporter_action = decision.details.get("teleporter_action")
        if decision.details.get("mog_close"):
            close_zone = self.config.mog_close_click_zone
            allowed = [close_zone] if close_zone is not None else []
        if teleporter_action in {"place", "return"}:
            allowed = [
                zone
                for zone in self.config.teleporter_click_zones[teleporter_action]
                if zone is not None
            ]
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
        if decision.details.get("mog_close"):
            if not self._mog_menu_pending:
                return False
            self._mog_menu_pending = False
        if teleporter_action in {"place", "return"}:
            step = decision.details.get("teleporter_step")
            if not isinstance(step, int):
                return False
            sequence = self._teleporter_sequence
            if sequence != (teleporter_action, step):
                return False
            zones = self.config.teleporter_click_zones[teleporter_action]
            next_step = step + 1
            if next_step < len(zones):
                self._teleporter_sequence = (teleporter_action, next_step)
            else:
                self._teleporter_sequence = None
                if teleporter_action == "place":
                    self.maze.place_teleporter()
                    self.maze_transition_status = (
                        f"Teleporter placed at {self.maze.position}"
                    )
                else:
                    self.maze.return_to_teleporter()
                    self._clear_pending_move()
                    self.maze_transition_status = (
                        f"Returned to teleporter at {self.maze.position}"
                    )
                self.on_maze_update()
        if decision.state_name == "exploring":
            if decision.details.get("mog_detected"):
                self._mog_interacted_tiles.add(self.maze.position)
                self._mog_menu_pending = True
            index = decision.details.get("zone_index")
            if isinstance(index, int):
                direction = self.config.active_click_zone_name(
                    "exploring", index
                ).lower()
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
