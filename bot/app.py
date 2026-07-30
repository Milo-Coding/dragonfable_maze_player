from __future__ import annotations

import base64
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2

from .calibration import ScreenSelector
from .config import Config
from .controller import BotController
from .models import Decision, Mode, Point, Region
from .vision import BossDetector, PlayerDetector, StateDetector


class App:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("DragonFable Bot")
        self.root.geometry("780x680")
        self.root.minsize(700, 600)
        self.root.attributes("-topmost", True)
        self.config = Config()
        self.status = tk.StringVar(value="Idle - clicks disabled")
        self.active_mode = tk.StringVar(value="Active mode: IDLE")
        self.state = tk.StringVar(value="State: unknown")
        self.target = tk.StringVar(value="Target: none")
        self.learning_enabled = tk.BooleanVar(value=False)
        self.training_status = tk.StringVar(value="Learning disabled")
        self.policy_stats = tk.StringVar()
        self.binding_status = tk.StringVar(
            value="Choose Reassign, then press a keyboard or mouse button."
        )
        self.training_binding = tk.StringVar()
        self.live_binding = tk.StringVar()
        self.maze_status = tk.StringVar(value="Current: (0, 0)  Searching for boss")
        self.teleporter_status = tk.StringVar()
        self.maze_exit_vars = {
            direction: tk.BooleanVar(value=False)
            for direction in ("north", "east", "south", "west")
        }
        self.manual_maze_x = tk.IntVar(value=0)
        self.manual_maze_y = tk.IntVar(value=0)
        self.controller = BotController(
            self.config,
            self._queue_update,
            self._queue_emergency_stop,
            self._queue_training_event,
            self._queue_maze_update,
            self._queue_mode_hotkey,
            self._queue_binding_captured,
        )
        self._build_marker()
        self._build()
        self.controller.start()
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_marker(self) -> None:
        self.marker = tk.Toplevel(self.root)
        self.marker.withdraw()
        self.marker.overrideredirect(True)
        self.marker.attributes("-topmost", True)
        canvas = tk.Canvas(
            self.marker,
            width=24,
            height=24,
            bg="#00ff55",
            highlightthickness=2,
            highlightbackground="#004d1a",
        )
        canvas.pack()
        canvas.create_oval(3, 3, 21, 21, fill="#00ff55", outline="white", width=2)
        canvas.create_line(12, 5, 12, 19, fill="#004d1a", width=2)
        canvas.create_line(5, 12, 19, 12, fill="#004d1a", width=2)
        self.root.after(100, self._make_marker_click_through)

    def _build(self) -> None:
        notebook = ttk.Notebook(self.root)
        self.notebook = notebook
        notebook.pack(fill="both", expand=True, padx=8, pady=8)
        control = ttk.Frame(notebook, padding=16)
        training = ttk.Frame(notebook, padding=16)
        calibration = ttk.Frame(notebook, padding=12)
        maze = ttk.Frame(notebook, padding=12)
        teleporter = ttk.Frame(notebook, padding=16)
        sprites = ttk.Frame(notebook, padding=12)
        controls = ttk.Frame(notebook, padding=16)
        notebook.add(control, text="Control")
        notebook.add(training, text="Training")
        notebook.add(calibration, text="Calibration")
        notebook.add(maze, text="Maze")
        notebook.add(teleporter, text="Teleporter")
        notebook.add(sprites, text="Sprites")
        notebook.add(controls, text="Controls")
        self._build_control_tab(control)
        self._build_training_tab(training)
        self._build_calibration_tab(calibration)
        self._build_maze_tab(maze)
        self._build_teleporter_tab(teleporter)
        self._build_sprites_tab(sprites)
        self._build_controls_tab(controls)
        notebook.bind("<<NotebookTabChanged>>", self._tab_changed)

    def _build_control_tab(self, frame: ttk.Frame) -> None:
        ttk.Label(frame, text="Mode", font=("", 12, "bold")).pack(anchor="w")
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=8)
        for mode in Mode:
            ttk.Button(
                buttons,
                text=mode.value.title(),
                command=lambda selected=mode: self._set_mode(selected),
            ).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Separator(frame).pack(fill="x", pady=8)
        ttk.Label(frame, textvariable=self.status).pack(anchor="w")
        ttk.Label(frame, textvariable=self.state).pack(anchor="w", pady=(8, 0))
        ttk.Label(frame, textvariable=self.target, wraplength=580).pack(anchor="w")
        ttk.Button(
            frame, text="Test training marker", command=self._test_marker
        ).pack(fill="x", pady=(18, 2))
        ttk.Label(
            frame,
            text="ESC is the global emergency stop.\n"
            "Live clicks only inside state-specific allowlisted zones.",
            foreground="#9b1c1c",
        ).pack(anchor="w", pady=(18, 0))

    def _build_training_tab(self, frame: ttk.Frame) -> None:
        ttk.Label(
            frame, text="Imitation policy training", font=("", 12, "bold")
        ).pack(anchor="w")
        ttk.Checkbutton(
            frame,
            text="Learn from my combat clicks while in Training mode",
            variable=self.learning_enabled,
            command=self._toggle_learning,
        ).pack(anchor="w", pady=(12, 8))
        ttk.Label(
            frame,
            text="The green marker is the prediction. Click normally in one of the\n"
            "configured zones. A match earns reward 1; another configured zone\n"
            "earns reward 0 and teaches the policy which action you chose.",
        ).pack(anchor="w")
        ttk.Separator(frame).pack(fill="x", pady=14)
        ttk.Label(frame, textvariable=self.training_status, wraplength=580).pack(
            anchor="w"
        )
        ttk.Label(frame, textvariable=self.maze_status, foreground="#176b2c").pack(
            anchor="w", pady=(10, 0)
        )
        ttk.Label(frame, textvariable=self.policy_stats).pack(anchor="w", pady=(10, 0))
        ttk.Button(frame, text="Reset learned policy", command=self._reset_policy).pack(
            anchor="w", pady=(14, 0)
        )
        self._refresh_policy_stats()

    def _build_controls_tab(self, frame: ttk.Frame) -> None:
        ttk.Label(
            frame, text="Mode shortcuts", font=("", 12, "bold")
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="Shortcuts work globally while the bot is running. Escape is "
            "reserved for emergency stop.",
            wraplength=620,
        ).pack(anchor="w", pady=(6, 16))
        bindings = ttk.LabelFrame(frame, text="Assignments", padding=12)
        bindings.pack(fill="x")
        for row, (mode, variable) in enumerate(
            (
                (Mode.TRAINING, self.training_binding),
                (Mode.LIVE, self.live_binding),
            )
        ):
            ttk.Label(bindings, text=mode.value.title(), width=12).grid(
                row=row, column=0, sticky="w", pady=6
            )
            ttk.Label(bindings, textvariable=variable, width=24).grid(
                row=row, column=1, sticky="w", padx=8
            )
            ttk.Button(
                bindings,
                text="Reassign",
                command=lambda selected=mode: self._begin_binding_capture(selected),
            ).grid(row=row, column=2, padx=(8, 0), pady=6)
        ttk.Label(
            frame, textvariable=self.binding_status, wraplength=620,
            foreground="#0d47a1"
        ).pack(anchor="w", pady=(14, 0))
        self._refresh_bindings()

    def _build_calibration_tab(self, frame: ttk.Frame) -> None:
        toolbar = ttk.Frame(frame)
        toolbar.pack(fill="x")
        for text, command in (
            ("Add anchor", self._capture_anchor),
            ("Add click zone", self._capture_click_zone),
            ("Add visual region", self._capture_visual_region),
            ("Lobby target", self._capture_lobby_point),
        ):
            ttk.Button(toolbar, text=text, command=command).pack(
                side="left", padx=2
            )
        columns = ("kind", "state", "name", "coordinates")
        self.items = ttk.Treeview(
            frame, columns=columns, show="headings", height=17, selectmode="browse"
        )
        for column, width in zip(columns, (90, 90, 150, 230)):
            self.items.heading(column, text=column.title())
            self.items.column(column, width=width, stretch=column == "coordinates")
        self.items.pack(fill="both", expand=True, pady=10)
        actions = ttk.Frame(frame)
        actions.pack(fill="x")
        for text, command in (
            ("Preview", self._preview_selected),
            ("Retake", self._retake_selected),
            ("Rename", self._rename_selected),
            ("Delete", self._delete_selected),
        ):
            ttk.Button(actions, text=text, command=command).pack(side="left", padx=2)
        ttk.Button(actions, text="Refresh", command=self._refresh_items).pack(
            side="right", padx=2
        )
        ttk.Button(actions, text="Remove state", command=self._remove_state).pack(
            side="right", padx=2
        )
        ttk.Button(actions, text="Add state", command=self._add_state).pack(
            side="right", padx=2
        )
        self._refresh_items()

    def _build_maze_tab(self, frame: ttk.Frame) -> None:
        ttk.Label(frame, text="10 x 10 exploration map", font=("", 12, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            frame, textvariable=self.active_mode, foreground="#0d47a1",
            font=("", 10, "bold")
        ).pack(anchor="w", pady=(3, 0))
        ttk.Label(frame, textvariable=self.maze_status).pack(anchor="w", pady=(4, 8))
        maze_body = ttk.Frame(frame)
        maze_body.pack(fill="both", expand=True)
        self.maze_canvas = tk.Canvas(
            maze_body, width=410, height=410, bg="white", highlightthickness=1
        )
        self.maze_canvas.pack(side="left", anchor="n")
        controls = ttk.LabelFrame(
            maze_body, text="Exits visible", padding=12
        )
        controls.pack(side="left", anchor="n", fill="y", padx=(16, 0))
        directions = ttk.Frame(controls)
        directions.pack(pady=(8, 16))
        positions = {
            "north": (0, 1),
            "west": (1, 0),
            "east": (1, 2),
            "south": (2, 1),
        }
        for direction, (row, column) in positions.items():
            ttk.Checkbutton(
                directions,
                text=direction.title(),
                variable=self.maze_exit_vars[direction],
            ).grid(row=row, column=column, padx=5, pady=14, sticky="nsew")
        for column in range(3):
            directions.columnconfigure(column, weight=1, minsize=70)
        ttk.Button(
            controls,
            text="Submit checked\ntile layout",
            command=self._submit_checked_layout,
        ).pack(fill="x", pady=(4, 8))
        ttk.Button(
            controls,
            text="Save checked exits\nto current maze tile",
            command=self._save_checked_maze_exits,
        ).pack(fill="x", pady=(0, 8))
        ttk.Button(
            controls, text="Reset maze", command=self.controller.reset_maze
        ).pack(fill="x")
        location = ttk.LabelFrame(controls, text="Manual location", padding=8)
        location.pack(fill="x", pady=(14, 0))
        coordinates = ttk.Frame(location)
        coordinates.pack(fill="x")
        ttk.Label(coordinates, text="X").grid(row=0, column=0, padx=(0, 3))
        ttk.Spinbox(
            coordinates, from_=0, to=9, width=3, textvariable=self.manual_maze_x
        ).grid(row=0, column=1, padx=(0, 10))
        ttk.Label(coordinates, text="Y").grid(row=0, column=2, padx=(0, 3))
        ttk.Spinbox(
            coordinates, from_=0, to=9, width=3, textvariable=self.manual_maze_y
        ).grid(row=0, column=3)
        ttk.Button(
            location, text="Set current tile", command=self._set_manual_maze_location
        ).pack(fill="x", pady=(8, 0))
        ttk.Separator(controls).pack(fill="x", pady=14)
        ttk.Button(
            controls, text="Capture boss sprite", command=self._capture_boss_sprite
        ).pack(fill="x")
        transition_zones = ttk.LabelFrame(
            controls, text="Scene edge regions", padding=6
        )
        transition_zones.pack(fill="x", pady=(10, 0))
        for index, direction in enumerate(("north", "south", "east", "west")):
            ttk.Button(
                transition_zones,
                text=f"Set {direction.title()}",
                command=lambda selected=direction: self._capture_visual_region(
                    "exploring", f"scene_change_{selected}"
                ),
            ).grid(
                row=index // 2,
                column=index % 2,
                padx=3,
                pady=3,
                sticky="ew",
            )
        transition_zones.columnconfigure(0, weight=1)
        transition_zones.columnconfigure(1, weight=1)
        ttk.Label(
            transition_zones,
            text="Place each region over the corresponding room entry edge.",
            wraplength=190,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self._draw_maze()

    def _build_teleporter_tab(self, frame: ttk.Frame) -> None:
        ttk.Label(
            frame, text="Teleporter action calibration", font=("", 12, "bold")
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="Capture each click in execution order. Placement uses three "
            "clicks and return uses two. These clicks have no pathfinding cost.",
            wraplength=620,
        ).pack(anchor="w", pady=(6, 14))
        for action, count in (("place", 3), ("return", 2)):
            group = ttk.LabelFrame(
                frame,
                text=(
                    "Place teleporter"
                    if action == "place"
                    else "Return to teleporter"
                ),
                padding=12,
            )
            group.pack(fill="x", pady=(0, 12))
            for index in range(count):
                ttk.Label(group, text=f"Click {index + 1}", width=12).grid(
                    row=index, column=0, sticky="w", pady=4
                )
                ttk.Button(
                    group,
                    text="Set zone",
                    command=lambda selected=action, step=index:
                        self._capture_teleporter_zone(selected, step),
                ).grid(row=index, column=1, padx=4, pady=4)
                ttk.Button(
                    group,
                    text="Preview",
                    command=lambda selected=action, step=index:
                        self._preview_teleporter_zone(selected, step),
                ).grid(row=index, column=2, padx=4, pady=4)
                ttk.Button(
                    group,
                    text="Clear",
                    command=lambda selected=action, step=index:
                        self._clear_teleporter_zone(selected, step),
                ).grid(row=index, column=3, padx=4, pady=4)
        ttk.Label(
            frame, textvariable=self.teleporter_status, foreground="#0d47a1"
        ).pack(anchor="w", pady=(4, 0))
        self._refresh_teleporter_status()

    def _capture_teleporter_zone(self, action: str, index: int) -> None:
        self.controller.set_mode(Mode.IDLE)
        ScreenSelector(
            self.root,
            int(self.config.data["monitor"]),
            f"Drag around teleporter {action} click {index + 1}",
            on_region=lambda region, _crop:
                self._save_teleporter_zone(action, index, region),
        )

    def _save_teleporter_zone(
        self, action: str, index: int, region: Region
    ) -> None:
        self.config.set_teleporter_click_zone(action, index, region)
        self._refresh_teleporter_status()
        self.status.set(f"Saved teleporter {action} click {index + 1}")

    def _preview_teleporter_zone(self, action: str, index: int) -> None:
        zone = self.config.teleporter_click_zones[action][index]
        if zone is None:
            messagebox.showinfo(
                "Zone not set",
                f"Teleporter {action} click {index + 1} has not been captured.",
                parent=self.root,
            )
            return
        self.controller.set_mode(Mode.IDLE)
        ScreenSelector(
            self.root,
            int(self.config.data["monitor"]),
            f"Teleporter {action} click {index + 1}",
            highlight_region=zone,
        )

    def _clear_teleporter_zone(self, action: str, index: int) -> None:
        self.config.delete_teleporter_click_zone(action, index)
        self._refresh_teleporter_status()

    def _refresh_teleporter_status(self) -> None:
        zones = self.config.teleporter_click_zones
        place = sum(zone is not None for zone in zones["place"])
        returns = sum(zone is not None for zone in zones["return"])
        self.teleporter_status.set(
            f"Placement: {place}/3 zones set    Return: {returns}/2 zones set"
        )

    def _build_sprites_tab(self, frame: ttk.Frame) -> None:
        ttk.Label(frame, text="Sprite management", font=("", 12, "bold")).pack(
            anchor="w"
        )
        toolbar = ttk.Frame(frame)
        toolbar.pack(fill="x", pady=(10, 8))
        ttk.Button(
            toolbar, text="Capture boss sprite", command=self._capture_boss_sprite
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            toolbar, text="Capture player sprite", command=self._capture_player_sprite
        ).pack(side="left", padx=4)
        self.sprite_items = ttk.Treeview(
            frame,
            columns=("kind", "name", "path"),
            show="headings",
            height=18,
            selectmode="browse",
        )
        for column, width in (("kind", 100), ("name", 150), ("path", 430)):
            self.sprite_items.heading(column, text=column.title())
            self.sprite_items.column(column, width=width, stretch=column == "path")
        self.sprite_items.pack(fill="both", expand=True)
        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=(8, 0))
        ttk.Button(actions, text="Preview", command=self._preview_sprite).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(actions, text="Retake", command=self._retake_sprite).pack(
            side="left", padx=4
        )
        ttk.Button(actions, text="Delete", command=self._delete_sprite).pack(
            side="left", padx=4
        )
        ttk.Button(actions, text="Refresh", command=self._refresh_sprites).pack(
            side="right"
        )
        self._refresh_sprites()

    def _ask_state(self) -> str | None:
        prompt = f"Enter one of: {', '.join(self.config.states)}"
        state = simpledialog.askstring("Game state", prompt, parent=self.root)
        if state is None:
            return None
        state = state.strip().lower()
        allowed = set(self.config.states)
        if state not in allowed:
            messagebox.showerror(
                "Invalid state",
                f"Configured states: {', '.join(self.config.states)}",
                parent=self.root,
            )
            return None
        return state

    def _capture_anchor(self, state: str | None = None) -> None:
        state = state or self._ask_state()
        if not state:
            return
        self.controller.set_mode(Mode.IDLE)
        ScreenSelector(
            self.root,
            int(self.config.data["monitor"]),
            f"Drag around a distinctive, unchanging {state} feature",
            on_region=lambda region, crop: self._save_anchor(state, region, crop),
        )

    def _save_anchor(self, state: str, region: Region, crop) -> None:
        templates = Path("templates")
        templates.mkdir(exist_ok=True)
        path = templates / f"{state}.png"
        cv2.imwrite(str(path), crop)
        self.config.set_region(f"{state}_anchor", region)
        self.config.data["templates"][state] = path.as_posix()
        self.config.save()
        self._reload_detector()
        self._refresh_items()
        self.status.set(f"Saved {state} anchor")

    def _capture_click_zone(
        self, state: str | None = None, replace_index: int | None = None
    ) -> None:
        state = state or self._ask_state()
        if not state:
            return
        self.controller.set_mode(Mode.IDLE)
        ScreenSelector(
            self.root,
            int(self.config.data["monitor"]),
            f"Drag where {state} action {replace_index + 1 if replace_index is not None else ''} is allowed",
            on_region=lambda region, _crop: self._save_click_zone(
                state, region, replace_index
            ),
        )

    def _save_click_zone(
        self, state: str, region: Region, replace_index: int | None = None
    ) -> None:
        if replace_index is None:
            self.config.add_click_zone(state, region)
        else:
            self.config.set_click_zone(state, replace_index, region)
        self._refresh_items()
        self.status.set(f"Saved {state} click zone")

    def _capture_visual_region(
        self, state: str | None = None, name: str | None = None
    ) -> None:
        state = state or self._ask_state()
        if not state:
            return
        if name is None:
            name = simpledialog.askstring(
                "Visual region name",
                "Name what this region shows (for example player_health,\n"
                "enemy_1_health, north_passage, or action_1_available):",
                parent=self.root,
            )
        if not name or not name.strip():
            return
        name = name.strip().lower().replace(" ", "_")
        self.controller.set_mode(Mode.IDLE)
        ScreenSelector(
            self.root,
            int(self.config.data["monitor"]),
            f"Drag around {state}: {name}",
            on_region=lambda region, _crop: self._save_visual_region(
                state, name, region
            ),
        )

    def _save_visual_region(self, state: str, name: str, region: Region) -> None:
        self.config.set_visual_region(state, name, region)
        self._refresh_items()
        self.status.set(f"Saved visual region {state}/{name}")

    def _capture_lobby_point(self) -> None:
        self.controller.set_mode(Mode.IDLE)
        ScreenSelector(
            self.root,
            int(self.config.data["monitor"]),
            "Click the center of the Start Quest button",
            on_point=self._save_lobby_point,
        )

    def _save_lobby_point(self, point: Point) -> None:
        self.config.set_lobby_start_point(point)
        self._refresh_items()
        self.status.set(f"Lobby target saved at {point.x}, {point.y}")

    def _capture_boss_sprite(self) -> None:
        self.controller.set_mode(Mode.IDLE)
        ScreenSelector(
            self.root,
            int(self.config.data["monitor"]),
            "Drag tightly around the boss sprite only",
            on_region=self._save_boss_sprite,
        )

    def _save_boss_sprite(self, _region: Region, crop) -> None:
        templates = Path("templates")
        templates.mkdir(exist_ok=True)
        path = templates / "boss.png"
        cv2.imwrite(str(path), crop)
        self.config.data["boss_template"] = path.as_posix()
        self.config.save()
        self.controller.boss_detector = BossDetector(
            path.as_posix(),
            float(self.config.data.get("boss_match_threshold", 0.82)),
        )
        self.controller.boss_status = (
            "Boss sprite saved; add boss_search_area if it is not configured"
        )
        self._refresh_sprites()
        self._draw_maze()

    def _capture_player_sprite(self, replace_index: int | None = None) -> None:
        self.controller.set_mode(Mode.IDLE)
        ScreenSelector(
            self.root,
            int(self.config.data["monitor"]),
            "Drag tightly around the player sprite only",
            on_region=lambda region, crop: self._save_player_sprite(
                region, crop, replace_index
            ),
        )

    def _save_player_sprite(
        self, _region: Region, crop, replace_index: int | None = None
    ) -> None:
        templates = Path("templates")
        templates.mkdir(exist_ok=True)
        paths = list(self.config.data.get("player_templates", []))
        if replace_index is not None and replace_index < len(paths):
            path = Path(paths[replace_index])
        else:
            number = 1
            while (templates / f"player_{number:03d}.png").exists():
                number += 1
            path = templates / f"player_{number:03d}.png"
            paths.append(path.as_posix())
        cv2.imwrite(str(path), crop)
        self.config.data["player_templates"] = paths
        self.config.save()
        self.controller.player_detector = PlayerDetector(
            paths,
            float(self.config.data.get("player_match_threshold", 0.78)),
        )
        self.status.set(f"Saved player sprite {path.name}")
        self._refresh_sprites()

    def _refresh_sprites(self) -> None:
        if not hasattr(self, "sprite_items"):
            return
        self.sprite_items.delete(*self.sprite_items.get_children())
        boss = self.config.data.get("boss_template")
        if boss:
            self.sprite_items.insert(
                "", "end", iid="boss|0",
                values=("boss", "boss sprite", boss),
            )
        for index, path in enumerate(self.config.data.get("player_templates", [])):
            self.sprite_items.insert(
                "", "end", iid=f"player|{index}",
                values=("player", f"appearance {index + 1}", path),
            )

    def _selected_sprite(self) -> tuple[str, int] | None:
        selection = self.sprite_items.selection()
        if not selection:
            return None
        kind, index = selection[0].split("|", 1)
        return kind, int(index)

    def _sprite_path(self, selected: tuple[str, int]) -> str | None:
        kind, index = selected
        if kind == "boss":
            return self.config.data.get("boss_template")
        paths = self.config.data.get("player_templates", [])
        return paths[index] if index < len(paths) else None

    def _preview_sprite(self) -> None:
        selected = self._selected_sprite()
        if selected is None:
            return
        path = self._sprite_path(selected)
        image = cv2.imread(str(path)) if path else None
        if image is None:
            messagebox.showerror("Missing sprite", "The sprite file could not be read.")
            return
        height, width = image.shape[:2]
        scale = min(1.0, 500 / max(width, height))
        if scale < 1.0:
            image = cv2.resize(
                image, (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            return
        window = tk.Toplevel(self.root)
        window.title("Sprite preview")
        window.attributes("-topmost", True)
        window.photo = tk.PhotoImage(
            data=base64.b64encode(encoded).decode("ascii")
        )
        ttk.Label(window, image=window.photo).pack(padx=12, pady=12)
        ttk.Button(window, text="Close", command=window.destroy).pack(pady=(0, 12))

    def _retake_sprite(self) -> None:
        selected = self._selected_sprite()
        if selected is None:
            return
        kind, index = selected
        if kind == "boss":
            self._capture_boss_sprite()
        else:
            self._capture_player_sprite(index)

    def _delete_sprite(self) -> None:
        selected = self._selected_sprite()
        if selected is None:
            return
        kind, index = selected
        path = self._sprite_path(selected)
        if not messagebox.askyesno(
            "Delete sprite",
            f"Remove this {kind} sprite from the detector and delete its file?",
            parent=self.root,
        ):
            return
        if kind == "boss":
            self.config.data["boss_template"] = None
            self.controller.boss_detector = BossDetector(None)
            self.controller.boss_status = "Boss detector not configured"
        else:
            paths = list(self.config.data.get("player_templates", []))
            if index < len(paths):
                paths.pop(index)
            self.config.data["player_templates"] = paths
            self.controller.player_detector = PlayerDetector(
                paths,
                float(self.config.data.get("player_match_threshold", 0.78)),
            )
        self.config.save()
        self._remove_sprite_file(path)
        self._refresh_sprites()
        self._draw_maze()

    @staticmethod
    def _remove_sprite_file(path: str | None) -> None:
        if not path:
            return
        candidate = Path(path).resolve()
        templates = (Path.cwd() / "templates").resolve()
        if candidate.is_relative_to(templates) and candidate.exists():
            candidate.unlink()

    def _refresh_items(self) -> None:
        if not hasattr(self, "items"):
            return
        self.items.delete(*self.items.get_children())
        for state in sorted(self.config.states):
            region = self.config.regions.get(f"{state}_anchor")
            if region:
                self._insert_item(
                    "anchor", state, self.config.anchor_name(state), region
                )
        for state, zones in sorted(self.config.click_zones.items()):
            for index, region in enumerate(zones):
                self._insert_item(
                    "zone",
                    state,
                    self.config.click_zone_name(state, index),
                    region,
                    str(index),
                )
        for state, regions in sorted(self.config.visual_regions.items()):
            for name, region in sorted(regions.items()):
                self._insert_item("visual", state, name, region, name)
        point = self.config.lobby_start_point
        if point:
            self.items.insert(
                "", "end", iid="point|lobby|target", values=(
                    "point", "lobby", "start target", f"x={point.x}, y={point.y}"
                )
            )

    def _insert_item(
        self, kind: str, state: str, name: str, region: Region, key: str = ""
    ) -> None:
        coordinates = (
            f"x={region.left}, y={region.top}, "
            f"w={region.width}, h={region.height}"
        )
        self.items.insert(
            "", "end", iid=f"{kind}|{state}|{key}", values=(
                kind, state, name, coordinates
            )
        )

    def _selected(self) -> tuple[str, str, str] | None:
        selection = self.items.selection()
        return tuple(selection[0].split("|", 2)) if selection else None

    def _selected_region(self, selected: tuple[str, str, str]) -> Region | None:
        kind, state, key = selected
        if kind == "anchor":
            return self.config.regions.get(f"{state}_anchor")
        if kind == "zone":
            return self.config.click_zones[state][int(key)]
        if kind == "visual":
            return self.config.visual_regions[state][key]
        point = self.config.lobby_start_point
        return Region(point.x - 14, point.y - 14, 28, 28) if point else None

    def _preview_selected(self) -> None:
        selected = self._selected()
        if not selected:
            return
        region = self._selected_region(selected)
        if region:
            self.controller.set_mode(Mode.IDLE)
            ScreenSelector(
                self.root,
                int(self.config.data["monitor"]),
                "Selected item highlighted; click or Escape to close",
                highlight_region=region,
            )

    def _retake_selected(self) -> None:
        selected = self._selected()
        if not selected:
            return
        kind, state, key = selected
        if kind == "anchor":
            self._capture_anchor(state)
        elif kind == "zone":
            self._capture_click_zone(state, int(key))
        elif kind == "visual":
            self._capture_visual_region(state, key)
        else:
            self._capture_lobby_point()

    def _rename_selected(self) -> None:
        selected = self._selected()
        if not selected:
            return
        kind, state, key = selected
        if kind == "point":
            messagebox.showinfo(
                "Fixed name", "The lobby start target has a fixed name.", parent=self.root
            )
            return
        current = self.items.item(self.items.selection()[0], "values")[2]
        name = simpledialog.askstring(
            "Rename item", "New name:", initialvalue=current, parent=self.root
        )
        if not name or not name.strip():
            return
        name = name.strip().lower().replace(" ", "_")
        if "|" in name:
            messagebox.showerror("Invalid name", "Names cannot contain |.", parent=self.root)
            return
        if kind == "anchor":
            self.config.rename_anchor(state, name)
        elif kind == "zone":
            self.config.rename_click_zone(state, int(key), name)
        elif kind == "visual":
            if name in self.config.visual_regions.get(state, {}) and name != key:
                messagebox.showerror("Duplicate name", "That region already exists.")
                return
            self.config.rename_visual_region(state, key, name)
        self._refresh_items()

    def _add_state(self) -> None:
        name = simpledialog.askstring(
            "Add state",
            "New state name (letters, numbers, and underscores):",
            parent=self.root,
        )
        if not name:
            return
        name = name.strip().lower().replace(" ", "_")
        if not name.replace("_", "").isalnum() or "|" in name:
            messagebox.showerror("Invalid state", "Use letters, numbers, and underscores.")
            return
        self.config.add_state(name)
        self._refresh_items()
        self.status.set(f"Added state {name}; capture its anchor and click zones")

    def _remove_state(self) -> None:
        state = simpledialog.askstring(
            "Remove state",
            f"State to remove:\n{', '.join(self.config.states)}",
            parent=self.root,
        )
        if not state:
            return
        state = state.strip().lower()
        if state not in self.config.states:
            messagebox.showerror("Unknown state", "That state is not configured.")
            return
        if not messagebox.askyesno(
            "Remove state",
            f"Remove {state} and all of its anchors, regions, and zones?",
            parent=self.root,
        ):
            return
        self.config.remove_state(state)
        self._reload_detector()
        self._refresh_items()

    def _delete_selected(self) -> None:
        selected = self._selected()
        if not selected:
            return
        if not messagebox.askyesno(
            "Delete calibration", "Delete the selected item?", parent=self.root
        ):
            return
        kind, state, key = selected
        if kind == "anchor":
            self.config.delete_anchor(state)
            self._reload_detector()
        elif kind == "zone":
            self.config.delete_click_zone(state, int(key))
        elif kind == "visual":
            self.config.delete_visual_region(state, key)
        else:
            self.config.delete_lobby_start_point()
        self._refresh_items()

    def _checked_exits(self) -> set[str]:
        return {
            direction
            for direction, variable in self.maze_exit_vars.items()
            if variable.get()
        }

    def _submit_checked_layout(self) -> None:
        self.controller.submit_tile_layout(self._checked_exits())

    def _save_checked_maze_exits(self) -> None:
        self.controller.confirm_maze_exits(self._checked_exits())
        for variable in self.maze_exit_vars.values():
            variable.set(False)

    def _set_manual_maze_location(self) -> None:
        try:
            x = int(self.manual_maze_x.get())
            y = int(self.manual_maze_y.get())
            self.controller.set_maze_position(x, y)
        except (ValueError, tk.TclError):
            messagebox.showerror(
                "Invalid location", "X and Y must both be between 0 and 9."
            )

    def _queue_maze_update(self) -> None:
        self.root.after(0, self._draw_maze)

    def _draw_maze(self) -> None:
        if not hasattr(self, "maze_canvas"):
            return
        canvas = self.maze_canvas
        canvas.delete("all")
        memory = self.controller.maze
        recommendation = self.controller.boss_direction or memory.recommended_action(
            self.controller._teleporter_ready("place"),
            self.controller._teleporter_ready("return"),
        )
        current_x, current_y = memory.position
        recommended_cell = None
        offsets = {
            "north": (0, -1), "east": (1, 0),
            "south": (0, 1), "west": (-1, 0),
        }
        if recommendation in offsets:
            dx, dy = offsets[recommendation]
            recommended_cell = (current_x + dx, current_y + dy)
        size, margin = 38, 15
        for y in range(10):
            for x in range(10):
                left, top = margin + x * size, margin + y * size
                fill = "white"
                if (x, y) in memory.tiles:
                    fill = "#cfd8dc"
                if (x, y) == recommended_cell:
                    fill = "#ffd54f"
                if (x, y) == memory.position:
                    fill = "#00c853"
                canvas.create_rectangle(
                    left, top, left + size, top + size, fill=fill, outline="#607d8b"
                )
                tile = memory.tiles.get((x, y))
                if tile:
                    cx, cy = left + size // 2, top + size // 2
                    if tile.layout_id:
                        layout_label = tile.layout_id.replace("layout_", "L")
                        if tile.layout_provisional:
                            layout_label = f"~{layout_label}"
                        canvas.create_text(
                            left + 3,
                            top + 3,
                            anchor="nw",
                            text=layout_label,
                            fill="#37474f",
                            font=("", 7),
                        )
                    for direction in tile.exits:
                        dx, dy = offsets[direction]
                        canvas.create_line(
                            cx, cy, cx + dx * 16, cy + dy * 16,
                            fill="#1565c0", width=3
                        )
                if (x, y) == memory.teleporter_position:
                    canvas.create_text(
                        left + size - 4,
                        top + 3,
                        anchor="ne",
                        text="T",
                        fill="#6a1b9a",
                        font=("", 10, "bold"),
                    )
                if (x, y) == memory.boss_position:
                    canvas.create_text(
                        left + 4,
                        top + size - 3,
                        anchor="sw",
                        text="B",
                        fill="#b71c1c",
                        font=("", 10, "bold"),
                    )
        current_tile = memory.tiles.get(memory.position)
        current_layout = (
            (
                f"{current_tile.layout_id} (provisional)"
                if current_tile.layout_provisional
                else current_tile.layout_id
            )
            if current_tile and current_tile.layout_id
            else "unlinked"
        )
        self.maze_status.set(
            f"Clears: {self.controller.clears}  "
            f"Current: ({current_x}, {current_y})  "
            f"Objective: find boss  "
            f"Recommendation: {recommendation or 'none'}"
            f"{' (BOSS)' if self.controller.boss_direction else ''}  "
            f"Layout: {current_layout}  "
            f"Explored tiles: {len(memory.tiles)}  "
            f"Teleporter: {memory.teleporter_position or 'not placed'}  "
            f"Boss tile: {memory.boss_position or 'unknown'}  "
            f"Submitted layouts: {len(self.controller.tile_layouts.layouts)}\n"
            f"{self.controller.maze_transition_status}\n"
            f"{self.controller.boss_status}\n"
            f"{self.controller.layout_status}"
        )

    def _reload_detector(self) -> None:
        self.controller.detector = StateDetector(
            self.config.data["templates"], self.config.data["confidence_threshold"]
        )

    def _toggle_learning(self) -> None:
        enabled = self.learning_enabled.get()
        self.controller.set_training_enabled(enabled)
        self.training_status.set(
            "Learning enabled - combat clicks will train the policy"
            if enabled else "Learning disabled"
        )

    def _reset_policy(self) -> None:
        if messagebox.askyesno(
            "Reset policy", "Delete all learned policy examples?", parent=self.root
        ):
            self.controller.policy.reset()
            self.training_status.set("Learned policy reset")
            self._refresh_policy_stats()

    def _refresh_policy_stats(self) -> None:
        policy = self.controller.policy
        accuracy = policy.matches / max(1, policy.examples)
        self.policy_stats.set(
            f"Examples: {policy.examples}    Rewards: {policy.matches}    "
            f"Match rate: {accuracy:.0%}"
        )

    def _queue_training_event(self, message: str) -> None:
        self.root.after(0, self._render_training_event, message)

    def _render_training_event(self, message: str) -> None:
        self.training_status.set(message)
        self._refresh_policy_stats()

    def _set_mode(self, mode: Mode) -> None:
        self.controller.set_mode(mode)
        self.active_mode.set(f"Active mode: {mode.value.upper()}")
        if mode != Mode.TRAINING:
            self.marker.withdraw()
        descriptions = {
            Mode.IDLE: "Idle - clicks disabled",
            Mode.TRAINING: "Training - observing; clicks disabled",
            Mode.LIVE: "Live - allowlisted clicks enabled",
        }
        self.status.set(descriptions[mode])

    @staticmethod
    def _binding_label(binding: str | None) -> str:
        if not binding:
            return "Unassigned"
        device, _, name = binding.partition(":")
        readable = {
            "x1": "Thumb mouse button 1",
            "x2": "Thumb mouse button 2",
        }.get(name, name.replace("_", " ").upper())
        return readable if device == "mouse" and name in {"x1", "x2"} else (
            f"{device.title()}: {readable}"
        )

    def _refresh_bindings(self) -> None:
        bindings = self.config.data.get("mode_bindings", {})
        self.training_binding.set(
            self._binding_label(bindings.get(Mode.TRAINING.value))
        )
        self.live_binding.set(self._binding_label(bindings.get(Mode.LIVE.value)))

    def _begin_binding_capture(self, mode: Mode) -> None:
        self.controller.begin_binding_capture(mode)
        self.binding_status.set(
            f"Waiting for the new {mode.value.title()} keyboard or mouse button..."
        )

    def _queue_mode_hotkey(self, mode: Mode) -> None:
        self.root.after(0, self._set_mode, mode)

    def _queue_binding_captured(self, mode: Mode, binding: str) -> None:
        self.root.after(0, self._render_binding_captured, mode, binding)

    def _render_binding_captured(self, mode: Mode, binding: str) -> None:
        self._refresh_bindings()
        self.binding_status.set(
            f"{mode.value.title()} assigned to {self._binding_label(binding)}."
        )

    def _test_marker(self) -> None:
        point = self.config.lobby_start_point
        if point is None:
            messagebox.showinfo("No target", "Set the lobby target first.")
            return
        self.controller.set_mode(Mode.IDLE)
        self.status.set("Marker test - showing lobby target for 3 seconds")
        self._show_marker(point)
        self.root.after(3000, self.marker.withdraw)

    def _show_marker(self, point: Point) -> None:
        self.marker.geometry(f"28x28+{point.x - 14}+{point.y - 14}")
        self.marker.deiconify()
        self.marker.lift()

    def _queue_update(self, decision: Decision) -> None:
        self.root.after(0, self._render_decision, decision)

    def _render_decision(self, decision: Decision) -> None:
        self.state.set(
            f"State: {decision.state_name} ({decision.confidence:.0%} confidence)"
        )
        text = (
            f"{decision.point.x}, {decision.point.y} - {decision.reason}"
            if decision.point else decision.reason or "none"
        )
        self.target.set(f"Target: {text}")
        if self.controller.mode == Mode.TRAINING and decision.point:
            self._show_marker(decision.point)
        else:
            self.marker.withdraw()

    def _queue_emergency_stop(self) -> None:
        self.root.after(0, self._emergency_stop)

    def _emergency_stop(self) -> None:
        self.marker.withdraw()
        self.active_mode.set("Active mode: IDLE (Escape stop)")
        self.status.set("EMERGENCY STOP - clicks disabled")
        self.binding_status.set(
            "Shortcut capture canceled. Escape remains the emergency stop."
        )

    def _tab_changed(self, _event=None) -> None:
        mode = self.controller.mode
        self.active_mode.set(f"Active mode: {mode.value.upper()}")

    def _make_marker_click_through(self) -> None:
        import ctypes

        hwnd = self.marker.winfo_id()
        style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
        ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x20)

    def _close(self) -> None:
        self.controller.shutdown()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    try:
        App().run()
    except Exception as exc:
        messagebox.showerror("DragonFable Bot", str(exc))


if __name__ == "__main__":
    main()
