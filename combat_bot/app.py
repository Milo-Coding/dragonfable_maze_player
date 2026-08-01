from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2

from bot.calibration import ScreenSelector
from bot.models import Decision, Mode, Point, Region
from .config import CombatConfig, VALID_MOUSE_BINDINGS
from .controller import CombatController


class CombatApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("DragonFable Combat Bot 2.0")
        self.root.geometry("760x650"); self.root.minsize(680, 560)
        self.root.attributes("-topmost", True)
        self.config = CombatConfig()
        self.mode_text = tk.StringVar(value="Mode: IDLE")
        self.scene_text = tk.StringVar(value="Combat anchor: waiting")
        self.action_text = tk.StringVar(value="Action: none")
        self.training_text = tk.StringVar(value="No demonstrations yet")
        self.stats_text = tk.StringVar()
        self.maze_text = tk.StringVar(value="Maze: waiting for exploring task")
        self.impact_vars: dict[str, tk.BooleanVar] = {}
        self.controller = CombatController(self.config, self._queue_update,
                                           self._queue_mode, self._queue_training,
                                           self._queue_maze)
        self._build_marker(); self._build(); self._refresh_all()
        self.controller.start(); self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_marker(self) -> None:
        self.marker = self._make_target("#00ef68", 24)
        self.importance_markers = [self._make_target("#ff2020", 24) for _ in range(3)]
        self.root.after(100, self._click_through)

    def _make_target(self, color: str, size: int) -> tk.Toplevel:
        marker = tk.Toplevel(self.root); marker.withdraw()
        marker.overrideredirect(True); marker.attributes("-topmost", True)
        transparent = "#ff00ff"
        marker.configure(bg=transparent)
        marker.attributes("-transparentcolor", transparent)
        canvas = tk.Canvas(
            marker, width=size, height=size, bg=transparent,
            highlightthickness=0, borderwidth=0,
        )
        canvas.pack()
        center = size // 2
        canvas.create_oval(3, 3, size - 3, size - 3, outline=color, width=3)
        canvas.create_line(center, 3, center, size - 3, fill=color, width=2)
        canvas.create_line(3, center, size - 3, center, fill=color, width=2)
        return marker

    def _build(self) -> None:
        book = ttk.Notebook(self.root); book.pack(fill="both", expand=True, padx=8, pady=8)
        control, tasks, scanning, maze, calibration, impact, settings, controls = (ttk.Frame(book, padding=16) for _ in range(8))
        for frame, label in ((control, "Control"), (calibration, "Calibration"),
                             (tasks, "Tasks"), (scanning, "Scanning"),
                             (maze, "Maze"),
                             (impact, "Last-click impact"), (settings, "Network"),
                             (controls, "Controls")):
            book.add(frame, text=label)
        self._build_control(control); self._build_tasks(tasks); self._build_scanning(scanning)
        self._build_maze(maze)
        self._build_calibration(calibration)
        self.impact_frame = impact; self._build_settings(settings)
        self._build_controls(controls)

    def _build_control(self, frame: ttk.Frame) -> None:
        ttk.Label(frame, text="Combat agent", font=("", 14, "bold")).pack(anchor="w")
        buttons = ttk.Frame(frame); buttons.pack(fill="x", pady=14)
        for mode in Mode:
            ttk.Button(buttons, text=mode.value.title(),
                       command=lambda value=mode: self._set_mode(value)).pack(side="left", expand=True, fill="x", padx=3)
        ttk.Label(frame, textvariable=self.mode_text, font=("", 11, "bold")).pack(anchor="w", pady=(10, 6))
        ttk.Label(frame, textvariable=self.scene_text).pack(anchor="w")
        ttk.Label(frame, textvariable=self.action_text, wraplength=650).pack(anchor="w", pady=(6, 0))
        ttk.Separator(frame).pack(fill="x", pady=18)
        ttk.Label(frame, textvariable=self.training_text, wraplength=650).pack(anchor="w")
        ttk.Label(frame, textvariable=self.stats_text).pack(anchor="w", pady=(8, 0))
        ttk.Label(frame, text="Training observes your clicks and never clicks for you. Live clicks only inside enabled action zones. Escape always returns to Idle.",
                  foreground="#8b1616", wraplength=650).pack(anchor="w", pady=(24, 0))

    def _build_calibration(self, frame: ttk.Frame) -> None:
        toolbar = ttk.Frame(frame); toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Add viewing zone", command=lambda: self._capture_zone("view")).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Add action zone", command=lambda: self._capture_zone("action")).pack(side="left", padx=2)
        self.items = ttk.Treeview(frame, columns=("kind", "name", "enabled", "region"), show="headings", height=17)
        for column, width in (("kind", 100), ("name", 190), ("enabled", 80), ("region", 300)):
            self.items.heading(column, text=column.title()); self.items.column(column, width=width)
        self.items.pack(fill="both", expand=True, pady=12)
        actions = ttk.Frame(frame); actions.pack(fill="x")
        for label, command in (("Preview", self._preview), ("Retake", self._retake),
                               ("Enable / Disable", self._toggle), ("Delete", self._delete)):
            ttk.Button(actions, text=label, command=command).pack(side="left", padx=2)

    def _build_tasks(self, frame: ttk.Frame) -> None:
        ttk.Label(frame, text="Prioritized tasks", font=("", 13, "bold")).pack(anchor="w")
        ttk.Label(frame, text="The first enabled task whose anchor is visible wins. Higher rows have higher priority.",
                  wraplength=650).pack(anchor="w", pady=(5, 10))
        self.task_items = ttk.Treeview(frame, columns=("priority", "name", "behavior", "enabled", "anchor"),
                                       show="headings", height=15, selectmode="browse")
        for column, width in (("priority", 65), ("name", 160), ("behavior", 190), ("enabled", 70), ("anchor", 100)):
            self.task_items.heading(column, text=column.title()); self.task_items.column(column, width=width)
        self.task_items.pack(fill="both", expand=True, pady=10)
        first = ttk.Frame(frame); first.pack(fill="x")
        for label, command in (("Add task", self._add_task), ("Remove", self._remove_task),
                               ("Rename", self._rename_task), ("Enable / Disable", self._toggle_task)):
            ttk.Button(first, text=label, command=command).pack(side="left", padx=2)
        second = ttk.Frame(frame); second.pack(fill="x", pady=(6, 0))
        for label, command in (("Move up", lambda: self._move_task(-1)), ("Move down", lambda: self._move_task(1)),
                               ("Add anchor", self._capture_task_anchor), ("Preview anchor", self._preview_task_anchor),
                               ("Remove anchor", self._remove_task_anchor), ("Set simple click", self._capture_simple_point)):
            ttk.Button(second, text=label, command=command).pack(side="left", padx=2)

    def _build_scanning(self, frame: ttk.Frame) -> None:
        ttk.Label(frame, text="Scan targets", font=("", 13, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Add one or more sprite samples and search zones. Matching detections are randomly selected.",
                  wraplength=650).pack(anchor="w", pady=(5, 10))
        self.scan_items = ttk.Treeview(frame, columns=("kind", "target", "detail"), show="headings",
                                       height=16, selectmode="browse")
        for column, width in (("kind", 100), ("target", 190), ("detail", 350)):
            self.scan_items.heading(column, text=column.title()); self.scan_items.column(column, width=width)
        self.scan_items.pack(fill="both", expand=True, pady=10)
        actions = ttk.Frame(frame); actions.pack(fill="x")
        for label, command in (("Add target", self._add_scan_target), ("Rename target", self._rename_scan_target),
                               ("Add sprite sample", self._capture_scan_sprite), ("Add scan zone", self._capture_scan_zone),
                               ("Preview", self._preview_scan_item), ("Remove", self._remove_scan_item)):
            ttk.Button(actions, text=label, command=command).pack(side="left", padx=2)

    def _build_maze(self, frame: ttk.Frame) -> None:
        scroller = tk.Canvas(frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=scroller.yview)
        content = ttk.Frame(scroller, padding=(2, 2, 12, 16))
        window = scroller.create_window((0, 0), window=content, anchor="nw")
        scroller.configure(yscrollcommand=scrollbar.set)
        scroller.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        content.bind("<Configure>", lambda _event: scroller.configure(scrollregion=scroller.bbox("all")))
        scroller.bind("<Configure>", lambda event: scroller.itemconfigure(window, width=event.width))
        scroller.bind("<MouseWheel>", lambda event: scroller.yview_scroll(int(-event.delta / 120), "units"))
        ttk.Label(content, text="Exploring behavior", font=("", 13, "bold")).pack(anchor="w")
        ttk.Label(content, text="For each direction, capture the visual that means an exit is active, then click the movement destination. The original bounded-depth solver chooses among detected exits.",
                  wraplength=680).pack(anchor="w", pady=(5, 10))
        grid = ttk.LabelFrame(content, text="Directional calibration", padding=10); grid.pack(fill="x")
        self.maze_calibration_labels: dict[str, tk.StringVar] = {}
        for row, direction in enumerate(("north", "east", "south", "west")):
            ttk.Label(grid, text=direction.title(), width=10).grid(row=row, column=0, sticky="w", pady=4)
            variable = tk.StringVar(); self.maze_calibration_labels[direction] = variable
            ttk.Label(grid, textvariable=variable, width=30).grid(row=row, column=1, sticky="w")
            ttk.Button(grid, text="Scan active exit", command=lambda d=direction: self._capture_maze_exit(d)).grid(row=row, column=2, padx=3)
            ttk.Button(grid, text="Set movement click", command=lambda d=direction: self._capture_maze_move(d)).grid(row=row, column=3, padx=3)
        teleporter = ttk.LabelFrame(content, text="Teleporter sequence", padding=8); teleporter.pack(fill="x", pady=(8, 0))
        self.teleporter_labels: dict[tuple[str, int], tk.StringVar] = {}
        column = 0
        for action, count in (("place", 3), ("return", 2)):
            for index in range(count):
                variable = tk.StringVar(); self.teleporter_labels[(action, index)] = variable
                ttk.Button(teleporter, textvariable=variable,
                           command=lambda a=action, i=index: self._capture_teleporter_point(a, i)).grid(row=0, column=column, padx=3)
                column += 1
        ttk.Label(content, textvariable=self.maze_text, wraplength=680, foreground="#135f2b").pack(anchor="w", pady=(12, 5))
        self.maze_canvas = tk.Canvas(content, width=310, height=310, bg="white", highlightthickness=1)
        self.maze_canvas.pack(anchor="w", pady=5)
        controls = ttk.Frame(content); controls.pack(fill="x")
        ttk.Button(controls, text="Reset maze", command=self.controller.reset_maze).pack(side="left", padx=2)
        ttk.Button(controls, text="Set current position", command=self._set_maze_position).pack(side="left", padx=2)

    def _build_settings(self, frame: ttk.Frame) -> None:
        ttk.Label(frame, text="Dense policy", font=("", 13, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        self.settings_vars = {}
        fields = (("hidden_layers", "Hidden layers (comma-separated)"), ("learning_rate", "Learning rate"),
                  ("tick_seconds", "Scan interval (seconds)"), ("click_interval_seconds", "Live click interval"),
                  ("anchor_threshold", "Anchor threshold"), ("maze_exit_threshold", "Maze exit-match threshold"),
                  ("maze_transition_seconds", "Maze transition wait"))
        for row, (key, label) in enumerate(fields, 1):
            value = self.config.data[key]
            variable = tk.StringVar(value=", ".join(map(str, value)) if isinstance(value, list) else str(value))
            self.settings_vars[key] = variable
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=7)
            ttk.Entry(frame, textvariable=variable, width=24).grid(row=row, column=1, sticky="w", padx=10)
        button_row = len(fields) + 2
        ttk.Button(frame, text="Save settings", command=self._save_settings).grid(row=button_row, column=0, sticky="w", pady=20)
        ttk.Button(frame, text="Reset learned policy", command=self._reset_policy).grid(row=button_row, column=1, sticky="w", pady=20)

    def _build_controls(self, frame: ttk.Frame) -> None:
        ttk.Label(frame, text="Global mode buttons", font=("", 13, "bold")).pack(anchor="w")
        ttk.Label(frame, text="These mouse buttons work even while the game has focus. Escape is always reserved for Idle/emergency stop.",
                  wraplength=650).pack(anchor="w", pady=(6, 18))
        readable = {
            "mouse:x1": "Thumb button 1 (X1)",
            "mouse:x2": "Thumb button 2 (X2)",
            "mouse:middle": "Middle mouse button",
        }
        self.binding_vars: dict[str, tk.StringVar] = {}
        bindings = self.config.data.setdefault("mode_bindings", {})
        box = ttk.LabelFrame(frame, text="Assignments", padding=12); box.pack(fill="x")
        for row, mode in enumerate((Mode.TRAINING, Mode.LIVE)):
            ttk.Label(box, text=mode.value.title(), width=14).grid(row=row, column=0, sticky="w", pady=8)
            variable = tk.StringVar(value=bindings.get(mode.value, "mouse:x1" if mode == Mode.TRAINING else "mouse:x2"))
            self.binding_vars[mode.value] = variable
            chooser = ttk.Combobox(box, state="readonly", width=25,
                                   values=list(VALID_MOUSE_BINDINGS), textvariable=variable)
            chooser.grid(row=row, column=1, padx=8, pady=8)
            ttk.Label(box, textvariable=variable, foreground="#555").grid(row=row, column=2, padx=8)
        ttk.Button(frame, text="Save mode buttons", command=self._save_bindings).pack(anchor="w", pady=16)
        ttk.Label(frame, text="Default: X1 = Training, X2 = Live", foreground="#0d5b2a").pack(anchor="w")

    def _save_bindings(self) -> None:
        training = self.binding_vars[Mode.TRAINING.value].get()
        live = self.binding_vars[Mode.LIVE.value].get()
        if training == live:
            messagebox.showerror("Duplicate button", "Training and Live must use different mouse buttons.")
            return
        self.config.set_mode_binding(Mode.TRAINING.value, training)
        self.config.set_mode_binding(Mode.LIVE.value, live)
        self.training_text.set("Global mode-button assignments saved")

    def _selected_task(self) -> dict | None:
        selection = self.task_items.selection()
        return self.config.task(selection[0]) if selection else None

    def _add_task(self) -> None:
        name = simpledialog.askstring("Add task", "Task name:", parent=self.root)
        if not name or not name.strip(): return
        targets = {target["name"]: target["id"] for target in self.config.scan_targets}
        choices = ["combat", "exploring", "simple", *(f"scanning_for_{target}" for target in targets)]
        behavior_name = simpledialog.askstring("Task behavior", "Behavior:\n" + "\n".join(choices), parent=self.root)
        if not behavior_name: return
        cleaned = behavior_name.strip()
        if cleaned == "combat": behavior = "combat"
        elif cleaned == "exploring": behavior = "exploring"
        elif cleaned == "simple": behavior = "simple"
        elif cleaned.startswith("scanning_for_") and cleaned.removeprefix("scanning_for_") in targets:
            behavior = "scanning:" + targets[cleaned.removeprefix("scanning_for_")]
        else:
            messagebox.showerror("Unknown behavior", "Choose combat, exploring, simple, or one of the listed scanning behaviors.")
            return
        task_id = self.config.add_task(name.strip(), behavior)
        self._refresh_tasks(); self.task_items.selection_set(task_id)

    def _remove_task(self) -> None:
        task = self._selected_task()
        if task and messagebox.askyesno("Remove task", f"Remove {task['name']}?", parent=self.root):
            self.config.delete_task(task["id"]); self._refresh_tasks()

    def _rename_task(self) -> None:
        task = self._selected_task()
        if not task: return
        name = simpledialog.askstring("Rename task", "New task name:", initialvalue=task["name"], parent=self.root)
        if name and name.strip(): task["name"] = name.strip(); self.config.save(); self._refresh_tasks()

    def _toggle_task(self) -> None:
        task = self._selected_task()
        if task: task["enabled"] = not task.get("enabled", True); self.config.save(); self._refresh_tasks()

    def _move_task(self, offset: int) -> None:
        task = self._selected_task()
        if task: self.config.move_task(task["id"], offset); self._refresh_tasks(); self.task_items.selection_set(task["id"])

    def _capture_task_anchor(self) -> None:
        task = self._selected_task()
        if not task: return
        self._set_mode(Mode.IDLE)
        ScreenSelector(self.root, int(self.config.data["monitor"]), f"Drag around the anchor for {task['name']}",
                       on_region=lambda region, crop: self._save_task_anchor(task["id"], region, crop))

    def _save_task_anchor(self, task_id: str, region: Region, crop) -> None:
        task = self.config.task(task_id)
        if task is None: return
        Path("templates").mkdir(exist_ok=True)
        path = f"templates/task_{task_id}_anchor_{len(task.get('anchors', [])) + 1:03d}.png"
        if not cv2.imwrite(path, crop): raise RuntimeError("Could not save task anchor")
        self.config.add_task_anchor(task_id, region, path); self._refresh_tasks()

    def _choose_anchor_index(self, task: dict, title: str) -> int | None:
        anchors = task.get("anchors", [])
        if not anchors: messagebox.showinfo("No anchors", "This task has no anchors."); return None
        value = simpledialog.askinteger(title, f"Anchor number (1-{len(anchors)}):", minvalue=1,
                                        maxvalue=len(anchors), parent=self.root)
        return value - 1 if value is not None else None

    def _preview_task_anchor(self) -> None:
        task = self._selected_task()
        if not task: return
        index = self._choose_anchor_index(task, "Preview anchor")
        if index is None: return
        self._set_mode(Mode.IDLE)
        ScreenSelector(self.root, int(self.config.data["monitor"]), f"{task['name']} anchor {index + 1}",
                       highlight_region=Region(**task["anchors"][index]["region"]))

    def _remove_task_anchor(self) -> None:
        task = self._selected_task()
        if not task: return
        index = self._choose_anchor_index(task, "Remove anchor")
        if index is not None and messagebox.askyesno("Remove anchor", f"Remove anchor {index + 1}?", parent=self.root):
            self.config.delete_task_anchor(task["id"], index); self._refresh_tasks()

    def _capture_simple_point(self) -> None:
        task = self._selected_task()
        if not task or task.get("behavior") != "simple":
            messagebox.showinfo("Simple tasks only", "Select a task with the simple behavior."); return
        self._set_mode(Mode.IDLE)
        ScreenSelector(self.root, int(self.config.data["monitor"]), f"Click the action point for {task['name']}",
                       on_point=lambda point: self._save_simple_point(task["id"], point))

    def _save_simple_point(self, task_id: str, point: Point) -> None:
        task = self.config.task(task_id)
        if task: task["simple_point"] = {"x": point.x, "y": point.y}; self.config.save(); self._refresh_tasks()

    def _behavior_label(self, task: dict) -> str:
        behavior = task.get("behavior", "simple")
        if not behavior.startswith("scanning:"): return behavior
        target = self.config.scan_target(behavior.partition(":")[2])
        return f"scanning_for_{target['name']}" if target else "scanning_for_[missing]"

    def _refresh_tasks(self) -> None:
        self.task_items.delete(*self.task_items.get_children())
        for index, task in enumerate(self.config.tasks, 1):
            self.task_items.insert("", "end", iid=task["id"], values=(index, task["name"], self._behavior_label(task),
                "Yes" if task.get("enabled", True) else "No", f"{len(task.get('anchors', []))} anchor(s)"))

    def _capture_maze_exit(self, direction: str) -> None:
        self._set_mode(Mode.IDLE)
        ScreenSelector(self.root, int(self.config.data["monitor"]), f"Drag the zone showing an ACTIVE {direction} exit",
                       on_region=lambda region, crop: self._save_maze_exit(direction, region, crop))

    def _save_maze_exit(self, direction: str, region: Region, crop) -> None:
        Path("templates").mkdir(exist_ok=True); path = f"templates/maze_exit_{direction}.png"
        if not cv2.imwrite(path, crop): raise RuntimeError("Could not save exit scan")
        self.config.data.setdefault("maze_exit_scans", {})[direction] = {
            "region": {"left": region.left, "top": region.top, "width": region.width, "height": region.height},
            "template": path,
        }
        self.config.save(); self._refresh_maze()

    def _capture_maze_move(self, direction: str) -> None:
        self._set_mode(Mode.IDLE)
        ScreenSelector(self.root, int(self.config.data["monitor"]), f"Click the simple movement point for {direction}",
                       on_point=lambda point: self._save_maze_move(direction, point))

    def _save_maze_move(self, direction: str, point: Point) -> None:
        self.config.data.setdefault("maze_move_points", {})[direction] = {"x": point.x, "y": point.y}
        self.config.save(); self._refresh_maze()

    def _capture_teleporter_point(self, action: str, index: int) -> None:
        self._set_mode(Mode.IDLE)
        ScreenSelector(self.root, int(self.config.data["monitor"]),
                       f"Click teleporter {action} step {index + 1}",
                       on_point=lambda point: self._save_teleporter_point(action, index, point))

    def _save_teleporter_point(self, action: str, index: int, point: Point) -> None:
        values = self.config.data.setdefault("maze_teleporter_points", {}).setdefault(
            action, [None] * (3 if action == "place" else 2))
        values[index] = {"x": point.x, "y": point.y}
        self.config.save(); self._refresh_maze()

    def _set_maze_position(self) -> None:
        value = simpledialog.askstring("Maze position", "Coordinates as x,y (each 0-9):", parent=self.root)
        if not value: return
        try:
            x, y = (int(part.strip()) for part in value.split(",", 1))
            self.controller.set_maze_position(x, y)
        except (ValueError, TypeError): messagebox.showerror("Invalid position", "Enter two values from 0 to 9, such as 3,4.")

    def _queue_maze(self) -> None: self.root.after(0, self._refresh_maze)

    def _refresh_maze(self) -> None:
        if not hasattr(self, "maze_canvas"): return
        for direction, variable in self.maze_calibration_labels.items():
            scan = self.config.data.get("maze_exit_scans", {}).get(direction)
            point = self.config.data.get("maze_move_points", {}).get(direction)
            variable.set(f"exit scan: {'set' if scan else 'missing'} · click: {'set' if point else 'missing'}")
        teleporter = self.config.data.get("maze_teleporter_points", {})
        for (action, index), variable in self.teleporter_labels.items():
            values = teleporter.get(action, [])
            configured = index < len(values) and values[index] is not None
            variable.set(f"{action.title()} {index + 1}: {'set' if configured else 'set click'}")
        self.maze_text.set(self.controller.maze_status)
        canvas = self.maze_canvas; canvas.delete("all")
        memory = self.controller.maze; size, margin = 29, 10
        recommendation = memory.recommended_action(False, False)
        offsets = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
        recommended_cell = None
        if recommendation in offsets:
            dx, dy = offsets[recommendation]; recommended_cell = (memory.position[0] + dx, memory.position[1] + dy)
        for y in range(10):
            for x in range(10):
                left, top = margin + x * size, margin + y * size
                fill = "#d8e1e5" if (x, y) in memory.tiles else "white"
                if (x, y) == recommended_cell: fill = "#ffd54f"
                if (x, y) == memory.position: fill = "#00c853"
                canvas.create_rectangle(left, top, left + size, top + size, fill=fill, outline="#607d8b")
                tile = memory.tiles.get((x, y))
                if tile:
                    cx, cy = left + size // 2, top + size // 2
                    for direction in tile.exits:
                        dx, dy = offsets[direction]
                        canvas.create_line(cx, cy, cx + dx * 11, cy + dy * 11, fill="#1565c0", width=3)
                if (x, y) == memory.teleporter_position:
                    canvas.create_text(left + size - 3, top + 2, anchor="ne", text="T", fill="#6a1b9a", font=("", 9, "bold"))

    def _selected_scan(self) -> tuple[str, str, int | None] | None:
        selection = self.scan_items.selection()
        if not selection: return None
        parts = selection[0].split("|")
        return parts[0], parts[1], int(parts[2]) if len(parts) > 2 else None

    def _selected_scan_target(self) -> dict | None:
        selected = self._selected_scan()
        return self.config.scan_target(selected[1]) if selected else None

    def _add_scan_target(self) -> None:
        name = simpledialog.askstring("Scan target", "Target name:", parent=self.root)
        if name and name.strip(): self.config.add_scan_target(name.strip().lower().replace(" ", "_")); self._refresh_scanning()

    def _rename_scan_target(self) -> None:
        target = self._selected_scan_target()
        if not target: return
        name = simpledialog.askstring("Rename target", "New name:", initialvalue=target["name"], parent=self.root)
        if name and name.strip(): target["name"] = name.strip().lower().replace(" ", "_"); self.config.save(); self._refresh_scanning(); self._refresh_tasks()

    def _capture_scan_sprite(self) -> None:
        target = self._selected_scan_target()
        if not target: messagebox.showinfo("Select target", "Select a target or one of its rows first."); return
        self._set_mode(Mode.IDLE)
        ScreenSelector(self.root, int(self.config.data["monitor"]), f"Drag tightly around a {target['name']} sprite",
                       on_region=lambda region, crop: self._save_scan_sprite(target["id"], region, crop))

    def _save_scan_sprite(self, target_id: str, region: Region, crop) -> None:
        target = self.config.scan_target(target_id)
        if not target: return
        Path("templates").mkdir(exist_ok=True); number = len(target["sprites"]) + 1
        path = f"templates/scan_{target_id}_{number:03d}.png"
        if not cv2.imwrite(path, crop): raise RuntimeError("Could not save scan sprite")
        target["sprites"].append({"path": path, "source_region": {"left": region.left, "top": region.top,
                                                                   "width": region.width, "height": region.height}})
        self.config.save(); self._refresh_scanning()

    def _capture_scan_zone(self) -> None:
        target = self._selected_scan_target()
        if not target: messagebox.showinfo("Select target", "Select a target or one of its rows first."); return
        self._set_mode(Mode.IDLE)
        ScreenSelector(self.root, int(self.config.data["monitor"]), f"Drag a search zone for {target['name']}",
                       on_region=lambda region, _crop: self._save_scan_zone(target["id"], region))

    def _save_scan_zone(self, target_id: str, region: Region) -> None:
        target = self.config.scan_target(target_id)
        if target: target["scan_zones"].append({"left": region.left, "top": region.top, "width": region.width, "height": region.height}); self.config.save(); self._refresh_scanning()

    def _preview_scan_item(self) -> None:
        selected = self._selected_scan(); target = self._selected_scan_target()
        if not selected or not target or selected[2] is None: return
        kind, _, index = selected
        value = target["sprites"][index].get("source_region") if kind == "sprite" else target["scan_zones"][index]
        if value: self._set_mode(Mode.IDLE); ScreenSelector(self.root, int(self.config.data["monitor"]), "Selected scan calibration", highlight_region=Region(**value))

    def _remove_scan_item(self) -> None:
        selected = self._selected_scan(); target = self._selected_scan_target()
        if not selected or not target: return
        kind, target_id, index = selected
        if not messagebox.askyesno("Remove", "Remove the selected scan item?", parent=self.root): return
        if kind == "target": self.config.delete_scan_target(target_id)
        elif kind == "sprite" and index is not None: del target["sprites"][index]; self.config.save()
        elif kind == "zone" and index is not None: del target["scan_zones"][index]; self.config.save()
        self._refresh_scanning(); self._refresh_tasks()

    def _refresh_scanning(self) -> None:
        self.scan_items.delete(*self.scan_items.get_children())
        for target in self.config.scan_targets:
            target_id = target["id"]
            self.scan_items.insert("", "end", iid=f"target|{target_id}", values=("target", target["name"],
                f"{len(target['sprites'])} sprite(s), {len(target['scan_zones'])} scan zone(s)"))
            for index, sprite in enumerate(target["sprites"]):
                self.scan_items.insert("", "end", iid=f"sprite|{target_id}|{index}", values=("sprite", target["name"], sprite.get("path", "")))
            for index, raw in enumerate(target["scan_zones"]):
                self.scan_items.insert("", "end", iid=f"zone|{target_id}|{index}", values=("scan zone", target["name"], self._coords(Region(**raw))))

    def _refresh_all(self) -> None:
        self._refresh_items(); self._refresh_tasks(); self._refresh_scanning()
        self._refresh_maze(); self._refresh_impact(); self._refresh_stats()

    def _refresh_items(self) -> None:
        self.items.delete(*self.items.get_children())
        for kind, values in (("view", self.config.view_zones), ("action", self.config.action_zones)):
            for index, item in enumerate(values):
                self.items.insert("", "end", iid=f"{kind}|{index}", values=(kind, item["name"],
                    "Yes" if item.get("enabled", True) else "No", self._coords(self.config.region(item))))

    def _refresh_impact(self) -> None:
        for child in self.impact_frame.winfo_children(): child.destroy()
        canvas = tk.Canvas(self.impact_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.impact_frame, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas, padding=(2, 2, 12, 12))
        window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        canvas.bind("<MouseWheel>", lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"))
        ttk.Label(content, text="What mattered to your last click?", font=("", 13, "bold")).pack(anchor="w")
        ttk.Label(
            content,
            text="After demonstrating an action, select the viewing zones that most influenced it. Apply updates only those visual branches; unchecked zones receive no gradient for the focused replay. It does not count as a second example.",
            wraplength=650,
            justify="left",
        ).pack(anchor="w", fill="x", pady=(6, 14))
        self.impact_vars = {}
        for item in self.config.enabled("view"):
            variable = tk.BooleanVar(value=False); self.impact_vars[item["name"]] = variable
            tk.Checkbutton(
                content,
                text=item["name"],
                variable=variable,
                anchor="w",
                justify="left",
                wraplength=520,
                padx=3,
                pady=3,
            ).pack(anchor="w", fill="x")
        ttk.Button(content, text="Apply impact to last click", command=self._apply_impact).pack(anchor="w", pady=18)

    @staticmethod
    def _coords(region: Region) -> str:
        return f"x={region.left}, y={region.top}, w={region.width}, h={region.height}"

    def _capture_anchor(self) -> None:
        task = next((task for task in self.config.tasks if task.get("behavior") == "combat"), None)
        if task is None:
            task_id = self.config.add_task("combat", "combat")
            task = self.config.task(task_id)
            self._refresh_tasks()
        assert task is not None
        self._set_mode(Mode.IDLE)
        ScreenSelector(self.root, int(self.config.data["monitor"]), "Drag around a stable combat-only anchor",
                       on_region=lambda region, crop: self._save_task_anchor(task["id"], region, crop))

    def _save_anchor(self, region: Region, crop) -> None:
        Path("templates").mkdir(exist_ok=True); path = "templates/combat_v2_anchor.png"
        if not cv2.imwrite(path, crop): raise RuntimeError("Could not save anchor template")
        self.config.set_anchor(region, path); self._refresh_items()

    def _capture_zone(self, kind: str, replace: int | None = None) -> None:
        name = None if replace is not None else simpledialog.askstring("Zone name", f"Name for this {kind} zone:", parent=self.root)
        if replace is None and (not name or not name.strip()): return
        self._set_mode(Mode.IDLE)
        ScreenSelector(self.root, int(self.config.data["monitor"]), f"Drag around the {kind} zone",
                       on_region=lambda region, _crop: self._save_zone(kind, name, region, replace))

    def _save_zone(self, kind: str, name: str | None, region: Region, replace: int | None) -> None:
        if replace is None:
            assert name is not None
            existing = [x["name"] for x in (self.config.view_zones if kind == "view" else self.config.action_zones)]
            clean = name.strip().lower().replace(" ", "_")
            if clean in existing: messagebox.showerror("Duplicate", "Zone names must be unique within their type."); return
            self.config.add_zone(kind, clean, region)
        else: self.config.update_zone(kind, replace, region)
        self._refresh_all()

    def _selected(self):
        selection = self.items.selection()
        return selection[0].split("|") if selection else None

    def _selected_region(self, selected) -> Region | None:
        kind, raw = selected
        if kind == "anchor": return self.config.anchor_region
        values = self.config.view_zones if kind == "view" else self.config.action_zones
        return self.config.region(values[int(raw)])

    def _preview(self) -> None:
        selected = self._selected()
        if selected and (region := self._selected_region(selected)):
            self._set_mode(Mode.IDLE); ScreenSelector(self.root, int(self.config.data["monitor"]), "Selected zone", highlight_region=region)

    def _retake(self) -> None:
        selected = self._selected()
        if not selected: return
        if selected[0] == "anchor": self._capture_anchor()
        else: self._capture_zone(selected[0], int(selected[1]))

    def _toggle(self) -> None:
        selected = self._selected()
        if selected and selected[0] != "anchor": self.config.toggle_zone(selected[0], int(selected[1])); self._refresh_all()

    def _delete(self) -> None:
        selected = self._selected()
        if not selected or selected[0] == "anchor": return
        if messagebox.askyesno("Delete zone", "Delete the selected zone?", parent=self.root):
            self.config.delete_zone(selected[0], int(selected[1])); self._refresh_all()

    def _apply_impact(self) -> None:
        selected = {name for name, value in self.impact_vars.items() if value.get()}
        if not self.controller.reinforce_last(selected):
            messagebox.showinfo("Nothing applied", "Make a training click and select at least one viewing zone.")
        self._refresh_stats()

    def _save_settings(self) -> None:
        try:
            hidden = [int(x.strip()) for x in self.settings_vars["hidden_layers"].get().split(",")]
            if not hidden or any(x < 1 for x in hidden): raise ValueError
            self.config.data["hidden_layers"] = hidden
            for key in self.settings_vars.keys() - {"hidden_layers"}:
                self.config.data[key] = float(self.settings_vars[key].get())
            if not all(0 <= self.config.data[key] <= 1 for key in
                       ("anchor_threshold", "maze_exit_threshold")): raise ValueError
            self.config.save(); self.training_text.set("Network settings saved")
        except ValueError: messagebox.showerror("Invalid settings", "Use positive numbers; match thresholds must be between 0 and 1.")

    def _reset_policy(self) -> None:
        if messagebox.askyesno("Reset policy", "Delete all Combat Bot 2.0 learning?", parent=self.root):
            self.controller.policy.reset(); self.controller.last_example = None; self._refresh_stats()

    def _set_mode(self, mode: Mode) -> None:
        self.controller.set_mode(mode); self._render_mode(mode)

    def _queue_mode(self, mode: Mode) -> None: self.root.after(0, self._render_mode, mode)
    def _render_mode(self, mode: Mode) -> None:
        self.mode_text.set(f"Mode: {mode.value.upper()}")
        if mode != Mode.TRAINING:
            self.marker.withdraw()
            for marker in self.importance_markers: marker.withdraw()

    def _queue_update(self, decision: Decision) -> None: self.root.after(0, self._render_update, decision)
    def _render_update(self, decision: Decision) -> None:
        active = self.controller.active_task
        self.scene_text.set(
            f"Active task: {active['name'] if active else 'none'} "
            f"({'anchor found' if active else 'no anchor found'}, {decision.confidence:.0%})"
        )
        self.action_text.set("Action: " + (decision.reason or "none"))
        if self.controller.mode == Mode.TRAINING and decision.point:
            self.marker.geometry(f"24x24+{decision.point.x - 12}+{decision.point.y - 12}"); self.marker.deiconify(); self.marker.lift()
            important = decision.details.get("important_views", [])
            for index, marker in enumerate(self.importance_markers):
                if index < len(important):
                    point = important[index]
                    marker.geometry(f"24x24+{point.x - 12}+{point.y - 12}")
                    marker.deiconify(); marker.lift()
                else:
                    marker.withdraw()
        else:
            self.marker.withdraw()
            for marker in self.importance_markers: marker.withdraw()

    def _queue_training(self, message: str) -> None: self.root.after(0, self._render_training, message)
    def _render_training(self, message: str) -> None:
        self.training_text.set(message); self._refresh_stats()

    def _refresh_stats(self) -> None:
        policy = self.controller.policy
        rate = policy.matches / max(1, policy.examples)
        self.stats_text.set(f"Examples: {policy.examples} · matches: {policy.matches} ({rate:.0%}) · updates: {policy.updates} · loss: {policy.last_loss:.3f}")

    def _click_through(self) -> None:
        import ctypes
        for marker in (self.marker, *self.importance_markers):
            hwnd = marker.winfo_id(); style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x20)

    def _close(self) -> None:
        self.controller.shutdown(); self.root.destroy()

    def run(self) -> None: self.root.mainloop()


def main() -> None:
    try: CombatApp().run()
    except Exception as exc: messagebox.showerror("DragonFable Combat Bot 2.0", str(exc))


if __name__ == "__main__": main()
