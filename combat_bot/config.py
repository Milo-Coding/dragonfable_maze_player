from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import uuid

from bot.models import Region


DEFAULTS = {
    "monitor": 1,
    "tick_seconds": 0.35,
    "click_interval_seconds": 1.25,
    "anchor_threshold": 0.82,
    "anchor_region": None,
    "anchor_template": None,
    "view_zones": [],
    "action_zones": [],
    "hidden_layers": [64, 32],
    "learning_rate": 0.01,
    "mode_bindings": {
        "training": "mouse:x1",
        "live": "mouse:x2"
    },
    "tasks": [],
    "scan_targets": [],
    "scan_match_threshold": 0.82,
    "maze_exit_threshold": 0.88,
    "maze_transition_seconds": 1.5,
    "maze_exit_scans": {direction: None for direction in ("north", "east", "south", "west")},
    "maze_move_points": {direction: None for direction in ("north", "east", "south", "west")},
    "maze_teleporter_points": {"place": [None, None, None], "return": [None, None]},
}

VALID_MOUSE_BINDINGS = ("mouse:x1", "mouse:x2", "mouse:middle")


class CombatConfig:
    def __init__(self, path: str | Path = "combat_config.json") -> None:
        self.path = Path(path)
        self.data = deepcopy(DEFAULTS)
        if self.path.exists():
            self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
        if "epsilon" in self.data:
            self.data.pop("epsilon")
            self.save()
        self._migrate_legacy_combat_anchor()
        self._migrate_task_anchors()

    def _migrate_legacy_combat_anchor(self) -> None:
        """Make an existing v2 combat setup the first task exactly once."""
        if self.data.get("tasks") or not self.data.get("anchor_region"):
            return
        self.data["tasks"] = [{
            "id": self.new_id(),
            "name": "combat",
            "behavior": "combat",
            "enabled": True,
            "anchor_region": self.data.get("anchor_region"),
            "anchor_template": self.data.get("anchor_template"),
            "simple_point": None,
        }]
        self.save()

    def _migrate_task_anchors(self) -> None:
        changed = False
        for task in self.data.get("tasks", []):
            if "anchors" not in task:
                task["anchors"] = []
                if task.get("anchor_region") and task.get("anchor_template"):
                    task["anchors"].append({
                        "region": task["anchor_region"],
                        "template": task["anchor_template"],
                    })
                task.pop("anchor_region", None)
                task.pop("anchor_template", None)
                changed = True
        if changed:
            self.save()

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    @property
    def anchor_region(self) -> Region | None:
        value = self.data.get("anchor_region")
        return Region(**value) if value else None

    @property
    def view_zones(self) -> list[dict]:
        return self.data["view_zones"]

    @property
    def action_zones(self) -> list[dict]:
        return self.data["action_zones"]

    @staticmethod
    def region(item: dict) -> Region:
        return Region(**item["region"])

    def set_anchor(self, region: Region, template: str) -> None:
        self.data["anchor_region"] = asdict(region)
        self.data["anchor_template"] = template
        self.save()

    def add_zone(self, kind: str, name: str, region: Region) -> None:
        key = "view_zones" if kind == "view" else "action_zones"
        self.data[key].append({"name": name, "region": asdict(region), "enabled": True})
        self.save()

    def update_zone(self, kind: str, index: int, region: Region) -> None:
        key = "view_zones" if kind == "view" else "action_zones"
        self.data[key][index]["region"] = asdict(region)
        self.save()

    def toggle_zone(self, kind: str, index: int) -> None:
        key = "view_zones" if kind == "view" else "action_zones"
        item = self.data[key][index]
        item["enabled"] = not item.get("enabled", True)
        self.save()

    def delete_zone(self, kind: str, index: int) -> None:
        key = "view_zones" if kind == "view" else "action_zones"
        del self.data[key][index]
        self.save()

    def enabled(self, kind: str) -> list[dict]:
        source = self.view_zones if kind == "view" else self.action_zones
        return [item for item in source if item.get("enabled", True)]

    def set_mode_binding(self, mode: str, binding: str) -> None:
        if mode not in {"training", "live"} or binding not in VALID_MOUSE_BINDINGS:
            raise ValueError("Invalid mode binding")
        bindings = self.data.setdefault("mode_bindings", {})
        for other_mode, other_binding in list(bindings.items()):
            if other_mode != mode and other_binding == binding:
                bindings[other_mode] = None
        bindings[mode] = binding
        self.save()

    @property
    def tasks(self) -> list[dict]:
        return self.data.setdefault("tasks", [])

    @property
    def scan_targets(self) -> list[dict]:
        return self.data.setdefault("scan_targets", [])

    def add_task(self, name: str, behavior: str) -> str:
        task_id = self.new_id()
        self.tasks.append({
            "id": task_id, "name": name, "behavior": behavior,
            "enabled": True, "anchors": [], "simple_point": None,
        })
        self.save()
        return task_id

    def task(self, task_id: str) -> dict | None:
        return next((task for task in self.tasks if task["id"] == task_id), None)

    def move_task(self, task_id: str, offset: int) -> None:
        index = next(i for i, task in enumerate(self.tasks) if task["id"] == task_id)
        destination = max(0, min(len(self.tasks) - 1, index + offset))
        if destination != index:
            self.tasks.insert(destination, self.tasks.pop(index))
            self.save()

    def delete_task(self, task_id: str) -> None:
        self.data["tasks"] = [task for task in self.tasks if task["id"] != task_id]
        self.save()

    def add_task_anchor(self, task_id: str, region: Region, template: str) -> None:
        task = self.task(task_id)
        if task is None:
            raise KeyError(task_id)
        task.setdefault("anchors", []).append({"region": asdict(region), "template": template})
        self.save()

    def delete_task_anchor(self, task_id: str, index: int) -> None:
        task = self.task(task_id)
        if task is None:
            raise KeyError(task_id)
        del task.setdefault("anchors", [])[index]
        self.save()

    def add_scan_target(self, name: str) -> str:
        target_id = self.new_id()
        self.scan_targets.append({
            "id": target_id, "name": name, "sprites": [], "scan_zones": []
        })
        self.save()
        return target_id

    def scan_target(self, target_id: str) -> dict | None:
        return next((target for target in self.scan_targets if target["id"] == target_id), None)

    def delete_scan_target(self, target_id: str) -> None:
        self.data["scan_targets"] = [target for target in self.scan_targets if target["id"] != target_id]
        for task in self.tasks:
            if task.get("behavior") == f"scanning:{target_id}":
                task["enabled"] = False
        self.save()
