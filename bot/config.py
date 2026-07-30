from __future__ import annotations

from copy import deepcopy
import json
from dataclasses import asdict
from pathlib import Path

from .models import Point, Region


DEFAULTS = {
    "monitor": 1,
    "tick_seconds": 0.5,
    "confidence_threshold": 0.82,
    "regions": {},
    "visual_regions": {},
    "click_zones": {},
    "templates": {},
    "lobby_start_point": None,
    "states": ["menu", "lobby", "exploring", "combat"],
    "click_zone_names": {},
    "anchor_names": {},
    "boss_template": None,
    "boss_match_threshold": 0.82,
    "transition_pending_timeout_seconds": 10.0,
    "transition_sample_seconds": 0.1,
    "transition_edge_appearance_difference": 0.04,
    "transition_edge_stability_difference": 0.01,
    "transition_edge_stable_frames": 3,
    "mode_bindings": {
        "training": "mouse:x1",
        "live": "mouse:x2",
    },
}


class Config:
    def __init__(self, path: str | Path = "config.json") -> None:
        self.path = Path(path)
        self.data = deepcopy(DEFAULTS)
        self.load()

    def load(self) -> None:
        if self.path.exists():
            self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
        exploring = self.data.setdefault("visual_regions", {}).setdefault(
            "exploring", {}
        )
        changed = False
        if "scene_change_pending_timeout_seconds" in self.data:
            self.data["transition_pending_timeout_seconds"] = self.data.pop(
                "scene_change_pending_timeout_seconds"
            )
            changed = True
        migration = {
            "scene_change_1": "scene_change_north",
            "scene_change_2": "scene_change_south",
            "scene_change_3": "scene_change_east",
            "scene_change_4": "scene_change_west",
        }
        for old, new in migration.items():
            if old in exploring and new not in exploring:
                exploring[new] = exploring.pop(old)
                changed = True
        for key in (
            "room_transition_threshold",
            "room_transition_delay_seconds",
            "room_scene_stability_threshold",
            "scene_change_required_zones",
            "scene_change_big_threshold",
            "scene_change_pair_window_frames",
            "scene_change_pixel_threshold",
            "scene_change_fraction_threshold",
            "scene_change_direction_window_frames",
            "walkable_grid_columns",
            "walkable_grid_rows",
            "walkable_chunk_difference_threshold",
            "walkable_nonadjacent_min_chunks",
            "walkable_nonadjacent_span",
            "walkable_widespread_fraction",
            "walkable_stable_fraction",
            "walkable_stable_frames",
            "walkable_volatility_ignore_threshold",
            "player_match_threshold",
            "tile_layout_match_distance",
            "tile_layout_duplicate_distance",
            "tile_layout_ambiguity_margin",
        ):
            if key in self.data:
                self.data.pop(key)
                changed = True
        if changed:
            self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    @property
    def regions(self) -> dict[str, Region]:
        return {name: Region(**value) for name, value in self.data["regions"].items()}

    @property
    def click_zones(self) -> dict[str, list[Region]]:
        return {
            state: [Region(**value) for value in values]
            for state, values in self.data["click_zones"].items()
        }

    @property
    def visual_regions(self) -> dict[str, dict[str, Region]]:
        return {
            state: {name: Region(**value) for name, value in values.items()}
            for state, values in self.data["visual_regions"].items()
        }

    @property
    def lobby_start_point(self) -> Point | None:
        value = self.data.get("lobby_start_point")
        return Point(**value) if value else None

    def set_region(self, name: str, region: Region) -> None:
        self.data["regions"][name] = asdict(region)
        self.save()

    def add_click_zone(self, state: str, region: Region) -> None:
        self.data["click_zones"].setdefault(state, []).append(asdict(region))
        number = len(self.data["click_zones"][state])
        self.data["click_zone_names"].setdefault(state, []).append(f"zone_{number}")
        self.save()

    def set_click_zone(self, state: str, index: int, region: Region) -> None:
        self.data["click_zones"][state][index] = asdict(region)
        self.save()

    def delete_click_zone(self, state: str, index: int) -> None:
        del self.data["click_zones"][state][index]
        names = self.data["click_zone_names"].get(state, [])
        if index < len(names):
            del names[index]
        self.save()

    def click_zone_name(self, state: str, index: int) -> str:
        names = self.data["click_zone_names"].setdefault(state, [])
        while len(names) < len(self.data["click_zones"].get(state, [])):
            names.append(f"zone_{len(names) + 1}")
        return names[index]

    def rename_click_zone(self, state: str, index: int, name: str) -> None:
        self.click_zone_name(state, index)
        self.data["click_zone_names"][state][index] = name
        self.save()

    def delete_anchor(self, state: str) -> None:
        self.data["regions"].pop(f"{state}_anchor", None)
        self.data["templates"].pop(state, None)
        self.save()

    def anchor_name(self, state: str) -> str:
        return self.data["anchor_names"].get(state, f"{state} anchor")

    def rename_anchor(self, state: str, name: str) -> None:
        self.data["anchor_names"][state] = name
        self.save()

    def set_visual_region(self, state: str, name: str, region: Region) -> None:
        self.data["visual_regions"].setdefault(state, {})[name] = asdict(region)
        self.save()

    def delete_visual_region(self, state: str, name: str) -> None:
        self.data["visual_regions"].get(state, {}).pop(name, None)
        self.save()

    def rename_visual_region(self, state: str, old: str, new: str) -> None:
        regions = self.data["visual_regions"].setdefault(state, {})
        regions[new] = regions.pop(old)
        self.save()

    @property
    def states(self) -> list[str]:
        return list(self.data["states"])

    def add_state(self, state: str) -> None:
        if state not in self.data["states"]:
            self.data["states"].append(state)
            self.save()

    def remove_state(self, state: str) -> None:
        if state in self.data["states"]:
            self.data["states"].remove(state)
        self.data["regions"].pop(f"{state}_anchor", None)
        self.data["templates"].pop(state, None)
        self.data["click_zones"].pop(state, None)
        self.data["click_zone_names"].pop(state, None)
        self.data["visual_regions"].pop(state, None)
        self.data["anchor_names"].pop(state, None)
        self.save()

    def set_lobby_start_point(self, point: Point) -> None:
        self.data["lobby_start_point"] = asdict(point)
        self.save()

    def set_mode_binding(self, mode: str, binding: str) -> None:
        bindings = self.data.setdefault("mode_bindings", {})
        for other_mode, other_binding in list(bindings.items()):
            if other_mode != mode and other_binding == binding:
                bindings[other_mode] = None
        bindings[mode] = binding
        self.save()

    def delete_lobby_start_point(self) -> None:
        self.data["lobby_start_point"] = None
        self.save()
