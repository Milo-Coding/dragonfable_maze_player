from __future__ import annotations

from copy import deepcopy
import json
from dataclasses import asdict
from pathlib import Path

from .models import Point, Region


DEFAULTS = {
    "monitor": 1,
    "clears": 10,
    "tick_seconds": 0.5,
    "teleporter_click_interval_seconds": 0.25,
    "confidence_threshold": 0.82,
    "regions": {},
    "visual_regions": {},
    "disabled_visual_regions": {},
    "click_zones": {},
    "disabled_click_zones": {},
    "templates": {},
    "lobby_start_point": None,
    "states": ["menu", "lobby", "exploring", "combat"],
    "click_zone_names": {},
    "teleporter_click_zones": {
        "place": [None, None, None],
        "return": [None, None],
    },
    "anchor_names": {},
    "boss_template": None,
    "boss_match_threshold": 0.82,
    "mog_template": None,
    "mog_match_threshold": 0.82,
    "mog_close_click_zone": None,
    "player_templates": [],
    "player_match_threshold": 0.78,
    "transition_pending_timeout_seconds": 10.0,
    "transition_sample_seconds": 0.03,
    "transition_edge_activity_difference": 0.02,
    "transition_edge_stability_difference": 0.01,
    "transition_edge_stable_frames": 3,
    "transition_minimum_changed_regions": 2,
    "transition_whole_scene_activity_difference": 0.003,
    "transition_whole_scene_stability_difference": 0.0015,
    "transition_click_confirmation_seconds": 1,
    "transition_player_movement_pixels": 4.0,
    "transition_player_stable_frames": 4,
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
        if "walkable_ground" in exploring:
            exploring.pop("walkable_ground")
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
    def active_click_zones(self) -> dict[str, list[Region]]:
        disabled = self.data.setdefault("disabled_click_zones", {})
        return {
            state: [
                Region(**value)
                for index, value in enumerate(values)
                if index not in set(disabled.get(state, []))
            ]
            for state, values in self.data["click_zones"].items()
        }

    @property
    def teleporter_click_zones(self) -> dict[str, list[Region | None]]:
        configured = self.data.setdefault("teleporter_click_zones", {})
        result: dict[str, list[Region | None]] = {}
        for action, count in (("place", 3), ("return", 2)):
            values = list(configured.get(action, []))
            values = (values + [None] * count)[:count]
            result[action] = [
                Region(**value) if value is not None else None for value in values
            ]
        return result

    def set_teleporter_click_zone(
        self, action: str, index: int, region: Region
    ) -> None:
        required = {"place": 3, "return": 2}
        if action not in required or not 0 <= index < required[action]:
            raise ValueError("Invalid teleporter action step")
        zones = self.data.setdefault("teleporter_click_zones", {}).setdefault(
            action, [None] * required[action]
        )
        while len(zones) < required[action]:
            zones.append(None)
        zones[index] = asdict(region)
        self.save()

    def delete_teleporter_click_zone(self, action: str, index: int) -> None:
        required = {"place": 3, "return": 2}
        if action not in required or not 0 <= index < required[action]:
            raise ValueError("Invalid teleporter action step")
        zones = self.data.setdefault("teleporter_click_zones", {}).setdefault(
            action, [None] * required[action]
        )
        while len(zones) < required[action]:
            zones.append(None)
        zones[index] = None
        self.save()

    @property
    def visual_regions(self) -> dict[str, dict[str, Region]]:
        disabled = self.data.setdefault("disabled_visual_regions", {})
        return {
            state: {
                name: Region(**value)
                for name, value in values.items()
                if name not in set(disabled.get(state, []))
            }
            for state, values in self.data["visual_regions"].items()
        }

    @property
    def all_visual_regions(self) -> dict[str, dict[str, Region]]:
        return {
            state: {name: Region(**value) for name, value in values.items()}
            for state, values in self.data["visual_regions"].items()
        }

    @property
    def lobby_start_point(self) -> Point | None:
        value = self.data.get("lobby_start_point")
        return Point(**value) if value else None

    @property
    def mog_close_click_zone(self) -> Region | None:
        value = self.data.get("mog_close_click_zone")
        return Region(**value) if value else None

    def set_mog_close_click_zone(self, region: Region) -> None:
        self.data["mog_close_click_zone"] = asdict(region)
        self.save()

    def delete_mog_close_click_zone(self) -> None:
        self.data["mog_close_click_zone"] = None
        self.save()

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
        disabled = self.data.setdefault("disabled_click_zones", {}).get(state, [])
        self.data["disabled_click_zones"][state] = [
            value - 1 if value > index else value
            for value in disabled
            if value != index
        ]
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

    def click_zone_enabled(self, state: str, index: int) -> bool:
        return index not in set(
            self.data.setdefault("disabled_click_zones", {}).get(state, [])
        )

    def set_click_zone_enabled(
        self, state: str, index: int, enabled: bool
    ) -> None:
        disabled = set(
            self.data.setdefault("disabled_click_zones", {}).get(state, [])
        )
        if enabled:
            disabled.discard(index)
        else:
            disabled.add(index)
        self.data["disabled_click_zones"][state] = sorted(disabled)
        self.save()

    def active_click_zone_name(self, state: str, active_index: int) -> str:
        disabled = set(
            self.data.setdefault("disabled_click_zones", {}).get(state, [])
        )
        enabled_indices = [
            index
            for index in range(len(self.data["click_zones"].get(state, [])))
            if index not in disabled
        ]
        return self.click_zone_name(state, enabled_indices[active_index])

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
        disabled = self.data.setdefault("disabled_visual_regions", {}).get(state, [])
        self.data["disabled_visual_regions"][state] = [
            value for value in disabled if value != name
        ]
        self.save()

    def rename_visual_region(self, state: str, old: str, new: str) -> None:
        regions = self.data["visual_regions"].setdefault(state, {})
        regions[new] = regions.pop(old)
        disabled = self.data.setdefault("disabled_visual_regions", {}).get(state, [])
        self.data["disabled_visual_regions"][state] = [
            new if value == old else value for value in disabled
        ]
        self.save()

    def visual_region_enabled(self, state: str, name: str) -> bool:
        return name not in set(
            self.data.setdefault("disabled_visual_regions", {}).get(state, [])
        )

    def set_visual_region_enabled(
        self, state: str, name: str, enabled: bool
    ) -> None:
        disabled = set(
            self.data.setdefault("disabled_visual_regions", {}).get(state, [])
        )
        if enabled:
            disabled.discard(name)
        else:
            disabled.add(name)
        self.data["disabled_visual_regions"][state] = sorted(disabled)
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
        self.data["disabled_click_zones"].pop(state, None)
        self.data["disabled_visual_regions"].pop(state, None)
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
