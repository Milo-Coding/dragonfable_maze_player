from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


DIRECTIONS = {
    "north": (0, -1),
    "east": (1, 0),
    "south": (0, 1),
    "west": (-1, 0),
}
OPPOSITE = {"north": "south", "east": "west", "south": "north", "west": "east"}


@dataclass
class Tile:
    exits: set[str] = field(default_factory=set)
    tried: set[str] = field(default_factory=set)
    tile_type: str = "unknown"
    exits_confirmed: bool = False
    layout_id: str | None = None
    layout_provisional: bool = False


class MazeMemory:
    """A map of explored rooms with bounded-depth frontier exploration."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.position = (0, 0)
        self.tiles: dict[tuple[int, int], Tile] = {}
        self.teleporter_position: tuple[int, int] | None = None
        self.boss_position: tuple[int, int] | None = None

    def observe(self, exits: set[str], tile_type: str = "unknown") -> None:
        tile = self.tiles.setdefault(self.position, Tile())
        if not tile.exits_confirmed:
            tile.exits.update(exits)
        tile.tile_type = tile_type
        if tile_type == "boss":
            self.boss_position = self.position

    def set_position(self, x: int, y: int) -> None:
        if not (0 <= x < 10 and 0 <= y < 10):
            raise ValueError("Maze coordinates must be between 0 and 9")
        self.position = (x, y)
        self.tiles.setdefault(self.position, Tile())

    def confirm_exits(self, exits: set[str]) -> None:
        """Replace inferred exits with the trainer's authoritative labels."""
        tile = self.tiles.setdefault(self.position, Tile())
        tile.exits = set(exits)
        tile.exits_confirmed = True

    def link_layout(
        self, layout_id: str, exits: set[str], provisional: bool = False
    ) -> None:
        tile = self.tiles.setdefault(self.position, Tile())
        tile.layout_id = layout_id
        tile.layout_provisional = provisional
        tile.exits = set(exits)
        tile.exits_confirmed = True

    def update_linked_layout(self, layout_id: str, exits: set[str]) -> int:
        updated = 0
        for tile in self.tiles.values():
            if tile.layout_id == layout_id:
                tile.exits = set(exits)
                tile.exits_confirmed = True
                updated += 1
        return updated

    def choose_exit(self) -> str | None:
        tile = self.tiles.setdefault(self.position, Tile())
        untried = [
            direction
            for direction in ("north", "east", "south", "west")
            if direction in tile.exits and direction not in tile.tried
        ]
        if untried:
            # A boss room is a dead end. At a branch, inspect the path with the
            # smallest remaining possible search area first so short,
            # constrained branches are eliminated before deep ones.
            return min(
                untried,
                key=lambda direction: (
                    self.maximum_possible_depth(direction),
                    tuple(DIRECTIONS).index(direction),
                ),
            )

        # The current room is exhausted. Travel by the shortest known route to
        # the nearest room that still has an unexplored branch.
        queue = deque([(self.position, None)])
        visited = {self.position}
        while queue:
            position, first_direction = queue.popleft()
            candidate = self.tiles[position]
            if position != self.position and any(
                direction in candidate.exits and direction not in candidate.tried
                for direction in DIRECTIONS
            ):
                return first_direction
            for direction, (dx, dy) in DIRECTIONS.items():
                if direction not in candidate.tried:
                    continue
                neighbor = (position[0] + dx, position[1] + dy)
                if neighbor not in self.tiles or neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, first_direction or direction))
        return None

    def _frontier_route(
        self, start: tuple[int, int]
    ) -> tuple[int, str | None] | None:
        """Return walking distance and first direction to the nearest frontier."""
        if start not in self.tiles:
            return None
        queue = deque([(start, 0, None)])
        visited = {start}
        while queue:
            position, distance, first_direction = queue.popleft()
            candidate = self.tiles[position]
            if any(
                direction in candidate.exits and direction not in candidate.tried
                for direction in DIRECTIONS
            ):
                return distance, first_direction
            for direction, (dx, dy) in DIRECTIONS.items():
                if direction not in candidate.tried:
                    continue
                neighbor = (position[0] + dx, position[1] + dy)
                if neighbor not in self.tiles or neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(
                    (neighbor, distance + 1, first_direction or direction)
                )
        return None

    def _route_to(self, target: tuple[int, int]) -> str | None:
        """Return the first direction on the shortest known route to a tile."""
        if target == self.position:
            return None
        if self.position not in self.tiles or target not in self.tiles:
            return None
        queue = deque([(self.position, None)])
        visited = {self.position}
        while queue:
            position, first_direction = queue.popleft()
            tile = self.tiles[position]
            for direction, (dx, dy) in DIRECTIONS.items():
                if direction not in tile.tried:
                    continue
                neighbor = (position[0] + dx, position[1] + dy)
                if neighbor not in self.tiles or neighbor in visited:
                    continue
                next_direction = first_direction or direction
                if neighbor == target:
                    return next_direction
                visited.add(neighbor)
                queue.append((neighbor, next_direction))
        return None

    def recommended_action(
        self, can_place: bool = False, can_return: bool = False
    ) -> str | None:
        """Use the teleporter only to remove walk-backs from a branch."""
        if self.boss_position is not None and self.boss_position != self.position:
            boss_route = self._route_to(self.boss_position)
            if boss_route is not None:
                return boss_route
        tile = self.tiles.setdefault(self.position, Tile())
        untried = [
            direction
            for direction in DIRECTIONS
            if direction in tile.exits and direction not in tile.tried
        ]
        anchor = (
            self.tiles.get(self.teleporter_position)
            if self.teleporter_position is not None
            else None
        )
        anchor_has_untried = anchor is not None and any(
            direction in anchor.exits and direction not in anchor.tried
            for direction in DIRECTIONS
        )
        if (
            len(untried) >= 2
            and can_place
            and self.teleporter_position != self.position
            and not anchor_has_untried
        ):
            return "place_teleporter"

        if untried:
            return self.choose_exit()

        if (
            can_return
            and self.teleporter_position is not None
            and self.teleporter_position != self.position
            and anchor_has_untried
        ):
            return "return_teleporter"
        walking = self._frontier_route(self.position)
        return walking[1] if walking is not None else None

    def place_teleporter(self) -> None:
        self.teleporter_position = self.position

    def return_to_teleporter(self) -> bool:
        if self.teleporter_position is None:
            return False
        self.position = self.teleporter_position
        return True

    def maximum_possible_depth(self, direction: str) -> int:
        """Upper-bound a branch's depth by its reachable unexplored grid cells.

        Unknown passages may ultimately be walls, so the most conservative
        assumption is that every adjacent unknown cell connects. Known rooms
        form barriers; this lets edges and previously explored rooms constrain
        a branch before its actual layout is visible.
        """
        if direction not in DIRECTIONS:
            raise ValueError(f"Unknown maze direction: {direction}")
        dx, dy = DIRECTIONS[direction]
        start = (self.position[0] + dx, self.position[1] + dy)
        if not (0 <= start[0] < 10 and 0 <= start[1] < 10):
            return 0
        if start in self.tiles:
            return 0

        known = set(self.tiles)
        reachable = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for step_x, step_y in DIRECTIONS.values():
                neighbor = (x + step_x, y + step_y)
                if not (0 <= neighbor[0] < 10 and 0 <= neighbor[1] < 10):
                    continue
                if neighbor in known or neighbor in reachable:
                    continue
                reachable.add(neighbor)
                queue.append(neighbor)
        return len(reachable)

    def recommendation(self) -> str | None:
        return self.choose_exit()

    def moved(self, direction: str) -> bool:
        current = self.tiles.setdefault(self.position, Tile())
        dx, dy = DIRECTIONS[direction]
        next_position = (self.position[0] + dx, self.position[1] + dy)
        if not (0 <= next_position[0] < 10 and 0 <= next_position[1] < 10):
            return False
        current.tried.add(direction)
        self.position = next_position
        self.tiles.setdefault(next_position, Tile()).tried.add(OPPOSITE[direction])
        return True
