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
    """A small DFS map for randomized mazes."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.position = (0, 0)
        self.tiles: dict[tuple[int, int], Tile] = {}

    def observe(self, exits: set[str], tile_type: str = "unknown") -> None:
        tile = self.tiles.setdefault(self.position, Tile())
        if not tile.exits_confirmed:
            tile.exits.update(exits)
        tile.tile_type = tile_type

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
        for direction in ("north", "east", "south", "west"):
            if direction in tile.exits and direction not in tile.tried:
                return direction

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
