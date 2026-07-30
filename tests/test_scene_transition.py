import unittest

import numpy as np

from bot.maze import MazeMemory, Tile
from bot.vision import (
    PlayerMovementTracker,
    SceneTransitionDetector,
    WholeSceneMotionDetector,
)


def regions(value: float = 0.0) -> dict[str, np.ndarray]:
    return {
        direction: np.full((8, 8), value, dtype=np.float32)
        for direction in ("north", "east", "south", "west")
    }


class SceneTransitionDetectorTests(unittest.TestCase):
    def test_requires_changed_scene_then_stability(self) -> None:
        detector = SceneTransitionDetector(0.10, 0.02, 3, 2)
        detector.reset(regions())
        changed = regions(0.5)

        self.assertFalse(detector.update(changed))
        self.assertFalse(detector.update(changed))
        self.assertFalse(detector.update(changed))
        self.assertTrue(detector.update(changed))

    def test_ignores_localized_animation(self) -> None:
        detector = SceneTransitionDetector(0.10, 0.02, 2, 2)
        detector.reset(regions())

        for index in range(8):
            animated = regions()
            if index % 2:
                animated["north"].fill(0.8)
            self.assertFalse(detector.update(animated))

    def test_ignores_unchanged_stable_room(self) -> None:
        detector = SceneTransitionDetector(0.10, 0.02, 2, 2)
        origin = regions()
        detector.reset(origin)

        for _ in range(5):
            self.assertFalse(detector.update(origin))

    def test_detects_coordinated_subtle_scene_change(self) -> None:
        detector = SceneTransitionDetector(0.02, 0.01, 3, 2)
        detector.reset(regions())
        similar_room = regions(0.008)

        self.assertFalse(detector.update(similar_room))
        self.assertFalse(detector.update(similar_room))
        self.assertFalse(detector.update(similar_room))
        self.assertTrue(detector.update(similar_room))
        self.assertEqual("coordinated subtle", detector.change_mode)

    def test_single_region_change_requires_extra_stability(self) -> None:
        detector = SceneTransitionDetector(0.02, 0.01, 3, 2)
        detector.reset(regions())
        similar_room = regions()
        similar_room["east"].fill(0.10)

        for _ in range(6):
            self.assertFalse(detector.update(similar_room))
        self.assertTrue(detector.update(similar_room))
        self.assertEqual("single-region", detector.change_mode)


class WholeSceneMotionDetectorTests(unittest.TestCase):
    def test_detects_transition_that_returns_to_identical_scene(self) -> None:
        detector = WholeSceneMotionDetector(0.02, 0.005, 3)
        origin = np.zeros((20, 30), dtype=np.float32)
        detector.reset(origin)

        self.assertFalse(detector.update(np.full_like(origin, 0.5)))
        self.assertFalse(detector.update(np.full_like(origin, 0.8)))
        self.assertFalse(detector.update(origin))
        self.assertFalse(detector.update(origin))
        self.assertFalse(detector.update(origin))
        self.assertTrue(detector.update(origin))

    def test_requires_motion_before_stability(self) -> None:
        detector = WholeSceneMotionDetector(0.02, 0.005, 2)
        origin = np.zeros((20, 30), dtype=np.float32)
        detector.reset(origin)

        for _ in range(6):
            self.assertFalse(detector.update(origin))


class MazeMemoryTests(unittest.TestCase):
    def test_successful_move_creates_destination_tile(self) -> None:
        maze = MazeMemory()
        self.assertTrue(maze.moved("east"))
        self.assertEqual((1, 0), maze.position)
        self.assertIn((1, 0), maze.tiles)
        self.assertIn("west", maze.tiles[(1, 0)].tried)

    def test_out_of_bounds_move_does_not_mark_exit_tried(self) -> None:
        maze = MazeMemory()
        self.assertFalse(maze.moved("north"))
        self.assertNotIn("north", maze.tiles[(0, 0)].tried)

    def test_branch_prefers_smaller_reachable_unexplored_region(self) -> None:
        maze = MazeMemory()
        maze.set_position(1, 1)
        maze.observe({"north", "east"})
        # Known tiles form a wall that traps the north branch in one cell.
        for position in ((0, 0), (2, 0)):
            maze.set_position(*position)
        maze.set_position(1, 1)

        self.assertEqual(1, maze.maximum_possible_depth("north"))
        self.assertGreater(maze.maximum_possible_depth("east"), 1)
        self.assertEqual("north", maze.recommendation())

    def test_equal_branch_depth_preserves_direction_order(self) -> None:
        maze = MazeMemory()
        maze.set_position(5, 5)
        maze.observe({"east", "south"})

        self.assertEqual("east", maze.recommendation())

    def test_places_teleporter_at_branch(self) -> None:
        maze = MazeMemory()
        maze.set_position(5, 5)
        maze.observe({"east", "south"})

        self.assertEqual(
            "place_teleporter",
            maze.recommended_action(can_place=True, can_return=True),
        )
        maze.place_teleporter()
        self.assertEqual(
            "east", maze.recommended_action(can_place=True, can_return=True)
        )

    def test_returns_to_unfinished_anchored_branch(self) -> None:
        maze = MazeMemory()
        maze.tiles = {
            (0, 0): Tile(exits={"east", "south"}, tried={"east"}),
            (1, 0): Tile(exits={"west"}, tried={"west"}),
        }
        maze.position = (1, 0)
        maze.teleporter_position = (0, 0)

        self.assertEqual(
            "return_teleporter",
            maze.recommended_action(can_place=True, can_return=True),
        )
        self.assertTrue(maze.return_to_teleporter())
        self.assertEqual((0, 0), maze.position)

    def test_does_not_use_teleporter_for_unrelated_frontier(self) -> None:
        maze = MazeMemory()
        maze.tiles = {
            (0, 0): Tile(exits={"east"}, tried={"east"}),
            (1, 0): Tile(exits={"east", "west"}, tried={"east", "west"}),
            (2, 0): Tile(exits={"west", "south"}, tried={"west"}),
        }
        maze.position = (0, 0)
        maze.teleporter_position = (1, 0)

        self.assertEqual(
            "east", maze.recommended_action(can_place=True, can_return=True)
        )

    def test_known_boss_tile_overrides_teleporter_and_frontiers(self) -> None:
        maze = MazeMemory()
        maze.tiles = {
            (0, 0): Tile(exits={"east", "south"}, tried={"east"}),
            (1, 0): Tile(exits={"east", "west"}, tried={"east", "west"}),
            (2, 0): Tile(exits={"west"}, tried={"west"}, tile_type="boss"),
        }
        maze.position = (0, 0)
        maze.teleporter_position = (1, 0)
        maze.boss_position = (2, 0)

        self.assertEqual(
            "east", maze.recommended_action(can_place=True, can_return=True)
        )

    def test_observing_boss_persists_its_coordinate(self) -> None:
        maze = MazeMemory()
        maze.set_position(7, 4)
        maze.observe({"west"}, tile_type="boss")

        self.assertEqual((7, 4), maze.boss_position)
        maze.set_position(0, 0)
        self.assertEqual((7, 4), maze.boss_position)

    def test_teleporter_stays_disabled_at_known_boss_tile(self) -> None:
        maze = MazeMemory()
        maze.set_position(5, 5)
        maze.observe({"east", "south"}, tile_type="boss")
        maze.teleporter_position = (1, 1)

        self.assertIsNone(
            maze.recommended_action(can_place=True, can_return=True)
        )


class PlayerMovementTrackerTests(unittest.TestCase):
    def test_requires_movement_then_stable_position(self) -> None:
        tracker = PlayerMovementTracker(5.0, 3)
        tracker.reset((100, 100))

        self.assertFalse(tracker.update((110, 100)))
        self.assertFalse(tracker.update((120, 100)))
        self.assertFalse(tracker.update((120, 100)))
        self.assertFalse(tracker.update((120, 100)))
        self.assertTrue(tracker.update((120, 100)))

    def test_confirms_stable_position_without_prior_movement(self) -> None:
        tracker = PlayerMovementTracker(5.0, 2)
        tracker.reset((100, 100))

        self.assertFalse(tracker.update((100, 100)))
        self.assertTrue(tracker.update((100, 100)))

    def test_detects_slow_cumulative_movement(self) -> None:
        tracker = PlayerMovementTracker(5.0, 2)
        tracker.reset((100, 100))

        self.assertFalse(tracker.update((103, 100)))
        self.assertFalse(tracker.update((106, 100)))
        self.assertFalse(tracker.update((106, 100)))
        self.assertTrue(tracker.update((106, 100)))


if __name__ == "__main__":
    unittest.main()
