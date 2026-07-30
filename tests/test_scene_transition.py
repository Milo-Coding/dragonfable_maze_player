import unittest

import numpy as np

from bot.maze import MazeMemory
from bot.vision import SceneTransitionDetector, WholeSceneMotionDetector


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


if __name__ == "__main__":
    unittest.main()
