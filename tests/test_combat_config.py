import tempfile
import unittest
import json
from pathlib import Path

from combat_bot.config import CombatConfig


class CombatBindingTests(unittest.TestCase):
    def test_thumb_buttons_are_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = CombatConfig(Path(temporary) / "combat.json")
            self.assertEqual("mouse:x1", config.data["mode_bindings"]["training"])
            self.assertEqual("mouse:x2", config.data["mode_bindings"]["live"])

    def test_binding_cannot_be_assigned_to_both_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = CombatConfig(Path(temporary) / "combat.json")
            config.set_mode_binding("training", "mouse:x2")
            self.assertIsNone(config.data["mode_bindings"]["live"])


class CombatTaskTests(unittest.TestCase):
    def test_task_order_is_priority_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = CombatConfig(Path(temporary) / "combat.json")
            low = config.add_task("low", "simple")
            high = config.add_task("high", "combat")
            config.move_task(high, -1)
            self.assertEqual([high, low], [task["id"] for task in config.tasks])

    def test_deleting_scan_target_disables_dependent_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = CombatConfig(Path(temporary) / "combat.json")
            target = config.add_scan_target("treasure")
            task = config.add_task("find treasure", f"scanning:{target}")
            config.delete_scan_target(target)
            self.assertFalse(config.task(task)["enabled"])

    def test_single_legacy_anchor_migrates_to_anchor_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "combat.json"
            path.write_text(json.dumps({"tasks": [{
                "id": "task1", "name": "old", "behavior": "combat", "enabled": True,
                "anchor_region": {"left": 1, "top": 2, "width": 3, "height": 4},
                "anchor_template": "old.png", "simple_point": None,
            }]}), encoding="utf-8")
            config = CombatConfig(path)
            task = config.task("task1")
            self.assertEqual(1, len(task["anchors"]))
            self.assertEqual("old.png", task["anchors"][0]["template"])
            self.assertNotIn("anchor_region", task)

    def test_task_accepts_multiple_anchors(self) -> None:
        from bot.models import Region
        with tempfile.TemporaryDirectory() as temporary:
            config = CombatConfig(Path(temporary) / "combat.json")
            task_id = config.add_task("multi", "simple")
            config.add_task_anchor(task_id, Region(0, 0, 10, 10), "one.png")
            config.add_task_anchor(task_id, Region(20, 20, 10, 10), "two.png")
            self.assertEqual(2, len(config.task(task_id)["anchors"]))

    def test_first_matching_task_wins_priority(self) -> None:
        import numpy as np
        from combat_bot.controller import CombatController
        with tempfile.TemporaryDirectory() as temporary:
            config = CombatConfig(Path(temporary) / "combat.json")
            first = config.add_task("first", "simple")
            config.add_task("second", "simple")
            controller = CombatController.__new__(CombatController)
            controller.config = config
            controller._anchor_score = lambda _frame, _task: 0.99
            active, _score = controller._active_task(np.zeros((2, 2, 3), dtype=np.uint8))
            self.assertEqual(first, active["id"])


if __name__ == "__main__":
    unittest.main()
