import tempfile
import unittest
from pathlib import Path

from bot.config import Config
from bot.models import Region


class ZoneEnablementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.config = Config(Path(self.temporary.name) / "config.json")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_disabled_click_zone_is_saved_but_not_active(self) -> None:
        for index in range(3):
            self.config.add_click_zone("combat", Region(index * 10, 0, 5, 5))
            self.config.rename_click_zone("combat", index, f"action_{index}")

        self.config.set_click_zone_enabled("combat", 1, False)

        self.assertEqual(3, len(self.config.click_zones["combat"]))
        self.assertEqual(2, len(self.config.active_click_zones["combat"]))
        self.assertEqual(
            "action_2", self.config.active_click_zone_name("combat", 1)
        )

    def test_disabled_visual_region_is_saved_but_not_active(self) -> None:
        self.config.set_visual_region(
            "combat", "enemy_health", Region(0, 0, 10, 10)
        )
        self.config.set_visual_region(
            "combat", "unused_detail", Region(10, 0, 10, 10)
        )

        self.config.set_visual_region_enabled("combat", "unused_detail", False)

        self.assertIn(
            "unused_detail", self.config.all_visual_regions["combat"]
        )
        self.assertNotIn("unused_detail", self.config.visual_regions["combat"])


if __name__ == "__main__":
    unittest.main()
