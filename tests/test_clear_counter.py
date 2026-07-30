import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from bot.config import Config
from bot.controller import BotController


class ClearCounterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = BotController.__new__(BotController)
        self.controller.clears = 0
        self.controller._quest_in_progress = False
        self.controller.config = Mock()
        self.controller.config.data = {}

    def test_counts_return_to_lobby_after_exploration(self) -> None:
        self.controller._track_clear("exploring")
        self.controller._track_clear("combat")
        self.controller._track_clear("lobby")

        self.assertEqual(1, self.controller.clears)
        self.assertEqual(1, self.controller.config.data["clears"])
        self.controller.config.save.assert_called_once_with()

    def test_repeated_lobby_frames_do_not_add_clears(self) -> None:
        self.controller._track_clear("exploring")
        self.controller._track_clear("lobby")
        self.controller._track_clear("lobby")

        self.assertEqual(1, self.controller.clears)

    def test_lobby_at_startup_is_not_a_clear(self) -> None:
        self.controller._track_clear("lobby")

        self.assertEqual(0, self.controller.clears)
        self.controller.config.save.assert_not_called()

    def test_clear_count_survives_config_reload(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = Config(path)
            self.controller.config = config
            self.controller._track_clear("combat")
            self.controller._track_clear("lobby")

            reloaded = Config(path)

        self.assertEqual(1, reloaded.data["clears"])


if __name__ == "__main__":
    unittest.main()
