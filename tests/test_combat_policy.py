import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from bot.models import Region
from combat_bot.policy import (
    DensePolicy,
    FEATURES_PER_CELL,
    FEATURES_PER_ZONE,
    GRID_SIZE,
    HUE_BINS,
    SATURATION_BINS,
    visual_features,
)


class CombatFeatureTests(unittest.TestCase):
    def test_each_spatial_cell_has_three_one_hot_color_features(self) -> None:
        frame = np.zeros((20, 40, 3), dtype=np.uint8)
        frame[:, :20] = (0, 0, 255)
        frame[:, 20:] = (0, 255, 0)
        features = visual_features(
            frame, [Region(0, 0, 20, 20), Region(20, 0, 20, 20)]
        )
        self.assertEqual(2 * FEATURES_PER_ZONE, len(features))
        for group in features.reshape(2, FEATURES_PER_ZONE):
            for cell in group.reshape(GRID_SIZE * GRID_SIZE, FEATURES_PER_CELL):
                self.assertEqual(1.0, cell[:HUE_BINS].sum())
                self.assertEqual(1.0, cell[HUE_BINS:HUE_BINS + SATURATION_BINS].sum())
                self.assertEqual(1.0, cell[HUE_BINS + SATURATION_BINS:].sum())

    def test_red_and_green_use_different_hue_categories(self) -> None:
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        frame[:, :10] = (0, 0, 255)
        frame[:, 10:] = (0, 255, 0)
        values = visual_features(frame, [Region(0, 0, 10, 10), Region(10, 0, 10, 10)])
        groups = values.reshape(2, FEATURES_PER_ZONE)
        self.assertNotEqual(np.argmax(groups[0, :HUE_BINS]), np.argmax(groups[1, :HUE_BINS]))


class DensePolicyTests(unittest.TestCase):
    def test_training_increases_demonstrated_action_probability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            policy = DensePolicy(Path(temporary) / "policy.json")
            policy.ensure(["health"], ["attack", "heal"], [8, 4])
            inputs = np.zeros(FEATURES_PER_ZONE, dtype=np.float32)
            inputs[[0, HUE_BINS, HUE_BINS + SATURATION_BINS]] = 1
            before = float(policy.probabilities(inputs)[1])
            for _ in range(20):
                policy.train(inputs, actual=1, predicted=0, learning_rate=0.05)
            self.assertGreater(float(policy.probabilities(inputs)[1]), before)

    def test_relevance_mask_freezes_unselected_zone_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            policy = DensePolicy(Path(temporary) / "policy.json")
            policy.ensure(["health", "cooldown"], ["attack", "heal"], [8, 4])
            inputs = np.zeros(2 * FEATURES_PER_ZONE, dtype=np.float32)
            inputs[0] = inputs[FEATURES_PER_ZONE] = 1.0
            selected_before = [weight.copy() for weight in policy.branch_weights[0]]
            ignored_before = [weight.copy() for weight in policy.branch_weights[1]]
            policy.train(inputs, actual=1, predicted=0, learning_rate=0.05,
                         zone_mask=np.asarray([True, False]))
            self.assertTrue(any(not np.array_equal(old, new) for old, new in
                                zip(selected_before, policy.branch_weights[0])))
            self.assertTrue(all(np.array_equal(old, new) for old, new in
                               zip(ignored_before, policy.branch_weights[1])))

    def test_zone_importance_ranks_stronger_action_margin_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            policy = DensePolicy(Path(temporary) / "policy.json")
            policy.ensure(["health", "cooldown"], ["attack", "heal"], [2])
            for zone in range(2):
                policy.branch_weights[zone][0].fill(0.0)
                policy.branch_biases[zone][0].fill(1.0)
            policy.branch_weights[0][1][:] = [[3.0, 3.0], [0.0, 0.0]]
            policy.branch_weights[1][1][:] = [[0.5, 0.5], [0.0, 0.0]]
            importance = policy.zone_importance(
                np.zeros(2 * FEATURES_PER_ZONE, dtype=np.float32), action=0
            )
            self.assertEqual((2,), importance.shape)
            self.assertGreater(importance[0], importance[1])


if __name__ == "__main__":
    unittest.main()
