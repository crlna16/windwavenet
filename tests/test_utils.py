# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import torch

from src.utils.utils import get_bin_mask, get_metric_value, circular_mae, circular_rmse


class TestGetBinMask:
    """Test suite for get_bin_mask utility function."""

    def test_first_bin_numpy(self):
        """Test first bin mask (values < bins[0]) with numpy arrays."""
        values = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        bins = [0.5, 1.0, 1.5]

        mask = get_bin_mask(values, bins, bin_idx=0)

        expected = np.array([True, False, False, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_first_bin_torch(self):
        """Test first bin mask with torch tensors."""
        values = torch.tensor([0.1, 0.5, 1.0, 1.5, 2.0])
        bins = [0.5, 1.0, 1.5]

        mask = get_bin_mask(values, bins, bin_idx=0)

        expected = torch.tensor([True, False, False, False, False])
        assert torch.equal(mask, expected)

    def test_middle_bin_numpy(self):
        """Test middle bin mask (bins[i-1] <= values < bins[i]) with numpy."""
        values = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        bins = [0.5, 1.0, 1.5]

        # Bin 1: [0.5, 1.0)
        mask = get_bin_mask(values, bins, bin_idx=1)
        expected = np.array([False, True, False, False, False])
        np.testing.assert_array_equal(mask, expected)

        # Bin 2: [1.0, 1.5)
        mask = get_bin_mask(values, bins, bin_idx=2)
        expected = np.array([False, False, True, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_last_bin_numpy(self):
        """Test last bin mask (values >= bins[-1]) with numpy."""
        values = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
        bins = [0.5, 1.0, 1.5]

        mask = get_bin_mask(values, bins, bin_idx=3)  # Last bin

        expected = np.array([False, False, False, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_last_bin_torch(self):
        """Test last bin mask with torch tensors."""
        values = torch.tensor([0.1, 0.5, 1.0, 1.5, 2.0])
        bins = [0.5, 1.0, 1.5]

        mask = get_bin_mask(values, bins, bin_idx=3)

        expected = torch.tensor([False, False, False, True, True])
        assert torch.equal(mask, expected)

    def test_single_bin_edge(self):
        """Test with a single bin edge."""
        values = np.array([0.0, 0.5, 1.0])
        bins = [0.5]

        # Bin 0: < 0.5
        mask0 = get_bin_mask(values, bins, bin_idx=0)
        np.testing.assert_array_equal(mask0, np.array([True, False, False]))

        # Bin 1: >= 0.5
        mask1 = get_bin_mask(values, bins, bin_idx=1)
        np.testing.assert_array_equal(mask1, np.array([False, True, True]))

    def test_negative_values(self):
        """Test with negative values and bins."""
        values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        bins = [-1.0, 0.0, 1.0]

        # Bin 0: < -1.0
        mask = get_bin_mask(values, bins, bin_idx=0)
        np.testing.assert_array_equal(mask, np.array([True, False, False, False, False]))

        # Bin 1: [-1.0, 0.0)
        mask = get_bin_mask(values, bins, bin_idx=1)
        np.testing.assert_array_equal(mask, np.array([False, True, False, False, False]))

    def test_all_values_in_one_bin(self):
        """Test when all values fall into one bin."""
        values = np.array([5.0, 6.0, 7.0, 8.0])
        bins = [0.0, 1.0, 2.0]

        # All values should be in the last bin (>= 2.0)
        mask = get_bin_mask(values, bins, bin_idx=3)
        np.testing.assert_array_equal(mask, np.array([True, True, True, True]))


class TestGetMetricValue:
    """Test suite for get_metric_value utility function."""

    def test_get_existing_metric(self):
        """Test retrieving an existing metric."""
        metric_dict = {"loss": torch.tensor(0.5), "accuracy": torch.tensor(0.95)}

        value = get_metric_value(metric_dict, "loss")
        assert value == pytest.approx(0.5)

    def test_get_none_metric_name(self):
        """Test that None metric name returns None."""
        metric_dict = {"loss": torch.tensor(0.5)}

        value = get_metric_value(metric_dict, None)
        assert value is None

    def test_get_missing_metric_raises(self):
        """Test that missing metric raises Exception."""
        metric_dict = {"loss": torch.tensor(0.5)}

        with pytest.raises(Exception, match="Metric value not found"):
            get_metric_value(metric_dict, "accuracy")

    def test_get_metric_converts_to_float(self):
        """Test that metric value is converted to float."""
        metric_dict = {"val/loss": torch.tensor(0.123456789)}

        value = get_metric_value(metric_dict, "val/loss")
        assert isinstance(value, float)
        assert value == pytest.approx(0.123456789)


class TestCircularMetrics:
    """Test suite for circular_mae and circular_rmse."""

    def test_circular_mae_no_wraparound(self):
        """Standard angular difference without wraparound."""
        pred = np.array([90.0])
        truth = np.array([80.0])
        assert circular_mae(pred, truth) == pytest.approx(10.0)

    def test_circular_mae_wraparound(self):
        """1° vs 359° should give MAE of 2°, not 358°."""
        pred = np.array([1.0])
        truth = np.array([359.0])
        assert circular_mae(pred, truth) == pytest.approx(2.0)

    def test_circular_mae_zero_is_360(self):
        """0° and 360° are the same direction."""
        pred = np.array([0.0])
        truth = np.array([360.0])
        assert circular_mae(pred, truth) == pytest.approx(0.0, abs=1e-5)

    def test_circular_mae_180_degrees(self):
        """Exactly opposite directions give MAE of 180°."""
        pred = np.array([0.0])
        truth = np.array([180.0])
        assert circular_mae(pred, truth) == pytest.approx(180.0)

    def test_circular_mae_perfect_prediction(self):
        """Perfect prediction gives zero error."""
        angles = np.array([0.0, 45.0, 90.0, 180.0, 270.0, 359.0])
        assert circular_mae(angles, angles) == pytest.approx(0.0, abs=1e-5)

    def test_circular_rmse_wraparound(self):
        """1° vs 359° should give RMSE of 2°, not 358°."""
        pred = np.array([1.0])
        truth = np.array([359.0])
        assert circular_rmse(pred, truth) == pytest.approx(2.0)

    def test_circular_rmse_multiple_samples(self):
        """RMSE over multiple samples with one wraparound case."""
        # errors: 10°, 2° (via wraparound)  →  RMSE = sqrt((100 + 4) / 2) = sqrt(52)
        pred = np.array([90.0, 1.0])
        truth = np.array([80.0, 359.0])
        expected = np.sqrt((10.0**2 + 2.0**2) / 2)
        assert circular_rmse(pred, truth) == pytest.approx(expected)
