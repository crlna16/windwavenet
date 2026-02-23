# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
import torch

from src.utils.utils import get_bin_mask, get_metric_value


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
