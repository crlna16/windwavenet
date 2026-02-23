# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from src.data.BaseWWMDataModule import BaseWWMDataModule
from src.data.SequenceWWMDataModule import SequenceWWMDataModule
from src.data.WWMDataModule import WWMDataModule


class TestWWMDataModule:
    """Test suite for WWMDataModule."""

    @pytest.fixture
    def cicd_data_path(self):
        """Return path to CICD test data."""
        return Path(__file__).parent / "cicd_data" / "cicd.zarr"

    @pytest.fixture
    def node_idx_csv_path(self):
        """Return path to node index CSV."""
        return Path(__file__).parent / "cicd_data" / "node_idx.csv"

    def test_init(self, cicd_data_path):
        """Test basic initialization."""
        dm = WWMDataModule(
            data_dir=str(cicd_data_path),
            train_days=[1, 2],
            val_days=[2, 3],
            test_days=[3, 4],
            features=["windSpeedX", "windSpeedY"],
            targets=["sigWaveHeight"],
            batch_size=32,
            num_workers=0,
        )
        assert dm.batch_size == 32
        assert dm.features == ["windSpeedX", "windSpeedY"]
        assert dm.targets == ["sigWaveHeight"]

    def test_prepare_data_raises_on_missing_dir(self, tmp_path):
        """Test that prepare_data raises error for missing directory."""
        dm = WWMDataModule(
            data_dir=str(tmp_path / "nonexistent"),
            train_days=[1, 2],
            val_days=[2, 3],
            test_days=[3, 4],
            features=["windSpeedX"],
            targets=["sigWaveHeight"],
        )
        with pytest.raises(FileNotFoundError):
            dm.prepare_data()

    def test_setup_fit_creates_datasets(self, cicd_data_path, node_idx_csv_path):
        """Test that setup creates train, val, and test datasets."""
        dm = WWMDataModule(
            data_dir=str(cicd_data_path),
            train_days=[1, 2],
            val_days=[2, 3],
            test_days=[3, 4],
            features=["windSpeedX", "windSpeedY"],
            targets=["sigWaveHeight"],
            node_idx_csv=str(node_idx_csv_path),
            batch_size=32,
            num_workers=0,
        )
        dm.setup(stage="fit")

        assert dm.ds_train is not None
        assert dm.ds_val is not None
        assert dm.ds_test is not None

    def test_setup_test_creates_dataset(self, cicd_data_path):
        """Test that setup with stage='test' creates test dataset."""
        dm = WWMDataModule(
            data_dir=str(cicd_data_path),
            train_days=[1, 2],
            val_days=[2, 3],
            test_days=[3, 4],
            features=["windSpeedX", "windSpeedY"],
            targets=["sigWaveHeight"],
            batch_size=32,
            num_workers=0,
        )
        dm.setup(stage="test")

        assert dm.ds_test is not None
        # test uses all nodes (no node_idx_csv)
        assert dm.ds_train is None

    def test_setup_predict_creates_dataset(self, cicd_data_path):
        """Test that setup with stage='predict' creates predict dataset."""
        dm = WWMDataModule(
            data_dir=str(cicd_data_path),
            train_days=[1, 2],
            val_days=[2, 3],
            test_days=[3, 4],
            features=["windSpeedX", "windSpeedY"],
            targets=["sigWaveHeight"],
            batch_size=32,
            num_workers=0,
        )
        dm.setup(stage="predict")

        assert dm.ds_pred is not None

    def test_setup_invalid_stage_raises(self, cicd_data_path):
        """Test that setup with invalid stage raises ValueError."""
        dm = WWMDataModule(
            data_dir=str(cicd_data_path),
            train_days=[1, 2],
            val_days=[2, 3],
            test_days=[3, 4],
            features=["windSpeedX", "windSpeedY"],
            targets=["sigWaveHeight"],
        )
        with pytest.raises(ValueError):
            dm.setup(stage="invalid")

    def test_setup_validate_not_implemented(self, cicd_data_path):
        """Test that setup with stage='validate' raises NotImplementedError."""
        dm = WWMDataModule(
            data_dir=str(cicd_data_path),
            train_days=[1, 2],
            val_days=[2, 3],
            test_days=[3, 4],
            features=["windSpeedX", "windSpeedY"],
            targets=["sigWaveHeight"],
        )
        with pytest.raises(NotImplementedError):
            dm.setup(stage="validate")

    def test_dataloaders(self, cicd_data_path, node_idx_csv_path):
        """Test that dataloaders are created correctly."""
        dm = WWMDataModule(
            data_dir=str(cicd_data_path),
            train_days=[1, 2],
            val_days=[2, 3],
            test_days=[3, 4],
            features=["windSpeedX", "windSpeedY"],
            targets=["sigWaveHeight"],
            node_idx_csv=str(node_idx_csv_path),
            batch_size=32,
            num_workers=0,
        )
        dm.setup(stage="fit")

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Check batch size
        assert train_loader.batch_size == 32
        assert val_loader.batch_size == 128  # 4 * batch_size
        assert test_loader.batch_size == 128

    def test_predict_dataloader(self, cicd_data_path):
        """Test that predict dataloader is created correctly."""
        dm = WWMDataModule(
            data_dir=str(cicd_data_path),
            train_days=[1, 2],
            val_days=[2, 3],
            test_days=[3, 4],
            features=["windSpeedX", "windSpeedY"],
            targets=["sigWaveHeight"],
            batch_size=32,
            num_workers=0,
        )
        dm.setup(stage="predict")
        pred_loader = dm.predict_dataloader()

        assert pred_loader is not None
        assert pred_loader.batch_size == 32


class TestSequenceWWMDataModule:
    """Test suite for SequenceWWMDataModule."""

    @pytest.fixture
    def cicd_data_path(self):
        """Return path to CICD test data."""
        return Path(__file__).parent / "cicd_data" / "cicd.zarr"

    @pytest.fixture
    def node_idx_csv_path(self):
        """Return path to node index CSV."""
        return Path(__file__).parent / "cicd_data" / "node_idx.csv"

    def test_init(self, cicd_data_path):
        """Test basic initialization with seq_len."""
        dm = SequenceWWMDataModule(
            data_dir=str(cicd_data_path),
            train_days=[1, 2],
            val_days=[2, 3],
            test_days=[3, 4],
            features=["windSpeedX", "windSpeedY"],
            targets=["sigWaveHeight"],
            seq_len=5,
            batch_size=32,
            num_workers=0,
        )
        assert dm.seq_len == 5
        assert dm.batch_size == 32

    def test_setup_fit_creates_sequence_datasets(self, cicd_data_path, node_idx_csv_path):
        """Test that setup creates sequence datasets."""
        dm = SequenceWWMDataModule(
            data_dir=str(cicd_data_path),
            train_days=[1, 2],
            val_days=[2, 3],
            test_days=[3, 4],
            features=["windSpeedX", "windSpeedY"],
            targets=["sigWaveHeight"],
            seq_len=3,
            node_idx_csv=str(node_idx_csv_path),
            batch_size=32,
            num_workers=0,
        )
        dm.setup(stage="fit")

        assert dm.ds_train is not None
        assert dm.ds_val is not None
        assert dm.ds_test is not None

        # Check that datasets have seq_len attribute
        assert hasattr(dm.ds_train, "seq_len")
        assert dm.ds_train.seq_len == 3

    def test_dataloaders_return_sequences(self, cicd_data_path, node_idx_csv_path):
        """Test that dataloaders return proper sequence tensors."""
        dm = SequenceWWMDataModule(
            data_dir=str(cicd_data_path),
            train_days=[1, 2],
            val_days=[2, 3],
            test_days=[3, 4],
            features=["windSpeedX", "windSpeedY"],
            targets=["sigWaveHeight"],
            seq_len=3,
            node_idx_csv=str(node_idx_csv_path),
            batch_size=4,
            num_workers=0,
        )
        dm.setup(stage="fit")

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        X, y = batch
        # X should be (batch, seq_len, n_features)
        assert len(X.shape) == 3
        assert X.shape[1] == 3  # seq_len
        # y should be (batch, n_targets)
        assert len(y.shape) == 2


class TestAbstractBaseClass:
    """Test that BaseWWMDataModule is properly abstract."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseWWMDataModule cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseWWMDataModule(
                data_dir="/tmp/data",
                train_days=[1, 2],
                val_days=[2, 3],
                test_days=[3, 4],
                features=["windSpeedX"],
                targets=["sigWaveHeight"],
            )
