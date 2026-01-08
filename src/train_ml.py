# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Training script for classical ML models (XGBoost, RandomForest, etc.)."""

from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import numpy as np
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig

from src.utils.pylogger import RankedLogger
from src.utils.utils import print_config_tree, task_wrapper

# Initialize rank for single-process execution to use RankedLogger
rank_zero_only.rank = 0

log = RankedLogger(__name__, rank_zero_only=True)


def extract_data_from_datamodule(datamodule, stage: str = "fit"):
    """Extract numpy arrays from Lightning DataModule.

    Args:
        datamodule: Lightning DataModule
        stage: Stage to setup ('fit', 'test', etc.)

    Returns:
        Tuple of (X_train, y_train, weights_train, X_val, y_val, weights_val, X_test, y_test, weights_test)
    """
    datamodule.setup(stage=stage)

    # Extract training data
    train_dataset = datamodule.ds_train
    X_train = train_dataset.X.cpu().numpy()
    y_train = train_dataset.y.cpu().numpy()
    weights_train = (
        train_dataset.weights.cpu().numpy()
        if train_dataset.weights is not None
        else None
    )

    # Extract validation data
    val_dataset = datamodule.ds_val
    X_val = val_dataset.X.cpu().numpy()
    y_val = val_dataset.y.cpu().numpy()
    weights_val = (
        val_dataset.weights.cpu().numpy() if val_dataset.weights is not None else None
    )

    # Extract test data
    test_dataset = datamodule.ds_test
    X_test = test_dataset.X.cpu().numpy()
    y_test = test_dataset.y.cpu().numpy()
    weights_test = (
        test_dataset.weights.cpu().numpy() if test_dataset.weights is not None else None
    )

    log.info(
        f"Extracted data - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    return (
        X_train,
        y_train,
        weights_train,
        X_val,
        y_val,
        weights_val,
        X_test,
        y_test,
        weights_test,
    )


@task_wrapper
def train_ml(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train a classical ML model.

    Args:
        cfg: Hydra configuration

    Returns:
        Tuple of (metrics_dict, objects_dict)
    """
    # Set random seed
    if cfg.get("seed"):
        np.random.seed(cfg.seed)
        log.info(f"Random seed set to {cfg.seed}")

    log.info("Printing config tree")
    print_config_tree(cfg, resolve=True, save_to_file=True)

    # Instantiate datamodule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    # Extract data
    log.info("Extracting data from datamodule...")
    (
        X_train,
        y_train,
        weights_train,
        X_val,
        y_val,
        weights_val,
        X_test,
        y_test,
        weights_test,
    ) = extract_data_from_datamodule(datamodule, stage="fit")

    # Instantiate ML model
    log.info(f"Instantiating ML model <{cfg.model._target_}>")
    ml_model = hydra.utils.instantiate(cfg.model)

    # Flatten sample weights if needed (most ML models expect 1D weights)
    if weights_train is not None and weights_train.ndim > 1:
        # Use weights from first target column (assuming same weights for all targets)
        weights_train = weights_train[:, 0]
        log.info("Flattened sample weights for training")

    # Train model
    if cfg.get("train"):
        log.info("Starting training...")
        train_metrics = ml_model.fit(
            X_train, y_train, X_val=X_val, y_val=y_val, sample_weight=weights_train
        )

        # Log training metrics
        log.info("Training metrics:")
        for key, value in train_metrics.items():
            log.info(f"  {key}: {value:.6f}")

    # Test model
    test_metrics = {}
    if cfg.get("test"):
        log.info("Starting testing...")
        test_metrics = ml_model.evaluate(X_test, y_test, prefix="test")

        # Log test metrics
        log.info("Test metrics:")
        for key, value in test_metrics.items():
            log.info(f"  {key}: {value:.6f}")

    # Save model
    if cfg.get("save_model", True):
        save_path = Path(cfg.paths.output_dir) / "model.joblib"
        ml_model.save(str(save_path))

    # Merge metrics
    metric_dict = {**train_metrics, **test_metrics}

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": ml_model,
    }

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_ml.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for ML training.

    Args:
        cfg: Hydra configuration
    """
    # Train the model
    metrics, _ = train_ml(cfg)

    # Return metrics for hyperparameter optimization
    return metrics.get("val/rmse", metrics.get("test/rmse"))


if __name__ == "__main__":
    main()
