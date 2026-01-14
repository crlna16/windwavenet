# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Training script for classical ML models (XGBoost, RandomForest, etc.)."""

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import hydra
import numpy as np
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig
import wandb
from wandb.integration.xgboost import WandbCallback

from src.utils.pylogger import RankedLogger
from src.utils.utils import (
    print_config_tree,
    task_wrapper,
    extract_data_from_datamodule,
    generate_predictions_ml,
)

# Initialize rank for single-process execution to use RankedLogger
rank_zero_only.rank = 0

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train_ml(cfg: DictConfig) -> Union[None, Tuple[Dict[str, Any], Dict[str, Any]]]:
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
    if len(cfg.data.targets > 1):
        log.info(
            f"ML pipeline works only with single target. Received {cfg.data.targets}."
        )
        return None
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

    if cfg.logger:
        log.info("Initializing WandB...")
        wandb_run = wandb.init(**cfg.logger.wandb, config=dict(cfg))
        callbacks = [WandbCallback(log_model=False)]
    else:
        wandb_run = None
        callbacks = None

    # Instantiate ML model
    log.info(f"Instantiating ML model <{cfg.model._target_}>")
    ml_model = hydra.utils.instantiate(cfg.model, callbacks=callbacks)

    log.info("Starting training...")
    train_metrics = ml_model.fit(
        X_train, y_train, X_val=X_val, y_val=y_val, sample_weight=weights_train
    )

    log.info("Training metrics:")
    for key, value in train_metrics.items():
        log.info(f"  {key}: {value:.6f}")

    # Log to WandB
    if wandb_run is not None:
        wandb.log(train_metrics)

    # Test model
    test_metrics = {}
    if cfg.get("test"):
        log.info("Starting testing...")
        test_metrics = ml_model.evaluate(X_test, y_test, prefix="test")

        # Log test metrics
        log.info("Test metrics:")
        for key, value in test_metrics.items():
            log.info(f"  {key}: {value:.6f}")

        # Log to WandB
        if wandb_run is not None:
            wandb.log(test_metrics)

    # Save model
    if cfg.get("save_model", True):
        save_path = Path(cfg.paths.output_dir) / "model.joblib"
        ml_model.save(str(save_path))

    # Generate predictions
    if cfg.get("predict", True):
        log.info("Generating predictions on test set...")
        ds_pred = generate_predictions_ml(
            ml_model=ml_model,
            datamodule=datamodule,
            target_names=cfg.data.targets,
            model_target=cfg.model._target_,
        )

        # Save predictions to netCDF
        output_path = Path(cfg.paths.output_dir) / "predictions.nc"
        log.info(f"Saving predictions to {output_path}")
        ds_pred.to_netcdf(output_path)
        log.info(f"Predictions saved successfully to {output_path}")

    # Finish WandB run
    if wandb_run is not None:
        wandb.finish()
        log.info("WandB run finished")

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
