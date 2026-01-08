# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from lightning_utilities.core.rank_zero import rank_zero_only
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

from src.utils.pylogger import RankedLogger
from src.utils.utils import get_bin_mask

# Initialize rank for single-process execution to use RankedLogger
rank_zero_only.rank = 0

log = RankedLogger(__name__, rank_zero_only=False)


class MLModelWrapper:
    """Wrapper for classical ML models (XGBoost, RandomForest, etc.) to work with Hydra config.

    This provides a similar interface to the Lightning models for consistency.
    """

    def __init__(
        self,
        model: BaseEstimator,
        target_names: List[str],
        target_bins: Optional[Dict[str, List[float]]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> None:
        """Initialize the ML model wrapper.

        Args:
            model: Scikit-learn compatible model (XGBoost, RandomForest, etc.)
            target_names: List of target variable names
            target_bins: Optional dict mapping target names to bin edges for binned metrics
            callbacks: Optional list of callbacks (e.g., WandB callback for XGBoost)
        """
        self.model = model
        self.target_names = target_names
        self.target_bins = target_bins if target_bins is not None else {}
        self.num_targets = len(target_names)
        self.callbacks = callbacks if callbacks is not None else []

        log.info(f"Initialized ML model: {type(model).__name__}")
        log.info(f"Target names: {target_names}")
        if self.callbacks:
            log.info(f"Callbacks: {[type(cb).__name__ for cb in self.callbacks]}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for early stopping if supported)
            y_val: Validation targets
            sample_weight: Sample weights for training

        Returns:
            Dictionary of training metrics
        """
        log.info(f"Training {type(self.model).__name__} on {len(X_train)} samples...")

        if X_val is None or y_val is None:
            eval_set = None
        else:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=True,
        )

        log.info("Training complete")

        # Compute training metrics
        train_preds = self.model.predict(X_train)
        train_metrics = self._compute_metrics(y_train, train_preds, prefix="train")

        # Compute validation metrics if provided
        if X_val is not None:
            val_preds = self.model.predict(X_val)
            val_metrics = self._compute_metrics(y_val, val_preds, prefix="val")
            train_metrics.update(val_metrics)

        return train_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        return self.model.predict(X)

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, prefix: str = "test"
    ) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            X: Input features
            y: True targets
            prefix: Prefix for metric names (e.g., 'test', 'val')

        Returns:
            Dictionary of evaluation metrics
        """
        preds = self.predict(X)
        return self._compute_metrics(y, preds, prefix=prefix)

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "train"
    ) -> Dict[str, float]:
        """Compute evaluation metrics.

        Args:
            y_true: True targets
            y_pred: Predicted targets
            prefix: Prefix for metric names

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Ensure 2D arrays
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        # Overall metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        metrics[f"{prefix}/loss"] = mse
        metrics[f"{prefix}/rmse"] = rmse

        # Per-target metrics
        for i, target_name in enumerate(self.target_names):
            mse_target = mean_squared_error(y_true[:, i], y_pred[:, i])
            rmse_target = np.sqrt(mse_target)
            metrics[f"{prefix}/rmse_{target_name}"] = rmse_target
            metrics[f"{prefix}/loss_{target_name}"] = mse_target

            # Binned metrics if bins are specified
            if target_name in self.target_bins:
                bins = self.target_bins[target_name]
                for bin_idx in range(len(bins) + 1):
                    mask = get_bin_mask(y_true[:, i], bins, bin_idx)

                    if mask.sum() > 0:
                        rmse_bin = np.sqrt(
                            mean_squared_error(y_true[mask, i], y_pred[mask, i])
                        )

                        # Create bin label
                        if bin_idx == 0:
                            bin_label = f"<{bins[0]}"
                        elif bin_idx == len(bins):
                            bin_label = f">={bins[-1]}"
                        else:
                            bin_label = f"[{bins[bin_idx - 1]},{bins[bin_idx]})"

                        metrics[f"{prefix}/rmse_{target_name}_{bin_label}"] = rmse_bin

        return metrics

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        log.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to load the model from
        """
        self.model = joblib.load(path)
        log.info(f"Model loaded from {path}")
