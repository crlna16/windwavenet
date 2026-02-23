# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError

from src.utils.pylogger import RankedLogger
from src.utils.utils import get_bin_mask

log = RankedLogger(__name__, rank_zero_only=False)


class WWMEmulator(L.LightningModule):
    """Lightning Module for Wave Wind Model emulation with flexible backbone.

    This model uses a configurable backbone architecture that can be swapped out
    via Hydra configuration.
    """

    def __init__(
        self,
        backbone: nn.Module,
        target_names: List[str],
        optimizer: torch.optim.Optimizer,  # accessed via hparams
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,  # accessed via hparams
        loss_weights: Optional[Dict[str, float]] = None,
        target_bins: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """Initialize the WWM Emulator.

        Args:
            backbone: The neural network backbone (can be any nn.Module).
            target_names: List of target variable names (e.g., ['sigWaveHeight']).
            optimizer: PyTorch optimizer (partially instantiated via Hydra).
            scheduler: PyTorch learning rate scheduler (partially instantiated via Hydra).
            loss_weights: Optional dict mapping target names to loss weights (default: equal weights of 1.0).
            target_bins: Optional dict mapping target names to bin edges for binned metrics.
                        E.g., {'sigWaveHeight': [-1.0, 0.0, 0.5]} creates bins: <-1, [-1,0), [0,0.5), >=0.5

        Note:
            Sample weights are now computed and stored in the dataset. Configure them via the data module
            using sample_weight_strategy and sample_weight_power parameters.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        # we need this for optimizer and scheduler, which can only be instantiated after the model is initialised
        self.save_hyperparameters(logger=False, ignore=["backbone"])

        self.backbone = backbone
        self.target_names = target_names
        self.num_targets = len(target_names)

        # Store which targets are delta predictions (for reconstruction during inference)
        self.is_delta_target = {name: name.startswith("delta_") for name in target_names}

        # Initialize loss weights (default to 1.0 for all targets)
        if loss_weights is None:
            self.loss_weights = {name: 1.0 for name in target_names}
        else:
            self.loss_weights = loss_weights
            # Validate that all targets have weights
            for name in target_names:
                if name not in self.loss_weights:
                    log.warning(f"No loss weight specified for '{name}', defaulting to 1.0")
                    self.loss_weights[name] = 1.0

        log.info(f"Loss weights: {self.loss_weights}")

        # Store target bins for binned metrics
        self.target_bins = target_bins if target_bins is not None else {}
        if self.target_bins:
            log.info(f"Target bins: {self.target_bins}")

        # Loss function (reduction='none' to get per-sample loss)
        self.criterion = nn.MSELoss(reduction="none")

        # Track best validation loss
        self.best_val_loss = float("inf")

        # Per-variable RMSE metrics
        self.train_rmse = nn.ModuleDict({name: MeanSquaredError(squared=False) for name in target_names})
        self.val_rmse = nn.ModuleDict({name: MeanSquaredError(squared=False) for name in target_names})
        self.test_rmse = nn.ModuleDict({name: MeanSquaredError(squared=False) for name in target_names})

        # Binned RMSE metrics for validation (only create if bins are specified)
        self.val_rmse_binned = nn.ModuleDict()
        for name in target_names:
            log.info(f"Checking if {name} has target bins ...")
            if name in self.target_bins:
                bins = self.target_bins[name]
                # Create one metric per bin range
                for i in range(len(bins) + 1):
                    bin_key = f"{name}_bin{i}"
                    self.val_rmse_binned[bin_key] = MeanSquaredError(squared=False)
                    log.info(f"Added validation metric {bin_key} for {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor.

        Returns:
            Model predictions
        """
        output = self.backbone(x)
        return output

    def model_step(
        self, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Args:
            batch: A batch of data containing input, target tensors, and optionally sample weights.

        Returns:
            A tuple containing (total_loss, per_variable_losses, predictions, targets).
        """
        # Unpack batch - could be (x, y) or (x, y, weights)
        if len(batch) == 3:
            x, y, sample_weights = batch
        else:
            x, y = batch
            sample_weights = torch.ones_like(y)

        preds = self.forward(x)

        # Compute per-sample, per-variable MSE loss
        per_sample_loss = self.criterion(preds, y)  # shape: (batch, num_targets)

        # Apply sample weights to the loss
        weighted_loss = per_sample_loss * sample_weights  # shape: (batch, num_targets)

        # Compute per-variable loss (mean over batch)
        per_variable_losses = {}
        for i, name in enumerate(self.target_names):
            per_variable_losses[name] = weighted_loss[:, i].mean()

        # Total loss is weighted sum of per-variable losses
        total_loss = sum(per_variable_losses[name] * self.loss_weights[name] for name in self.target_names)

        return total_loss, per_variable_losses, preds, y

    def _shared_step(
        self,
        batch: Tuple[torch.Tensor, ...],
        prefix: str,
        rmse_metrics: nn.ModuleDict,
        log_on_step: bool = False,
    ) -> torch.Tensor:
        """Shared logic for training, validation, and test steps.

        Args:
            batch: A batch of data containing input, target tensors, and optionally sample weights.
            prefix: Logging prefix ('train', 'val', or 'test').
            rmse_metrics: ModuleDict containing RMSE metrics for each target.
            log_on_step: Whether to log on step (True for training).

        Returns:
            Total loss tensor.
        """
        total_loss, per_var_losses, preds, targets = self.model_step(batch)

        # Log total loss
        self.log(f"{prefix}/loss", total_loss, on_step=log_on_step, on_epoch=True, prog_bar=True)

        # Log per-variable losses and RMSE
        for i, name in enumerate(self.target_names):
            self.log(f"{prefix}/loss_{name}", per_var_losses[name], on_step=False, on_epoch=True)
            rmse_metrics[name](preds[:, i], targets[:, i])
            self.log(f"{prefix}/rmse_{name}", rmse_metrics[name], on_step=False, on_epoch=True)

        return total_loss, preds, targets

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data.

        Args:
            batch: A batch of data (a tuple) containing input and target tensors.
            batch_idx: The index of the current batch.

        Returns:
            A tensor of losses between model predictions and targets.
        """
        total_loss, _, _ = self._shared_step(batch, "train", self.train_rmse, log_on_step=True)
        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data.

        Args:
            batch: A batch of data (a tuple) containing input and target tensors.
            batch_idx: The index of the current batch.
        """
        _, preds, targets = self._shared_step(batch, "val", self.val_rmse, log_on_step=False)

        # Compute binned RMSE (validation only)
        for i, name in enumerate(self.target_names):
            if name in self.target_bins:
                bins = self.target_bins[name]
                target_vals = targets[:, i]
                pred_vals = preds[:, i]

                for bin_idx in range(len(bins) + 1):
                    mask = get_bin_mask(target_vals, bins, bin_idx)

                    # Only update metric if there are samples in this bin
                    if mask.sum() > 0:
                        bin_key = f"{name}_bin{bin_idx}"
                        self.val_rmse_binned[bin_key](pred_vals[mask], target_vals[mask])

                        # Create descriptive bin label for logging
                        if bin_idx == 0:
                            bin_label = f"<{bins[0]}"
                        elif bin_idx == len(bins):
                            bin_label = f">={bins[-1]}"
                        else:
                            bin_label = f"[{bins[bin_idx - 1]},{bins[bin_idx]})"

                        self.log(
                            f"val/rmse_{name}_{bin_label}",
                            self.val_rmse_binned[bin_key],
                            on_step=False,
                            on_epoch=True,
                        )

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch to track best validation loss."""
        # Get the current epoch's validation loss
        val_loss = self.trainer.callback_metrics.get("val/loss")

        if val_loss is not None:
            # Update best validation loss if current loss is better
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss.item()

            # Log the best validation loss
            self.log("val/best_loss", self.best_val_loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data.

        Args:
            batch: A batch of data (a tuple) containing input and target tensors.
            batch_idx: The index of the current batch.
        """
        self._shared_step(batch, "test", self.test_rmse, log_on_step=False)

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single predict step on a batch of data.

        TODO: Refactor for use with dataset that does not return target.

        Args:
            batch: A batch of data (a tuple) containing input and target tensors.
            batch_idx: The index of the current batch.
        """
        _, _, preds, _ = self.model_step(batch)

        return preds

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning-rate schedulers.

        Returns:
            A dict containing the configured optimizers and learning-rate schedulers.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}
