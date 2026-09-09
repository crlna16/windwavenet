# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

import math
from typing import Any, Dict, List, Optional, Tuple, cast

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanSquaredError

from src.utils.pylogger import RankedLogger
from src.utils.utils import get_bin_mask

log = RankedLogger(__name__, rank_zero_only=False)


class WWMEmulator(L.LightningModule):
    """Lightning Module for Wind Wave Model emulation with flexible backbone.

    This model uses a configurable backbone architecture that can be swapped out
    via Hydra configuration.
    """

    # Registry of supported output constraint functions.
    # "unit_circle" is a paired constraint applied to (sin_*, cos_*) target pairs
    # jointly inside _apply_output_constraints; the lambda here is a no-op sentinel.
    CONSTRAINT_FUNCTIONS = {
        "semipositive": lambda x: torch.relu(x),
        "positive": lambda x: F.softplus(x),
        "angle_deg": lambda x: torch.remainder(x, 360.0),
        "angle_rad": lambda x: torch.remainder(x, 2.0 * math.pi),
        "sigmoid": lambda x: torch.sigmoid(x),
        "unit_circle": lambda x: x,  # handled as a paired operation below
    }

    def __init__(
        self,
        backbone: nn.Module,
        target_names: List[str],
        optimizer: torch.optim.Optimizer,  # accessed via hparams
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,  # accessed via hparams
        loss_weights: Optional[Dict[str, float]] = None,
        target_bins: Optional[Dict[str, List[float]]] = None,
        output_constraints: Optional[Dict[str, str]] = None,
        focal_gamma: Optional[bool] = True,
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
            output_constraints: Optional dict mapping target names to constraint types.
                        Supported types: 'semipositive' (ReLU, >=0), 'positive' (softplus, >0),
                        'angle_deg' (mod 360), 'angle_rad' (mod 2*pi), 'sigmoid' ((0,1)).
                        E.g., {'sigWaveHeight': 'semipositive', 'meanWaveDirection': 'angle_deg'}
            focal_gamma: Kept for legacy so that existing ckpts can be loaded.

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

        # Initialize output constraints
        self.output_constraints = output_constraints if output_constraints is not None else {}
        if self.output_constraints:
            for name, constraint in self.output_constraints.items():
                if name not in target_names:
                    raise ValueError(f"Output constraint target '{name}' not found in target_names: {target_names}")
                if constraint not in self.CONSTRAINT_FUNCTIONS:
                    raise ValueError(
                        f"Unknown constraint type '{constraint}' for target '{name}'. "
                        f"Supported types: {list(self.CONSTRAINT_FUNCTIONS.keys())}"
                    )
            log.info(f"Output constraints: {self.output_constraints}")

        # Store target bins for binned metrics
        self.target_bins = target_bins if target_bins is not None else {}
        if self.target_bins:
            log.info(f"Target bins: {self.target_bins}")

        # Loss function (reduction='none' to get per-sample loss)
        self.criterion = nn.MSELoss(reduction="none")

        # Track best validation loss
        self.best_val_loss = float("inf")
        self.best_val_rmse = {name: float("inf") for name in target_names}

        # Per-variable RMSE metrics
        self.train_rmse = nn.ModuleDict({name: MeanSquaredError(squared=False) for name in target_names})
        self.val_rmse = nn.ModuleDict({name: MeanSquaredError(squared=False) for name in target_names})
        self.test_rmse = nn.ModuleDict({name: MeanSquaredError(squared=False) for name in target_names})

        # Binned RMSE metrics for validation (only create if bins are specified)
        self.val_rmse_binned = nn.ModuleDict()
        self.test_rmse_binned = nn.ModuleDict()
        for name in target_names:
            log.info(f"Checking if {name} has target bins ...")
            if name in self.target_bins:
                bins = self.target_bins[name]
                # Create one metric per bin range
                for i in range(len(bins) + 1):
                    bin_key = f"{name}_bin{i}"
                    self.val_rmse_binned[bin_key] = MeanSquaredError(squared=False)
                    self.test_rmse_binned[bin_key] = MeanSquaredError(squared=False)
                    log.info(f"Added validation and test metric {bin_key} for {name}")

    def _apply_output_constraints(self, preds: torch.Tensor) -> torch.Tensor:
        """Apply output constraints to predictions.

        Individual constraints (``semipositive``, ``positive``, ``angle_deg``,
        ``angle_rad``, ``sigmoid``) are applied per-target channel.

        ``unit_circle`` is a *paired* constraint: for every ``sin_*`` /
        ``cos_*`` target pair that both carry the ``unit_circle`` constraint,
        the two channels are jointly projected onto the unit circle via
        ``(sin, cos) / max(norm, eps)``.  This keeps predictions in [-1, 1]
        and ensures sin² + cos² ≈ 1 without hard saturation.

        Args:
            preds: Model predictions of shape (batch_size, num_targets).

        Returns:
            Constrained predictions with the same shape.
        """
        if not self.output_constraints:
            return preds

        # Build result tensor element-by-element to avoid in-place operations
        result = []

        # Track which indices have already been handled by the unit_circle pair logic.
        handled: set = set()

        # Build lookup: target_name -> index
        name_to_idx = {name: i for i, name in enumerate(self.target_names)}

        for i, name in enumerate(self.target_names):
            if i in handled:
                continue
            constraint = self.output_constraints.get(name)
            if constraint is None:
                result.append(preds[:, i].unsqueeze(-1))
                continue

            if constraint == "unit_circle":
                # Detect the complementary channel: sin_X pairs with cos_X
                if name.startswith("sin_"):
                    base = name[len("sin_") :]
                    partner = f"cos_{base}"
                elif name.startswith("cos_"):
                    base = name[len("cos_") :]
                    partner = f"sin_{base}"
                else:
                    partner = None

                if (
                    partner is not None
                    and partner in name_to_idx
                    and self.output_constraints.get(partner) == "unit_circle"
                ):
                    j = name_to_idx[partner]
                    norm = torch.sqrt(preds[:, i] ** 2 + preds[:, j] ** 2).clamp(min=1e-8)
                    sin_normalized = preds[:, i] / norm
                    cos_normalized = preds[:, j] / norm
                    result.append(sin_normalized.unsqueeze(-1))
                    result.append(cos_normalized.unsqueeze(-1))
                    handled.add(i)
                    handled.add(j)
                else:
                    result.append(preds[:, i].unsqueeze(-1))
                # If no valid partner found, fall through and leave channel unchanged.
            else:
                constraint_fn = self.CONSTRAINT_FUNCTIONS[constraint]
                result.append(constraint_fn(preds[:, i]).unsqueeze(-1))

        return torch.cat(result, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor.

        Returns:
            Model predictions
        """
        output = self.backbone(x)
        return self._apply_output_constraints(output)

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

    def log_step_metrics(
        self,
        total_loss: torch.Tensor,
        per_var_losses: Dict[str, torch.Tensor],
        preds: torch.Tensor,
        targets: torch.Tensor,
        prefix: str,
        rmse_metrics: nn.ModuleDict,
        log_on_step: bool = False,
    ) -> None:
        """Log loss and RMSE metrics for a step.

        Args:
            total_loss: Total weighted loss.
            per_var_losses: Dict of per-variable losses.
            preds: Model predictions.
            targets: Ground truth targets.
            prefix: Logging prefix ('train', 'val', or 'test').
            rmse_metrics: ModuleDict containing RMSE metrics for each target.
            log_on_step: Whether to log on step (True for training).
        """
        sync_dist = prefix in ("val", "test")

        # Log total loss
        self.log(f"{prefix}/loss", total_loss, on_step=log_on_step, on_epoch=True, prog_bar=True, sync_dist=sync_dist)

        # Log per-variable losses and RMSE
        for i, name in enumerate(self.target_names):
            self.log(f"{prefix}/loss_{name}", per_var_losses[name], on_step=False, on_epoch=True, sync_dist=sync_dist)
            rmse_metrics[name](preds[:, i], targets[:, i])
            self.log(f"{prefix}/rmse_{name}", rmse_metrics[name], on_step=False, on_epoch=True, sync_dist=sync_dist)

    def log_binned_rmse_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        prefix: str,
        binned_rmse_metrics: nn.ModuleDict,
    ) -> None:
        """Log per-bin RMSE metrics for configured targets.

        Args:
            preds: Model predictions.
            targets: Ground truth targets.
            prefix: Logging prefix ('val' or 'test').
            binned_rmse_metrics: ModuleDict containing one RMSE metric per target bin.
        """
        sync_dist = prefix in ("val", "test")

        for i, name in enumerate(self.target_names):
            if name not in self.target_bins:
                continue

            bins = self.target_bins[name]
            target_vals = targets[:, i]
            pred_vals = preds[:, i]

            for bin_idx in range(len(bins) + 1):
                mask = get_bin_mask(target_vals, bins, bin_idx)

                # Only update metric if there are samples in this bin
                if mask.sum() == 0:
                    continue

                bin_key = f"{name}_bin{bin_idx}"
                binned_rmse_metrics[bin_key](pred_vals[mask], target_vals[mask])

                # Create descriptive bin label for logging
                if bin_idx == 0:
                    bin_label = f"<{bins[0]}"
                elif bin_idx == len(bins):
                    bin_label = f">={bins[-1]}"
                else:
                    bin_label = f"[{bins[bin_idx - 1]},{bins[bin_idx]})"

                self.log(
                    f"{prefix}/rmse_{name}_{bin_label}",
                    cast(Any, binned_rmse_metrics[bin_key]),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=sync_dist,
                )

    def log_circular_metrics(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        prefix: str,
    ) -> None:
        """Log circular MAE (in degrees) for sin/cos target pairs.

        For every ``sin_X`` / ``cos_X`` pair in ``target_names`` both carrying
        the ``unit_circle`` constraint, reconstructs the angle via
        ``atan2(sin, cos)`` for both predictions and ground truth, then logs
        the circular mean absolute error in degrees.

        Args:
            preds: Model predictions, shape (batch, num_targets).
            targets: Ground-truth targets, shape (batch, num_targets).
            prefix: Logging prefix (``'val'`` or ``'test'``).
        """
        sync_dist = prefix in ("val", "test")
        name_to_idx = {n: i for i, n in enumerate(self.target_names)}
        sin_names = {n[len("sin_") :]: n for n in self.target_names if n.startswith("sin_")}
        cos_names = {n[len("cos_") :]: n for n in self.target_names if n.startswith("cos_")}

        for base_var in sin_names:
            if base_var not in cos_names:
                continue
            sin_name = sin_names[base_var]
            cos_name = cos_names[base_var]
            # Only log if both carry the unit_circle constraint
            if (
                self.output_constraints.get(sin_name) != "unit_circle"
                or self.output_constraints.get(cos_name) != "unit_circle"
            ):
                continue

            si = name_to_idx[sin_name]
            ci = name_to_idx[cos_name]

            angle_pred = torch.atan2(preds[:, si], preds[:, ci]) * (180.0 / math.pi)
            angle_truth = torch.atan2(targets[:, si], targets[:, ci]) * (180.0 / math.pi)

            diff = (angle_pred - angle_truth).abs() % 360.0
            circular_mae_val = torch.where(diff > 180.0, 360.0 - diff, diff).mean()

            self.log(
                f"{prefix}/circular_mae_{base_var}", circular_mae_val, on_step=False, on_epoch=True, sync_dist=sync_dist
            )

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data.

        Args:
            batch: A batch of data (a tuple) containing input and target tensors.
            batch_idx: The index of the current batch.

        Returns:
            A tensor of losses between model predictions and targets.
        """
        total_loss, per_var_losses, preds, targets = self.model_step(batch)
        self.log_step_metrics(total_loss, per_var_losses, preds, targets, "train", self.train_rmse, log_on_step=True)
        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data.

        Args:
            batch: A batch of data (a tuple) containing input and target tensors.
            batch_idx: The index of the current batch.
        """
        total_loss, per_var_losses, preds, targets = self.model_step(batch)

        # Skip logging during the sanity check so its tiny, non-representative
        # batch subset can't pollute real metrics (e.g. `best_rmse` tracking).
        if self.trainer.sanity_checking:
            return

        self.log_step_metrics(total_loss, per_var_losses, preds, targets, "val", self.val_rmse, log_on_step=False)
        self.log_binned_rmse_metrics(preds, targets, "val", self.val_rmse_binned)
        self.log_circular_metrics(preds, targets, "val")

    def on_validation_epoch_end(self) -> None:
        """Called at the end of the validation epoch to track best validation loss."""
        if self.trainer.sanity_checking:
            return

        # Get the current epoch's validation loss
        val_loss = self.trainer.callback_metrics.get("val/loss")

        if val_loss is not None:
            # Update best validation loss if current loss is better
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss.item()

            # Log the best validation loss
            self.log("val/best_loss", self.best_val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        for name in self.target_names:
            val_rmse = self.trainer.callback_metrics.get(f"val/rmse_{name}")
            if val_rmse is not None:
                if val_rmse < self.best_val_rmse[name]:
                    self.best_val_rmse[name] = val_rmse.item()

                self.log(
                    f"val/best_rmse_{name}",
                    self.best_val_rmse[name],
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                )

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data.

        Args:
            batch: A batch of data (a tuple) containing input and target tensors.
            batch_idx: The index of the current batch.
        """
        total_loss, per_var_losses, preds, targets = self.model_step(batch)
        self.log_step_metrics(total_loss, per_var_losses, preds, targets, "test", self.test_rmse, log_on_step=False)
        self.log_binned_rmse_metrics(preds, targets, "test", self.test_rmse_binned)
        self.log_circular_metrics(preds, targets, "test")

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single predict step on a batch of data.

        Note:
            This currently reuses ``model_step``, which unconditionally unpacks a target
            tensor ``y`` from ``batch`` (see ``model_step``'s ``x, y = batch`` fallback). It
            therefore only works with datasets/datamodules that always return targets (as all
            current WWM datasets do), even though the target is discarded below. A dataset that
            yields inputs only (no target, e.g. true out-of-sample forecasting) is NOT
            supported yet and would raise when unpacking the batch; this would require either
            an optional-target branch here or in ``model_step``.

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
