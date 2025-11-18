# SPDX-FileCopyrightText: 2025 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Tuple

import lightning as L
import torch
import torch.nn as nn


class WWMEmulator(L.LightningModule):
    """Lightning Module for Wave Wind Model emulation with flexible backbone.

    This model uses a configurable backbone architecture that can be swapped out
    via Hydra configuration.
    """

    def __init__(
        self,
        backbone: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        compile: bool = False,
    ) -> None:
        """Initialize the WWM Emulator.

        Args:
            backbone: The neural network backbone (can be any nn.Module).
            optimizer: PyTorch optimizer (partially instantiated via Hydra).
            scheduler: PyTorch learning rate scheduler (partially instantiated via Hydra).
            compile: Whether to compile the model with torch.compile().
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["backbone"])

        self.backbone = backbone

        # Loss function
        self.criterion = nn.MSELoss()

        # Metric tracking (you can add more metrics here)
        # from torchmetrics import MeanSquaredError
        # self.train_mse = MeanSquaredError()
        # self.val_mse = MeanSquaredError()
        # self.test_mse = MeanSquaredError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor.

        Returns:
            Model predictions.
        """
        return self.backbone(x)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Args:
            batch: A batch of data (a tuple) containing input and target tensors.

        Returns:
            A tuple containing (loss, predictions, targets).
        """
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data.

        Args:
            batch: A batch of data (a tuple) containing input and target tensors.
            batch_idx: The index of the current batch.

        Returns:
            A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data.

        Args:
            batch: A batch of data (a tuple) containing input and target tensors.
            batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Update and log metrics
        # self.val_mse(preds, targets)
        # self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data.

        Args:
            batch: A batch of data (a tuple) containing input and target tensors.
            batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Update and log metrics
        # self.test_mse(preds, targets)
        # self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)

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
