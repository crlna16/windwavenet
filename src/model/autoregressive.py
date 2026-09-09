# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Mixin adding free-running autoregressive rollout training to WWM LightningModules.

Combine with ``WWMEmulator`` (pointwise/LSTM backbones) or
``GraphWWMEmulator`` (graph backbones) to train a model that rolls out
``rollout_steps`` predictions, feeding each one back into the
``prev_<target>`` input slot of the next step, and averages the loss over
all rollout steps. See ``src/model/ARWWMEmulator.py`` for the pointwise/LSTM
subclass and ``configs/experiment/*_ar.yaml`` for example configs.
"""

from typing import Dict, List, Tuple

import torch

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)

FeedbackMap = List[Tuple[int, int, float, float]]


class AutoregressiveMixin:
    """Adds rollout training to a WWMEmulator-like LightningModule.

    Subclasses must set ``self.window_len`` (1 for pointwise models,
    ``seq_len`` for sequence models) and be combined with a LightningModule
    that provides ``self.criterion``, ``self.target_names``,
    ``self.loss_weights``, and ``self.forward``. Requires a datamodule that
    exposes ``feedback_map`` (e.g. ``ARWWMDataModule``).
    """

    window_len: int = 1

    def _init_autoregressive(self, rollout_steps: int, rollout_detach: bool = True) -> None:
        """Store rollout config and placeholder feedback-map buffers.

        Args:
            rollout_steps: Number of future steps (T) to roll out and average
                the loss over.
            rollout_detach: If True (default), stop gradients through fed-back
                predictions (cheaper, more stable). If False, backpropagate
                through the full rollout (BPTT).
        """
        if rollout_steps < 1:
            raise ValueError(f"rollout_steps must be >= 1, got {rollout_steps}")
        self.rollout_steps = rollout_steps
        self.rollout_detach = rollout_detach
        self._feedback_map_ready = False
        self.register_buffer("feedback_feat_idx", torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer("feedback_target_idx", torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer("feedback_mean", torch.zeros(0), persistent=False)
        self.register_buffer("feedback_std", torch.zeros(0), persistent=False)

    def on_fit_start(self) -> None:
        """Load the feedback map from the datamodule before training/validation begins."""
        super().on_fit_start()
        self._load_feedback_map()

    def _load_feedback_map(self) -> None:
        """Fetch ``feedback_map`` from the datamodule and register it as index/scale buffers."""
        datamodule = getattr(self.trainer, "datamodule", None)
        feedback_map: FeedbackMap = getattr(datamodule, "feedback_map", None)
        if feedback_map is None:
            raise RuntimeError(
                "Autoregressive training requires a datamodule exposing `feedback_map` "
                "(list of (feature_idx, target_idx, mean, std)), e.g. ARWWMDataModule."
            )
        if len(feedback_map) == 0:
            log.warning("Feedback map is empty: no `prev_<target>` feature will be updated during rollout.")

        feat_idx, tgt_idx, means, stds = zip(*feedback_map) if feedback_map else ((), (), (), ())
        device = self.device
        self.feedback_feat_idx = torch.tensor(feat_idx, dtype=torch.long, device=device)
        self.feedback_target_idx = torch.tensor(tgt_idx, dtype=torch.long, device=device)
        self.feedback_mean = torch.tensor(means, dtype=torch.float32, device=device)
        self.feedback_std = torch.tensor(stds, dtype=torch.float32, device=device)
        self._feedback_map_ready = True

    def _forward_step(self, x_buf: torch.Tensor, s: int) -> torch.Tensor:
        """Extract inputs for rollout step ``s`` from ``x_buf`` and run forward.

        Default implementation for pointwise (``window_len == 1``) and
        sequence/LSTM (``window_len == seq_len``) models: slices a plain
        tensor and forwards it directly. Graph models override this to wrap
        the slice into a PyG ``Data``/``Batch`` object before forwarding
        (see ``ARGraphWWMEmulator``).

        Args:
            x_buf: Rolling input buffer, see ``_rollout``.
            s: Current rollout step index (0-indexed).

        Returns:
            Predictions for step ``s``, shape ``(rows, n_targets)``.
        """
        window_len = self.window_len
        x_step = x_buf[:, s, :] if window_len == 1 else x_buf[:, s : s + window_len, :]
        return self.forward(x_step)

    def _rollout(self, x_buf: torch.Tensor) -> torch.Tensor:
        """Roll out ``rollout_steps`` predictions, feeding each back into ``x_buf``.

        Args:
            x_buf: Rolling input buffer of shape ``(rows, window_len + T - 1,
                n_features)``, where ``rows`` is ``batch`` for pointwise/LSTM
                models or ``total_nodes_in_pyg_batch`` for graph models. For
                pointwise models (``window_len == 1``) this is simply
                ``(rows, T, n_features)``. A clone is mutated internally with
                fed-back predictions; the input is not modified.

        Returns:
            Predictions of shape ``(rows, T, n_targets)``.
        """
        if not self._feedback_map_ready:
            raise RuntimeError("Feedback map not loaded; autoregressive training requires `trainer.fit`.")

        x_buf = x_buf.clone()
        window_len = self.window_len
        has_feedback = self.feedback_feat_idx.numel() > 0
        preds = []
        for s in range(self.rollout_steps):
            pred = self._forward_step(x_buf, s)
            preds.append(pred)

            # Last step's prediction feeds no further step, so it needs no write-back.
            if has_feedback and s < self.rollout_steps - 1:
                fed_back = pred[:, self.feedback_target_idx]
                if self.rollout_detach:
                    fed_back = fed_back.detach()
                scaled = (fed_back - self.feedback_mean) / self.feedback_std
                x_buf[:, s + window_len, self.feedback_feat_idx] = scaled

        return torch.stack(preds, dim=1)

    def _compute_ar_loss(
        self, x_buf: torch.Tensor, y_win: torch.Tensor, sample_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Roll out predictions and average the loss over all ``rollout_steps``.

        Shared by the tuple-batch ``model_step`` below and graph AR emulators,
        which unpack their PyG ``Batch`` into the same ``(x_buf, y_win,
        sample_weights)`` shapes before calling this.

        Args:
            x_buf: Rolling input buffer, see ``_rollout``.
            y_win: Targets, shape ``(rows, T, n_targets)``.
            sample_weights: Per-step sample weights, shape ``(rows, T, n_targets)``.

        Returns:
            Tuple of (total_loss, per_variable_losses, predictions, targets), with
            predictions/targets flattened to ``(rows * T, num_targets)`` so that
            the base class's logging methods (which expect 2D tensors) work unchanged.
        """
        preds = self._rollout(x_buf)  # (rows, T, C)
        # Stashed for validation_step's single-step (lead time 0) loss log.
        self._last_ar_preds = preds
        self._last_ar_targets = y_win

        per_sample_loss = self.criterion(preds, y_win)  # (rows, T, C)

        weighted_loss = per_sample_loss * sample_weights

        per_variable_losses = {}
        for i, name in enumerate(self.target_names):
            per_variable_losses[name] = weighted_loss[:, :, i].mean()

        total_loss = sum(per_variable_losses[name] * self.loss_weights[name] for name in self.target_names)

        b, t, c = preds.shape
        return total_loss, per_variable_losses, preds.reshape(b * t, c), y_win.reshape(b * t, c)

    def model_step(
        self, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Roll out predictions over ``rollout_steps`` and average the loss over all of them.

        Args:
            batch: ``(x_buf, y_win)`` or ``(x_buf, y_win, sample_weights)``, where
                ``y_win`` and ``sample_weights`` have shape ``(batch, T, n_targets)``.

        Returns:
            Tuple of (total_loss, per_variable_losses, predictions, targets), with
            predictions/targets flattened to ``(batch * T, num_targets)`` so that
            the base class's logging methods (which expect 2D tensors) work unchanged.
        """
        if len(batch) == 3:
            x_buf, y_win, sample_weights = batch
        else:
            x_buf, y_win = batch
            sample_weights = torch.ones_like(y_win)

        return self._compute_ar_loss(x_buf, y_win, sample_weights)

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        """Run the T-step validation step, plus log the single-step (lead time 0) loss.

        Args:
            batch: Same format as ``model_step``.
            batch_idx: The index of the current batch.
        """
        super().validation_step(batch, batch_idx)
        if self.trainer.sanity_checking:
            return
        loss_step0 = self.criterion(self._last_ar_preds[:, 0, :], self._last_ar_targets[:, 0, :]).mean()
        self.log("val/loss_step0", loss_step0, on_step=False, on_epoch=True, sync_dist=True)
