# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Training-time noise injection on the ``prev_<target>`` input channels.

At rollout time the ``prev_*`` inputs are the model's own predictions and
therefore carry error; at training time they are ground truth. The model
consequently learns to trust them more than it should, and a single bad step
can throw the whole trajectory off. Corrupting them during training closes
that gap: the model is trained to recover the clean target from a perturbed
previous state, so it leans on the forcings rather than on persistence.

This is the cheap counterpart to autoregressive rollout training. With
detached feedback (as in :class:`~src.model.autoregressive.AutoregressiveMixin`)
rollout training corrupts the same channels with the model's *own* errors at
~T x the cost; noise injection approximates that with an explicit sigma at
~1x the cost.

References:
    Sanchez-Gonzalez et al., "Learning to Simulate Complex Physics with Graph
    Networks", ICML 2020 -- input-state noise as the key ingredient for stable
    long rollouts in learned simulators.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class PrevNoiseInjector:
    """Adds i.i.d. Gaussian noise to the ``prev_<target>`` channels of a feature tensor.

    Sigmas are given in **physical units** (e.g. metres for wave height) and
    converted into the z-scored feature space using the same per-feature
    mean/std that closes the autoregressive loop.

    Args:
        features: Feature names in model input order.
        noise_std: Mapping of *target base name* to sigma in physical units,
            e.g. ``{"sigWaveHeight": 0.08}``. Names that resolve to no
            ``prev_`` feature are reported and skipped.
        feedback_map: ``(feature_idx, target_idx, mean, std)`` tuples from
            :func:`src.data.dataset_utils.build_feedback_map`, supplying the
            feature-space scaling of each ``prev_`` channel.
    """

    def __init__(
        self,
        features: Sequence[str],
        noise_std: Dict[str, float],
        feedback_map: Sequence[Tuple[int, int, float, float]],
    ):
        features = list(features)
        std_by_feat_idx = {int(f_idx): float(std) for f_idx, _, _, std in feedback_map}

        idx: List[int] = []
        sigma: List[float] = []

        for name, value in noise_std.items():
            value = float(value)
            if value < 0:
                raise ValueError(f"prev_noise_std[{name!r}] must be >= 0, got {value}")
            if value == 0.0:
                continue

            feat = f"prev_{name}"
            if feat not in features:
                log.warning(f"prev_noise_std[{name!r}]: no feature '{feat}' — skipped.")
                continue
            f_idx = features.index(feat)
            if f_idx not in std_by_feat_idx:
                log.warning(f"prev_noise_std[{name!r}]: '{feat}' is not a feedback channel — skipped.")
                continue

            idx.append(f_idx)
            # Features are z-scored, targets are not: convert sigma to feature space.
            sigma.append(value / std_by_feat_idx[f_idx])
            log.info(f"prev noise: '{feat}' +/- {value:.3g} (physical) = {sigma[-1]:.4f} (scaled)")

        self.feat_idx = torch.tensor(idx, dtype=torch.long)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)

    def __bool__(self) -> bool:
        """True if this injector would actually perturb anything."""
        return bool(self.feat_idx.numel())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return a copy of ``x`` with the configured ``prev_`` channels perturbed.

        Args:
            x: Feature tensor with features last -- ``(batch, n_features)``,
                ``(total_nodes, n_features)`` or ``(batch, seq_len, n_features)``.
                Sequence inputs get independent noise at every step.

        Returns:
            A new tensor; ``x`` is not modified.
        """
        if not self:
            return x

        x = x.clone()
        idx = self.feat_idx.to(x.device)
        sigma = self.sigma.to(device=x.device, dtype=x.dtype)
        noise = torch.randn(x.shape[:-1] + (idx.numel(),), device=x.device, dtype=x.dtype)
        x[..., idx] = x[..., idx] + sigma * noise
        return x


def build_prev_noise_injector(datamodule, noise_std: Optional[Dict[str, float]]) -> Optional[PrevNoiseInjector]:
    """Build a :class:`PrevNoiseInjector` from a datamodule, or ``None`` if disabled.

    Args:
        datamodule: Datamodule exposing ``features`` and the ``feedback_map``
            built during ``setup(stage="fit")``.
        noise_std: Target-name to sigma mapping, or ``None``/empty to disable.

    Returns:
        The injector, or ``None`` if noise is disabled or resolves to nothing.
    """
    if not noise_std:
        return None
    if datamodule is None:
        raise RuntimeError("prev_noise_std requires a datamodule exposing `features` and `feedback_map`.")

    # Without scaling_config, build_feedback_map falls back to mean=0/std=1 and the
    # sigma conversion below would silently be off by the feature's std.
    if getattr(datamodule, "scaling_config", None) is None:
        raise RuntimeError(
            "prev_noise_std requires data.scaling_config: sigma is given in physical units "
            "and needs each prev_ feature's std to convert into the z-scored feature space."
        )

    feedback_map = getattr(datamodule, "feedback_map", None)
    if not feedback_map:
        raise RuntimeError(
            "prev_noise_std is set but the datamodule has an empty `feedback_map` — "
            "no prev_<target> feature to perturb. Check that the features include "
            "prev_<target> entries and that scaling_config is set."
        )

    injector = PrevNoiseInjector(
        features=datamodule.features,
        noise_std=noise_std,
        feedback_map=feedback_map,
    )
    if not injector:
        log.warning("prev_noise_std resolved to no active channels; training without noise injection.")
        return None
    return injector
