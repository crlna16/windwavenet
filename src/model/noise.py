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

This is a cheap approximation of the error a model would accumulate from its
own compounding mistakes during multi-step rollout, without the T x cost of
actually rolling out multiple steps during training. See
``AutoregressiveTraining.md`` at the repo root for background on that
(removed) approach.

References:
    Sanchez-Gonzalez et al., "Learning to Simulate Complex Physics with Graph
    Networks", ICML 2020 -- input-state noise as the key ingredient for stable
    long rollouts in learned simulators.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# Output constraints that bound a channel, as (lower, upper) in physical units.
_CONSTRAINT_BOUNDS = {
    "semipositive": (0.0, None),
    "positive": (0.0, None),
    "seminegative": (None, 0.0),
    "negative": (None, 0.0),
    "sigmoid": (0.0, 1.0),
}


class PrevNoiseInjector:
    """Adds i.i.d. Gaussian noise to the ``prev_<target>`` channels of a feature tensor.

    Sigmas are given in **physical units** (e.g. metres for wave height) and
    converted into the z-scored feature space using the same per-feature
    mean/std that closes the autoregressive loop.

    Optionally, the noised channels are pushed back into the range the model's
    own predictions can take, since that is what they are at rollout time:
    channels whose target has a sign or ``sigmoid`` constraint are clamped to
    it, and ``sin_*``/``cos_*`` pairs with the ``unit_circle`` constraint are
    projected back onto the unit circle.

    Args:
        features: Feature names in model input order.
        noise_std: Mapping of *target base name* to sigma in physical units,
            e.g. ``{"sigWaveHeight": 0.08}``. Names that resolve to no
            ``prev_`` feature are reported and skipped.
        feedback_map: ``(feature_idx, target_idx, mean, std)`` tuples from
            :func:`src.data.dataset_utils.build_feedback_map`, supplying the
            feature-space scaling of each ``prev_`` channel.
        output_constraints: The model's target-name to constraint mapping
            (see ``WWMEmulator``), or ``None`` to leave the noised values
            unconstrained.
    """

    def __init__(
        self,
        features: Sequence[str],
        noise_std: Dict[str, float],
        feedback_map: Sequence[Tuple[int, int, float, float]],
        output_constraints: Optional[Dict[str, str]] = None,
    ):
        features = list(features)
        std_by_feat_idx = {int(f_idx): float(std) for f_idx, _, _, std in feedback_map}
        mean_by_feat_idx = {int(f_idx): float(mean) for f_idx, _, mean, _ in feedback_map}

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

        constraints = output_constraints or {}

        # Per-channel bounds in feature (z-scored) space; +/-inf = unbounded.
        lower = [-float("inf")] * len(idx)
        upper = [float("inf")] * len(idx)
        for k, f_idx in enumerate(idx):
            base = features[f_idx][len("prev_") :]
            bounds = _CONSTRAINT_BOUNDS.get(constraints.get(base))
            if bounds is None:
                continue
            mean, std = mean_by_feat_idx[f_idx], std_by_feat_idx[f_idx]
            lo, hi = bounds
            if lo is not None:
                lower[k] = (lo - mean) / std
            if hi is not None:
                upper[k] = (hi - mean) / std
            log.info(f"prev noise: '{features[f_idx]}' clamped to the '{constraints[base]}' range.")
        self.lower = torch.tensor(lower, dtype=torch.float32)
        self.upper = torch.tensor(upper, dtype=torch.float32)
        self._clamp = any(v != -float("inf") for v in lower) or any(v != float("inf") for v in upper)

        # unit_circle pairs with at least one noised channel:
        # (sin_idx, cos_idx, sin_mean, sin_std, cos_mean, cos_std).
        self.circle_pairs: List[Tuple[int, int, float, float, float, float]] = []
        seen = set()
        for f_idx in idx:
            base = features[f_idx][len("prev_") :]
            if not base.startswith(("sin_", "cos_")):
                continue
            angle = base[len("sin_") :]
            if angle in seen:
                continue
            seen.add(angle)
            sin_name, cos_name = f"sin_{angle}", f"cos_{angle}"
            if constraints.get(sin_name) != "unit_circle" or constraints.get(cos_name) != "unit_circle":
                continue
            sin_feat, cos_feat = f"prev_{sin_name}", f"prev_{cos_name}"
            if sin_feat not in features or cos_feat not in features:
                log.warning(f"prev noise: '{angle}' has no prev_ sin/cos pair — not projected onto the unit circle.")
                continue
            si, ci = features.index(sin_feat), features.index(cos_feat)
            if si not in std_by_feat_idx or ci not in std_by_feat_idx:
                log.warning(f"prev noise: '{angle}' sin/cos are not both feedback channels — not projected.")
                continue
            self.circle_pairs.append(
                (si, ci, mean_by_feat_idx[si], std_by_feat_idx[si], mean_by_feat_idx[ci], std_by_feat_idx[ci])
            )
            log.info(f"prev noise: '{sin_feat}'/'{cos_feat}' projected back onto the unit circle.")

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
        noised = x[..., idx] + sigma * noise
        if self._clamp:
            lower = self.lower.to(device=x.device, dtype=x.dtype)
            upper = self.upper.to(device=x.device, dtype=x.dtype)
            noised = torch.maximum(torch.minimum(noised, upper), lower)
        x[..., idx] = noised

        # Project sin/cos pairs back onto the unit circle in physical units.
        for si, ci, s_mean, s_std, c_mean, c_std in self.circle_pairs:
            s = x[..., si] * s_std + s_mean
            c = x[..., ci] * c_std + c_mean
            norm = torch.sqrt((s * s + c * c).clamp(min=1e-16))
            x[..., si] = (s / norm - s_mean) / s_std
            x[..., ci] = (c / norm - c_mean) / c_std
        return x


def build_prev_noise_injector(
    datamodule,
    noise_std: Optional[Dict[str, float]],
    output_constraints: Optional[Dict[str, str]] = None,
) -> Optional[PrevNoiseInjector]:
    """Build a :class:`PrevNoiseInjector` from a datamodule, or ``None`` if disabled.

    Args:
        datamodule: Datamodule exposing ``features`` and the ``feedback_map``
            built during ``setup(stage="fit")``.
        noise_std: Target-name to sigma mapping, or ``None``/empty to disable.
        output_constraints: The model's output constraints, used to clamp the
            noised channels and project sin/cos pairs back onto the unit
            circle; ``None`` leaves them unconstrained.

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
        output_constraints=output_constraints,
    )
    if not injector:
        log.warning("prev_noise_std resolved to no active channels; training without noise injection.")
        return None
    return injector
