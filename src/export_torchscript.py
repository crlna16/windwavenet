# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Export the final SWH-1h checkpoints to TorchScript.

Covers the same 10 checkpoints as ``scripts/submit_test_swh_1h_array.sh``
(DenseNet, LSTM x2 sequence lengths, MultiHeadGAT, MultiHeadGATVN x2 sub-graph
sizes, MultiHeadGATLSTMVN x2 sequence lengths x2 sub-graph sizes). For each
checkpoint this script:

1. Builds the experiment's Hydra config and loads the real trained checkpoint,
   exactly as ``scripts/submit_test_swh_1h_array.sh`` does for ``src/predict.py``
   / ``src/predict_gcn.py`` (same ``experiment=`` / ``ckpt_path=`` overrides).
2. Fetches two independent real batches from the zarr dataset via the
   datamodule's ``predict_dataloader()``: one for tracing, one for
   verification.
3. Wraps the model (graph models need a thin adapter, see
   ``GraphExportWrapper``, since ``torch.jit.trace`` cannot safely trace a
   forward that takes a PyG ``Data``/``Batch`` object) and traces it with
   ``torch.jit.trace``.
4. Saves the traced module with ``torch.jit.save``.
5. Verifies the export: reloads it with ``torch.jit.load`` and compares its
   output against the eager model's output on the SECOND, independent batch
   (never seen during tracing), to catch tensors that got silently baked in
   as trace-time constants instead of staying dynamic inputs.

Usage::

    python src/export_torchscript.py
    python src/export_torchscript.py --labels densenet lstm-6
    python src/export_torchscript.py --output-dir checkpoints/torchscript --device cpu
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn as nn
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig

from src.model.WWMEmulator import WWMEmulator
from src.utils import predict_utils as pu
from src.utils.pylogger import RankedLogger

# Initialize the distributed rank so RankedLogger works without a Lightning trainer.
if getattr(rank_zero_only, "rank", None) is None:
    rank_zero_only.rank = 0

# This script uses hydra.initialize()/compose() instead of the @hydra.main decorator, so Hydra's
# job logging config never runs and RankedLogger's log.info() calls would otherwise go nowhere.
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

log = RankedLogger(__name__, rank_zero_only=True)


# ---------------------------------------------------------------------------
# Experiment definitions (mirrors scripts/submit_test_swh_1h_array.sh)
# ---------------------------------------------------------------------------


@dataclass
class ExperimentSpec:
    """One checkpoint to export: a backbone / hyperparameter combination."""

    label: str
    config_name: str  # base Hydra config: "predict", "predict_lstm", or "predict_gcn"
    experiment: str  # experiment=<name> override
    ckpt_path: str  # relative to the repository root
    family: str  # "dense", "lstm", "gat", "gat_vn", or "gat_lstm_vn"
    seq_len: Optional[int] = None
    max_nodes: Optional[int] = None


EXPERIMENTS: List[ExperimentSpec] = [
    ExperimentSpec(
        label="densenet",
        config_name="predict",
        experiment="swh_1h_mlp",
        ckpt_path="logs/train/swh_1h_densenet/runs/2026-09-10_14-22-52/wwm-emulator-hourly-final/"
        "sytiiae9/checkpoints/epoch=58-step=214229.ckpt",
        family="dense",
    ),
    ExperimentSpec(
        label="lstm-6",
        config_name="predict_lstm",
        experiment="swh_1h_lstm",
        ckpt_path="logs/train/swh_1h_lstm/runs/2026-09-10_14-22-52/wwm-emulator-hourly-final/"
        "k6ydmtnc/checkpoints/epoch=41-step=152208.ckpt",
        family="lstm",
        seq_len=6,
    ),
    ExperimentSpec(
        label="lstm-48",
        config_name="predict_lstm",
        experiment="swh_1h_lstm",
        ckpt_path="logs/train/swh_1h_lstm/runs/2026-09-10_14-22-52/wwm-emulator-hourly-final/"
        "0pq97b9h/checkpoints/epoch=23-step=85728.ckpt",
        family="lstm",
        seq_len=48,
    ),
    ExperimentSpec(
        label="gat-500",
        config_name="predict_gcn",
        experiment="swh_1h_gat",
        ckpt_path="logs/train/swh_1h_gat/runs/2026-09-10_14-22-52/wwm-emulator-hourly-final/"
        "zwu7keey/checkpoints/epoch=5-step=87126.ckpt",
        family="gat",
        max_nodes=500,
    ),
    ExperimentSpec(
        label="gat-vn-500",
        config_name="predict_gcn",
        experiment="swh_1h_gat-vn",
        ckpt_path="logs/train/swh_1h_gat-vn/runs/2026-09-11_09-41-02/wwm-emulator-hourly-final/"
        "52stxhrx/checkpoints/epoch=14-step=217815.ckpt",
        family="gat_vn",
        max_nodes=500,
    ),
    ExperimentSpec(
        label="gat-vn-50",
        config_name="predict_gcn",
        experiment="swh_1h_gat-vn",
        ckpt_path="logs/train/swh_1h_gat-vn/runs/2026-09-10_19-03-48/wwm-emulator-hourly-final/"
        "k6kbc2nd/checkpoints/epoch=25-step=377546.ckpt",
        family="gat_vn",
        max_nodes=50,
    ),
    ExperimentSpec(
        label="gat-lstm-vn-6-500",
        config_name="predict_gcn",
        experiment="swh_1h_gat-lstm-vn",
        ckpt_path="logs/train/swh_1h_gat-lstm-vn/runs/2026-09-11_16-56-08/wwm-emulator-hourly-final/"
        "p5bi2bnv/checkpoints/epoch=19-step=289920.ckpt",
        family="gat_lstm_vn",
        seq_len=6,
        max_nodes=500,
    ),
    ExperimentSpec(
        label="gat-lstm-vn-6-50",
        config_name="predict_gcn",
        experiment="swh_1h_gat-lstm-vn",
        ckpt_path="logs/train/swh_1h_gat-lstm-vn/runs/2026-09-10_23-14-40/wwm-emulator-hourly-final/"
        "wypqt1z9/checkpoints/epoch=23-step=347904.ckpt",
        family="gat_lstm_vn",
        seq_len=6,
        max_nodes=50,
    ),
    ExperimentSpec(
        label="gat-lstm-vn-48-500",
        config_name="predict_gcn",
        experiment="swh_1h_gat-lstm-vn",
        ckpt_path="logs/train/swh_1h_gat-lstm-vn/runs/2026-09-11_17-04-00/wwm-emulator-hourly-final/"
        "n1ye9zqj/checkpoints/epoch=11-step=171456.ckpt",
        family="gat_lstm_vn",
        seq_len=48,
        max_nodes=500,
    ),
    ExperimentSpec(
        label="gat-lstm-vn-48-50",
        config_name="predict_gcn",
        experiment="swh_1h_gat-lstm-vn",
        ckpt_path="logs/train/swh_1h_gat-lstm-vn/runs/2026-09-14_09-23-07/wwm-emulator-hourly-final/"
        "1mosibic/checkpoints/epoch=8-step=128592.ckpt",
        family="gat_lstm_vn",
        seq_len=48,
        max_nodes=50,
    ),
]


# ---------------------------------------------------------------------------
# Graph model export adapter
# ---------------------------------------------------------------------------


class _BackboneExportBase(nn.Module):
    """Holds only the emulator's backbone plus the metadata its output constraints need.

    The ``WWMEmulator`` (a ``LightningModule``) must NOT be registered as a submodule here:
    ``torch.jit.trace`` recursively probes every submodule's attributes with ``hasattr`` (looking
    for ``@torch.jit.export`` methods), and ``LightningModule.trainer`` raises ``RuntimeError``
    (not ``AttributeError``) when no ``Trainer`` is attached. So the plain ``nn.Module`` backbone
    is traced directly and ``WWMEmulator._apply_output_constraints`` is re-used on this object,
    which carries the three attributes that method reads.
    """

    CONSTRAINT_FUNCTIONS = WWMEmulator.CONSTRAINT_FUNCTIONS

    def __init__(self, model: WWMEmulator) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.target_names = list(model.target_names)
        self.output_constraints = dict(model.output_constraints)

    def _apply_output_constraints(self, preds: torch.Tensor) -> torch.Tensor:
        return WWMEmulator._apply_output_constraints(self, preds)


class GraphExportWrapper(_BackboneExportBase):
    """Adapts a ``GraphWWMEmulator`` (PyG-``Data``-object forward) to explicit tensor args.

    ``torch.jit.trace`` cannot safely trace a model whose forward takes a
    custom Python object (e.g. a PyG ``Batch``): tensors reachable only
    through such a non-tensor argument get silently baked in as trace-time
    constants instead of staying real graph inputs. This wrapper instead
    exposes the tensors the graph backbones actually read (``x``,
    ``edge_index``, ``edge_weight``, ``batch``[, ``x_center_seq``]) as
    ordinary positional arguments and re-assembles a lightweight namespace
    for the wrapped model to consume.
    """

    def __init__(self, model: WWMEmulator, needs_x_center_seq: bool) -> None:
        super().__init__(model)
        self.needs_x_center_seq = needs_x_center_seq

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch: torch.Tensor,
        x_center_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        data = SimpleNamespace(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        if self.needs_x_center_seq:
            data.x_center_seq = x_center_seq
        return self._apply_output_constraints(self.backbone(data))


class DenseLSTMExportWrapper(_BackboneExportBase):
    """Mirrors ``WWMEmulator.forward`` (backbone + output constraints) without the LightningModule."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_output_constraints(self.backbone(x))


def build_traceable_module(model: WWMEmulator, family: str) -> nn.Module:
    """Wraps the loaded model so it can be handed to ``torch.jit.trace`` with plain tensors."""
    if family in ("dense", "lstm"):
        return DenseLSTMExportWrapper(model)
    return GraphExportWrapper(model, needs_x_center_seq=(family == "gat_lstm_vn"))


def build_example_inputs(family: str, batch: Any) -> Tuple[torch.Tensor, ...]:
    """Extracts the positional tensor args ``build_traceable_module``'s output expects from a batch."""
    if family in ("dense", "lstm"):
        return (batch,)  # batch is already the raw (batch_size, ...) feature tensor
    if family in ("gat", "gat_vn"):
        return (batch.x, batch.edge_index, batch.edge_weight, batch.batch)
    if family == "gat_lstm_vn":
        return (batch.x, batch.edge_index, batch.edge_weight, batch.batch, batch.x_center_seq)
    raise ValueError(f"Unknown family: {family}")


# ---------------------------------------------------------------------------
# Config / data helpers
# ---------------------------------------------------------------------------


def build_overrides(spec: ExperimentSpec, repo_root: Path) -> List[str]:
    """Builds the list of Hydra override strings for one experiment spec."""
    ckpt_abs = str((repo_root / spec.ckpt_path).resolve())
    # Checkpoint filenames contain "=" (e.g. "epoch=2-step=3972.ckpt"), which
    # Hydra's override grammar would otherwise choke on. Quoting the value
    # forces Hydra to treat it as a single opaque string.
    overrides = [
        f"experiment={spec.experiment}",
        f"ckpt_path='{ckpt_abs}'",
        "data.num_workers=0",
        "data.persistent_workers=false",
        # Dataset structure is time-invariant and seq_len maxes out at 48h, so a 5-day slice of the
        # configured test range is enough for tracing/verification without loading the full 2 months.
        "data.test_days=[336,341]",
    ]
    if spec.seq_len is not None:
        overrides.append(f"data.seq_len={spec.seq_len}")
    if spec.max_nodes is not None:
        overrides.append(f"data.max_nodes={spec.max_nodes}")
    return overrides


def _unwrap_batch(batch: Any) -> Any:
    """Reduces a raw DataLoader batch to the object the model's forward expects."""
    if hasattr(batch, "edge_index"):
        return batch  # PyG Batch, returned as-is
    if isinstance(batch, (list, tuple)):
        first = batch[0]
        if hasattr(first, "edge_index"):
            return first  # PyG DataLoader wrapped the Batch in a list
        return first  # (x, y[, weights]) tuple -> keep features only
    return batch


def get_batches(cfg: DictConfig, n: int) -> List[Any]:
    """Instantiates the datamodule and fetches ``n`` independent real batches."""
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup(stage="predict")
    loader = datamodule.predict_dataloader()
    it = iter(loader)
    batches = [_unwrap_batch(next(it)) for _ in range(n)]
    return batches


# ---------------------------------------------------------------------------
# Export + verification
# ---------------------------------------------------------------------------


def verify_export(
    label: str,
    traceable: nn.Module,
    output_path: Path,
    family: str,
    verify_batches: List[Any],
    device: torch.device,
    atol: float,
    rtol: float,
) -> bool:
    """Compares the reloaded TorchScript module against the eager module on several fresh batches.

    Each batch is independent from the tracing batch and, for graph families, generally has a
    different node/edge count, so passing on all of them is evidence the trace stayed shape-dynamic
    rather than baking the trace-time shapes in as constants.
    """
    loaded = torch.jit.load(str(output_path), map_location=device)
    loaded.eval()

    all_close = True
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    for i, verify_batch in enumerate(verify_batches):
        example_inputs = build_example_inputs(family, verify_batch.to(device))

        with torch.no_grad():
            eager_out = traceable(*example_inputs)
            traced_out = loaded(*example_inputs)

        abs_diff = (eager_out - traced_out).abs()
        # Relative to the eager magnitude, mirroring allclose's rtol * |eager| term.
        rel_diff = abs_diff / eager_out.abs().clamp_min(atol)
        batch_abs = abs_diff.max().item()
        batch_rel = rel_diff.max().item()
        max_abs_diff = max(max_abs_diff, batch_abs)
        max_rel_diff = max(max_rel_diff, batch_rel)
        if not torch.allclose(eager_out, traced_out, atol=atol, rtol=rtol):
            all_close = False
        log.info(
            f"[{label}] batch {i} (input {tuple(example_inputs[0].shape)}): "
            f"max abs diff {batch_abs:.3e}, max rel diff {batch_rel:.3e}"
        )

    if all_close:
        log.info(
            f"[{label}] verification OK over {len(verify_batches)} fresh batches - "
            f"max abs diff {max_abs_diff:.3e}, max rel diff {max_rel_diff:.3e}"
        )
    else:
        log.error(
            f"[{label}] verification FAILED over {len(verify_batches)} fresh batches - "
            f"max abs diff {max_abs_diff:.3e}, max rel diff {max_rel_diff:.3e} (atol={atol}, rtol={rtol})"
        )
    return all_close


def export_one(
    spec: ExperimentSpec,
    cfg: DictConfig,
    device: torch.device,
    output_dir: Path,
    atol: float,
    rtol: float,
    num_verify_batches: int,
) -> bool:
    """Loads, traces, saves, and verifies a single checkpoint."""
    model = pu.load_model_from_checkpoint(cfg, device)

    batches = get_batches(cfg, n=1 + num_verify_batches)
    trace_batch = batches[0].to(device)
    verify_batches = batches[1:]

    traceable = build_traceable_module(model, spec.family)
    example_inputs = build_example_inputs(spec.family, trace_batch)

    log.info(f"[{spec.label}] tracing ...")
    with torch.no_grad():
        traced = torch.jit.trace(traceable, example_inputs, check_trace=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{spec.label}.pt"
    torch.jit.save(traced, str(output_path))
    log.info(f"[{spec.label}] saved TorchScript module to {output_path}")

    return verify_export(spec.label, traceable, output_path, spec.family, verify_batches, device, atol, rtol)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help=f"Subset of experiment labels to export (default: all). Choices: {[s.label for s in EXPERIMENTS]}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/trained_windwavenet_checkpoints/torchscript",
        help="Output directory for the exported .pt files, relative to the repository root.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to load, trace, and verify the model on (default: cpu, for portability).",
    )
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for verification.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for verification.")
    parser.add_argument(
        "--num-verify-batches",
        type=int,
        default=3,
        help="Number of independent fresh batches to verify each export against (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(args)
    repo_root = Path(__file__).resolve().parents[1]
    print(repo_root)
    output_dir = (repo_root / args.output_dir).resolve()
    print(output_dir)

    device = torch.device(args.device)
    print(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        log.warning("--device=cuda requested but CUDA is not available; falling back to cpu.")
        device = torch.device("cpu")

    specs = EXPERIMENTS
    if args.labels:
        wanted = set(args.labels)
        known = {s.label for s in EXPERIMENTS}
        unknown = wanted - known
        if unknown:
            raise ValueError(f"Unknown experiment label(s): {sorted(unknown)}. Choices: {sorted(known)}")
        specs = [s for s in EXPERIMENTS if s.label in wanted]

    results: Dict[str, Optional[bool]] = {}
    for spec in specs:
        log.info(f"=== {spec.label} ===")
        ckpt_abs = repo_root / spec.ckpt_path
        if not ckpt_abs.is_file():
            log.warning(f"[{spec.label}] checkpoint not found at {ckpt_abs}; skipping.")
            results[spec.label] = None
            continue

        overrides = build_overrides(spec, repo_root)
        with hydra.initialize(version_base="1.3", config_path="../configs"):
            cfg = hydra.compose(config_name=spec.config_name, overrides=overrides)

        results[spec.label] = export_one(spec, cfg, device, output_dir, args.atol, args.rtol, args.num_verify_batches)

    log.info("=== Summary ===")
    for label, ok in results.items():
        status = "SKIPPED (checkpoint not found)" if ok is None else ("OK" if ok else "FAILED verification")
        log.info(f"  {label}: {status}")

    if any(ok is False for ok in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
