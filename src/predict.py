# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Full-domain prediction entrypoint for pointwise/sequence WWM emulator models (MLP/LSTM).

Invoked via Hydra using ``configs/predict.yaml`` as the base config. Loads a trained
checkpoint, runs single-step or teacher-forced sequence prediction over the requested
day ranges, and assembles the results into an ``xarray.Dataset``. For graph-based
backbones (GAT/GAT-VN/GAT-LSTM-VN), use ``src/predict_gcn.py`` instead.

If ``data.test_days`` is a list of ``[start, end]`` ranges (e.g. ``[[183, 213], [336, 366]]``),
prediction runs once per range and each result is saved to its own
``predictions_<start>-<end>.nc`` file.
"""

import copy
from typing import Any, List, Union, cast

import hydra
import numpy as np
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

import warnings

import torch
import xarray as xr

from src.utils.instantiators import instantiate_loggers
from src.utils.pylogger import RankedLogger
from src.utils.utils import log_hyperparameters, print_config_tree
from src.data import dataset_utils as du

log = RankedLogger(__name__, rank_zero_only=True)

warnings.filterwarnings("ignore")


def predict(cfg: DictConfig) -> Union[xr.Dataset, List[xr.Dataset]]:
    """Predicts given checkpoint on a datamodule testset.

    If ``cfg.data.test_days`` specifies multiple day ranges, the prediction
    pipeline runs once per range (each with its own datamodule instance,
    since the reshaping logic assumes a single contiguous range) and each
    result is saved to its own ``predictions_<start>-<end>.nc`` file.

    Args:
      cfg (DictConfig): Configuration composed by hydra.

    Returns:
      xr.Dataset or List[xr.Dataset]: Dataset(s) with predictions, one per
        test day range.
    """
    assert cfg.ckpt_path

    test_ranges = du.normalize_day_ranges(list(cfg.data.test_days))

    log.info("Printing config tree with Rich <cfg.extras.print_config=True>")
    print_config_tree(cfg, resolve=True, save_to_file=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    if logger:
        log.info("Logging hyperparameters")
        log_hyperparameters({"cfg": cfg, "model": model, "logger": logger, "trainer": trainer})

    results = [
        _predict_single_range(cfg, model, trainer, day_range, is_multi_range=len(test_ranges) > 1)
        for day_range in test_ranges
    ]

    return results[0] if len(results) == 1 else results


def _predict_single_range(
    cfg: DictConfig,
    model: LightningModule,
    trainer: Trainer,
    day_range: List[int],
    is_multi_range: bool,
) -> xr.Dataset:
    """Runs prediction for a single contiguous day range and saves the result.

    Args:
        cfg: Base Hydra configuration (``data.test_days`` is overridden per range).
        model: Instantiated Lightning model, shared across ranges.
        trainer: Instantiated Lightning trainer, shared across ranges.
        day_range: Single ``[start_day, end_day]`` range to predict on.
        is_multi_range: Whether multiple ranges are being processed, which
            determines the output filename.

    Returns:
        xr.Dataset: Dataset with predictions for this range.
    """
    range_cfg = copy.deepcopy(cfg)
    with open_dict(range_cfg.data):
        range_cfg.data.test_days = day_range

    log.info(f"Instantiating datamodule <{range_cfg.data._target_}> for day range {day_range}")
    datamodule: LightningDataModule = hydra.utils.instantiate(range_cfg.data)

    log.info(f"Starting predictions for day range {day_range}")
    predictions = cast(
        list[torch.Tensor],
        trainer.predict(model=model, datamodule=datamodule, ckpt_path=range_cfg.ckpt_path),
    )
    predictions = torch.concatenate(predictions)  # Shape: (n_samples, n_targets)

    # run test metrics that are synced to wandb
    if range_cfg.test_metrics:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=range_cfg.ckpt_path)

    log.info("Retrieving metadata from source dataset")
    datamodule_data = cast(Any, datamodule)
    if datamodule_data.ds_pred is None:
        # trainer.predict()'s teardown hook frees ds_pred after the predict stage
        # finishes, so it must be rebuilt here before we can read its metadata.
        log.info("ds_pred was freed by teardown; re-instantiating predict dataset")
        datamodule.setup(stage="predict")
    ds_pred_data = datamodule_data.ds_pred
    ds = ds_pred_data.ds
    features = getattr(ds_pred_data, "features", [])
    scaling_params = getattr(ds_pred_data, "scaling_params", None)

    # Check if using sequence dataset (LSTM) - predictions start at seq_len timestep
    is_sequence_dataset = hasattr(ds_pred_data, "seq_len")
    if is_sequence_dataset:
        seq_len = ds_pred_data.seq_len
        n_valid_times = ds_pred_data.n_valid_times
        log.info(f"Sequence dataset detected: seq_len={seq_len}, n_valid_times={n_valid_times}")
        # Predictions are for the last step of each sequence: timesteps [seq_len-1:]
        pred_time_slice = slice(seq_len - 1, None)
        pred_n_time = n_valid_times
    else:
        seq_len = 0
        pred_time_slice = slice(None)
        pred_n_time = ds.sizes["time"]
        log.info(f"Simple dataset detected: pred_n_time={pred_n_time}")

    n_nodes = ds.sizes["nSCHISM_hgrid_node"]
    log.info(f"Dataset dimensions: time={ds.sizes['time']}, nSCHISM_hgrid_node={n_nodes}")
    log.info(f"Prediction dimensions: pred_n_time={pred_n_time}, n_nodes={n_nodes}")

    # Get target names from config and verify they match the model checkpoint
    config_targets = range_cfg.data.targets
    model_targets = model.target_names

    log.info(f"Config targets: {config_targets}")
    log.info(f"Model checkpoint targets: {model_targets}")

    if config_targets != model_targets:
        raise ValueError(
            f"Target mismatch! Config specifies {config_targets} but model was trained on {model_targets}. "
            f"Please ensure the config targets match the model checkpoint."
        )

    target_names = config_targets

    # Reshape predictions to (n_time, n_nodes, n_targets) numpy array
    predictions_np = predictions.cpu().numpy().reshape(pred_n_time, n_nodes, len(target_names))

    # NaN-out boundary nodes if configured (fair comparison with graph models)
    boundary_nodes_csv = range_cfg.data.get("boundary_nodes_csv", None)
    if boundary_nodes_csv is not None:
        boundary_node_ids = du.load_node_indices(boundary_nodes_csv)
        ds_node_ids = ds.nSCHISM_hgrid_node.values
        id_to_pos = {int(nid): pos for pos, nid in enumerate(ds_node_ids)}
        boundary_positions = np.array([id_to_pos[int(nid)] for nid in boundary_node_ids if int(nid) in id_to_pos])
        predictions_np[:, boundary_positions, :] = np.nan
        log.info(f"Set {len(boundary_positions)} boundary node predictions to NaN.")

    # For sequence datasets, slice the source dataset to matching time range
    if is_sequence_dataset:
        ds = ds.isel(time=pred_time_slice)

    ds_pred = du.build_prediction_dataset(
        predictions=predictions_np,
        target_names=target_names,
        ds=ds,
        features=features,
        scaling_params=scaling_params,
        time_coords=ds.time,
        extra_attrs={
            "checkpoint": range_cfg.ckpt_path,
            "model": range_cfg.model._target_,
            "seq_len": seq_len,
        },
    )

    # Save predictions to netCDF
    if is_multi_range:
        output_name = f"predictions_{day_range[0]}-{day_range[1]}.nc"
    else:
        output_name = "predictions.nc"
    output_path = f"{range_cfg.paths.output_dir}/{output_name}"
    log.info(f"Saving predictions to {output_path}")
    ds_pred.to_netcdf(output_path)
    log.info(f"Predictions saved successfully to {output_path}")

    return ds_pred


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        cfg [DictConfig]: DictConfig configuration composed by Hydra.
    """
    predict(cfg)


if __name__ == "__main__":
    main()
