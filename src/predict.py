# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

"""Full-domain prediction entrypoint for pointwise/sequence WWM emulator models (MLP/LSTM).

Invoked via Hydra using ``configs/predict.yaml`` as the base config. Loads a trained
checkpoint, runs single-step or teacher-forced sequence prediction over the requested
day ranges, and assembles the results into an ``xarray.Dataset``. For graph-based
backbones (GAT/GAT-VN/GAT-LSTM-VN), use ``src/predict_gcn.py`` instead.
"""

from typing import Any, List, cast

import hydra
import numpy as np
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import warnings

import torch
import xarray as xr

from src.utils.instantiators import instantiate_loggers
from src.utils.pylogger import RankedLogger
from src.utils.utils import log_hyperparameters, print_config_tree
from src.data import dataset_utils as du

log = RankedLogger(__name__, rank_zero_only=True)

warnings.filterwarnings("ignore")


def predict(cfg: DictConfig) -> xr.Dataset:
    """Predicts given checkpoint on a datamodule testset.

    Args:
      cfg (DictConfig): Configuration composed by hydra.

    Returns:
      xr.Dataset: Dataset with predictions.
    """
    assert cfg.ckpt_path

    if len(du.normalize_day_ranges(list(cfg.data.test_days))) > 1:
        raise NotImplementedError(
            "predict.py assumes a contiguous test range when reshaping predictions; "
            "run once per range instead, e.g. data.test_days=[212,242]."
        )

    log.info("Printing config tree with Rich <cfg.extras.print_config=True>")
    print_config_tree(cfg, resolve=True, save_to_file=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters")
        log_hyperparameters(object_dict)

    log.info("Starting predictions")
    predictions = cast(
        list[torch.Tensor],
        trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path),
    )
    predictions = torch.concatenate(predictions)  # Shape: (n_samples, n_targets)

    # run test metrics that are synced to wandb
    if cfg.test_metrics:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

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
    config_targets = cfg.data.targets
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
    boundary_nodes_csv = cfg.data.get("boundary_nodes_csv", None)
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
            "checkpoint": cfg.ckpt_path,
            "model": cfg.model._target_,
            "seq_len": seq_len,
        },
    )

    # Save predictions to netCDF
    output_path = f"{cfg.paths.output_dir}/predictions.nc"
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
