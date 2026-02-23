# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from typing import List

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

import warnings

import torch
import xarray as xr

from src.utils.instantiators import instantiate_loggers
from src.utils.pylogger import RankedLogger
from src.utils.utils import log_hyperparameters, print_config_tree
from src.data.SimpleWWMDataset import SimpleWWMDataset

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
    predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    predictions = torch.concatenate(predictions)  # Shape: (n_samples, n_targets)

    log.info("Retrieving metadata from source dataset")
    ds = datamodule.ds_pred.ds

    # Check if using sequence dataset (LSTM) - predictions start at seq_len timestep
    is_sequence_dataset = hasattr(datamodule.ds_pred, "seq_len")
    if is_sequence_dataset:
        seq_len = datamodule.ds_pred.seq_len
        n_valid_times = datamodule.ds_pred.n_valid_times
        log.info(f"Sequence dataset detected: seq_len={seq_len}, n_valid_times={n_valid_times}")
        # Predictions are for timesteps [seq_len:] only
        pred_time_slice = slice(seq_len, None)
        pred_n_time = n_valid_times
    else:
        seq_len = 0
        pred_time_slice = slice(None)
        pred_n_time = ds.sizes["time"]

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

    # Create xarray Dataset with predictions
    data_vars = {}

    for i, target_name in enumerate(target_names):
        # Extract predictions for this target
        target_predictions = predictions[:, i]

        # Reshape to match spatial and temporal dimensions
        # For sequence datasets, pred_n_time < ds.sizes["time"]
        predictions_reshaped = target_predictions.reshape(pred_n_time, n_nodes)

        # If this is a delta prediction, reconstruct the full values
        if target_name.startswith("delta_"):
            original_var = target_name.replace("delta_", "")
            log.info(f"Reconstructing {original_var} from delta predictions")

            # For sequence datasets, slice the source dataset to matching time range
            ds_for_reconstruction = ds.isel(time=pred_time_slice) if is_sequence_dataset else ds

            delta_predictions = predictions_reshaped.cpu().numpy()
            reconstructed, ds_for_reconstruction = SimpleWWMDataset.reconstruct_from_delta(
                delta_predictions=delta_predictions,
                dataset=ds_for_reconstruction,
                target_name=target_name,
                features=datamodule.ds_pred.features,
                scaling_params=datamodule.ds_pred.scaling_params,
            )

            # Store both the reconstructed values and the delta predictions
            data_vars[original_var] = (
                ["time", "nSCHISM_hgrid_node"],
                reconstructed,
            )
            data_vars[target_name] = (
                ["time", "nSCHISM_hgrid_node"],
                delta_predictions,
            )
        else:
            # Direct prediction of the target variable
            data_vars[target_name] = (
                ["time", "nSCHISM_hgrid_node"],
                predictions_reshaped.cpu().numpy(),
            )

    otn = [tn.replace("delta_", "") for tn in target_names]

    # For sequence datasets, use sliced time coordinates
    pred_time_coords = ds.time.isel(time=pred_time_slice)

    ds_pred = xr.Dataset(
        data_vars,
        coords={
            "time": pred_time_coords,
            "nSCHISM_hgrid_node": ds.nSCHISM_hgrid_node,
            "nSCHISM_hgrid_node_x": ds.SCHISM_hgrid_node_x,
            "nSCHISM_hgrid_node_y": ds.SCHISM_hgrid_node_y,
        },
        attrs={
            "description": f"Predictions for {', '.join(otn)}",
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
