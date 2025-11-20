# SPDX-FileCopyrightText: 2025 Hereon
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

log = RankedLogger(__name__, rank_zero_only=True)

warnings.filterwarnings("ignore")


def predict(cfg: DictConfig) -> None:
    """Predicts given checkpoint on a datamodule testset.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
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
    predictions = trainer.predict(
        model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path
    )
    predictions = torch.concatenate(predictions)

    log.info("Retrieving metadata from source dataset")
    ds = datamodule.ds_pred.ds

    ds_dims = ds.sizes
    log.info(
        f"Dataset dimensions: time={ds_dims['time']}, nSCHISM_hgrid_node={ds_dims['nSCHISM_hgrid_node']}"
    )

    # Reshape predictions to match spatial and temporal dimensions
    predictions_reshaped = predictions.reshape(
        ds_dims["time"], ds_dims["nSCHISM_hgrid_node"]
    )

    # Create xarray Dataset with predictions
    ds_pred = xr.Dataset(
        {
            cfg.data.targets[0]: (
                ["time", "nSCHISM_hgrid_node"],
                predictions_reshaped.cpu().numpy(),
            )
        },
        coords={
            "time": ds.time,
            "nSCHISM_hgrid_node": ds.nSCHISM_hgrid_node,
            "nSCHISM_hgrid_node_x": ds.SCHISM_hgrid_node_x,
            "nSCHISM_hgrid_node_y": ds.SCHISM_hgrid_node_y,
        },
        attrs={
            "description": f"Predictions for {cfg.data.targets[0]}",
            "checkpoint": cfg.ckpt_path,
            "model": cfg.model._target_,
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
