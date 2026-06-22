# SPDX-FileCopyrightText: 2021 ashleve
# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from pathlib import Path
from importlib.util import find_spec
from lightning_utilities.core.rank_zero import rank_zero_only
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sequence

import rich
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
import xarray as xr

from src.utils import pylogger
from src.data import dataset_utils as du

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    Args:
        task_func (Callable): The task function to be wrapped.

    Returns:
        Callable: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    Args:
        metric_dict (Dict[str, Any]): Dictionary containing metric values.
        metric_name (str, optional): Name of the metric to retrieve.

    Returns:
        float: Value of the requested metric.
    """
    if not metric_name:
        log.info("Metric name (`optimized_metric`) is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    Args:
        object_dict (Dict[str, Any]): A dictionary containing the following objects:
            - `"cfg"`: A DictConfig object containing the main config.
            - `"model"`: The Lightning model.
            - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    Args:
        cfg: A DictConfig composed by Hydra.
        print_order: Determines in what order config components are printed. Default is ``("data", "model",
            "callbacks", "logger", "trainer", "paths")``.
        resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
        save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


def get_bin_mask(
    values: Union[np.ndarray, torch.Tensor], bins: List[float], bin_idx: int
) -> Union[np.ndarray, torch.Tensor]:
    """Get boolean mask for samples in a specific bin.

    First bin: values < bins[0]
    Middle bins: bins[bin_idx-1] <= values < bins[bin_idx]
    Last bin: values >= bins[-1]

    Args:
        values: Tensor of values to bin.
        bins: List of bin edges (sorted).
        bin_idx: Index of the bin (0 = < first edge, 1 = [first, second), etc.).

    Returns:
        Boolean mask for samples in the specified bin.
    """
    if bin_idx == 0:
        return values < bins[0]
    elif bin_idx == len(bins):
        return values >= bins[-1]
    else:
        return (values >= bins[bin_idx - 1]) & (values < bins[bin_idx])


def circular_mae(pred_deg: np.ndarray, truth_deg: np.ndarray) -> float:
    """Mean absolute error for circular quantities given in degrees.

    Accounts for wrap-around at 0°/360° by taking the minimum of the direct
    angular difference and its complement.

    Args:
        pred_deg: Predicted angles in degrees (any shape).
        truth_deg: Ground-truth angles in degrees (same shape).

    Returns:
        Scalar mean circular absolute error in degrees.
    """
    diff = np.abs(pred_deg - truth_deg) % 360.0
    diff = np.where(diff > 180.0, 360.0 - diff, diff)
    return float(np.mean(diff))


def circular_rmse(pred_deg: np.ndarray, truth_deg: np.ndarray) -> float:
    """Root mean squared error for circular quantities given in degrees.

    Accounts for wrap-around at 0°/360°.

    Args:
        pred_deg: Predicted angles in degrees (any shape).
        truth_deg: Ground-truth angles in degrees (same shape).

    Returns:
        Scalar circular RMSE in degrees.
    """
    diff = np.abs(pred_deg - truth_deg) % 360.0
    diff = np.where(diff > 180.0, 360.0 - diff, diff)
    return float(np.sqrt(np.mean(diff**2)))


def generate_predictions_ml(
    ml_model,
    datamodule,
    target_names: List[str],
    model_target: str,
) -> xr.Dataset:
    """Generate predictions with a machine learning model.

    Args:
        ml_model: Trained ML model wrapper
        datamodule: Lightning DataModule
        target_names: List of target variable names
        model_target: Model target name for attributes

    Returns:
        xr.Dataset with predictions
    """
    log.info("Generating predictions on test set...")

    test_dataset = datamodule.ds_test
    X_test = test_dataset.X

    predictions = ml_model.predict(X_test)  # Shape: (n_samples, n_targets)
    if len(predictions.shape) == 1:
        predictions = predictions[:, np.newaxis]

    ds = test_dataset.ds
    n_time = ds.sizes["time"]
    n_nodes = ds.sizes["nSCHISM_hgrid_node"]
    predictions = predictions.reshape(n_time, n_nodes, len(target_names))

    ds_pred = du.build_prediction_dataset(
        predictions=predictions,
        target_names=target_names,
        ds=ds,
        features=[],
        scaling_params=None,
        time_coords=ds.time,
        extra_attrs={"model": model_target},
    )

    return ds_pred
