# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from typing import List, Optional

from torch.utils.data import Dataset

from src.data.BaseWWMDataModule import BaseWWMDataModule
from src.data.WWMDatasets import CenterHistoryGraphWWMDataset


class CenterHistoryGraphWWMDataModule(BaseWWMDataModule):
    """Lightning DataModule for center-history graph-based data.

    Each sample is a graph subgraph where the center node carries a
    temporal sequence of ``seq_len`` time steps, while all other nodes
    carry only current-time features.  Designed for the
    ``VirtualNodeCenterLSTMGCNNet`` backbone.
    """

    def __init__(
        self,
        data_dir: str,
        train_days: List[int],
        val_days: List[int],
        test_days: List[int],
        features: List[str],
        targets: List[str],
        graph_features: List[str],
        seq_len: int,
        scaling_config: Optional[str] = None,
        node_idx_csv: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        sample_weight_strategy: Optional[str] = None,
        sample_weight_power: Optional[float] = 1.0,
        subsample_prob: Optional[float] = None,
        preload_data: bool = False,
        max_radius: float = 10000.0,
        max_nodes: int = 1000,
        max_links: int = 10,
        edge_weight_scale: float = 1000.0,
    ) -> None:
        self.seq_len = seq_len
        self.graph_features = graph_features
        self.max_radius = max_radius
        self.max_nodes = max_nodes
        self.max_links = max_links
        self.edge_weight_scale = edge_weight_scale

        super().__init__(
            data_dir=data_dir,
            train_days=train_days,
            val_days=val_days,
            test_days=test_days,
            features=features,
            targets=targets,
            scaling_config=scaling_config,
            node_idx_csv=node_idx_csv,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            subsample_prob=subsample_prob,
            preload_data=preload_data,
        )

    def _create_dataset(
        self,
        data_range: List[int],
        node_idx_csv: Optional[str],
        sample_weight_strategy: Optional[str],
        sample_weight_power: Optional[float],
        subsample_prob: Optional[float],
        preload_data: bool,
    ) -> Dataset:
        return CenterHistoryGraphWWMDataset(
            data_dir=self.data_dir,
            data_range=data_range,
            features=self.features,
            targets=self.targets,
            graph_features=self.graph_features,
            seq_len=self.seq_len,
            scaling_config=self.scaling_config,
            node_idx_csv=node_idx_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            subsample_prob=subsample_prob,
            preload_data=preload_data,
            max_radius=self.max_radius,
            max_nodes=self.max_nodes,
            max_links=self.max_links,
            edge_weight_scale=self.edge_weight_scale,
        )
