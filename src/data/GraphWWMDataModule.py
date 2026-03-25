# SPDX-FileCopyrightText: 2025-2026 Hereon
# SPDX-FileContributor: Caroline Arnold
#
# SPDX-License-Identifier: MIT

from typing import List, Optional

from torch.utils.data import Dataset

from src.data.BaseGraphWWMDataModule import BaseGraphWWMDataModule
from src.data.WWMDatasets import GraphWWMDataset


class GraphWWMDataModule(BaseGraphWWMDataModule):
    """Lightning DataModule for graph-based Wave Wind Model data.

    Each sample is a graph subgraph at a single time step, built from
    the SCHISM mesh via BFS around center nodes.
    """

    def _create_dataset(
        self,
        data_range: List[int],
        node_idx_csv: Optional[str],
        sample_weight_strategy: Optional[str],
        sample_weight_power: Optional[float],
        subsample_prob: Optional[float],
        preload_data: bool,
    ) -> Dataset:
        return GraphWWMDataset(
            data_dir=self.data_dir,
            data_range=data_range,
            features=self.features,
            targets=self.targets,
            graph_features=self.graph_features,
            scaling_config=self.scaling_config,
            node_idx_csv=node_idx_csv,
            sample_weight_strategy=sample_weight_strategy,
            sample_weight_power=sample_weight_power,
            subsample_prob=subsample_prob,
            preload_data=preload_data,
            max_radius=self.max_radius,
            max_nodes=self.max_nodes,
            max_links=self.max_links,
            full_graph_target=self.full_graph_target,
        )
