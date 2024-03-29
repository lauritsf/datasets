from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform


class GetTarget(BaseTransform):
    def __init__(self, target: Optional[int] = None) -> None:
        self.target = [target]

    def forward(self, data: Data) -> Data:
        if self.target is not None:
            data.y = data.y[:, self.target]
        return data


class QM9DataModule(pl.LightningDataModule):
    target_types = ["atomwise" for _ in range(19)]
    target_types[0] = "dipole_moment"
    target_types[5] = "electronic_spatial_extent"

    # Specify unit conversions (eV to meV).
    unit_conversion = {
        i: (lambda t: 1000 * t) if i not in [0, 1, 5, 11, 16, 17, 18] else (lambda t: t) for i in range(19)
    }

    def __init__(
        self,
        target: int = 0,
        data_dir: str = "data/",
        batch_size_train: int = 32,
        batch_size_inference: int = 32,
        num_workers: int = 0,
        splits: Union[List[int], List[float]] | None = None,  # default: [0.8, 0.1, 0.1]
        seed: int = 0,
        subset_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.target = target
        self.data_dir = Path(data_dir) / "qm9"
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.num_workers = num_workers
        self.splits = [0.8, 0.1, 0.1] if splits is None else splits
        self.seed = seed
        self.subset_size = subset_size

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self) -> None:
        # Download data
        QM9(root=self.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = QM9(root=self.data_dir, transform=GetTarget(self.target))

        # Shuffle dataset
        rng = np.random.default_rng(seed=self.seed)
        dataset = dataset[rng.permutation(len(dataset))]

        # Subset dataset
        if self.subset_size is not None:
            dataset = dataset[: self.subset_size]

        # Split dataset
        if all(isinstance(split, int) for split in self.splits):
            split_sizes = self.splits
        elif all(isinstance(split, float) for split in self.splits):
            split_sizes = [int(len(dataset) * prop) for prop in self.splits]

        split_idx = np.cumsum(split_sizes)
        self.data_train = dataset[: split_idx[0]]
        self.data_val = dataset[split_idx[0] : split_idx[1]]
        self.data_test = dataset[split_idx[1] :]

    def get_target_stats(self, remove_atom_refs=False, divide_by_atoms=False):
        atom_refs = self.data_train.atomref(self.target)

        ys = []
        for batch in self.train_dataloader(shuffle=False):
            y = batch.y.clone()
            if remove_atom_refs and atom_refs is not None:
                y.index_add_(dim=0, index=batch.batch, source=-atom_refs[batch.z])
            if divide_by_atoms:
                _, num_atoms = torch.unique(batch.batch, return_counts=True)
                y = y / num_atoms.unsqueeze(-1)
            ys.append(y)

        y = torch.cat(ys, dim=0)
        return y.mean(), y.std(), atom_refs

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
