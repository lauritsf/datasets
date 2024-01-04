import lightning as L
import torch
from torch.utils.data import DataLoader, Subset, random_split

from .dataset_loader import get_dataset


class UciDataModule(L.LightningDataModule):
    def __init__(
        self,
        name: str,
        data_dir: str = "data/",
        train_val_test_split: tuple[int, int, int] | tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int | None = None,  # Set seed for reproducible splits
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Subset | None = None
        self.data_val: Subset | None = None
        self.data_test: Subset | None = None

        self.num_input_features: int | None = None
        self.num_output_features: int | None = None

    def prepare_data(self):
        """Download data if not already downloaded."""
        dataset = get_dataset(name=self.hparams["name"], data_dir=self.hparams["data_dir"])
        self.num_input_features = dataset.num_input_features
        self.num_output_features = dataset.num_output_features

    def setup(self, stage: str | None = None):
        del stage  # unused

        dataset = get_dataset(name=self.hparams["name"], data_dir=self.hparams["data_dir"])
        self.num_input_features = dataset.num_input_features
        self.num_output_features = dataset.num_output_features

        split = self.hparams["train_val_test_split"]
        if isinstance(split[0], float):
            # Convert from float to int
            split = [int(len(dataset) * s) for s in split]
            split[0] += len(dataset) - sum(split)

        seed = self.hparams["seed"]
        self.data_train, self.data_val, self.data_test = random_split(
            dataset,
            split,
            generator=torch.Generator().manual_seed(seed) if seed is not None else None,
        )

    def train_dataloader(self) -> DataLoader:
        if self.data_train is None:
            raise ValueError("Dataset not loaded.")
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.data_val is None:
            raise ValueError("Dataset not loaded.")
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.data_test is None:
            raise ValueError("Dataset not loaded.")
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
        )
