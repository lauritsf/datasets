import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split

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
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    def prepare_data(self):
        """Download data if not already downloaded."""
        _ = get_dataset(name=self.hparams["name"], data_dir=self.hparams["data_dir"])

    def setup(self, stage: str | None = None):
        del stage  # unused

        dataset = get_dataset(name=self.hparams["name"], data_dir=self.hparams["data_dir"])

        split = self.hparams["train_val_test_split"]
        if isinstance(split[0], float):
            # Convert from float to int
            split = [int(len(dataset) * s) for s in split]
            split[0] += len(dataset) - sum(split)

        self.data_train, self.data_val, self.data_test = random_split(
            dataset,
            split,
        )

    def train_dataloader(self):
        if self.data_train is None:
            raise ValueError("Dataset not loaded.")
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=True,
        )

    def val_dataloader(self):
        if self.data_val is None:
            raise ValueError("Dataset not loaded.")
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
        )

    def test_dataloader(self):
        if self.data_test is None:
            raise ValueError("Dataset not loaded.")
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["pin_memory"],
            shuffle=False,
        )
