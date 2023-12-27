from .datamodule import UciDataModule
from .dataset_loader import UciDataset, get_dataset

__all__ = ["get_dataset", "get_datamodule", "UciDataset", "UciDataModule"]


def get_datamodule(
    name: str,
    data_dir: str = "data/",
    train_val_test_split: tuple[int, int, int] | tuple[float, float, float] = (0.8, 0.1, 0.1),
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> UciDataModule:
    """Get a UCI dataset as a PyTorch Lightning DataModule.

    Args:
        name: Name of the dataset.
        data_dir: Directory to store the dataset.
        train_val_test_split: Split of the dataset into train, validation, and test set.
        batch_size: Batch size.
        num_workers: Number of workers for the DataLoader.
        pin_memory: Pin memory for the DataLoader.

    Returns:
        A PyTorch Lightning DataModule.
    """
    return UciDataModule(
        name=name,
        data_dir=data_dir,
        train_val_test_split=train_val_test_split,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
