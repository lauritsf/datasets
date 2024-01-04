# QM9 dataset

**Note:**
This dataset contains 130k samples, which is fewer than the full 134k samples in the original dataset (Some molecules have been filtered)

The QM9 dataset is available from [`torch_geometric.datasets.QM9`](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.QM9).

This directory contains Lightning module implementations for the QM9 dataset.
By default, the datamodule will return the `Dipole moment` property (index 0) as the target.

The data is downloaded to the `<data_dir>/qm9` directory.

## Usage (LightningModule)

```python
from dataset_baselines.datasets.qm9 import QM9DataModule

dm = QM9DataModule(data_dir="<data_dir>", target=7) # eg. target=7 for Internal energy at 0K
dm.prepare_data() # Optional (downloads the dataset)
dm.setup()
train_dataloader = dm.train_dataloader()
val_dataloader = dm.val_dataloader()
test_dataloader = dm.test_dataloader()
```
