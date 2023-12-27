# UCI Datasets

This directory contains datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).

The file `dataset_specs.yaml` lists datasets with their specifications, which are used to automate the downloading and processing of the datasets into a standardized CSV format. By default, datasets are downloaded to `data/uci/some_dataset/raw/` and processed into `data/uci/some_dataset/processed/`.


```
data/
└── uci/
    ├── some_dataset/
    │   ├── raw/
    │   │   ├── some_dataset.zip
    │   │   ├── some_dataset.data
    │   │   └── ...
    │   └── processed/
    │       └── some_dataset.csv
    └── ...
```

## Usage

### Dataset (PyTorch)

```python
from dataset_baselines.datasets.uci import get_dataset

dataset = get_dataset('some_dataset')
```

### LightningDataModule (PyTorch Lightning)

```python
from dataset_baselines.datasets.uci import get_datamodule

datamodule = get_datamodule('some_dataset')
```

## Datasets

Refer to `dataset_specs.yaml` for full specifications. Below is a summary of the datasets.

| Name                  | # Examples | # Features | # Targets | Label                      | Notes               |
| --------------------- | ---------- | ---------- | --------- | -------------------------- | ------------------- |
| `boston_housing`      | 506        | 13         | 1         | `MEDV`                     |                     |
| `concrete_strength`   | 1030       | 8          | 1         | `concrete_compresive_strength` |                     |
| `energy_efficiency`   | 768        | 8          | 2         | `Y1`                       | `Y2` is excluded    |
| `kin8nm`              | 8192       | 8          | 1         | `y`                        |                     |
| `naval_propulsion`    | 11934      | 16         | 2         | `GTTC`                     | `GTCD` is excluded  |
| `power_plant`         | 9568       | 4          | 1         | `PE`                       |                     |
| `protein_structure`   | 45730      | 9          | 1         | `RMSD`                     |                     |
| `wine_quality_red`    | 1599       | 11         | 1         | `quality`                  |                     |
| `yacht_hydrodynamics` | 308        | 6          | 1         | `resid_resist`             |                     |
