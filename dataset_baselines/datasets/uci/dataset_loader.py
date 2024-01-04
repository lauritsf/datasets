import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
import torch
import yaml
from scipy.io import arff
from torch.utils.data import Dataset


def read_arff(filepath):
    data, meta = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    df.columns = meta.names()
    return df


# function that removes trailing whitespace from a file
def remove_trailing_space(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    lines = [line.replace(" \n", "\n") for line in lines]
    if lines[-1].endswith(" "):
        # Remove trailing space from last line
        lines[-1] = lines[-1].rstrip()
    filepath = filepath.parent / (filepath.stem + "_no_trailing_space" + filepath.suffix)
    with open(filepath, "w") as f:
        f.writelines(lines)
    return filepath


@dataclass
class DataSpec:
    url: str
    read_function: str
    read_params: dict | None
    columns: list[str] | None
    preprocess: str | None
    file_in_zip: str | None
    label: str
    exclude: list[str] | None
    num_examples: int
    num_input_features: int
    num_output_features: int


def load_dataset_specs() -> dict[str, DataSpec]:
    with open(Path(__file__).parent / "dataset_specs.yaml") as file:
        specs = yaml.safe_load(file)
        return {name: DataSpec(**values) for name, values in specs.items()}


def process_dataset(dataset_name: str, data_spec: DataSpec, data_dir: Path) -> Path:
    """Download and process a dataset.

    The folder structure is as follows:
    data_dir/
    ├── raw/
    │   └── [downloaded raw dataset files]
    └── processed/
        └── dataset_name.csv

    Skip downloading and processing if the processed dataset file already exists.

    Returns:
        Path: Path to the processed dataset file.
    """
    raw_data_path = data_dir / "raw"
    raw_data_path.mkdir(exist_ok=True, parents=True)
    processed_data_path = data_dir / "processed"
    processed_data_path.mkdir(exist_ok=True, parents=True)

    processed_dataset_file = processed_data_path / f"{dataset_name}.csv"
    if processed_dataset_file.exists():
        return processed_dataset_file

    raw_dataset_file = raw_data_path / data_spec.url.split("/")[-1]

    if not raw_dataset_file.exists():
        response = requests.get(data_spec.url, timeout=5)
        response.raise_for_status()
        with open(raw_dataset_file, "wb") as file:
            file.write(response.content)

    if data_spec.file_in_zip:
        with zipfile.ZipFile(raw_dataset_file, "r") as zip_ref:
            raw_dataset_file = Path(zip_ref.extract(data_spec.file_in_zip, raw_dataset_file.parent))

    if data_spec.preprocess:
        preprocess_function = globals()[data_spec.preprocess]
        raw_dataset_file = preprocess_function(raw_dataset_file)

    # Dynamically determine the read function (either a global function or a module method)
    if "." in data_spec.read_function:
        module_name, function_name = data_spec.read_function.split(".")
        module = globals().get(module_name)
        if not module:
            raise ValueError(f"Module '{module_name}' not found.")
        read_function = getattr(module, function_name, None)
        if not read_function:
            raise ValueError(f"Function '{function_name}' not found in module '{module_name}'.")
    else:
        read_function = globals().get(data_spec.read_function)
        if not read_function:
            raise ValueError(f"Function '{data_spec.read_function}' not found.")

    dataset_df = read_function(raw_dataset_file, **(data_spec.read_params or {}))
    if data_spec.columns:
        dataset_df.columns = data_spec.columns

    dataset_df.to_csv(processed_dataset_file, index=False, header=True)

    return processed_dataset_file


class UciDataset(Dataset):
    """Dataset for UCI datasets."""

    def __init__(
        self,
        name: str,
        spec: DataSpec,
        data_dir: str = "data/",
    ):
        self.name = name
        self.spec = spec
        self.data_dir = Path(data_dir) / "uci" / name
        self.data, self.label = self._load_data()

        self.num_input_features = spec.num_input_features
        self.num_output_features = spec.num_output_features

        assert (
            self.num_input_features == self.data.shape[1]
        ), f"Wrong number of input features: {self.num_input_features} != {self.data.shape[1]}"
        assert (
            self.num_output_features == self.label.shape[1]
        ), f"Wrong number of output features: {self.num_output_features} != {self.label.shape[1]}"

    def _load_data(self):
        data_filepath = process_dataset(self.name, self.spec, self.data_dir)
        df = pd.read_csv(data_filepath, header=0, sep=",")

        # Remove excluded columns
        if self.spec.exclude:
            df = df.drop(columns=self.spec.exclude)

        # Split into data and label
        data = torch.FloatTensor(df.drop(columns=[self.spec.label]).values)
        label = torch.FloatTensor(df[self.spec.label].values)
        # Ensure that label is 2D
        if len(label.shape) == 1:
            label = label.unsqueeze(1)

        return data, label

    def __len__(self):
        if self.data is None:
            raise ValueError("Dataset not loaded.")
        return len(self.data)

    def __getitem__(self, idx):
        if self.data is None or self.label is None:
            raise ValueError("Dataset not loaded.")
        return self.data[idx], self.label[idx]


def get_dataset(name: str, data_dir="data/"):
    """Get a UCI dataset.

    Args:
        name (str): Name of the dataset.
        data_dir (str, optional): Directory to store the dataset. Defaults to "data/".

    Raises:
        ValueError: If the dataset is not found.

    Returns:
        UciDataset: The dataset.
    """
    specs = load_dataset_specs()
    if name not in specs:
        raise ValueError(f"Dataset '{name}' not found in '{specs.keys()}'.")

    spec = specs[name]
    return UciDataset(name=name, spec=spec, data_dir=data_dir)
