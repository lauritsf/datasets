import pytest
import pandas as pd
from pathlib import Path
from get_datasets import remove_trailing_space, process_dataset, UCI_DATASETS

# Test for remove_trailing_space function
def test_remove_trailing_space(tmp_path):
    # Create a sample file with trailing spaces
    sample_file = tmp_path / "sample.txt"
    with open(sample_file, 'w') as f:
        f.write("Some text with trailing space \nAnother line ")

    
    
    new_path = remove_trailing_space(sample_file)
    with open(new_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        print(line)
        assert not line.endswith(" "), line

@pytest.mark.parametrize("name, spec", UCI_DATASETS.items())
def test_process_dataset(name, spec, tmp_path):
    process_dataset(name, spec, tmp_path)
    dataset_folder = tmp_path / name
    data_filepath = dataset_folder / f"{name}.csv"
    assert data_filepath.exists()
    assert pd.read_csv(data_filepath).shape[0] > 0

def test_process_dataset_yacht_hydrodynamics(tmp_path):
    name = 'yacht_hydrodynamics'
    spec = UCI_DATASETS[name]
    process_dataset(name, spec, tmp_path)
    dataset_folder = tmp_path / name
    data_filepath = dataset_folder / f"{name}.csv"
    assert data_filepath.exists()
    assert pd.read_csv(data_filepath).shape[0] > 0
