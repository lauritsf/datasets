import pytest

from dataset_baselines.datasets.uci import get_datamodule, get_dataset
from dataset_baselines.datasets.uci.datamodule import UciDataModule
from dataset_baselines.datasets.uci.dataset_loader import DataSpec, load_dataset_specs


def test_load_dataset_specs():
    specs = load_dataset_specs()
    for name, spec in specs.items():
        assert isinstance(name, str), f"Wrong type: {type(name)}"
        assert isinstance(spec, DataSpec), f"Wrong type: {type(spec)}"


@pytest.mark.slow
@pytest.mark.parametrize("name", load_dataset_specs().keys())
def test_load_dataset(name):
    spec = load_dataset_specs()[name]
    dataset = get_dataset(name)
    assert dataset.name == name
    assert dataset.spec == spec
    assert dataset.num_input_features == spec.num_input_features
    assert dataset.num_output_features == spec.num_output_features
    assert len(dataset) == spec.num_examples


@pytest.mark.slow
@pytest.mark.parametrize("name", load_dataset_specs().keys())
def test_get_dataset(name):
    dataset = get_dataset(name)
    assert dataset.name == name


@pytest.mark.slow
@pytest.mark.parametrize("name", load_dataset_specs().keys())
def test_get_datamodule(name):
    datamodule = get_datamodule(name)
    datamodule.prepare_data()
    datamodule.setup()
    assert datamodule.hparams["name"] == name
    assert isinstance(datamodule, UciDataModule)
    assert datamodule.data_train is not None
    assert datamodule.data_val is not None
    assert datamodule.data_test is not None
