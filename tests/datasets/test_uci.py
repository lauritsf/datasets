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


# Test that the setup function returns the same split for the same seed
@pytest.mark.slow
@pytest.mark.parametrize("name", load_dataset_specs().keys())
def test_get_datamodule_setup(name):
    datamodule1 = get_datamodule(name, seed=42)
    datamodule1.prepare_data()
    datamodule1.setup()
    datamodule2 = get_datamodule(name, seed=42)
    datamodule2.prepare_data()
    datamodule2.setup()
    datamodule2.setup()  # Call setup twice to ensure that it does not change the split

    assert datamodule1.data_train is not None
    assert datamodule1.data_val is not None
    assert datamodule1.data_test is not None
    assert datamodule2.data_train is not None
    assert datamodule2.data_val is not None
    assert datamodule2.data_test is not None

    assert datamodule1.data_train.indices == datamodule2.data_train.indices
    assert datamodule1.data_val.indices == datamodule2.data_val.indices
    assert datamodule1.data_test.indices == datamodule2.data_test.indices
