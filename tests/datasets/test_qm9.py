import pytest

from dataset_baselines.datasets.qm9.datamodule import QM9DataModule


@pytest.mark.slow
@pytest.mark.parametrize("target", range(19))
def test_get_target_stats(target):
    dm = QM9DataModule(target=target)
    dm.prepare_data()
    dm.setup()

    stats = dm.get_target_stats(remove_atom_refs=False, divide_by_atoms=False)
    assert len(stats) == 3
    mean, std, atom_refs = stats
    # mean and std are singular tensor values
    assert mean.shape == std.shape == ()


@pytest.mark.slow
@pytest.mark.parametrize("target", range(19))
def test_datamoudle_dataloaders(target):
    dm = QM9DataModule(target=target)
    dm.prepare_data()
    dm.setup()

    assert dm.train_dataloader() is not None
    assert dm.val_dataloader() is not None
    assert dm.test_dataloader() is not None

    next(iter(dm.train_dataloader()))
    next(iter(dm.val_dataloader()))
    next(iter(dm.test_dataloader()))
