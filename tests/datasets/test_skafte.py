import pytest
import torch

from dataset_baselines.datasets.skafte.dataset import SkafteDataset


class TestSkafteDataset:
    @pytest.mark.parametrize("num_samples", [1, 10, 100])
    @pytest.mark.parametrize("heteroscedastic", [True, False])
    @pytest.mark.parametrize("x_bounds", [(0, 1), (0, 10), (2.5, 12.5)])
    @pytest.mark.parametrize("seed", [None, 0, 42])
    def test_skafte_dataset(self, num_samples, heteroscedastic, x_bounds, seed):
        dataset = SkafteDataset(num_samples=num_samples, heteroscedastic=heteroscedastic, x_bounds=x_bounds, seed=seed)
        assert len(dataset) == num_samples
        x, y = dataset[0]
        assert x.shape == y.shape == torch.Size([1])

        # Assert all x values are within bounds
        assert (x >= x_bounds[0]).all()
        assert (x <= x_bounds[1]).all()

    def test_x_bounds(self):
        with pytest.raises(ValueError):
            SkafteDataset(x_bounds=(0, 1, 2))

    def test_reproducibility(self):
        dataset1 = SkafteDataset(num_samples=100, seed=0)
        dataset2 = SkafteDataset(num_samples=100, seed=0)
        assert torch.allclose(dataset1.x, dataset2.x), "X values are not equal"
        assert torch.allclose(dataset1.y, dataset2.y), "Y values are not equal"
