import pytest
import torch

from dataset_baselines.utils import StandardScaler


class TestStandardScaler:
    @pytest.fixture
    def scaler(self):
        return StandardScaler()

    @pytest.fixture
    def data_mean_std(self):
        # data [batch_size=6, num_features=3]
        # feature 1: Normal(5, 3)
        # feature 2: Normal(6, 1)
        # feature 3: Normal(7, 2)

        data = torch.tensor(
            [
                [2.0, 5.0, 7.0],
                [5.0, 6.0, 7.0],
                [8.0, 7.0, 7.0],
                [11.0, 8.0, 7.0],
                [14.0, 9.0, 7.0],
                [17.0, 10.0, 7.0],
            ]
        )
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        assert mean.shape == std.shape == (3,)
        return data, mean, std

    def test_fit_iterable(self, scaler, data_mean_std):
        data, mean, std = data_mean_std
        # batch data in batches of 2
        data_iterable = [data[i : i + 2] for i in range(0, len(data), 2)]
        scaler.fit_batched(data_iterable)
        assert torch.allclose(scaler.mean, mean), f"{scaler.mean} != {mean}"
        assert torch.allclose(scaler.std, std), f"{scaler.std} != {std}"

    def test_fit(self, scaler, data_mean_std):
        data, mean, std = data_mean_std
        scaler.fit(data)
        assert torch.allclose(scaler.mean, mean)
        assert torch.allclose(scaler.std, std)

    def test_transform(self, scaler):
        data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        scaler.mean = torch.tensor([4.0, 5.0, 6.0])
        scaler.std = torch.tensor([3.0, 3.0, 3.0])
        transformed_data = scaler.transform(data)
        expected_data = torch.tensor([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        assert torch.allclose(transformed_data, expected_data)

    def test_transform_std(self, scaler):
        std = torch.tensor([1.0, 1.0, 1.0])
        scaler.mean = torch.tensor([4.0, 5.0, 6.0])
        scaler.std = torch.tensor([3.0, 3.0, 3.0])
        transformed_std = scaler.transform_std(std)
        expected_std = torch.tensor([0.33333333, 0.33333333, 0.33333333])
        assert torch.allclose(transformed_std, expected_std)

    def test_fit_transform(self, scaler):
        data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        transformed_data = scaler.fit_transform(data)
        expected_data = torch.tensor([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        assert torch.allclose(transformed_data, expected_data)

    def test_inverse_transform(self, scaler):
        scaled_data = torch.tensor([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        scaler.mean = torch.tensor([4.0, 5.0, 6.0])
        scaler.std = torch.tensor([3.0, 3.0, 3.0])
        unscaled_data = scaler.inverse_transform(scaled_data)
        expected_data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        assert torch.allclose(unscaled_data, expected_data)

    def test_inverse_transform_std(self, scaler):
        scaled_std = torch.tensor([0.33333333, 0.33333333, 0.33333333])
        scaler.mean = torch.tensor([4.0, 5.0, 6.0])
        scaler.std = torch.tensor([3.0, 3.0, 3.0])
        unscaled_std = scaler.inverse_transform_std(scaled_std)
        expected_std = torch.tensor([1.0, 1.0, 1.0])
        assert torch.allclose(unscaled_std, expected_std)
