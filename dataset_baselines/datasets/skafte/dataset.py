import torch
from torch.utils.data import Dataset


class SkafteDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 1000,
        heteroscedastic: bool = True,
        x_bounds: tuple[float, float] = (2.5, 12.5),
        seed: int | None = None,
    ):
        super().__init__()

        if len(x_bounds) != 2:
            raise ValueError("x_bounds must be a tuple of length 2")

        self.num_samples = num_samples
        self.heteroscedastic = heteroscedastic
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        self.x = torch.rand(self.num_samples, generator=generator) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
        self.noise = torch.randn(self.x.shape, generator=generator) * self.scale_function(self.x)
        self.y = self.target_function(self.x) + self.noise

    @classmethod
    def target_function(cls, x):
        return x * torch.sin(x)

    def scale_function(self, x):
        scale = 0.1 + torch.abs(0.5 * x)
        if self.heteroscedastic:
            return scale
        else:
            return scale.mean()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.x[idx].unsqueeze(-1)
        y = self.y[idx].unsqueeze(-1)
        return x, y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = SkafteDataset(num_samples=1000, heteroscedastic=True, seed=0)
    x, y = dataset[:]

    linspace = torch.linspace(2.5, 12.5, 1000)
    true_y = SkafteDataset.target_function(linspace)
    true_scale = dataset.scale_function(linspace)

    plt.scatter(x, y, label="Data", alpha=0.5, color="blue", edgecolor="grey", facecolor="none")
    plt.plot(linspace, true_y, label="True function", color="black")
    plt.plot(linspace, true_y + true_scale, label="True error", color="grey", linestyle="--")
    plt.plot(linspace, true_y - true_scale, color="grey", label="_nolegend_", linestyle="--")
    plt.legend()
    plt.show()
