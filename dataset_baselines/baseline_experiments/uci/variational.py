import argparse
import logging
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from sklearn.model_selection import KFold
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from dataset_baselines.datasets.uci import get_dataset
from dataset_baselines.utils import StandardScaler

# Model


class BayesianLinear(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.weight_rho = nn.Parameter(torch.Tensor(output_dim, input_dim))

        if use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(output_dim))
            self.bias_rho = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with small random values
        mu_range = 1 / math.sqrt(self.input_dim)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_rho.data.normal_(-3, 1e-2)

        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_rho.data.normal_(-3, 1e-2)

    def forward(self, x):
        weight = Normal(self.weight_mu, F.softplus(self.weight_rho)).rsample()

        bias = None
        if self.bias_mu is not None:
            bias = Normal(self.bias_mu, F.softplus(self.bias_rho)).rsample()

        return F.linear(x, weight, bias)

    def kl_divergence(self, prior_std: float):
        weight_kl = kl_divergence(Normal(self.weight_mu, F.softplus(self.weight_rho)), Normal(0, prior_std)).sum()
        bias_kl = (
            0
            if self.bias_mu is None
            else kl_divergence(Normal(self.bias_mu, F.softplus(self.bias_rho)), Normal(0, prior_std)).sum()
        )
        return weight_kl + bias_kl


class Rank1BayesianLinear(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        # TODO: add the ensemble element of the rank1 bayesian linear
        raise NotImplementedError("Rank1BayesianLinear not implemented")
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear = nn.Linear(input_dim, output_dim, bias=False)

        self.alpha_mu = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.alpha_rho = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.beta_mu = nn.Parameter(torch.Tensor(output_dim))
        self.beta_rho = nn.Parameter(torch.Tensor(output_dim))

        if use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(output_dim))
            self.bias_rho = nn.Parameter(torch.Tensor(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with small random values
        mu_range = 1 / math.sqrt(self.input_dim)
        self.alpha_mu.data.uniform_(-mu_range, mu_range)
        self.alpha_rho.data.uniform_(-mu_range, mu_range)
        self.beta_mu.data.uniform_(-mu_range, mu_range)
        self.beta_rho.data.uniform_(-mu_range, mu_range)

        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_rho.data.uniform_(-mu_range, mu_range)

    def forward(self, x):
        raise NotImplementedError("Rank1BayesianLinear not implemented")
        alpha = Normal(self.alpha_mu, F.softplus(self.alpha_rho)).rsample()
        gamma = Normal(self.beta_mu, F.softplus(self.beta_rho)).rsample()
        x = x * alpha
        x = self.linear(x)
        x = x + gamma

        bias = None
        if self.bias_mu is not None:
            bias = Normal(self.bias_mu, F.softplus(self.bias_rho)).rsample()
            x = x + bias

        return x

    def kl_divergence(self, prior_std: float):
        raise NotImplementedError("Rank1BayesianLinear not implemented")
        alpha_kl = kl_divergence(Normal(self.alpha_mu, F.softplus(self.alpha_rho)), Normal(0, prior_std)).sum()
        beta_kl = kl_divergence(Normal(self.beta_mu, F.softplus(self.beta_rho)), Normal(0, prior_std)).sum()
        bias_kl = (
            0
            if self.bias_mu is None
            else kl_divergence(Normal(self.bias_mu, F.softplus(self.bias_rho)), Normal(0, prior_std)).sum()
        )
        return alpha_kl + beta_kl + bias_kl

    def l2_regularization(self):
        l2 = torch.linear.weight.pow(2).sum()
        if self.bias_mu is not None:
            l2 += torch.linear.bias.pow(2).sum()
        return l2


class BayesianNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        layer_module: type[BayesianLinear] | type[Rank1BayesianLinear],
        use_bias=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.layer_module = layer_module

        self.input_layer = self.layer_module(input_dim, hidden_dim, use_bias=use_bias)

        self.output_mu_layer = self.layer_module(hidden_dim, output_dim, use_bias=use_bias)
        self.output_rho_layer = self.layer_module(hidden_dim, output_dim, use_bias=use_bias)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.gelu(x)
        mu = self.output_mu_layer(x)
        rho = self.output_rho_layer(x)
        return mu, rho

    def kl_divergence(self, prior_std: float) -> torch.Tensor:
        input_layer_kl = self.input_layer.kl_divergence(prior_std)
        output_mu_layer_kl = self.output_mu_layer.kl_divergence(prior_std)
        output_rho_layer_kl = self.output_rho_layer.kl_divergence(prior_std)
        return input_layer_kl + output_mu_layer_kl + output_rho_layer_kl

    def l2_regularization(self) -> torch.Tensor:
        l2 = torch.tensor(0.0)
        if hasattr(self.input_layer, "l2_regularization"):
            l2 += self.input_layer.l2_regularization()
        if hasattr(self.output_mu_layer, "l2_regularization"):
            l2 += self.output_mu_layer.l2_regularization()
        if hasattr(self.output_rho_layer, "l2_regularization"):
            l2 += self.output_rho_layer.l2_regularization()
        return l2


def build_model(args: argparse.Namespace) -> BayesianNetwork:
    match args.dataset:
        case "protein_structure":
            hidden_dim = 100
        case _:
            hidden_dim = 50
    match args.layer_type:
        case "full":
            layer_module = BayesianLinear
        case "rank1":
            layer_module = Rank1BayesianLinear
        case _:
            raise ValueError(f"Invalid layer type: {args.layer_type}")

    model = BayesianNetwork(
        input_dim=args.in_dim, output_dim=args.out_dim, hidden_dim=hidden_dim, layer_module=layer_module
    )

    return model


# Logging and parsing
def setup_logging(log_dir: Path, retrain: bool = False, print_to_console: bool = True):
    if log_dir.exists() and retrain:
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_dir / "log.txt",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S.%f",
    )
    logger = logging.getLogger()
    if print_to_console:
        logger.addHandler(logging.StreamHandler())
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Data and Logging Settings
    data_group = parser.add_argument_group("Data and Logging")
    data_group.add_argument("--dataset", type=str, default="yacht_hydrodynamics")
    data_group.add_argument("--data_dir", type=str, default="data/")
    data_group.add_argument("--log_dir", type=str, default="logs/")
    data_group.add_argument("--seed", type=int, default=0)

    # Model Settings
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--hidden_dim", type=int, default=50)
    model_group.add_argument("--layer_type", type=str, default="full", choices=["full", "rank1"])

    # Training Settings
    training_group = parser.add_argument_group("Training Settings")
    training_group.add_argument("--num_epochs", type=int, default=5000)
    training_group.add_argument("--start_lr", type=float, default=1e-2)
    training_group.add_argument("--end_lr", type=float, default=1e-5)
    training_group.add_argument("--batch_size", type=int, default=256)
    training_group.add_argument("--k_fold_cv", type=int, default=5)
    training_group.add_argument(
        "--prior_std_values", nargs="+", type=float, default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
    )
    training_group.add_argument("--mc_samples", type=int, default=1000)
    training_group.add_argument("--retrain", action="store_true")
    training_group.add_argument("--gpu", action="store_true")
    training_group.add_argument("--fast_dev_run", action="store_true")

    args = parser.parse_args()
    return args


# Tuning Training and Evaluation


def tune_prior_std(
    args: argparse.Namespace, train_dataset: torch.utils.data.Dataset, logger: logging.Logger, log_path: Path
) -> dict[float, list[float]]:
    # Setup kfold splits
    kfold = KFold(n_splits=args.k_fold_cv, shuffle=True, random_state=args.seed)
    kfold_splits: list[np.ndarray] = list(kfold.split(train_dataset))  # type: ignore

    tuning_losses: dict[float, list[float]] = {}
    for prior_std in args.prior_std_values:
        logger.info(f"Prior std: {prior_std}")
        tuning_losses[prior_std] = []
        for fold, (train_indices, val_indices) in enumerate(kfold_splits):
            logger.info(f"\tFold {fold +1} of {args.k_fold_cv}")
            train_dataset_fold = torch.utils.data.Subset(train_dataset, train_indices)
            eval_dataset_fold = torch.utils.data.Subset(train_dataset, val_indices)

            # Train and evaluate model
            model = build_model(args)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.num_epochs, eta_min=args.end_lr
            )
            input_scaler = StandardScaler().fit_batched((x for x, _ in train_dataset_fold))
            output_scaler = StandardScaler().fit_batched((y for _, y in train_dataset_fold))
            train_dataloader = DataLoader(train_dataset_fold, batch_size=args.batch_size, shuffle=True)
            eval_dataloader = DataLoader(eval_dataset_fold, batch_size=args.batch_size, shuffle=False)

            try:
                model.train()
                for _ in range(args.num_epochs):
                    for x, y in train_dataloader:
                        with torch.no_grad():
                            x = input_scaler.transform(x)
                            y = output_scaler.transform(y)
                        optimizer.zero_grad()
                        mu, rho = model(x)
                        loss = -Normal(mu, F.softplus(rho)).log_prob(y).mean()
                        loss += model.kl_divergence(prior_std)
                        loss += model.l2_regularization()
                        loss.backward()
                        optimizer.step()
                    scheduler.step()

                    model.eval()
                    running_loss = 0.0
                    with torch.no_grad():
                        for x, y in eval_dataloader:
                            x = input_scaler.transform(x)
                            y = output_scaler.transform(y)
                            mu, rho = model(x)
                            loss = -Normal(mu, F.softplus(rho)).log_prob(y).sum(dim=1).mean()
                            running_loss += loss.item()
                    model_loss = running_loss / len(eval_dataloader)
                    tuning_losses[prior_std].append(model_loss)
                logging.info(f"\t\tLoss: {tuning_losses[prior_std][-1]}")

            except ValueError as e:
                if "to satisfy the constraint Real()," in str(e):
                    logger.info("ValueError: Encountered non-real value. Skipping fold and adding loss of inf")
                    tuning_losses[prior_std].append(float("inf"))
                    continue
                else:
                    raise
    return tuning_losses


def train_model(
    args: argparse.Namespace,
    model: BayesianNetwork,
    train_dataset: torch.utils.data.Dataset,
    prior_std: float,
    logger: logging.Logger,
    log_path: Path,
) -> BayesianNetwork:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.end_lr)
    input_scaler = StandardScaler().fit_batched((x for x, _ in train_dataset))
    output_scaler = StandardScaler().fit_batched((y for _, y in train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    tb_logger = TensorBoardLogger(log_path / "tb_logs", name=f"variational_{args.layer_type}")
    csv_logger = CSVLogger(log_path / "csv_logs", name=f"variational_{args.layer_type}")

    train_loss = MeanMetric()
    train_nll = MeanMetric()
    train_mse = MeanMetric()

    model.train()
    for epoch in range(args.num_epochs):
        for metric in [train_loss, train_nll, train_mse]:
            metric.reset()

        for x, y in train_dataloader:
            with torch.no_grad():
                x_scaled = input_scaler.transform(x)
                y_scaled = output_scaler.transform(y)
            optimizer.zero_grad()
            mu, rho = model(x_scaled)
            loss = -Normal(mu, F.softplus(rho)).log_prob(y_scaled).mean()
            loss += model.kl_divergence(prior_std)
            loss += model.l2_regularization()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss(loss)
                pred_mean = output_scaler.inverse_transform(mu)
                pred_std = output_scaler.inverse_transform(F.softplus(rho))
                train_nll(-Normal(pred_mean, pred_std).log_prob(y).mean())
                train_mse(F.mse_loss(pred_mean, y))
        scheduler.step()
        metrics = {
            "train_loss": train_loss.compute().item(),
            "train_nll": train_nll.compute().item(),
            "train_mse": train_mse.compute().item(),
        }
        tb_logger.log_metrics(metrics, step=epoch)
        csv_logger.log_metrics(metrics, step=epoch)

    return model


@torch.no_grad
def ensemble_predict(
    models: list[nn.Module], dataloader: DataLoader, input_scaler: StandardScaler, output_scaler: StandardScaler
):
    TensorList = list[torch.Tensor]
    y_list: TensorList = []
    pred_mean_list: TensorList = []
    pred_std_list: TensorList = []

    for x, y in dataloader:
        y_list.append(y)
        x = input_scaler.transform(x)
        y = output_scaler.transform(y)
        pred_mean, pred_std = zip(*[model(x) for model in models])
        pred_mean = torch.cat(pred_mean, dim=1)
        pred_std = torch.cat(pred_std, dim=1)
        pred_mean = output_scaler.inverse_transform(pred_mean)
        pred_std = output_scaler.inverse_transform_std(pred_std)
        pred_mean_list.append(pred_mean)
        pred_std_list.append(pred_std)

    y = torch.cat(y_list)
    pred_mean = torch.cat(pred_mean_list)
    pred_std = torch.cat(pred_std_list)

    return y, pred_mean, pred_std


def main():
    args = parse_args()

    log_path = Path(args.log_dir) / f"variational_{args.layer_type}"
    log_path.mkdir(parents=True, exist_ok=True)
    log_path = log_path / args.dataset
    log_path.mkdir(parents=True, exist_ok=True)
    log_path = log_path / f"seed_{args.seed}"
    logger = setup_logging(log_path, args.retrain)

    # Set seed
    torch.manual_seed(args.seed)

    # Load data
    dataset = get_dataset(args.dataset, args.data_dir)
    # register input and output dimensions
    args.in_dim = dataset.num_input_features
    args.out_dim = dataset.num_output_features

    # Split into 90% training, 10% testing
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)

    # Tune prior std
    if args.retrain or not (log_path / "tuning_losses.csv").exists():
        logger.info("--- Tuning prior std ---")
        tuning_losses: dict[float, list[float]] = tune_prior_std(args, train_dataset, logger, log_path)
        tuning_losses_df = pd.DataFrame(tuning_losses)
        tuning_losses_df.to_csv(log_path / "tuning_losses.csv", index=False)
    else:
        logger.info("--- Loading prior std tuning losses ---")
        tuning_losses_df = pd.read_csv(log_path / "tuning_losses.csv")
    best_prior_std = float(tuning_losses_df.mean().idxmin())
    logger.info(f"Best prior std: {best_prior_std}")

    # Train model
    checkpoint_path = log_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model = build_model(args)
    if (checkpoint_path / "model.pt").exists():
        logger.info("--- Loading best model ---")
        model.load_state_dict(torch.load(checkpoint_path / "model.pt"))
    else:
        logger.info("--- Training model ---")
        model = train_model(args, model, train_dataset, best_prior_std, logger, log_path)
        torch.save(model.state_dict(), checkpoint_path / "model.pt")

    # Evaluate model
    logger.info("--- Evaluating model ---")
    model.eval()

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    input_scaler = StandardScaler().fit_batched((x for x, _ in train_dataset))
    output_scaler = StandardScaler().fit_batched((y for _, y in train_dataset))

    y, pred_mean, pred_std = ensemble_predict([model], test_dataloader, input_scaler, output_scaler)
    y, pred_mean, pred_std = y[0], pred_mean[0], pred_std[0]

    predictions_path = log_path / "predictions"
    predictions_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "y": y,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
        }
    ).to_csv(predictions_path / "predictions.csv", index=False)
    pred_dist = Normal(pred_mean, pred_std)

    log_likelihood = pred_dist.log_prob(y)
    mc_samples = pred_dist.sample(torch.Size([args.mc_samples]))
    errors = mc_samples - y
    mse = errors.pow(2).mean()
    rmse = mse.sqrt()
    test_metrics = {
        "log_likelihood": log_likelihood.mean().item(),
        "mean_squared_error": mse.item(),
        "root_mean_squared_error": rmse.item(),
    }

    logger.info("--- Test Metrics ---")
    for metric, value in test_metrics.items():
        logger.info(f"\t{metric}: {value}")

    test_metrics_path = log_path / "test_metrics.csv"
    model_name = f"variational_{args.layer_type}"
    test_metrics_df = pd.DataFrame(
        {
            "model": model_name,
            "dataset": args.dataset,
            "seed": args.seed,
            **test_metrics,
            "prior_std": best_prior_std,
            "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        index=[0],
    )
    test_metrics_df.to_csv(test_metrics_path, index=False)


if __name__ == "__main__":
    main()
