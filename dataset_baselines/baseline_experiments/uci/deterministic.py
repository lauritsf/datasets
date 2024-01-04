import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torchmetrics import MeanMetric

from dataset_baselines.datasets.uci import get_dataset
from dataset_baselines.utils import StandardScaler


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


def parse_args():
    parser = argparse.ArgumentParser()

    # Data and Logging Settings
    data_group = parser.add_argument_group("Data and Logging")
    data_group.add_argument("--dataset", type=str, default="yacht_hydrodynamics")
    data_group.add_argument("--data_dir", type=str, default="data/")
    data_group.add_argument("--log_dir", type=str, default="logs/")
    data_group.add_argument("--seed", type=int, default=0)

    # Model Settings
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--ensemble_size", type=int, default=1)
    model_group.add_argument(
        "--noise_model", type=str, default="homoscedastic", choices=["homoscedastic", "heteroscedastic"]
    )

    # Training Settings
    training_group = parser.add_argument_group("Training Settings")
    training_group.add_argument("--num_epochs", type=int, default=5000)
    training_group.add_argument("--start_lr", type=float, default=0.01)
    training_group.add_argument("--end_lr", type=float, default=1e-5)
    training_group.add_argument("--batch_size", type=int, default=256)
    training_group.add_argument("--k_fold_cv", type=int, default=5)
    training_group.add_argument(
        "--weight_decay_values", type=float, nargs="+", default=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    )
    training_group.add_argument("--mc_samples", type=int, default=1000)
    training_group.add_argument("--retrain", action="store_true")
    training_group.add_argument("--gpu", action="store_true")
    training_group.add_argument("--fast_dev_run", action="store_true")

    args = parser.parse_args()
    return args


class NetHomoscedastic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.loc = nn.Linear(hidden_dim, output_dim)
        self.softplus_scale_param = nn.Parameter(torch.zeros(1))

        # Initialize weights (bias is initialized to 0 by default)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.loc.weight)

    def forward(self, x):
        x = self.linear(x)
        x = F.gelu(x)
        loc = self.loc(x)
        scale = F.softplus(self.softplus_scale_param)
        scale = scale.expand(x.shape[0], -1)
        return loc, scale


class NetHeteroscedastic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.loc = nn.Linear(hidden_dim, output_dim)
        self.softplus_scale = nn.Linear(hidden_dim, output_dim)

        # Initialize weights (bias is initialized to 0 by default)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.loc.weight)
        nn.init.xavier_uniform_(self.softplus_scale.weight)

    def forward(self, x):
        x = self.linear(x)
        x = F.gelu(x)
        loc = self.loc(x)
        scale = F.softplus(self.softplus_scale(x))
        return loc, scale


def train_and_evaluate(model, args, train_dataset, eval_dataset, weight_decay, loggers: list[Logger]):
    """
    Train the model on the training set and evaluate on the evaluation set.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.end_lr)

    input_scaler = StandardScaler().fit_batched((x for x, _ in train_dataset))
    output_scaler = StandardScaler().fit_batched((y for _, y in train_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)

    train_loss = MeanMetric()
    eval_loss = MeanMetric()
    eval_nll = MeanMetric()
    epoch_eval_loss = float("inf")
    for epoch in range(args.num_epochs):
        model.train()
        train_loss.reset()
        for x, y in train_dataloader:
            with torch.no_grad():
                x = input_scaler.transform(x)
                y = output_scaler.transform(y)
            optimizer.zero_grad()
            loc, scale = model(x)
            loss = -td.Normal(loc, scale).log_prob(y).mean()
            loss.backward()
            optimizer.step()
            train_loss(loss)
        scheduler.step()

        model.eval()
        eval_loss.reset()
        # Evaluate on the evaluation set
        with torch.no_grad():
            for x, y in eval_dataloader:
                orig_y = y
                x = input_scaler.transform(x)
                y = output_scaler.transform(y)
                loc, scale = model(x)
                loss = -td.Normal(loc, scale).log_prob(y).mean()
                eval_loss(loss)
                pred_mean = output_scaler.inverse_transform(loc)
                pred_std = output_scaler.inverse_transform_std(scale)
                nll = -td.Normal(pred_mean, pred_std).log_prob(orig_y).mean()
                eval_nll(nll)
            epoch_eval_loss = eval_loss.compute().item()
        for logger in loggers:
            logger.log_metrics(
                {
                    "train/loss": train_loss.compute().item(),
                    "val/loss": epoch_eval_loss,
                    "val/nll": eval_nll.compute().item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )
    return epoch_eval_loss


def train_model(
    model,
    args,
    train_dataloader,
    weight_decay,
    input_scaler,
    output_scaler,
    log_path,
    eval_dataloader=None,
    loggers: list[Logger] | None = None,
):
    if loggers is None:
        loggers = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.start_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.end_lr)

    epoch_loss = MeanMetric()
    epoch_nll = MeanMetric()
    epoch_mse = MeanMetric()

    eval_loss = MeanMetric()
    eval_nll = MeanMetric()
    eval_mse = MeanMetric()

    model.train()
    for epoch in range(args.num_epochs):
        for x, y in train_dataloader:
            y_orig = y
            with torch.no_grad():
                x = input_scaler.transform(x)
                y = output_scaler.transform(y)
            optimizer.zero_grad()
            loc, scale = model(x)
            loss = -td.Normal(loc, scale).log_prob(y).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                mean = output_scaler.inverse_transform(loc)
                std = output_scaler.inverse_transform_std(scale)
                nll = -td.Normal(mean, std).log_prob(y_orig).mean()
                mse = F.mse_loss(mean, y_orig)
            epoch_loss(loss)
            epoch_nll(nll)
            epoch_mse(mse)

        scheduler.step()
        metrics = {
            "train/learning_rate": scheduler.get_last_lr()[0],
            "train/loss": epoch_loss.compute().item(),
            "train/nll": epoch_nll.compute().item(),
            "train/mse": epoch_mse.compute().item(),
        }

        if eval_dataloader is not None:
            model.eval()
            eval_loss.reset()
            eval_nll.reset()
            eval_mse.reset()
            with torch.no_grad():
                for x, y in eval_dataloader:
                    y_orig = y
                    x = input_scaler.transform(x)
                    y = output_scaler.transform(y)
                    loc, scale = model(x)
                    loss = -td.Normal(loc, scale).log_prob(y).mean()
                    eval_loss(loss)
                    mean = output_scaler.inverse_transform(loc)
                    std = output_scaler.inverse_transform_std(scale)
                    nll = -td.Normal(mean, std).log_prob(y_orig).mean()
                    mse = F.mse_loss(mean, y_orig)
                    eval_nll(nll)
                    eval_mse(mse)
                metrics.update(
                    {
                        "val/loss": eval_loss.compute().item(),
                        "val/nll": eval_nll.compute().item(),
                        "val/mse": eval_mse.compute().item(),
                    }
                )
            model.train()

        for logger in loggers:
            logger.log_metrics(metrics, step=epoch)

        epoch_loss.reset()
        epoch_nll.reset()
        epoch_mse.reset()

        for logger in loggers:
            logger.save()

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


def tune_weight_decay(model_initializer, args, dataset, logger, log_path) -> dict[float, list[float]]:
    # Setup kfold splits
    kfold = KFold(n_splits=args.k_fold_cv, shuffle=True, random_state=args.seed)
    kfold_splits: list[np.ndarray] = list(kfold.split(dataset))  # type: ignore

    logger.info("--- Tuning weight decay ---")
    tuning_losses: dict[float, list[float]] = {}
    for weight_decay in args.weight_decay_values:
        logger.info(f"Weight decay: {weight_decay}")
        tuning_losses[weight_decay] = []
        for fold, (train_ids, eval_ids) in enumerate(kfold_splits):
            logger.info(f"\tFold {fold + 1}/{args.k_fold_cv}")
            train_dataset_fold = Subset(dataset, train_ids)
            eval_dataset_fold = Subset(dataset, eval_ids)
            model = model_initializer()
            csv_logger = CSVLogger(
                log_path / "tuning_logs", name=f"weight_decay={weight_decay}", version=f"fold_{fold}"
            )
            tb_logger = TensorBoardLogger(
                log_path / "tuning_logs", name=f"weight_decay={weight_decay}", version=f"fold_{fold}"
            )
            loss = train_and_evaluate(
                model, args, train_dataset_fold, eval_dataset_fold, weight_decay, loggers=[csv_logger, tb_logger]
            )
            logger.info(f"\t\tLoss: {loss}")
            tuning_losses[weight_decay].append(loss)

            # Log tuning loss with tensorboard and flush
            tb_logger.log_hyperparams(
                {"weight_decay": weight_decay}, {"val/tuning_loss": tuning_losses[weight_decay][-1]}
            )
            csv_logger.save()
            tb_logger.save()

    return tuning_losses


def main():
    args = parse_args()

    # Check and create log path step by step to avoid errors
    log_path = Path(args.log_dir) / f"deterministic_{args.noise_model}_ensemble_size={args.ensemble_size}"
    log_path.mkdir(parents=True, exist_ok=True)
    log_path = log_path / args.dataset
    log_path.mkdir(parents=True, exist_ok=True)
    log_path = log_path / f"seed={args.seed}"
    logger = setup_logging(log_path)

    logger.info("--- Arguments ---")
    for arg, value in vars(args).items():
        logger.info(f"\t{arg}: {value}")

    # Set seed
    torch.manual_seed(args.seed)

    # Load data
    dataset = get_dataset(args.dataset, data_dir=args.data_dir)

    # Split int 90% trainig, 10% test
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)

    # Setup model
    hidden_features = 50 if args.dataset != "protein_structure" else 100
    model_class = NetHeteroscedastic if args.noise_model == "heteroscedastic" else NetHomoscedastic

    # Tune weight decay
    if args.retrain or not (log_path / "tuning_losses.csv").exists():

        def model_initializer():
            return model_class(dataset.num_input_features, hidden_features, dataset.num_output_features)

        tuning_losses: dict[float, list[float]] = tune_weight_decay(
            model_initializer, args, train_dataset, logger, log_path
        )
        tuning_losses_df = pd.DataFrame(tuning_losses)
        tuning_losses_df.to_csv(log_path / "tuning_losses.csv", index=False)
    else:
        logger.info("--- Loading tuning losses from csv ---")
        tuning_losses_df = pd.read_csv(log_path / "tuning_losses.csv")

    # Find best weight decay
    best_weight_decay = float(tuning_losses_df.mean().idxmin())
    logger.info(f"Best weight decay: {best_weight_decay}")

    # Train and evaluate on test set
    logger.info("--- Training and test predictions ---")

    # Setup training
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    input_scaler = StandardScaler().fit_batched((x for x, _ in train_dataloader))
    output_scaler = StandardScaler().fit_batched((y for _, y in train_dataloader))

    # Train models
    models = []
    checkpoint_path = log_path / "checkpoints"
    for i in range(args.ensemble_size):
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        model_checkpoint_path = checkpoint_path / f"model_{i}.pt"
        model = model_class(dataset.num_input_features, hidden_features, dataset.num_output_features)
        if model_checkpoint_path.exists():
            model.load_state_dict(torch.load(model_checkpoint_path))
        else:
            csv_logger = CSVLogger(log_path / "training_logs", name=f"model_{i}", version="")
            tb_logger = TensorBoardLogger(log_path / "training_logs", name=f"model_{i}", version="")
            eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
            model = train_model(
                model,
                args,
                train_dataloader,
                best_weight_decay,
                input_scaler,
                output_scaler,
                log_path,
                eval_dataloader=eval_dataloader,
                loggers=[csv_logger, tb_logger],
            )
            torch.save(model.state_dict(), model_checkpoint_path)
        models.append(model)

    # Predict on test set
    y, pred_mean, pred_std = ensemble_predict(models, test_dataloader, input_scaler, output_scaler)

    # Write predictions to csv (split into different files for each ensemble member)
    predictions_path = log_path / "predictions"
    predictions_path.mkdir(parents=True, exist_ok=True)
    for i in range(args.ensemble_size):
        pd.DataFrame({"y": y.squeeze(-1), "pred_mean": pred_mean[:, i], "pred_std": pred_std[:, i]}).to_csv(
            predictions_path / f"predictions_ensemble_{i}.csv", index=False
        )

    pred_dist = td.MixtureSameFamily(  # Create mixture of gaussians with equal weights
        td.Categorical(torch.ones(pred_mean.size(0), args.ensemble_size)), td.Normal(pred_mean, pred_std)
    )

    log_likelihood = pred_dist.log_prob(y.squeeze(-1))  # [num_samples]

    print("--- Test predictions ---")
    mc_samples = pred_dist.sample(torch.Size([args.mc_samples]))  # [MC_SAMPLES, num_samples]
    errors = mc_samples - y.swapaxes(0, 1)  # [MC_SAMPLES, num_samples]
    mse = errors.pow(2).mean(dim=0)
    rmse = mse.sqrt()
    test_metrics = {
        "log_likelihood": log_likelihood.mean().item(),
        "mean_squared_error": mse.mean().item(),
        "root_mean_squared_error": rmse.mean().item(),
    }

    # Log metrics
    logger.info("--- Test metrics ---")
    for metric, value in test_metrics.items():
        logger.info(f"\t{metric}: {value}")

    # Write to csv
    test_metrics_path = log_path / "test_metrics.csv"
    model_name = f"deterministic_{args.noise_model}_ensemble_size={args.ensemble_size}"
    test_metrics_df = pd.DataFrame(
        {
            "model": model_name,
            "dataset": args.dataset,
            "seed": args.seed,
            "log_likelihood": test_metrics["log_likelihood"],
            "mean_squared_error": test_metrics["mean_squared_error"],
            "root_mean_squared_error": test_metrics["root_mean_squared_error"],
            "weight_decay": best_weight_decay,
            "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        index=[0],
    )
    test_metrics_df.to_csv(test_metrics_path, index=False)


if __name__ == "__main__":
    main()
