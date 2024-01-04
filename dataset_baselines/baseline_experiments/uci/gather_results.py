import argparse
import os

import pandas as pd


def parse_args() -> argparse.Namespace:
    # Get the arguments for
    # - Where to look for the results
    # - The name of the results files
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True, help="Directory to look for the results in")
    parser.add_argument("--results_name", type=str, default="test_metrics.csv", help="Name of the results files")
    parser.add_argument(
        "--output_type", type=str, default="markdown", help="The type of output to use", choices=["markdown"]
    )
    return parser.parse_args()


def gather_dataframes(folder_path: str, results_name: str) -> pd.DataFrame:
    """
    Gathers all the dataframes in the folder_path and combines them into a single dataframe

    :param folder_path: The path to the folder containing the dataframes
    :param results_name: The name of the dataframes
    :return: A single dataframe containing all the dataframes in the folder_path
    """
    dataframes = []

    # Traverse the folder_path (recursively) and find all the dataframes
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(results_name):
                # Read the dataframe and add it to the dataframes list
                dataframes.append(pd.read_csv(os.path.join(root, file)))

    # Combine the dataframes into a single dataframe
    dataframes = pd.concat(dataframes)

    return dataframes


def pivot_and_aggregate(results: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """
    Modifies the dataframe structure to include the number of seeds next to each value.

    Each value represents the mean +- sem of the metric across the different seeds,
    with the number of seeds indicated next to each value.

    :param results: The results dataframe
    :param metric_name: The name of the metric to pivot and aggregate
    :return: A dataframe with the dataset as the columns and the "model" as the rows
    """

    # Calculate the mean, sem, and count (number of seeds) for each dataset, model
    results = results.groupby(["dataset", "model"])[metric_name].agg(["mean", "sem", "count"])
    results = results.reset_index()

    # Format the mean and sem values to have 4 decimal places and include n_seeds
    results["mean +- sem (n)"] = results.apply(lambda x: f"{x['mean']:.4f} +- {x['sem']:.4f} (n={x['count']})", axis=1)

    # Drop the mean, sem, and count columns
    results = results.drop(columns=["mean", "sem", "count"])

    # Pivot the table
    results = results.pivot(index="model", columns="dataset", values="mean +- sem (n)")

    return results


dataset_pretty_names = {
    "boston_housing": "Boston Housing",
    "concrete_strength": "Concrete Strength",
    "energy_efficiency": "Energy Efficiency",
    "kin8nm": "Kin8nm",
    "naval_propulsion": "Naval Propulsion",
    "power_plant": "Power Plant",
    "protein_structure": "Protein Structure",
    "wine_quality_red": "Wine Quality (Red)",
    "yacht_hydrodynamics": "Yacht Hydrodynamics",
}


def get_pretty_model_name(model_name: str) -> str:
    if model_name.startswith("deterministic"):
        # deterministic_{noise_model}_ensemble_size={M}
        ensemble_size = int(model_name.split("=")[-1])
        noise_model = model_name.split("_")[1]
        if ensemble_size > 1:
            return f"Deterministic {noise_model} Ensemble (M={ensemble_size})"
        else:
            return f"Deterministic {noise_model}"
    elif model_name.startswith("variational"):
        layer_type = model_name.split("_")[1]
        return f"Variational {layer_type.capitalize()}"
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def row_ordering(a, b):
    a_deterministic = a.startswith("deterministic")
    b_deterministic = b.startswith("deterministic")

    if a_deterministic and not b_deterministic:
        return -1
    elif not a_deterministic and b_deterministic:
        return 1

    if not a_deterministic and not b_deterministic:
        # TODO: implement for stochastic models
        return 0

    # If both are deterministic, sort by ensemble size and noise model
    a_parts = a.split("_")
    b_parts = b.split("_")

    # Extract ensemble size (assuming the format is deterministic_[noise_model]_ensemble_size=[size])
    a_ensemble_size = int(a_parts[3].split("=")[-1]) if len(a_parts) > 2 else 1
    b_ensemble_size = int(b_parts[3].split("=")[-1]) if len(b_parts) > 2 else 1

    # Order by ensemble size (no ensemble first, then by size)
    if a_ensemble_size != b_ensemble_size:
        return -1 if a_ensemble_size < b_ensemble_size else 1

    # If ensemble sizes are the same, sort by noise model (homoscedastic first, then heteroscedastic)
    a_noise_model = a_parts[1] if len(a_parts) > 1 else ""
    b_noise_model = b_parts[1] if len(b_parts) > 1 else ""

    if a_noise_model != b_noise_model:
        return -1 if a_noise_model == "homoscedastic" else 1

    # If both models are the same in terms of ensemble size and noise model
    return 0


def main():
    # Will look for results in the log_dir and combine them, then output a single csv file
    args = parse_args()
    args.metrics = ["log_likelihood", "mean_squared_error", "root_mean_squared_error"]

    results = gather_dataframes(args.log_dir, args.results_name)

    for metric in args.metrics:
        # Pivot and aggregate the results
        pivoted_results = pivot_and_aggregate(results, metric)

        # Rename the columns to be pretty
        pivoted_results = pivoted_results.rename(columns=dataset_pretty_names)

        # Rename the rows to be pretty
        pivoted_results = pivoted_results.rename(index=get_pretty_model_name)

        if args.output_type == "markdown":
            # Output the results as a markdown table
            print("### " + metric.replace("_", " ").title())
            print(pivoted_results.to_markdown())


if __name__ == "__main__":
    main()
