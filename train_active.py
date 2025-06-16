import os
import numpy as np
import torch
import pickle
import argparse
from matplotlib import pyplot as plt

from utils.utils import DataLoader
from active_learning.active_learning import ActiveLearner
from active_learning.kernel import random, bait
from train_model import load_model


def find_init_idx(datasets, init_size):
    """
    Try to find even distributed initial data points from pool
    In this case, based on only the first dimension
    """
    # Get unique values from the first dimension
    control_rot = datasets["x_pool"][:, 0]
    unique_values = np.unique(control_rot)
    num_unique = len(unique_values)
    # Find indices of each unique value
    group_indices = {
        val: np.where(control_rot == val)[0] for val in unique_values
    }

    # Select indices from each group
    selected_indices = []
    for i in range(init_size):
        group_id = i % num_unique  # cycle through groups
        selected_idx = np.random.choice(group_indices[unique_values[group_id]])
        selected_indices.append(selected_idx)

    return selected_indices


def plot_results(results, batch_size=1):
    """Plot active learning results"""
    fig, ax = plt.subplots()

    for key in results.keys():
        avg_results = np.average(results[key], axis=0)
        sigma_results = np.std(results[key], axis=0)

        num_data = (np.arange(len(avg_results)) + 1) * batch_size
        ax.plot(num_data, avg_results, "o-", label=key)
        ax.fill_between(
            num_data,
            avg_results - sigma_results,
            avg_results + sigma_results,
            alpha=0.2,
        )
    ax.set_ylabel("Loss after convergence")
    ax.set_xlabel("Number of data")
    ax.legend()

    return fig


def train_active_learning(
    model,
    acq_funcs,
    n_experiments,
    n_loop,
    batch_size,
    data_loader,
    model_prefix="",
    save_model=True,
):
    """Start training process"""
    results = dict()
    idx_used = dict()

    # Dataset summary
    datasets = data_loader.load_data()
    print(f"\n---------- Dataset ----------")
    print(f"Pool set size: {len(datasets['x_pool'])}")
    print(f"Validation set size: {len(datasets['x_val'])}")

    for e in range(n_experiments):
        print(f"\nExperiment {e + 1} / {n_experiments}")
        # Select even distributed initial data points from pool
        # evenly based on the first dimension
        init_index = find_init_idx(datasets, batch_size)

        for acq_func in acq_funcs:
            acq_func_name = str(acq_func).split(" ")[1]
            print(f"\n---------- {acq_func_name} ----------")
            if save_model:
                model_name = f"saved_models/{model_prefix}_{acq_func_name}_{e}"
            else:
                model_name = ""

            # Active learning
            active_learner = ActiveLearner(model, acq_func, datasets)
            test_scores, pool_index = active_learner.learn(
                n_loop, batch_size, init_index, model_name, verbose=False
            )

            # Collect results
            if acq_func_name not in results:
                results[acq_func_name] = []
                idx_used[acq_func_name] = []
            results[acq_func_name].append(test_scores)
            idx_used[acq_func_name].append(pool_index)

    # Convert lists to arrays
    for key in results.keys():
        results[key] = np.array(results[key])
        idx_used[key] = np.array(idx_used[key])

    return results, idx_used


def main(
    object_name,
    model_class,
    acq_functions,
    n_experiments,
    n_query,  # number of total queries
    batch_size,  # number of queries per loop
    seed,
    device,
):
    """Main function"""
    # Seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    data_loader = DataLoader(object_name, "data", val_size=200)

    # Load model
    # batch size here is deel learning batch size
    # not the same as the batch size in the active learning
    model = load_model(
        model_class,
        in_dim=3,
        out_dim=3,
        dropout=0.1,
        pred_var=False,
        batch_size=16,
        device=device,
    )

    # Define training parameters
    n_loop = n_query // batch_size - 1
    print(
        f"Active learning with {model_class} and {object_name}"
        + f"\nRepeat {n_experiments} experiments"
        + f"\nQuery {n_loop} times, {batch_size} queries per loop"
        + f"\nSeed: {seed}"
    )

    # Train active learning
    os.makedirs("results/learning/", exist_ok=True)
    os.makedirs("results/models/", exist_ok=True)
    name = object_name + "_" + model_class
    results, idx_used = train_active_learning(
        model,
        acq_functions,
        n_experiments=n_experiments,
        n_loop=n_loop,
        batch_size=batch_size,
        data_loader=data_loader,
        model_prefix=name,
        save_model=True,
    )

    # Save results
    pickle.dump(results, open("results/learning/loss_" + name + ".pkl", "wb"))
    pickle.dump(
        idx_used, open("results/learning/idx_used_" + name + ".pkl", "wb")
    )
    # # Plot and save results
    # fig = plot_results(results, batch_size)
    # fig.savefig("results/learning/" + name + ".pdf", dpi=300)
    # plt.close(fig)


if __name__ == "__main__":
    # Provide arguments for running parallelization
    parser = argparse.ArgumentParser()
    parser.add_argument("object_name", help="Object name")
    parser.add_argument("model_class", help="Model class (residual or mlp)")
    parser.add_argument(
        "--n_exp", type=int, default=1, help="Number of experiments"
    )
    parser.add_argument(
        "--n_query", type=int, default=9000, help="Total number of queries"
    )
    parser.add_argument(
        "--batch_size", type=int, default=9000, help="Batch size"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda:0", help="Device index")
    args = parser.parse_args()

    # Run active learning
    acq_functions = [random, bait]
    # acq_functions = [random]
    main(
        args.object_name,
        args.model_class,
        acq_functions,
        n_experiments=args.n_exp,
        n_query=args.n_query,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
