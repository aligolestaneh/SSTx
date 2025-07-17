#!/usr/bin/env python3
"""
Plot Fusion planner cost improvements over time with step plot.

Reads fusion_solutions.txt and creates a step plot showing how the solution cost
remains constant until improved, extending to show the full planning duration.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def plot_fusion_solutions(filename="fusion_solutions.txt", max_time=100.0):
    """
    Plot cost vs time from Fusion planner solution log as a step function.

    Args:
        filename: Path to the solution log file
        max_time: Maximum time to show on x-axis (default 100 seconds)
    """
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        print("Run the Fusion planner first to generate the solution log.")
        return

    times = []
    costs = []
    types = []
    iterations = []

    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith("#") or not line:
                    continue

                parts = line.split()
                if len(parts) >= 3:
                    time = float(parts[0])

                    # Handle infinite cost
                    if parts[1] == "inf":
                        cost = float("inf")
                    else:
                        cost = float(parts[1])

                    solution_type = parts[2]

                    # Handle both old format (3 columns) and new format (4 columns)
                    if len(parts) >= 4:
                        iteration = int(parts[3])
                    else:
                        iteration = len(
                            times
                        )  # Use line number as iteration for old format

                    times.append(time)
                    costs.append(cost)
                    types.append(solution_type)
                    iterations.append(iteration)

    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Convert to numpy arrays for easier handling
    times = np.array(times)
    costs = np.array(costs)
    iterations = np.array(iterations)

    # Filter out infinite costs for meaningful plotting
    finite_mask = np.isfinite(costs) & (costs != np.inf) & (costs != -np.inf)
    finite_times = times[finite_mask]
    finite_costs = costs[finite_mask]
    finite_types = np.array(types)[finite_mask]
    finite_iterations = iterations[finite_mask]

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Extend max_time if needed
    if len(times) > 0:
        max_time = max(max_time, times[-1] * 1.1)

    if not finite_times.size:
        print("No finite solution data found in file!")
        # Plot empty chart showing no solutions found
        plt.axhline(
            y=0,
            color="red",
            linestyle="--",
            alpha=0.5,
            label="No Solutions Found",
        )
        plt.xlim(0, max_time)
        plt.ylim(0, 100)
        plt.text(
            max_time / 2,
            50,
            "No Solutions Found",
            ha="center",
            va="center",
            fontsize=16,
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
        )
    else:
        # Create step plot data - use all points for smooth visualization
        step_times = []
        step_costs = []

        # If planning starts with no solution, show infinite cost period
        inf_mask = ~finite_mask
        if np.any(inf_mask):
            first_finite_time = (
                finite_times[0] if len(finite_times) > 0 else max_time
            )
            if times[0] < first_finite_time:
                # There's a period with infinite cost
                step_times.extend([0.0, first_finite_time])
                step_costs.extend(
                    [np.nan, np.nan]
                )  # Don't plot infinite costs

        # Add finite costs
        if len(finite_costs) > 0:
            # Start from the first finite solution
            step_times.extend(finite_times.tolist())
            step_costs.extend(finite_costs.tolist())

            # Extend final cost to max_time
            if finite_times[-1] < max_time:
                step_times.append(max_time)
                step_costs.append(finite_costs[-1])

        # Convert to numpy arrays
        step_times = np.array(step_times)
        step_costs = np.array(step_costs)

        # Filter out NaN values for plotting
        valid_mask = ~np.isnan(step_costs)

        if np.any(valid_mask):
            # Plot the step function
            plt.step(
                step_times[valid_mask],
                step_costs[valid_mask],
                where="post",
                linewidth=3,
                color="royalblue",
                alpha=0.8,
                label="Cost Over Time",
            )

            # Fill area under the step function
            plt.fill_between(
                step_times[valid_mask],
                step_costs[valid_mask],
                alpha=0.2,
                color="royalblue",
                step="post",
            )

        # Show period with no solutions (if any) as shaded area
        if np.any(inf_mask):
            inf_times = times[inf_mask]
            if len(inf_times) > 0 and len(finite_costs) > 0:
                # Shade the no-solution period
                inf_start = inf_times[0]
                inf_end = (
                    inf_times[-1]
                    if len(finite_times) == 0
                    else finite_times[0]
                )
                plt.axvspan(
                    inf_start,
                    inf_end,
                    alpha=0.1,
                    color="red",
                    label="No Solution Period",
                )

        # Add small scatter points to show exactly when solutions are found
        exact_mask = finite_types == "exact"
        approx_mask = finite_types == "approximate"

        if np.any(exact_mask):
            exact_times_plot = finite_times[exact_mask]
            exact_costs_plot = finite_costs[exact_mask]
            plt.scatter(
                exact_times_plot,
                exact_costs_plot,
                s=20,  # Larger but still reasonable
                color="limegreen",  # Brighter color
                marker="o",
                edgecolors="darkgreen",
                linewidths=1,
                label="Exact Solutions",
                alpha=0.9,
                zorder=10,  # Higher z-order to appear on top
            )

        if np.any(approx_mask):
            approx_times_plot = finite_times[approx_mask]
            approx_costs_plot = finite_costs[approx_mask]
            plt.scatter(
                approx_times_plot,
                approx_costs_plot,
                s=15,  # Slightly smaller than exact
                color="gold",  # Brighter color
                marker="s",
                edgecolors="darkorange",
                linewidths=1,
                label="Approximate Solutions",
                alpha=0.9,
                zorder=10,  # Higher z-order to appear on top
            )

    # Formatting
    plt.xlabel("Time (seconds)", fontsize=14)
    plt.ylabel("Solution Cost", fontsize=14)
    plt.title(
        "Fusion Planner: Cost Evolution Over Time (Step Plot)",
        fontsize=16,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Set x-axis to show full planning duration
    plt.xlim(0, max_time)

    # Set reasonable y-axis limits
    if len(finite_costs) > 0:
        min_cost = min(finite_costs)
        max_cost = max(finite_costs)

        # Additional safety check for any remaining non-finite values
        if np.isfinite(min_cost) and np.isfinite(max_cost):
            if max_cost > min_cost:
                y_margin = (max_cost - min_cost) * 0.1
            else:
                # All costs are the same, use 10% of the cost value as margin
                y_margin = max(abs(min_cost) * 0.1, 0.1)

            plt.ylim(min_cost - y_margin, max_cost + y_margin)
        else:
            print(
                "Warning: Found non-finite values in filtered costs, using default y-limits"
            )
            plt.ylim(0, 10)  # Default reasonable limits

    # Add cost annotations for first and final exact solutions only
    if len(finite_costs) > 0:
        exact_mask = finite_types == "exact"
        if np.any(exact_mask):
            exact_times_ann = finite_times[exact_mask]
            exact_costs_ann = finite_costs[exact_mask]

            # Annotate first exact solution
            first_exact_time = exact_times_ann[0]
            first_exact_cost = exact_costs_ann[0]
            plt.annotate(
                f"First: {first_exact_cost:.3f}",
                xy=(first_exact_time, first_exact_cost),
                xytext=(
                    first_exact_time + max_time * 0.05,
                    first_exact_cost
                    + (max(finite_costs) - min(finite_costs)) * 0.1,
                ),
                fontsize=11,
                fontweight="bold",
                color="darkgreen",
                ha="left",
                va="bottom",
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5),
            )

            # Annotate final exact solution (if different from first)
            if len(exact_costs_ann) > 1:
                last_exact_time = exact_times_ann[-1]
                last_exact_cost = exact_costs_ann[-1]
                plt.annotate(
                    f"Final: {last_exact_cost:.3f}",
                    xy=(last_exact_time, last_exact_cost),
                    xytext=(
                        last_exact_time + max_time * 0.05,
                        last_exact_cost
                        - (max(finite_costs) - min(finite_costs)) * 0.1,
                    ),
                    fontsize=11,
                    fontweight="bold",
                    color="darkgreen",
                    ha="left",
                    va="top",
                    arrowprops=dict(
                        arrowstyle="->", color="darkgreen", lw=1.5
                    ),
                )

    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    plt.tight_layout()

    # Save the plot in the results folder
    base_filename = os.path.basename(filename).replace(
        ".txt", "_step_plot.png"
    )
    output_file = os.path.join(results_dir, base_filename)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Step plot saved as: {output_file}")

    # Show the plot
    plt.show()

    # Print summary statistics
    print("\n=== Solution Summary ===")
    if len(finite_costs) > 0:
        print(
            f"First solution found at: {finite_times[0]:.2f} seconds (cost: {finite_costs[0]:.3f})"
        )
        print(
            f"Last solution found at: {finite_times[-1]:.2f} seconds (cost: {finite_costs[-1]:.3f})"
        )
        print(f"Total planning time: {max_time:.0f} seconds")
        print(f"Number of finite solutions: {len(finite_costs)}")
        print(f"Total iterations logged: {len(times)}")

        exact_mask = finite_types == "exact"
        approx_mask = finite_types == "approximate"
        print(f"Exact solutions: {np.sum(exact_mask)}")
        print(f"Approximate solutions: {np.sum(approx_mask)}")

        if len(finite_costs) > 1:
            improvement = (
                (finite_costs[0] - finite_costs[-1]) / finite_costs[0]
            ) * 100
            print(f"Cost improvement: {improvement:.1f}%")
            print(
                f"Average time between solutions: {(finite_times[-1] - finite_times[0]) / (len(finite_costs) - 1):.2f} seconds"
            )

        # Show iteration information
        if len(finite_iterations) > 0:
            print(f"First solution at iteration: {finite_iterations[0]}")
            print(f"Last solution at iteration: {finite_iterations[-1]}")
            if len(times) > 0:
                print(f"Final iteration: {iterations[-1]}")
    else:
        print(
            f"No finite solutions found in {max_time:.0f} seconds of planning"
        )
        print(f"Total iterations logged: {len(times)}")


def main():
    """Main function to handle command line arguments."""
    filename = "fusion_solutions.txt"
    max_time = 100.0

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        max_time = float(sys.argv[2])

    plot_fusion_solutions(filename, max_time)


if __name__ == "__main__":
    main()

