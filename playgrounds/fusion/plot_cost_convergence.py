#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_cost_convergence(filename="fusion_cost_log.txt"):
    """Plot the cost convergence over time from the Fusion planner log file

    Args:
        filename: Path to the cost log file (default: "fusion_cost_log.txt")
    """

    if not os.path.exists(filename):
        print(f"Error: Cost log file '{filename}' not found!")
        print(
            "Make sure to run the fusion planner first to generate the log file."
        )
        return

    # Read the data
    times = []
    costs = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith("#") or not line:
                continue

            try:
                parts = line.split()
                if len(parts) >= 2:
                    time_val = float(parts[0])
                    cost_val = float(parts[1])

                    # Skip infinite costs for plotting
                    if (
                        cost_val != float("inf")
                        and cost_val < 1e10
                        and cost_val > 0
                    ):
                        times.append(time_val)
                        costs.append(cost_val)
            except ValueError:
                # Skip lines that can't be parsed
                continue

    if not times:
        print("No valid cost data found in the log file!")
        return

    print(f"Loaded {len(times)} cost data points")
    print(f"Time range: {min(times):.3f}s to {max(times):.3f}s")
    print(f"Cost range: {min(costs):.3f} to {max(costs):.3f}")

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Main cost convergence plot
    plt.subplot(2, 1, 1)
    plt.plot(times, costs, "b-", linewidth=2, alpha=0.8, label="Best Cost")
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Path Cost", fontsize=12)
    plt.title(
        "Fusion Planner Cost Convergence", fontsize=14, fontweight="bold"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add some statistics
    final_cost = costs[-1]
    initial_cost = costs[0]
    improvement = initial_cost - final_cost
    improvement_pct = (
        (improvement / initial_cost) * 100 if initial_cost > 0 else 0
    )

    plt.text(
        0.02,
        0.98,
        f"Initial Cost: {initial_cost:.3f}\nFinal Cost: {final_cost:.3f}\nImprovement: {improvement:.3f} ({improvement_pct:.1f}%)",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Zoomed-in view of the latter part (if there's enough data)
    if len(times) > 100:
        plt.subplot(2, 1, 2)
        # Show last 80% of the data for detail
        start_idx = len(times) // 5
        plt.plot(
            times[start_idx:],
            costs[start_idx:],
            "r-",
            linewidth=2,
            alpha=0.8,
            label="Best Cost (Detailed)",
        )
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel("Path Cost", fontsize=12)
        plt.title(
            "Cost Convergence (Detailed View)", fontsize=12, fontweight="bold"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()

    # Save the plot
    output_filename = "fusion_cost_convergence.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Cost convergence plot saved to: {output_filename}")

    # Show the plot
    plt.show()


def plot_multiple_sessions(filename="fusion_cost_log.txt"):
    """Plot cost convergence for multiple planning sessions (resolve iterations)

    Args:
        filename: Path to the cost log file (default: "fusion_cost_log.txt")
    """

    if not os.path.exists(filename):
        print(f"Error: Cost log file '{filename}' not found!")
        return

    # Read and separate different planning sessions
    sessions = []
    current_session_times = []
    current_session_costs = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("# End planning session"):
                # End of a session
                if current_session_times:
                    sessions.append(
                        (
                            current_session_times.copy(),
                            current_session_costs.copy(),
                        )
                    )
                    current_session_times.clear()
                    current_session_costs.clear()
                continue

            # Skip other comments and empty lines
            if line.startswith("#") or not line:
                continue

            try:
                parts = line.split()
                if len(parts) >= 2:
                    time_val = float(parts[0])
                    cost_val = float(parts[1])

                    # Skip infinite costs for plotting
                    if (
                        cost_val != float("inf")
                        and cost_val < 1e10
                        and cost_val > 0
                    ):
                        current_session_times.append(time_val)
                        current_session_costs.append(cost_val)
            except ValueError:
                continue

    # Add the last session if it wasn't explicitly ended
    if current_session_times:
        sessions.append((current_session_times, current_session_costs))

    if not sessions:
        print("No valid session data found!")
        return

    print(f"Found {len(sessions)} planning sessions")

    # Create the plot
    plt.figure(figsize=(14, 10))

    colors = [
        "blue",
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
    ]

    for i, (times, costs) in enumerate(sessions):
        if not times:
            continue

        color = colors[i % len(colors)]
        label = (
            f"Session {i+1} (Resolve {i})"
            if i > 0
            else f"Session {i+1} (Initial)"
        )

        plt.plot(
            times, costs, color=color, linewidth=2, alpha=0.8, label=label
        )

        # Mark the final cost for each session
        if times:
            plt.scatter(
                times[-1],
                costs[-1],
                color=color,
                s=100,
                marker="o",
                edgecolors="black",
                linewidth=2,
            )
            plt.annotate(
                f"{costs[-1]:.2f}",
                (times[-1], costs[-1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Path Cost", fontsize=12)
    plt.title(
        "Fusion Planner Cost Convergence - Multiple Resolve Sessions",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    # Save the plot
    output_filename = "fusion_multiple_sessions_convergence.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Multiple sessions plot saved to: {output_filename}")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    import sys

    print("Fusion Planner Cost Convergence Plotter")
    print("=" * 40)

    filename = "fusion_cost_log.txt"
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    # Try to plot multiple sessions first
    try:
        plot_multiple_sessions(filename)
    except Exception as e:
        print(f"Error plotting multiple sessions: {e}")
        print("Trying single session plot...")
        try:
            plot_cost_convergence(filename)
        except Exception as e:
            print(f"Error plotting cost convergence: {e}")
