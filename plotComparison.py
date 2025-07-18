import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot Fusion planner comparison results from CSV file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="fusion_comparison.csv",
        help="Input CSV file path (default: fusion_comparison.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output plot file path (default: input_filename_cost_comparison.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Plot DPI (default: 300)",
    )

    return parser.parse_args()


def load_and_validate_data(csv_file):
    """Load CSV data and validate required columns."""
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded data from {csv_file}")
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Check for required columns
        required_columns = [
            "planning_time",
            "initial_cost",
            "final_cost",
            "fusion_time",
            "cost_improvement",
        ]

        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None

        return df

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def create_cost_comparison_plot(df, output_file, dpi=300):
    """Create the cost comparison plot from the dataframe."""
    print(f"\n{'='*80}")
    print(f"CREATING COST COMPARISON PLOT")
    print(f"{'='*80}")

    # Extract data for plotting
    planning_times = df["planning_time"].values
    initial_costs = df["initial_cost"].values
    final_costs = df["final_cost"].values

    # Replace infinite values with None for plotting
    initial_costs_plot = [
        cost if cost != float("inf") else None for cost in initial_costs
    ]
    final_costs_plot = [
        cost if cost != float("inf") else None for cost in final_costs
    ]

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot initial costs (SST Cost - before replanning)
    initial_valid_indices = [
        i for i, cost in enumerate(initial_costs_plot) if cost is not None
    ]
    if initial_valid_indices:
        initial_times = [planning_times[i] for i in initial_valid_indices]
        initial_costs_valid = [
            initial_costs_plot[i] for i in initial_valid_indices
        ]
        plt.plot(
            initial_times,
            initial_costs_valid,
            "b-o",
            label="SST Cost (Before Replanning)",
            linewidth=2,
            markersize=4,
        )

    # Plot final costs (after replanning)
    final_valid_indices = [
        i for i, cost in enumerate(final_costs_plot) if cost is not None
    ]
    if final_valid_indices:
        final_times = [planning_times[i] for i in final_valid_indices]
        final_costs_valid = [final_costs_plot[i] for i in final_valid_indices]
        plt.plot(
            final_times,
            final_costs_valid,
            "r-s",
            label="After Replanning",
            linewidth=2,
            markersize=4,
        )

    # Customize the plot
    plt.xlabel("Planning Time (seconds)", fontsize=12)
    plt.ylabel("Path Cost", fontsize=12)
    plt.title(
        "Fusion Planner Cost Comparison: SST Cost vs After Replanning",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Calculate and add statistics to the plot
    initial_costs_valid = [
        cost for cost in initial_costs if cost != float("inf")
    ]
    final_costs_valid = [cost for cost in final_costs if cost != float("inf")]

    if initial_costs_valid and final_costs_valid:
        avg_initial = np.mean(initial_costs_valid)
        avg_final = np.mean(final_costs_valid)
        improvement = ((avg_initial - avg_final) / avg_initial) * 100

        stats_text = f"Average SST Cost: {avg_initial:.3f}\n"
        stats_text += f"Average Final Cost: {avg_final:.3f}\n"
        stats_text += f"Average Improvement: {improvement:.1f}%"

        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=10,
        )

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    print(f"Cost comparison plot saved to: {output_file}")

    # Close the plot to free memory
    plt.close()


def print_summary_statistics(df):
    """Print summary statistics from the data."""
    print(f"\nSUMMARY STATISTICS:")
    print(f"{'='*80}")

    # Count successful solutions
    initial_success = sum(
        1 for cost in df["initial_cost"] if cost != float("inf")
    )
    final_success = sum(1 for cost in df["final_cost"] if cost != float("inf"))
    total_tests = len(df)

    print(
        f"SST solutions found: {initial_success}/{total_tests} ({initial_success/total_tests*100:.1f}%)"
    )
    print(
        f"Final solutions found: {final_success}/{total_tests} ({final_success/total_tests*100:.1f}%)"
    )

    # Average costs (excluding infinite costs)
    initial_costs = [
        cost for cost in df["initial_cost"] if cost != float("inf")
    ]
    final_costs = [cost for cost in df["final_cost"] if cost != float("inf")]

    if initial_costs:
        print(f"Average SST cost: {np.mean(initial_costs):.4f}")
    if final_costs:
        print(f"Average final cost: {np.mean(final_costs):.4f}")

    if initial_costs and final_costs:
        avg_improvement = np.mean(
            [
                improvement
                for improvement in df["cost_improvement"]
                if improvement != 0
            ]
        )
        print(f"Average cost improvement: {avg_improvement:.4f}")

        # Calculate improvement percentage
        improvement_percentage = (
            (np.mean(initial_costs) - np.mean(final_costs))
            / np.mean(initial_costs)
        ) * 100
        print(f"Overall improvement: {improvement_percentage:.1f}%")

    # Planning time statistics
    print(
        f"\nPlanning time range: {df['planning_time'].min():.1f}s - {df['planning_time'].max():.1f}s"
    )
    print(f"Average planning time: {df['planning_time'].mean():.1f}s")


def main():
    # Parse arguments
    args = parse_args()

    print("=" * 80)
    print("FUSION PLANNER RESULTS PLOTTER")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output or 'auto-generated'}")
    print(f"DPI: {args.dpi}")
    print("=" * 80)

    # Load and validate data
    df = load_and_validate_data(args.input)
    if df is None:
        sys.exit(1)

    # Print summary statistics
    print_summary_statistics(df)

    # Determine output filename if not provided
    if args.output is None:
        output_file = args.input.replace(".csv", "_cost_comparison.png")
    else:
        output_file = args.output

    # Create the plot
    create_cost_comparison_plot(df, output_file, args.dpi)

    print(f"\n{'='*80}")
    print(f"PLOTTING COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"Results plotted from: {args.input}")
    print(f"Plot saved to: {output_file}")


if __name__ == "__main__":
    main()
