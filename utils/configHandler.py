import sys
import yaml
import argparse
import numpy as np


def load_config(config_file: str) -> dict:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args_and_config():
    """Parse command line arguments and load configuration from YAML file."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run fusion planning with YAML configuration"
    )
    parser.add_argument(
        "--planning-time",
        type=float,
        help="Planning time in seconds (overrides YAML)",
    )
    parser.add_argument(
        "--replanning-time",
        type=float,
        help="Replanning time in seconds (overrides YAML)",
    )
    parser.add_argument(
        "--planner-name", type=str, help="Planner name (overrides YAML)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization (overrides YAML)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization (overrides YAML)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate for optimization (overrides YAML)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of epochs for optimization (overrides YAML)",
    )
    parser.add_argument(
        "--sampling-num-states",
        type=int,
        help="Number of random states to sample (overrides YAML)",
    )
    parser.add_argument(
        "--sampling-position-std",
        type=float,
        help="Standard deviation for position sampling (overrides YAML)",
    )
    parser.add_argument(
        "--sampling-rotation-std",
        type=float,
        help="Standard deviation for rotation sampling (overrides YAML)",
    )

    args = parser.parse_args()

    # Use fixed config file name
    config_file = "config.yaml"

    # Load configuration from YAML file
    try:
        config = load_config(config_file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

    # Extract parameters from config with command line overrides
    system = config.get("system")
    objectName = config.get("objectName")
    startState = np.array(config.get("startState"))
    goalState = np.array(config.get("goalState"))

    # Use command line args if provided, otherwise use YAML values
    planningTime = (
        args.planning_time
        if args.planning_time is not None
        else config.get("planningTime")
    )
    replanningTime = (
        args.replanning_time
        if args.replanning_time is not None
        else config.get("replanningTime")
    )
    plannerName = (
        args.planner_name
        if args.planner_name is not None
        else config.get("plannerName")
    )

    # Handle visualize flag with explicit override options
    if args.visualize:
        visualize = True
    elif args.no_visualize:
        visualize = False
    else:
        visualize = config.get("visualize")

    # Extract optimization parameters
    learningRate = (
        args.learning_rate
        if args.learning_rate is not None
        else config.get("learning_rate")
    )
    numEpochs = (
        args.num_epochs
        if args.num_epochs is not None
        else config.get("num_epochs")
    )

    # Extract sampling parameters
    sampling_config = config.get("sampling", {})
    sampling_num_states = (
        args.sampling_num_states
        if args.sampling_num_states is not None
        else sampling_config.get("num_states", 1000)
    )
    sampling_position_std = (
        args.sampling_position_std
        if args.sampling_position_std is not None
        else sampling_config.get("position_std", 0.005)
    )
    sampling_rotation_std = (
        args.sampling_rotation_std
        if args.sampling_rotation_std is not None
        else sampling_config.get("rotation_std", 0.1)
    )
    sampling_max_distance = sampling_config.get("max_distance", 0.025)

    return {
        "system": system,
        "objectName": objectName,
        "startState": startState,
        "goalState": goalState,
        "planningTime": planningTime,
        "replanningTime": replanningTime,
        "plannerName": plannerName,
        "visualize": visualize,
        "learningRate": learningRate,
        "numEpochs": numEpochs,
        "sampling_num_states": sampling_num_states,
        "sampling_max_distance": sampling_max_distance,
        "sampling_position_std": sampling_position_std,
        "sampling_rotation_std": sampling_rotation_std,
    }
