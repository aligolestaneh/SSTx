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
    parser = argparse.ArgumentParser(description="Run fusion planning with YAML configuration")
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
    parser.add_argument("--planner-name", type=str, help="Planner name (overrides YAML)")
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
    parser.add_argument(
        "--propagation-step-size",
        type=float,
        help="Propagation step size for simulation (overrides YAML)",
    )
    parser.add_argument(
        "--min-control-duration",
        type=int,
        help="Minimum control duration in steps (overrides YAML)",
    )
    parser.add_argument(
        "--max-control-duration",
        type=int,
        help="Maximum control duration in steps (overrides YAML)",
    )

    args = parser.parse_args()

    # Use fixed launcher config file name
    launcher_config_file = "config.yaml"

    # Load configuration from YAML file
    try:
        launcher_config = load_config(launcher_config_file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{launcher_config_file}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

    # If the top-level config contains a 'config_file' pointer, load it and merge
    if isinstance(launcher_config, dict) and launcher_config.get("config_file"):
        try:
            pointed_config = load_config(launcher_config["config_file"])
        except Exception as e:
            print(f"Error loading referenced config '{launcher_config['config_file']}': {e}")
            sys.exit(1)
        # Merge: values in launcher_config override pointed_config (except 'config_file')
        config = dict(pointed_config or {})
        for k, v in launcher_config.items():
            if k == "config_file":
                continue
            if v is not None:
                config[k] = v
    else:
        config = launcher_config

    # Extract parameters from config with command line overrides
    system = config.get("system")
    objectName = config.get("objectName")
    startState = (
        np.array(config.get("startState")) if config.get("startState") is not None else None
    )
    goalState = np.array(config.get("goalState")) if config.get("goalState") is not None else None

    # Use command line args if provided, otherwise use YAML values
    planningTime = (
        args.planning_time if args.planning_time is not None else config.get("planningTime")
    )
    replanningTime = (
        args.replanning_time if args.replanning_time is not None else config.get("replanningTime")
    )
    plannerName = args.planner_name if args.planner_name is not None else config.get("plannerName")

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
        else config.get("optimizer_learning_rate", 0.001)
    )
    numEpochs = (
        args.num_epochs if args.num_epochs is not None else config.get("optimizer_num_steps", 1000)
    )

    # Ensure we have valid values
    if learningRate is None or learningRate <= 0:
        print(f"[WARNING] Invalid learningRate: {learningRate}, using default: 0.001")
        learningRate = 0.001
    if numEpochs is None or numEpochs <= 0:
        print(f"[WARNING] Invalid numEpochs: {numEpochs}, using default: 1000")
        numEpochs = 1000

    # Debug: Print loaded values
    print(f"[DEBUG] Config loading:")
    print(f"  - optimizer_learning_rate from YAML: {config.get('optimizer_learning_rate')}")
    print(f"  - optimizer_num_steps from YAML: {config.get('optimizer_num_steps')}")
    print(f"  - learning_rate from args: {args.learning_rate}")
    print(f"  - num_epochs from args: {args.num_epochs}")
    print(f"  - Final learningRate: {learningRate}")
    print(f"  - Final numEpochs: {numEpochs}")

    # Extract sampling parameters
    sampling_config = config.get("sampling", {}) or {}
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

    # Ensure we have valid sampling values
    if sampling_num_states is None or sampling_num_states <= 0:
        print(f"[WARNING] Invalid sampling_num_states: {sampling_num_states}, using default: 1000")
        sampling_num_states = 1000
    if sampling_position_std is None or sampling_position_std <= 0:
        print(
            f"[WARNING] Invalid sampling_position_std: {sampling_position_std}, using default: 0.005"
        )
        sampling_position_std = 0.005
    if sampling_rotation_std is None or sampling_rotation_std <= 0:
        print(
            f"[WARNING] Invalid sampling_rotation_std: {sampling_rotation_std}, using default: 0.1"
        )
        sampling_rotation_std = 0.1
    if sampling_max_distance is None or sampling_max_distance <= 0:
        print(
            f"[WARNING] Invalid sampling_max_distance: {sampling_max_distance}, using default: 0.025"
        )
        sampling_max_distance = 0.025

    # Extract propagation step size from config
    propagation_step_size = (
        args.propagation_step_size
        if args.propagation_step_size is not None
        else config.get("propagation_step_size", 1.0)
    )

    # Extract control duration parameters from config
    min_control_duration = (
        args.min_control_duration
        if args.min_control_duration is not None
        else config.get("min_control_duration", 1)
    )
    max_control_duration = (
        args.max_control_duration
        if args.max_control_duration is not None
        else config.get("max_control_duration", 5)
    )

    # Ensure we have valid propagation and control values
    if propagation_step_size is None or propagation_step_size <= 0:
        print(
            f"[WARNING] Invalid propagation_step_size: {propagation_step_size}, using default: 1.0"
        )
        propagation_step_size = 1.0
    if min_control_duration is None or min_control_duration <= 0:
        print(f"[WARNING] Invalid min_control_duration: {min_control_duration}, using default: 1")
        min_control_duration = 1
    if max_control_duration is None or max_control_duration <= 0:
        print(f"[WARNING] Invalid max_control_duration: {max_control_duration}, using default: 5")
        max_control_duration = 5
    if min_control_duration > max_control_duration:
        print(
            f"[WARNING] min_control_duration ({min_control_duration}) > max_control_duration ({max_control_duration}), swapping"
        )
        min_control_duration, max_control_duration = max_control_duration, min_control_duration

    # Debug: Print final returned values
    print(f"[DEBUG] Final config values:")
    print(f"  - learningRate: {learningRate}")
    print(f"  - numEpochs: {numEpochs}")
    print(f"  - sampling_num_states: {sampling_num_states}")
    print(f"  - sampling_max_distance: {sampling_max_distance}")
    print(f"  - sampling_position_std: {sampling_position_std}")
    print(f"  - sampling_rotation_std: {sampling_rotation_std}")
    print(f"  - propagation_step_size: {propagation_step_size}")
    print(f"  - min_control_duration: {min_control_duration}")
    print(f"  - max_control_duration: {max_control_duration}")

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
        "propagation_step_size": propagation_step_size,
        "min_control_duration": min_control_duration,
        "max_control_duration": max_control_duration,
    }
