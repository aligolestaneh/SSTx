import sys
import yaml
import argparse
import numpy as np
import torch
import time
import csv
from datetime import datetime

from ik import IK
from functools import partial
from sim_network import SimClient
from geometry.pose import Pose, SE2Pose
from geometry.random_push import generate_path_form_params
from utils.utils import visualize_tree_3d
from planning.planning_utils import isStateValid

from factories import (
    configurationSpace,
    pickControlSampler,
    pickObjectShape,
    pickPropagator,
    pickStartState,
    pickGoalState,
    pickPlanner,
)

from ompl import base as ob
from ompl import util as ou
from ompl import control as oc


def load_config(config_file: str) -> dict:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args_and_config():
    """Parse command line arguments and load configuration from YAML file."""
    parser = argparse.ArgumentParser(
        description="Compare Fusion planner costs before and after replanning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fusion_comparison.csv",
        help="Output CSV file path (default: fusion_comparison.csv)",
    )
    parser.add_argument(
        "--replanning-time",
        type=float,
        default=2.0,
        help="Replanning time for Fusion planner (default: 2.0)",
    )

    args = parser.parse_args()

    # Load configuration from YAML file
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

    return config, args


def plan_with_fusion(
    system: str,
    objectShape: np.ndarray,
    startState: np.ndarray,
    goalState: np.ndarray,
    propagator: oc.StatePropagatorFn,
    planningTime: float,
    replanningTime: float,
    use_replanning: bool = False,
):
    """Plan with Fusion planner and optionally use replanning."""
    space, cspace = configurationSpace(system)

    # Define a simple setup class
    ss = oc.SimpleSetup(cspace)
    ss.setStateValidityChecker(
        ob.StateValidityCheckerFn(
            partial(isStateValid, ss.getSpaceInformation())
        )
    )

    # Set the propagator
    ss.setStatePropagator(oc.StatePropagatorFn(propagator))

    # Set the control sampler
    controlSampler = pickControlSampler(system, objectShape)
    cspace.setControlSamplerAllocator(
        oc.ControlSamplerAllocator(controlSampler)
    )

    # Set control duration to 1 (both min and max)
    si = ss.getSpaceInformation()
    si.setMinMaxControlDuration(1, 1)

    # Create a start state
    start = pickStartState(system, space, startState)
    ss.setStartState(start)

    # Create a goal state
    goal = pickGoalState(system, goalState, startState, objectShape, ss)
    goal.setThreshold(0.02)
    ss.setGoal(goal)

    # Choose Fusion planner
    planner = pickPlanner("fusion", ss)
    planner.setPruningRadius(0.1)
    ss.setPlanner(planner)

    # Attempt to solve the problem
    solved = ss.solve(planningTime)

    if not solved:
        return None, None, None

    # Get initial solution info (before replanning)
    solution_path = ss.getSolutionPath()
    initial_solution_info = getSolutionInfo(solution_path, ss, "fusion")
    initial_cost = initial_solution_info["final_cost"]

    # For replanning, continue replanning until solution is short
    final_solution_info = initial_solution_info
    if use_replanning:
        print(
            f"  Initial solution has {len(initial_solution_info.get('controls', []))} controls, cost: {initial_cost:.4f}"
        )

        # Continue replanning while there are more than 1 control
        replan_count = 0
        while len(final_solution_info.get("controls", [])) > 1:
            replan_count += 1
            print(
                f"  Replan {replan_count}: {len(final_solution_info.get('controls', []))} controls remaining"
            )

            # Run replan
            replan_result = ss.getPlanner().replan(replanningTime)

            if not replan_result:
                print(f"  Replan {replan_count} failed")
                break

            # Get updated solution
            updated_solution = ss.getSolutionPath()
            if updated_solution and updated_solution.getStateCount() > 0:
                final_solution_info = getSolutionInfo(
                    updated_solution, ss, "fusion"
                )
                print(
                    f"  After replan {replan_count}: {len(final_solution_info.get('controls', []))} controls, cost: {final_solution_info['final_cost']:.4f}"
                )
            else:
                print(f"  No solution after replan {replan_count}")
                break

        print(
            f"  Fusion solution after {replan_count} replans: {len(final_solution_info.get('controls', []))} controls, cost: {final_solution_info['final_cost']:.4f}"
        )

    return initial_solution_info, final_solution_info, ss


def getSolutionInfo(solution_path, ss, plannerName="unknown"):
    """Extract solution information from a path."""
    solution_info = {}
    solution_info["state_count"] = solution_path.getStateCount()
    solution_info["control_count"] = solution_path.getControlCount()

    control_space = ss.getControlSpace()
    control_dimension = control_space.getDimension()

    # Extract all controls
    controls_list = []
    for i in range(solution_info["control_count"]):
        control = solution_path.getControl(i)
        control_values = []
        for j in range(control_dimension):
            control_values.append(control[j])
        controls_list.append(control_values)
    solution_info["controls"] = controls_list

    # Extract all states
    states_list = []
    for i in range(solution_info["state_count"]):
        state = solution_path.getState(i)
        state_values = [state.getX(), state.getY(), state.getYaw()]
        states_list.append(state_values)
    solution_info["states"] = states_list

    # Extract the final solution cost from the planner's internal tracking
    if plannerName == "fusion":
        # For Fusion planner, get the cost directly from the planner
        solution_info["final_cost"] = (
            ss.getPlanner().getBestSolutionCost().value()
        )
    else:
        # For other planners, try to get cost from the path or calculate manually
        if ss.getProblemDefinition().getOptimizationObjective():
            # Convert PathControl to PathGeometric and get the cost
            geometric_path = solution_path.asGeometric()
            solution_info["final_cost"] = geometric_path.cost(
                ss.getProblemDefinition().getOptimizationObjective()
            ).value()
        else:
            # Fallback: no cost available
            solution_info["final_cost"] = float("inf")

    return solution_info


def main():
    # Parse arguments and load configuration
    config, args = parse_args_and_config()

    # Extract parameters from config
    system = config.get("system")
    objectName = config.get("objectName")
    startState = np.array(config.get("startState"))
    goalState = np.array(config.get("goalState"))
    replanningTime = args.replanning_time

    print("=" * 80)
    print("FUSION PLANNER COMPARISON: SST COST vs AFTER REPLANNING")
    print("=" * 80)
    print(f"System: {system}")
    print(f"Object: {objectName}")
    print(f"Start: {startState}")
    print(f"Goal: {goalState}")
    print(f"Replanning time: {replanningTime}s")
    print(f"Output file: {args.output}")
    print("=" * 80)

    # Set up the connection to the simulation (for object shape and propagator)
    print("Setting up simulation connection...")
    client = SimClient()
    _, dt, _ = client.execute("get_sim_info")

    # Get the object shape
    print("Getting object shape...")
    objectShape = pickObjectShape(objectName)

    # Pick the propagator
    print("Setting up propagator...")
    propagator = pickPropagator(system, objectShape)

    # Planning times to test
    planning_times = np.arange(1.0, 10.5, 1.0)

    # Results storage
    results = []

    # Test each planning time
    for planning_time in planning_times:
        print(f"\n{'='*60}")
        print(f"Testing planning time: {planning_time:.1f}s")
        print(f"{'='*60}")

        # Test Fusion Planner with replanning
        print(f"\n--- Fusion Planner (with replanning) ---")
        start_time = time.time()
        initial_solution, final_solution, fusion_ss = plan_with_fusion(
            system=system,
            objectShape=objectShape,
            startState=startState,
            goalState=goalState,
            propagator=propagator,
            planningTime=planning_time,
            replanningTime=replanningTime,
            use_replanning=True,
        )
        fusion_time = time.time() - start_time

        # Extract results
        initial_cost = (
            initial_solution.get("final_cost", float("inf"))
            if initial_solution
            else float("inf")
        )
        initial_controls = (
            len(initial_solution.get("controls", []))
            if initial_solution
            else 0
        )
        final_cost = (
            final_solution.get("final_cost", float("inf"))
            if final_solution
            else float("inf")
        )
        final_controls = (
            len(final_solution.get("controls", [])) if final_solution else 0
        )

        # Print results
        if initial_cost != float("inf"):
            print(
                f"  SST Cost (before replanning): Cost = {initial_cost:.4f}, Controls = {initial_controls}"
            )
        else:
            print(f"  SST Cost: No solution found")

        if final_cost != float("inf"):
            print(
                f"  Fusion (after replanning): Cost = {final_cost:.4f}, Controls = {final_controls}, Time = {fusion_time:.2f}s"
            )
        else:
            print(f"  Fusion: No solution found, Time = {fusion_time:.2f}s")

        # Print head-to-head cost comparison
        print(f"\n{'='*50}")
        print(f"HEAD-TO-HEAD COST COMPARISON")
        print(f"{'='*50}")
        if initial_cost != float("inf") and final_cost != float("inf"):
            improvement = ((initial_cost - final_cost) / initial_cost) * 100
            print(
                f"Improvement:  {improvement:.1f}% ({initial_cost - final_cost:.4f} units)"
            )
            if final_cost < initial_cost:
                print(
                    f"🏆 FUSION IMPROVED by {initial_cost - final_cost:.4f} units"
                )
            elif initial_cost < final_cost:
                print(
                    f"⚠️  FUSION WORSENED by {final_cost - initial_cost:.4f} units"
                )
            else:
                print(f"🤝 NO CHANGE")
        elif final_cost != float("inf"):
            print(f"🏆 FUSION SUCCESSFUL (only Fusion solution found)")
        elif initial_cost != float("inf"):
            print(f"⚠️  FUSION FAILED (only SST solution found)")
        else:
            print(f"❌ No solution found")
        print(f"{'='*50}")

        # Store results
        results.append(
            {
                "planning_time": planning_time,
                "initial_cost": initial_cost,
                "initial_controls": initial_controls,
                "final_cost": final_cost,
                "final_controls": final_controls,
                "fusion_time": fusion_time,
                "cost_improvement": (
                    initial_cost - final_cost
                    if initial_cost != float("inf")
                    and final_cost != float("inf")
                    else 0
                ),
            }
        )

    # Save results to CSV
    print(f"\n{'='*80}")
    print(f"Saving results to {args.output}")
    print(f"{'='*80}")

    with open(args.output, "w", newline="") as csvfile:
        fieldnames = [
            "planning_time",
            "initial_cost",
            "initial_controls",
            "final_cost",
            "final_controls",
            "fusion_time",
            "cost_improvement",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    # Print summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"{'='*80}")

    # Count successful solutions
    initial_success = sum(
        1 for r in results if r["initial_cost"] != float("inf")
    )
    final_success = sum(1 for r in results if r["final_cost"] != float("inf"))

    print(
        f"SST solutions found: {initial_success}/{len(results)} ({initial_success/len(results)*100:.1f}%)"
    )
    print(
        f"Fusion solutions found: {final_success}/{len(results)} ({final_success/len(results)*100:.1f}%)"
    )

    # Average costs (excluding infinite costs)
    initial_costs = [
        r["initial_cost"] for r in results if r["initial_cost"] != float("inf")
    ]
    final_costs = [
        r["final_cost"] for r in results if r["final_cost"] != float("inf")
    ]

    if initial_costs:
        print(f"Average SST cost: {np.mean(initial_costs):.4f}")
    if final_costs:
        print(f"Average Fusion cost: {np.mean(final_costs):.4f}")

    if initial_costs and final_costs:
        avg_improvement = np.mean(
            [
                r["cost_improvement"]
                for r in results
                if r["cost_improvement"] != 0
            ]
        )
        print(f"Average cost improvement: {avg_improvement:.4f}")

    print(f"\nResults saved to: {args.output}")
    print(
        f"To create a plot, run: python plot_results.py --input {args.output}"
    )


if __name__ == "__main__":
    main()
