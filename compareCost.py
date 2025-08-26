#!/usr/bin/env python3

import sys
import numpy as np
import time
import csv
from datetime import datetime

# Import configuration handler
from utils.configHandler import parse_args_and_config

# Import factory functions
from factories import (
    configurationSpace,
    pickControlSampler,
    pickObjectShape,
    pickPropagator,
    pickStartState,
    pickGoalState,
    pickPlanner,
)

# Import OMPL
from ompl import base as ob
from ompl import control as oc
from planning.planning_utils import isStateValid
from functools import partial

# Import solution handler
from utils.solutionsHandler import getSolutionsInfo


def setup_ompl_problem(config):
    """Set up OMPL planning problem using factory functions."""
    print(f"Setting up OMPL problem for system: {config['system']}")

    # Get object shape
    objectShape = pickObjectShape(config["objectName"])

    # Create state and control spaces
    space, cspace = configurationSpace(config["system"])

    # Create space information and simple setup
    si = oc.SpaceInformation(space, cspace)
    ss = oc.SimpleSetup(si)

    # Use a safe state validity checker to avoid segfault
    def safe_state_valid(state):
        x, y, yaw = state.getX(), state.getY(), state.getYaw()
        # Check bounds manually - same as state space bounds
        return (-5.0 <= x <= 5.0) and (-5.0 <= y <= 5.0)

    ss.setStateValidityChecker(ob.StateValidityCheckerFn(safe_state_valid))

    # Set propagator
    propagator = pickPropagator(config["system"], objectShape)
    ss.setStatePropagator(oc.StatePropagatorFn(propagator))

    # Temporarily skip custom control sampler to avoid segfault
    # controlSampler = pickControlSampler(config["system"], objectShape)
    # if controlSampler is not None:
    #     cspace.setControlSamplerAllocator(oc.ControlSamplerAllocator(controlSampler))
    print("DEBUG: Using OMPL default control sampler")

    # Set control duration
    si.setMinMaxControlDuration(1, 1)

    # Create start state
    start = pickStartState(config["system"], space, np.array(config["startState"]))
    ss.setStartState(start)

    # Create goal state
    goal = pickGoalState(
        config["system"],
        np.array(config["goalState"]),
        np.array(config["startState"]),
        objectShape,
        ss,
    )
    goal.setThreshold(0.02)
    ss.setGoal(goal)

    # Set up Fusion planner
    planner = pickPlanner("fusion", ss)
    planner.setPruningRadius(0.1)
    ss.setPlanner(planner)

    return ss, objectShape


# Removed get_solution_info - using getSolutionInfo from solutionsHandler instead


def run_planning_experiment(config, planning_time):
    """Run a single planning experiment for the given time."""
    print(f"\n{'='*60}")
    print(f"Testing planning time: {planning_time:.1f}s")
    print(f"{'='*60}")

    # Set up OMPL problem
    ss, objectShape = setup_ompl_problem(config)

    # Phase 1: Initial solve (SST cost)
    print(f"Phase 1: Running initial solve for {planning_time:.1f}s...")
    start_time = time.time()
    solved = ss.solve(planning_time)
    solve_time = time.time() - start_time

    if not solved:
        print(f"❌ No initial solution found in {planning_time:.1f}s")
        return {
            "planning_time": planning_time,
            "sst_cost": float("inf"),
            "fusion_cost": float("inf"),
            "initial_controls": 0,
            "final_controls": 0,
            "solve_time": solve_time,
            "resolve_time": 0.0,
            "resolve_count": 0,
            "improvement": 0.0,
        }

    # Get initial solution info (SST cost)
    solutions_info = getSolutionsInfo(ss)
    if not solutions_info:
        print("❌ No solution info extracted")
        return {
            "planning_time": planning_time,
            "sst_cost": float("inf"),
            "fusion_cost": float("inf"),
            "initial_controls": 0,
            "final_controls": 0,
            "solve_time": solve_time,
            "resolve_time": 0.0,
            "resolve_count": 0,
            "improvement": 0.0,
        }

    initial_info = solutions_info[0]  # Best solution (sorted by cost)
    sst_cost = initial_info["cost"]
    initial_controls = len(initial_info["controls"])

    print(f"✅ Initial solution found:")
    print(f"   SST Cost: {sst_cost:.4f}")
    print(f"   Controls: {initial_controls}")

    # Phase 2: Resolve until only 1 control remains (Fusion cost)
    print(f"\nPhase 2: Running resolve until 1 control remains...")
    resolve_start_time = time.time()
    resolve_count = 0
    current_controls = initial_controls

    while current_controls > 1:
        resolve_count += 1
        print(f"   Resolve #{resolve_count}: {current_controls} controls remaining")

        # Run resolve with the same time as initial solve
        resolve_result = ss.getPlanner().resolve(planning_time)

        if not resolve_result:
            print(f"   ⚠️ Resolve #{resolve_count} failed")
            break

        # Get updated solution info
        updated_solutions = getSolutionsInfo(ss)
        if updated_solutions:
            updated_info = updated_solutions[0]  # Best solution
            current_controls = len(updated_info["controls"])
            print(
                f"   After resolve #{resolve_count}: {current_controls} controls, cost: {updated_info['cost']:.4f}"
            )
        else:
            print(f"   ⚠️ No solution after resolve #{resolve_count}")
            break

    resolve_time = time.time() - resolve_start_time

    # Get final solution info (Fusion cost)
    final_solutions = getSolutionsInfo(ss)
    if final_solutions:
        final_info = final_solutions[0]  # Best solution
        fusion_cost = final_info["cost"]
        final_controls = len(final_info["controls"])
    else:
        fusion_cost = float("inf")
        final_controls = 0

    print(f"\n📊 RESULTS:")
    print(f"   SST Cost (initial): {sst_cost:.4f}")
    print(f"   Fusion Cost (final): {fusion_cost:.4f}")
    print(f"   Controls: {initial_controls} → {final_controls}")
    print(f"   Resolve iterations: {resolve_count}")

    # Calculate improvement
    if sst_cost != float("inf") and fusion_cost != float("inf"):
        improvement = ((sst_cost - fusion_cost) / sst_cost) * 100
        print(f"   Improvement: {improvement:.1f}% ({sst_cost - fusion_cost:.4f} units)")

        if fusion_cost < sst_cost:
            print(f"   🏆 FUSION IMPROVED by {sst_cost - fusion_cost:.4f} units")
        elif sst_cost < fusion_cost:
            print(f"   ⚠️ FUSION WORSENED by {fusion_cost - sst_cost:.4f} units")
        else:
            print(f"   🤝 NO CHANGE")
    else:
        improvement = 0.0
        print(f"   ❌ Cannot calculate improvement (invalid costs)")

    return {
        "planning_time": planning_time,
        "sst_cost": sst_cost,
        "fusion_cost": fusion_cost,
        "initial_controls": initial_controls,
        "final_controls": final_controls,
        "solve_time": solve_time,
        "resolve_time": resolve_time,
        "resolve_count": resolve_count,
        "improvement": improvement,
    }


def save_results(results, output_file):
    """Save results to CSV file."""
    print(f"\n{'='*80}")
    print(f"Saving results to {output_file}")
    print(f"{'='*80}")

    fieldnames = [
        "planning_time",
        "sst_cost",
        "fusion_cost",
        "initial_controls",
        "final_controls",
        "solve_time",
        "resolve_time",
        "resolve_count",
        "improvement",
    ]

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    # Print summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"{'='*80}")

    # Count successful solutions
    sst_success = sum(1 for r in results if r["sst_cost"] != float("inf"))
    fusion_success = sum(1 for r in results if r["fusion_cost"] != float("inf"))

    print(
        f"SST solutions found: {sst_success}/{len(results)} ({sst_success/len(results)*100:.1f}%)"
    )
    print(
        f"Fusion solutions found: {fusion_success}/{len(results)} ({fusion_success/len(results)*100:.1f}%)"
    )

    # Average costs (excluding infinite costs)
    sst_costs = [r["sst_cost"] for r in results if r["sst_cost"] != float("inf")]
    fusion_costs = [r["fusion_cost"] for r in results if r["fusion_cost"] != float("inf")]

    if sst_costs:
        print(f"Average SST cost: {np.mean(sst_costs):.4f}")
    if fusion_costs:
        print(f"Average Fusion cost: {np.mean(fusion_costs):.4f}")

    if sst_costs and fusion_costs:
        improvements = [r["improvement"] for r in results if r["improvement"] != 0]
        if improvements:
            print(f"Average improvement: {np.mean(improvements):.1f}%")

    print(f"\nResults saved to: {output_file}")


def main():
    """Main function to run the planning comparison."""
    print("=" * 80)
    print("FUSION PLANNER COMPARISON: SST vs FUSION COSTS")
    print("=" * 80)

    # Load configuration
    config = parse_args_and_config()

    print(f"System: {config['system']}")
    print(f"Object: {config['objectName']}")
    print(f"Start: {config['startState']}")
    print(f"Goal: {config['goalState']}")
    print("=" * 80)

    # No simulation connection needed for this comparison script

    # Define planning times: 1.0 to 10.0 seconds in 0.5s increments
    planning_times = np.arange(1.0, 10.5, 0.5)
    print(
        f"Testing {len(planning_times)} planning times: {planning_times[0]:.1f}s to {planning_times[-1]:.1f}s"
    )

    # Run experiments
    results = []
    for planning_time in planning_times:
        try:
            result = run_planning_experiment(config, planning_time)
            results.append(result)
        except Exception as e:
            print(f"❌ Error at planning time {planning_time:.1f}s: {e}")
            # Add failed result
            results.append(
                {
                    "planning_time": planning_time,
                    "sst_cost": float("inf"),
                    "fusion_cost": float("inf"),
                    "initial_controls": 0,
                    "final_controls": 0,
                    "solve_time": 0.0,
                    "resolve_time": 0.0,
                    "resolve_count": 0,
                    "improvement": 0.0,
                }
            )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"fusion_comparison_{timestamp}.csv"
    save_results(results, output_file)


if __name__ == "__main__":
    main()
