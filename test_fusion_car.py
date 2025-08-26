#!/usr/bin/env python3
"""
Test script to confirm that the fusion planner is compatible with a simple car system.
This script tests the replanning functionality and shows the solution after each iteration.
"""

import numpy as np
from functools import partial

from factories import (
    configurationSpace,
    pickControlSampler,
    pickPropagator,
    pickStartState,
    pickGoalState,
    pickPlanner,
)

from utils.configHandler import parse_args_and_config
from utils.utils import arrayDistance, log
from ompl import base as ob
from ompl import control as oc


def isStateValid(spaceInformation, state):
    """Check if state is valid (within bounds and collision-free)."""
    return spaceInformation.satisfiesBounds(state)


def plan_with_fusion(
    system: str,
    startState: np.ndarray,
    goalState: np.ndarray,
    propagator: oc.StatePropagatorFn,
    planningTime: float = 2.0,
    replanningTime: float = 1.0,
    pruningRadius: float = 0.1,
):
    """Plan initial solution using fusion planner."""
    print(f"[INFO] Planning with fusion planner for system: {system}")
    print(f"     - Start state: {startState}")
    print(f"     - Goal state: {goalState}")
    print(f"     - Planning time: {planningTime}s")

    space, cspace = configurationSpace(system)

    # Create SimpleSetup
    ss = oc.SimpleSetup(cspace)

    # Set state validity checker to check the valid bounds of the state space
    ss.setStateValidityChecker(
        ob.StateValidityCheckerFn(partial(isStateValid, ss.getSpaceInformation()))
    )

    # Set the propagator to the given propagator
    ss.setStatePropagator(oc.StatePropagatorFn(propagator))
    ss.getSpaceInformation().setMinMaxControlDuration(1, 10)
    ss.getSpaceInformation().setPropagationStepSize(1.0)

    # Create start and goal states using the given system, space, and states
    start = pickStartState(system, space, startState)
    ss.setStartState(start)

    goal = pickGoalState(system, goalState, ss)
    ss.setGoal(goal)

    # Choose fusion planner and set the pruning radius
    planner = pickPlanner("fusion", ss, pruningRadius=pruningRadius)
    ss.setPlanner(planner)

    # Set optimization objective
    ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(ss.getSpaceInformation()))

    try:
        solved = ss.solve(planningTime)
        if solved:
            print("[INFO] Initial solution found!")
            return get_solution_info(ss, system), ss
        else:
            print("[ERROR] No initial solution found")
            return None, None
    except Exception as e:
        log(f"[ERROR] Planning failed: {e}", "error")
        return None, None


def get_solution_info(ss, system="simple_car"):
    """Extract solution information from SimpleSetup."""
    solution = ss.getSolutionPath()
    if solution is None:
        return None

    # Get states and controls
    states = []
    controls = []
    times = []

    for i in range(solution.getStateCount()):
        state = solution.getState(i)
        if system == "simple_car":
            # Convert to list format for SE2
            state_list = [
                state.getX(),
                state.getY(),
                state.getYaw(),
            ]
        else:
            # For other systems, use appropriate conversion
            state_list = [state.getX(), state.getY(), state.getZ()]
        states.append(state_list)

    for i in range(solution.getControlCount()):
        control = solution.getControl(i)
        # Get control dimension from the control space
        control_dim = ss.getControlSpace().getDimension()
        control_list = [control[j] for j in range(control_dim)]
        controls.append(control_list)
        times.append(solution.getControlDuration(i))

    # Get cost from the planner's solution information
    try:
        # Use the solutionsHandler to get the cost from planner data
        from utils.solutionsHandler import getSolutionsInfo

        planner_solutions = getSolutionsInfo(ss)
        if planner_solutions and len(planner_solutions) > 0:
            cost = planner_solutions[0]["cost"]
        else:
            # Fallback: use path length as cost
            cost = solution.length()
    except Exception as e:
        print(f"[WARNING] Could not get cost from planner: {e}, using path length")
        cost = solution.length()

    return {
        "states": states,
        "controls": controls,
        "time": times,
        "length": solution.length(),
        "cost": cost,
    }


def print_solution_info(solutionsInfo, iteration, description=""):
    """Print solution information in a formatted way."""
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration} {description}")
    print(f"{'='*60}")

    if solutionsInfo is None:
        print("[ERROR] No solution information available")
        return

    print(
        f"Solution found with {len(solutionsInfo['states'])} states and {len(solutionsInfo['controls'])} controls"
    )
    print(f"Path length: {solutionsInfo['length']:.6f}")
    print(f"Cost: {solutionsInfo['cost']:.6f}")

    print(f"\nStates:")
    for i, state in enumerate(solutionsInfo["states"]):
        if len(state) == 3:  # SE2
            print(f"  State {i}: pos=[{state[0]:.3f}, {state[1]:.3f}], yaw={state[2]:.3f}")
        else:  # Other formats
            print(f"  State {i}: {state}")

    print(f"\nControls:")
    for i, control in enumerate(solutionsInfo["controls"]):
        print(f"  Control {i}: {control} (duration: {solutionsInfo['time'][i]:.3f}s)")

    print(f"\nTotal execution time: {sum(solutionsInfo['time']):.3f}s")


def test_replanning_loop(ss, system, replanningTime=2.0):
    """Test the replanning loop until only one control remains."""
    print(f"\n{'='*60}")
    print("STARTING REPLANNING LOOP TEST")
    print(f"{'='*60}")

    # Get the initial solution to compare costs later
    initial_solution = get_solution_info(ss, system)
    initial_cost = initial_solution["cost"] if initial_solution else None
    print(f"[INFO] Initial solution cost: {initial_cost:.6f}")

    iteration = 0

    while True:  # Continue until we break out
        iteration += 1
        print(f"\n[INFO] Replanning iteration {iteration}")

        try:
            # Get current solution to check control count
            current_solution = get_solution_info(ss, system)
            if current_solution is None or len(current_solution["controls"]) <= 1:
                print(
                    f"[INFO] Only {len(current_solution['controls']) if current_solution else 0} control(s) remaining, breaking loop"
                )
                break

            # Use the duration of the first control for replanning time
            first_control_duration = current_solution["time"][0]
            print(
                f"[INFO] Using first control duration ({first_control_duration:.3f}s) for replanning time"
            )

            # Call the replan function
            replanThread = createResolverThread(ss, 1.0, system)
            replanThread.start()
            replanThread.join()

            # Get replan result
            solutionsInfo = replanThread.resultContainer["result"]
            replanCompleted = replanThread.resultContainer["completed"]

            print(f"[INFO] Replan {iteration} completed: {replanCompleted}")

            if solutionsInfo is None:
                print(f"[WARNING] Replan {iteration} returned no solution")
                break

            # Print solution info
            print_solution_info(solutionsInfo, iteration, f"(Replan {iteration})")

            # Print all solution costs from the planner
            print(f"\n📊 ALL SOLUTION COSTS AFTER ITERATION {iteration}:")
            try:
                from utils.solutionsHandler import getAllPlannerSolutionsInfo

                planner = ss.getPlanner()
                all_solutions = getAllPlannerSolutionsInfo(planner, ss)

                if all_solutions:
                    print(f"  Found {len(all_solutions)} solutions:")
                    for i, sol in enumerate(all_solutions):
                        print(
                            f"    Solution {i+1}: Cost = {sol['cost']:.6f}, States = {sol['state_count']}, Controls = {sol['control_count']}"
                        )
                    input("Press Enter to continue...")
                else:
                    print("  No solutions found in planner")
            except Exception as e:
                print(f"  [WARNING] Could not get all solution costs: {e}")

            # Check if we should break (only one control remaining)
            if len(solutionsInfo["controls"]) <= 1:
                print(
                    f"[INFO] Only {len(solutionsInfo['controls'])} control(s) remaining, breaking loop"
                )
                break

        except Exception as e:
            log(f"[ERROR] Error in replanning iteration {iteration}: {e}", "error")
            import traceback

            traceback.print_exc()
            break

    # Get final solution for cost comparison
    final_solution = get_solution_info(ss, system)
    final_cost = final_solution["cost"] if final_solution else None

    print(f"\n{'='*60}")
    print(f"REPLANNING LOOP COMPLETED AFTER {iteration} ITERATIONS")
    print(f"{'='*60}")

    # Compare initial and final costs
    if initial_cost is not None and final_cost is not None:
        cost_difference = final_cost - initial_cost
        cost_improvement = (
            ((initial_cost - final_cost) / initial_cost) * 100 if initial_cost > 0 else 0
        )

        print(f"\n📊 COST COMPARISON:")
        print(f"  Initial solution cost: {initial_cost:.6f}")
        print(f"  Final solution cost:   {final_cost:.6f}")
        print(f"  Cost difference:      {cost_difference:+.6f}")
        if cost_improvement > 0:
            print(f"  Cost improvement:     {cost_improvement:.2f}%")
        else:
            print(f"  Cost change:          {cost_improvement:.2f}%")
    else:
        print(f"\n⚠️ Could not compare costs - missing solution information")

    # Print final summary of all solutions
    print(f"\n🏁 FINAL SUMMARY OF ALL SOLUTIONS:")
    try:
        from utils.solutionsHandler import getAllPlannerSolutionsInfo

        planner = ss.getPlanner()
        all_solutions = getAllPlannerSolutionsInfo(planner, ss)

        if all_solutions:
            print(f"  Total solutions found: {len(all_solutions)}")
            print(f"  Best solution cost: {all_solutions[0]['cost']:.6f}")
            print(f"  Worst solution cost: {all_solutions[-1]['cost']:.6f}")
            print(f"  Cost range: {all_solutions[-1]['cost'] - all_solutions[0]['cost']:.6f}")
        else:
            print("  No solutions found in planner")
    except Exception as e:
        print(f"  [WARNING] Could not get final solution summary: {e}")


def createResolverThread(ss, replanningTime, system="simple_car"):
    """Create a thread to run the replanning function."""
    import threading

    resultContainer = {"result": None, "completed": False}

    def replan_wrapper():
        try:
            print(f"[DEBUG] Starting replanning with {replanningTime}s time limit...")

            # Get current planner and call the replan function
            planner = ss.getPlanner()

            # Call the proper replan function
            planner.replan(replanningTime)

            print(f"[DEBUG] Replanning completed, extracting new solution...")
            # Get the new solution information
            solutionsInfo = get_solution_info(ss, system)
            resultContainer["result"] = solutionsInfo
            resultContainer["completed"] = True
            print(f"[DEBUG] Replanning completed successfully")

        except Exception as e:
            log(f"[ERROR] in replan_wrapper: {e}", "error")
            import traceback

            traceback.print_exc()
            resultContainer["result"] = None
            resultContainer["completed"] = False

    thread = threading.Thread(target=replan_wrapper)
    thread.daemon = True
    thread.resultContainer = resultContainer
    return thread


def main():
    """Main function to test fusion planner with simple car."""
    print("=" * 60)
    print("FUSION PLANNER COMPATIBILITY TEST WITH SIMPLE CAR")
    print("=" * 60)

    # Parse configuration but force system to be simple_car for this test
    config = parse_args_and_config()
    system = "simple_car"  # Force simple_car system
    print(f"Testing system: {system}")

    # Get propagator
    propagator = pickPropagator(system, None)
    if propagator is None:
        log("[ERROR] Could not create propagator", "error")
        return

    # Define test start and goal states
    if system == "simple_car":
        # SE2 states: [x, y, yaw]
        startState = np.array([0.0, 0.0, 0.0])  # Start at origin, facing forward
        goalState = np.array([1.0, 1.0, np.pi / 4])  # Goal at (2,1) with 45° orientation
    else:
        # Default states for other systems
        startState = np.array([0.0, 0.0, 0.0])
        goalState = np.array([1.2, 0.8, 0.0])

    print(f"Start state: {startState}")
    print(f"Goal state: {goalState}")

    # Plan initial solution
    solutionsInfo, ss = plan_with_fusion(
        system=system,
        startState=startState,
        goalState=goalState,
        propagator=propagator,
        planningTime=0.5,
        replanningTime=1.0,
    )

    if solutionsInfo is None:
        log("[ERROR] Initial planning failed", "error")
        return

    # Print initial solution
    print_solution_info(solutionsInfo, 0, "(Initial Plan)")

    # Test replanning loop
    input("Press Enter to start replanning loop...")
    test_replanning_loop(ss, system, replanningTime=1.0)

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
