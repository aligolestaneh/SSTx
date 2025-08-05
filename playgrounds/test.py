from math import sin, cos
from functools import partial

try:
    from ompl import base as ob
    from ompl import control as oc
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys

    sys.path.insert(
        0, join(dirname(dirname(abspath(__file__))), "py-bindings")
    )
    from ompl import base as ob
    from ompl import control as oc

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from fusionPlanning import (
    getSolutionsInfo,
    getAllPlannerSolutionsInfo,
    printAllPlannerSolutions,
    getChildrenStates,
    state2list,
    isSE2Equal,
)


def getBestSolutionFromPlanner(ss):
    """Get the best solution using the new planner tracking feature, with fallback."""
    planner = ss.getPlanner()

    # Try to use the new solution tracking feature first
    if hasattr(planner, "getAllSolutions"):
        try:
            all_solution_infos = getAllPlannerSolutionsInfo(planner, ss)
            if len(all_solution_infos) > 0:
                # Return the best solution (first one after sorting by cost)
                return all_solution_infos[0], all_solution_infos
            else:
                print(
                    "⚠️ Enhanced tracking returned no solutions, falling back to original method..."
                )
        except Exception as e:
            print(
                f"⚠️ Error with enhanced solution tracking: {e}, falling back to original method..."
            )

    # Fallback to original method
    infos = getSolutionsInfo(ss)
    if len(infos) > 0:
        return infos[0], infos
    else:
        return None, []


def printBestSolution(best_solution, iteration_label=""):
    """Print detailed information about the best solution."""
    if not best_solution:
        print(f"❌ {iteration_label}: No solution available")
        return

    print(f"\n🏆 {iteration_label} - BEST SOLUTION:")
    print("=" * 50)
    print(f"  Cost: {best_solution['cost']:.5f}")
    print(f"  States: {best_solution['state_count']}")
    print(f"  Controls: {best_solution['control_count']}")

    print(f"\n  📍 States:")
    for i, state in enumerate(best_solution["states"]):
        print(
            f"    [{i}]: x={state[0]:.5f}, y={state[1]:.5f}, yaw={state[2]:.5f}"
        )

    print(f"\n  🎮 Controls:")
    if "controls" in best_solution and len(best_solution["controls"]) > 0:
        for i, control in enumerate(best_solution["controls"]):
            print(f"    [{i}]: [{', '.join(f'{c:.5f}' for c in control)}]")
    else:
        print("    (No controls - reached goal state)")
    print("=" * 50)


class MyDecomposition(oc.GridDecomposition):
    def __init__(self, length, bounds):
        super(MyDecomposition, self).__init__(length, 2, bounds)

    def project(self, s, coord):
        coord[0] = s.getX()
        coord[1] = s.getY()

    def sampleFullState(self, sampler, coord, s):
        sampler.sampleUniform(s)
        s.setXY(coord[0], coord[1])


def isStateValid(spaceInformation, state):
    # perform collision checking or check if other constraints are
    # satisfied
    return spaceInformation.satisfiesBounds(state)


def propagate(start, control, duration, state):
    state.setX(start.getX() + control[0] * duration * cos(start.getYaw()))
    state.setY(start.getY() + control[0] * duration * sin(start.getYaw()))
    state.setYaw(start.getYaw() + control[1] * duration)


def plan():
    # construct the state space we are planning in
    space = ob.SE2StateSpace()

    # set the bounds for the R^2 part of SE(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-1)
    bounds.setHigh(1)
    space.setBounds(bounds)

    # create a control space
    cspace = oc.RealVectorControlSpace(space, 2)
    # set the bounds for the control space
    cbounds = ob.RealVectorBounds(2)
    cbounds.setLow(-0.3)
    cbounds.setHigh(0.3)
    cspace.setBounds(cbounds)

    # define a simple setup class
    ss = oc.SimpleSetup(cspace)
    ss.setStateValidityChecker(
        ob.StateValidityCheckerFn(
            partial(isStateValid, ss.getSpaceInformation())
        )
    )
    ss.setStatePropagator(oc.StatePropagatorFn(propagate))
    # Set min and max control duration to 1
    ss.getSpaceInformation().setMinMaxControlDuration(1, 1)

    # create a start state
    start = ob.State(space)
    start().setX(-0.5)
    start().setY(0.0)
    start().setYaw(0.0)

    # create a goal state
    goal = ob.State(space)
    goal().setX(0.5)
    goal().setY(0.0)
    goal().setYaw(0.0)

    # set the start and goal states
    ss.setStartAndGoalStates(start, goal, 0.05)

    # (optionally) set planner
    si = ss.getSpaceInformation()
    planner = oc.Fusion(si)
    # Set the optimization objective to path length
    ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
    # planner = oc.SST(si)
    # planner = oc.RRT(si)
    # planner = oc.EST(si)
    # planner = oc.KPIECE1(si) # this is the default
    # SyclopEST and SyclopRRT require a decomposition to guide the search
    # decomp = MyDecomposition(32, bounds)
    # planner = oc.SyclopEST(si, decomp)
    # planner = oc.SyclopRRT(si, decomp)
    ss.setPlanner(planner)
    # (optionally) set propagation step size
    si.setPropagationStepSize(1)

    # attempt to solve the problem
    solved = ss.solve(2.0)

    if solved:
        print("Initial solution(s) found:")

        # Use the new solution tracking feature
        planner = ss.getPlanner()
        if hasattr(planner, "getAllSolutions"):
            printAllPlannerSolutions(
                planner, "All Solutions Found During Initial Planning"
            )

        # Get the best solution using the new tracking feature
        best_solution, all_infos = getBestSolutionFromPlanner(ss)
        if not best_solution:
            print("❌ No solutions found!")
            return

        printBestSolution(best_solution, "INITIAL PLANNING")

        # Test getChildrenStates function
        print("\n" + "=" * 60)
        print("🧪 TESTING getChildrenStates FUNCTION")
        print("=" * 60)

        # Test with the first state of the best solution
        if best_solution and len(best_solution["states"]) > 0:
            first_state = best_solution["states"][0]
            print(
                f"\n🔍 Testing getChildrenStates for first state: {first_state}"
            )

            children = getChildrenStates(ss, first_state)
            print(f"✅ Found {len(children)} children states:")
            for i, child in enumerate(children):
                print(f"  Child {i}: {child}")

            # Test with the second state if it exists
            if len(best_solution["states"]) > 1:
                second_state = best_solution["states"][1]
                print(
                    f"\n🔍 Testing getChildrenStates for second state: {second_state}"
                )

                children2 = getChildrenStates(ss, second_state)
                print(f"✅ Found {len(children2)} children states:")
                for i, child in enumerate(children2):
                    print(f"  Child {i}: {child}")

            # Test with a state that should not exist
            fake_state = [999.0, 999.0, 999.0]
            print(
                f"\n🔍 Testing getChildrenStates for fake state: {fake_state}"
            )
            children_fake = getChildrenStates(ss, fake_state)
            print(
                f"✅ Found {len(children_fake)} children states (should be 0):"
            )
            for i, child in enumerate(children_fake):
                print(f"  Child {i}: {child}")

        # Now call replan on the planner MULTIPLE TIMES
        planner = ss.getPlanner()
        if hasattr(planner, "replan"):

            # Continue replanning while there are more than 1 control
            replan_count = 0
            print("\n" + "=" * 80)
            print(
                "🔄 STARTING CONTINUOUS REPLAN TESTING WITH NEW SOLUTION TRACKING"
            )
            print("=" * 80)

            # Get initial solution info using the new tracking feature
            current_solution_info, _ = getBestSolutionFromPlanner(ss)
            if not current_solution_info:
                print("❌ No initial solution found!")
                return

            print(
                f"  Initial solution has {current_solution_info.get('control_count', 0)} controls, cost: {current_solution_info.get('cost', 0):.4f}"
            )

            while (
                current_solution_info
                and current_solution_info.get("control_count", 0) > 1
            ):
                replan_count += 1
                print(
                    f"\n  🔄 Replan {replan_count}: {current_solution_info.get('control_count', 0)} controls remaining"
                )

                # Get solutions BEFORE replanning using new tracking
                best_before, all_before = getBestSolutionFromPlanner(ss)
                if best_before:
                    printBestSolution(
                        best_before, f"BEFORE REPLAN {replan_count}"
                    )

                    # Show all solutions tracked by planner
                    if hasattr(planner, "getAllSolutions"):
                        printAllPlannerSolutions(
                            planner,
                            f"All Solutions Before Replan {replan_count}",
                        )

                    # Print detailed information for ALL solutions
                    print(f"\n📋 ALL SOLUTIONS BEFORE REPLAN {replan_count}:")
                    print("=" * 70)
                    for idx, solution_info in enumerate(all_before):
                        print(f"\n🔹 Solution {idx + 1}:")
                        print(f"   Cost: {solution_info['cost']:.5f}")
                        print(f"   States: {solution_info['state_count']}")
                        print(f"   Controls: {solution_info['control_count']}")

                        print(f"   📍 States:")
                        for i, state in enumerate(solution_info["states"]):
                            print(
                                f"     [{i}]: x={state[0]:.5f}, y={state[1]:.5f}, yaw={state[2]:.5f}"
                            )

                        print(f"   🎮 Controls:")
                        if (
                            "controls" in solution_info
                            and len(solution_info["controls"]) > 0
                        ):
                            for i, control in enumerate(
                                solution_info["controls"]
                            ):
                                print(
                                    f"     [{i}]: [{', '.join(f'{c:.5f}' for c in control)}]"
                                )
                        else:
                            print("     (No controls)")
                    print("=" * 70)

                # Call replan
                print(
                    f"\n🔄 Calling planner.replan() - iteration {replan_count}"
                )
                try:
                    replan_result = planner.replan(
                        1.00
                    )  # Very short replanning time to test fallback

                    if replan_result:
                        print(f"✅ Replan {replan_count} succeeded.")
                    else:
                        print(
                            f"⚠️ Replan {replan_count} returned false, but continuing anyway..."
                        )

                except Exception as e:
                    print(
                        f"❌ Replan {replan_count} failed with error: {e}, but continuing anyway..."
                    )

                # Always continue regardless of replan result

                # Get solutions AFTER replanning using new tracking
                best_after, all_after = getBestSolutionFromPlanner(ss)
                if best_after:
                    printBestSolution(
                        best_after, f"AFTER REPLAN {replan_count}"
                    )

                    # Show all solutions tracked by planner
                    if hasattr(planner, "getAllSolutions"):
                        printAllPlannerSolutions(
                            planner,
                            f"All Solutions After Replan {replan_count}",
                        )

                    # Print detailed information for ALL solutions
                    print(f"\n📋 ALL SOLUTIONS AFTER REPLAN {replan_count}:")
                    print("=" * 70)
                    for idx, solution_info in enumerate(all_after):
                        print(f"\n🔹 Solution {idx + 1}:")
                        print(f"   Cost: {solution_info['cost']:.5f}")
                        print(f"   States: {solution_info['state_count']}")
                        print(f"   Controls: {solution_info['control_count']}")

                        print(f"   📍 States:")
                        for i, state in enumerate(solution_info["states"]):
                            print(
                                f"     [{i}]: x={state[0]:.5f}, y={state[1]:.5f}, yaw={state[2]:.5f}"
                            )

                        print(f"   🎮 Controls:")
                        if (
                            "controls" in solution_info
                            and len(solution_info["controls"]) > 0
                        ):
                            for i, control in enumerate(
                                solution_info["controls"]
                            ):
                                print(
                                    f"     [{i}]: [{', '.join(f'{c:.5f}' for c in control)}]"
                                )
                        else:
                            print("     (No controls)")
                    print("=" * 70)

                    # Check if the path got shorter
                    states_before = (
                        best_before["state_count"] if best_before else 0
                    )
                    controls_before = (
                        best_before["control_count"] if best_before else 0
                    )
                    states_after = best_after["state_count"]
                    controls_after = best_after.get("control_count", 0)

                    print(f"\n🔍 COMPARISON:")
                    print(
                        f"  States: {states_before} → {states_after} (change: {states_after - states_before})"
                    )
                    print(
                        f"  Controls: {controls_before} → {controls_after} (change: {controls_after - controls_before})"
                    )
                    print(
                        f"  After replan {replan_count}: {controls_after} controls, cost: {best_after['cost']:.4f}"
                    )

                    # Update current solution info for next iteration
                    current_solution_info = best_after

                    # Check if no more controls (but don't break automatically)
                    if controls_after == 0:
                        print(
                            "🛑 No more controls left, but you can still continue replanning if desired"
                        )

                else:
                    print(
                        "📊 AFTER REPLAN: No solution available, but continuing..."
                    )
                    # Don't break, let user decide

                # Add input prompt for user control (always ask regardless of solution state)
                user_input = input(
                    f"\n⏸️  Press Enter to continue to replan {replan_count + 1}, or 'q' to quit: "
                )
                if user_input.lower() == "q":
                    print("👋 Stopping replanning by user request")
                    break

            print(f"\n🏁 FINAL RESULT:")
            if current_solution_info:
                print(
                    f"  Final solution after {replan_count} replans: {current_solution_info.get('control_count', 0)} controls, cost: {current_solution_info.get('cost', 0):.4f}"
                )
            else:
                print("  No final solution available")

        else:
            print("Planner does not support replan().")
    else:
        print("No solution found.")


if __name__ == "__main__":
    plan()
