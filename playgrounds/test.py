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
    getChildrenStates,
    state2list,
    isSE2Equal,
)


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
    goal().setX(0.0)
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
    solved = ss.solve(5.0)

    if solved:
        print("Initial solution(s) found:")
        infos = getSolutionsInfo(ss)
        for idx, info in enumerate(infos):
            print(f"Solution {idx}:")
            print(f"  Cost: {info['cost']:.5f}")
            print(f"  State count: {info['state_count']}")
            print(f"  Control count: {info['control_count']}")
            print(f"  States: [")
            for state in info["states"]:
                print(f"    [{state[0]:.5f}, {state[1]:.5f}, {state[2]:.5f}]")
            print(f"  ]")
            print(f"  Controls: [")
            for control in info["controls"]:
                print(f"    [{', '.join(f'{c:.5f}' for c in control)}]")
            print(f"  ]")

        # Test getChildrenStates function
        print("\n" + "=" * 60)
        print("🧪 TESTING getChildrenStates FUNCTION")
        print("=" * 60)

        # Test with the first state of the best solution
        if len(infos) > 0:
            best_solution = infos[0]  # Best solution (lowest cost)
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

        # Now call replan on the planner
        print("\n" + "=" * 60)
        print("🔄 CALLING REPLAN")
        print("=" * 60)
        planner = ss.getPlanner()
        if hasattr(planner, "replan"):
            planner.replan(5.0)  # You can adjust the replanning time as needed
            print("Replan finished.")

            # Test getChildrenStates again after replanning
            print("\n" + "=" * 60)
            print("🧪 TESTING getChildrenStates AFTER REPLAN")
            print("=" * 60)

            # Get all solutions after replanning
            infos = getSolutionsInfo(ss)
            print(f"\nAll solutions after replanning (count: {len(infos)}):")
            for idx, info in enumerate(infos):
                print(f"Solution {idx}:")
                print(f"  Cost: {info['cost']:.5f}")
                print(f"  State count: {info['state_count']}")
                print(f"  Control count: {info['control_count']}")
                print(f"  States: [")
                for state in info["states"]:
                    print(
                        f"    [{state[0]:.5f}, {state[1]:.5f}, {state[2]:.5f}]"
                    )
                print(f"  ]")
                print(f"  Controls: [")
                for control in info["controls"]:
                    print(f"    [{', '.join(f'{c:.5f}' for c in control)}]")
                print(f"  ]")

            # Test getChildrenStates with the new solution
            if len(infos) > 0:
                best_solution = infos[0]  # Best solution after replan
                first_state = best_solution["states"][0]
                print(
                    f"\n🔍 Testing getChildrenStates for first state after replan: {first_state}"
                )

                children = getChildrenStates(ss, first_state)
                print(
                    f"✅ Found {len(children)} children states after replan:"
                )
                for i, child in enumerate(children):
                    print(f"  Child {i}: {child}")
        else:
            print("Planner does not support replan().")
    else:
        print("No solution found.")


if __name__ == "__main__":
    plan()
