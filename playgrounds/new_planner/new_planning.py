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
    bounds.setLow(-10)
    bounds.setHigh(10)
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

    # create a start state
    start = ob.State(space)
    start().setX(-5)
    start().setY(0.0)
    start().setYaw(0.0)

    # create a goal state
    goal = ob.State(space)
    goal().setX(0.0)
    goal().setY(5)
    goal().setYaw(1.57)

    # set the start and goal states
    ss.setStartAndGoalStates(start, goal, 0.05)

    # (optionally) set planner
    si = ss.getSpaceInformation()
    planner = oc.SST(si)
    # planner = oc.RRT(si)
    # planner = oc.EST(si)
    # planner = oc.KPIECE1(si) # this is the default
    # SyclopEST and SyclopRRT require a decomposition to guide the search
    # decomp = MyDecomposition(32, bounds)
    # planner = oc.SyclopEST(si, decomp)
    # planner = oc.SyclopRRT(si, decomp)
    ss.setPlanner(planner)
    # (optionally) set propagation step size
    si.setPropagationStepSize(0.1)

    # attempt to solve the problem
    for i in range(10):
        print(f"Solving for {i+1}th time")
        solved = ss.solve(5.0)
        if solved:
            # print the final state
            final_state = ss.getSolutionPath().getStates()[-1]
            print(
                f"Found solution:\nFinal state: x={final_state.getX():.3f}, y={final_state.getY():.3f}, yaw={final_state.getYaw():.3f}"
            )

    # if solved:
    #     # print the path to screen
    #     print("Found solution:\n%s" % ss.getSolutionPath().printAsMatrix())


if __name__ == "__main__":
    plan()
