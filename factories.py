import numpy as np

from utils.utils import visualize_tree_3d

from train_model import load_model
from planning.planning_utils import BoxPropagator

# Import propagators
from planning.propagators import (
    propagateDublinsAirplane,
    propagateCar,
)

# Import control samplers
from planning.controlSamplers import PushingControlSampler

# Import OMPL modules
from ompl import base as ob
from ompl import control as oc
from ompl import util as ou

# Set OMPL log level
ou.setLogLevel(ou.LogLevel.LOG_INFO)


def pickObjectShape(objectName: str):
    if objectName == "crackerBox":
        return np.array([0.1628, 0.2139, 0.0676])
    else:
        raise ValueError(f"Unknown object for propagator: {objectName}")


def configurationSpace(system: str):
    print(f"DEBUG: configurationSpace called with system='{system}'")

    if system == "pushing":
        space = ob.SE2StateSpace()
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, -0.9)
        bounds.setHigh(0, 0.76)
        bounds.setLow(1, -0.9)
        bounds.setHigh(1, -0.3)
        space.setBounds(bounds)

        # Create control space - CORRECT OMPL method
        print(f"DEBUG: Creating RealVectorControlSpace(space, 3) for pushing")
        cspace = oc.RealVectorControlSpace(space, 3)  # Correct: state space + dimension
        print(f"DEBUG: Created cspace = {cspace}, type = {type(cspace)}")

        cbounds = ob.RealVectorBounds(3)
        cbounds.setLow(0, 0)  # minimum rotation
        cbounds.setHigh(0, 4)  # maximum rotation
        cbounds.setLow(1, -0.4)  # minimum side offset
        cbounds.setHigh(1, 0.4)  # maximum side offset
        cbounds.setLow(2, 0.0)  # minimum push distance
        cbounds.setHigh(2, 0.3)  # maximum push distance
        cspace.setBounds(cbounds)

        print(f"DEBUG: Final cspace = {cspace}, type = {type(cspace)}")

        return space, cspace

    elif system == "simple_car":
        print(f"DEBUG: Setting up simple_car system")
        space = ob.SE2StateSpace()
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, -5.0)
        bounds.setHigh(0, 5.0)
        bounds.setLow(1, -5.0)
        bounds.setHigh(1, 5.0)
        space.setBounds(bounds)

        cspace = oc.RealVectorControlSpace(space, 2)

        cbounds = ob.RealVectorBounds(2)
        cbounds.setLow(0, -1.0)
        cbounds.setHigh(0, 1.0)
        cbounds.setLow(1, -0.3)
        cbounds.setHigh(1, 0.3)
        cspace.setBounds(cbounds)

        return space, cspace

    elif system == "drone":
        print(f"DEBUG: Setting up drone system with droneStateSpace (SE3 + 6D)")

        # Instantiate compound state space (SE3 + 6D)
        space = ob.CompoundStateSpace()
        space.addSubspace(ob.SE3StateSpace(), 1.0)
        space.addSubspace(ob.RealVectorStateSpace(6), 0.002)

        # set the bounds using the function in the class
        posBounds = ob.RealVectorBounds(3)
        posBounds.setLow(0, -20.0)
        posBounds.setHigh(0, 20.0)
        posBounds.setLow(1, -20.0)
        posBounds.setHigh(1, 20.0)
        posBounds.setLow(2, 0.1)
        posBounds.setHigh(2, 20.0)

        velBounds = ob.RealVectorBounds(6)
        velBounds.setLow(-5.0)
        velBounds.setHigh(5.0)

        space.getSubspace(0).setBounds(posBounds)
        space.getSubspace(1).setBounds(velBounds)

        # Create control space for 4 RPM inputs (actual RPM values)
        cspace = oc.RealVectorControlSpace(space, 4)
        cbounds = ob.RealVectorBounds(4)
        base_rpm = 14468.429183500699
        offset_rpm = 0.001
        for i in range(4):
            cbounds.setLow(i, (1 - offset_rpm) * base_rpm)
            cbounds.setHigh(i, (1 + offset_rpm) * base_rpm)
        cspace.setBounds(cbounds)

        print("DEBUG: Created droneStateSpace with bounds and RPM control bounds")
        return space, cspace

    elif system == "dublin_airplane":
        space = ob.SE3StateSpace()
        bounds = ob.RealVectorBounds(3)
        bounds.setLow(0, -20.0)
        bounds.setHigh(0, 20.0)
        bounds.setLow(1, -20.0)
        bounds.setHigh(1, 20.0)
        bounds.setLow(2, 0.1)
        bounds.setHigh(2, 20.0)
        space.setBounds(bounds)

        cspace = oc.RealVectorControlSpace(space, 3)
        cbounds = ob.RealVectorBounds(3)
        cbounds.setLow(0, -0.2)
        cbounds.setHigh(0, 0.2)
        cbounds.setLow(1, -0.2)
        cbounds.setHigh(1, 0.2)
        cbounds.setLow(2, 0.0)
        cbounds.setHigh(2, 0.5)
        cspace.setBounds(cbounds)
        return space, cspace

    else:
        raise ValueError(f"Unknown system for space configuration: {system}")


def pickPropagator(system: str, object_shape: np.ndarray):

    if system == "pushing":
        model = load_model("residual", 3, 3)
        model.load(f"saved_models/crackerBoxRandom9000.pth")
        model = model.model  # use torch model directly
        model.eval()
        propagator = BoxPropagator(model, object_shape)
        return propagator.propagate

    elif system == "dublin_airplane":
        return propagateDublinsAirplane

    elif system == "simple_car":
        return propagateCar

    else:
        raise ValueError(f"Unknown system: {system}")


def pickStartState(system: str, space: ob.StateSpace, startState: np.ndarray):
    if system == "pushing":
        start_state = ob.State(space)
        start_state().setX(startState[0])
        start_state().setY(startState[1])
        start_state().setYaw(startState[2])
        return start_state

    elif system == "dublin_airplane":
        start_state = ob.State(space)
        start_state().setXYZ(startState[0], startState[1], startState[2])
        start_state()[1].x = startState[3]
        start_state()[1].y = startState[4]
        start_state()[1].z = startState[5]
        start_state()[1].w = startState[6]
        return start_state

    elif system == "simple_car":
        start_state = ob.State(space)
        start_state().setX(startState[0])
        start_state().setY(startState[1])
        start_state().setYaw(startState[2])
        return start_state
    else:
        raise ValueError(f"Unknown system for start state: {system}")


class SE2GoalState(ob.GoalState):
    def __init__(self, si, goal, ranges):
        super().__init__(si)
        self.ranges = ranges

        # Create a proper state object and set its values
        goal_state = ob.State(si.getStateSpace())
        goal_state().setX(goal[0])
        goal_state().setY(goal[1])
        goal_state().setYaw(goal[2])
        self.setState(goal_state)
        self.setThreshold(0.01)

    def distanceGoal(self, state: ob.State) -> float:
        try:
            x = state.getX()
            y = state.getY()
            yaw = state.getYaw()
            goal_x = self.getState().getX()
            goal_y = self.getState().getY()
            goal_yaw = self.getState().getYaw()

            # distance to goal in SE2 space
            x_dist = x - goal_x
            y_dist = y - goal_y
            yaw_dist = yaw - goal_yaw
            yaw_dist = (yaw_dist + np.pi) % (2 * np.pi) - np.pi

            # if in range, return a value smaller than 0.01
            # to indicate success, since threshold is by set to 0.01
            if (
                self.ranges[0][0] <= x_dist <= self.ranges[0][1]
                and self.ranges[1][0] <= y_dist <= self.ranges[1][1]
                and self.ranges[2][0] <= yaw_dist <= self.ranges[2][1]
            ):
                return 0
            else:
                dist = np.sqrt(x_dist**2 + y_dist**2 + yaw_dist**2)
                return dist
        except Exception as e:
            print(f"❌ ERROR in distanceGoal: {e}")
            print(f"  - state type: {type(state)}")
            print(f"  - self.getState() type: {type(self.getState())}")
            raise e


def pickGoalState(
    system: str,
    goalState: np.ndarray,
    ss: oc.SimpleSetup,
    threshold: float = 0.1,
):
    if system == "pushing":
        try:
            goal_state = SE2GoalState(
                ss.getSpaceInformation(),
                np.array([goalState[0], goalState[1], goalState[2]]),
                np.array([[-0.05, 0.05], [-0.05, 0.05], [-0.1, 0.1]]),
            )
            return goal_state
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e
    elif system == "dublin_airplane":
        g_state = ob.State(ss.getSpaceInformation())
        g_state().setXYZ(goalState[0], goalState[1], goalState[2])
        g_state()[1].x = goalState[3]
        g_state()[1].y = goalState[4]
        g_state()[1].z = goalState[5]
        g_state()[1].w = goalState[6]
        goal_state = ob.GoalState(ss.getSpaceInformation())
        goal_state.setState(g_state)
        return goal_state

    elif system == "simple_car":
        # Create a goal state
        g_state = ob.State(ss.getSpaceInformation())
        g_state().setX(goalState[0])
        g_state().setY(goalState[1])
        g_state().setYaw(goalState[2])

        # Create a goal region with threshold
        goal_region = ob.GoalState(ss.getSpaceInformation())
        goal_region.setState(g_state)
        goal_region.setThreshold(threshold)  # Set the threshold here
        return goal_region
    else:
        raise ValueError(f"Unknown system for goal state: {system}")


def pickPlanner(planner_name: str, ss: oc.SimpleSetup, pruningRadius: float = 0.1):
    if planner_name == "fusion":
        planner = oc.Fusion(ss.getSpaceInformation())
        planner.setPruningRadius(pruningRadius)
        return planner
    elif planner_name == "aorrt":
        planner = oc.AORRT(ss.getSpaceInformation())
        return planner
    elif planner_name == "sst":
        planner = oc.SST(ss.getSpaceInformation())
        planner.setPruningRadius(pruningRadius)
        return planner
    elif planner_name == "rrt":
        planner = oc.RRT(ss.getSpaceInformation())
        return planner
    else:
        raise ValueError(f"Unknown planner: {planner_name}")


def pickControlSampler(system: str, obj_shape: np.ndarray):
    if system == "pushing":

        def ControlSamplerAllocator(space):
            return PushingControlSampler(space, obj_shape)

        return ControlSamplerAllocator

    elif system == "dublin_airplane":
        # Use OMPL's default control sampler for now
        return None
    else:
        raise ValueError(f"Unknown system for control sampler: {system}")
