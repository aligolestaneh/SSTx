from functools import partial
import numpy as np

from utils.utils import visualize_tree_3d

from train_model import load_model
from planning.planning_utils import (
    BoxPropagator,
    isStateValid,
    GraspableRegion,
)

# Import propagators
from planning.propagators import (
    propagate_simple,
    propagate_complex,
    propagate_unstable,
    propagate_pendulum,
)

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

    if system == "pushing":
        space = ob.SE2StateSpace()
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, -0.9)
        bounds.setHigh(0, 0.76)
        bounds.setLow(1, -0.9)
        bounds.setHigh(1, -0.3)
        space.setBounds(bounds)

        cspace = oc.RealVectorControlSpace(space, 3)
        cbounds = ob.RealVectorBounds(3)
        cbounds.setLow(0, 0)  # minimum rotation
        cbounds.setHigh(0, 4)  # maximum rotation
        cbounds.setLow(1, -0.4)  # minimum side offset
        cbounds.setHigh(1, 0.4)  # maximum side offset
        cbounds.setLow(2, 0.0)  # minimum push distance
        cbounds.setHigh(2, 0.3)  # maximum push distance
        cspace.setBounds(cbounds)

        return space, cspace

    elif system == "car":
        pass

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

    elif system == "simple":
        return propagate_simple

    elif system == "complex":
        return propagate_complex

    elif system == "unstable":
        return propagate_unstable

    elif system == "pendulum":
        return propagate_pendulum

    else:
        raise ValueError(f"Unknown system: {system}")


def pickStartState(system: str, space: ob.StateSpace, startState: np.ndarray):
    if system == "pushing":
        start_state = ob.State(space)
        start_state().setX(startState[0])
        start_state().setY(startState[1])
        start_state().setYaw(startState[2])
        return start_state
    else:
        raise ValueError(f"Unknown system for start state: {system}")


class SE2GoalState(ob.GoalState):
    def __init__(self, si, goal, ranges):
        print(f"🔍 DEBUG: SE2GoalState.__init__ called")
        print(f"  - si type: {type(si)}")
        print(f"  - goal type: {type(goal)}, value: {goal}")
        print(f"  - ranges type: {type(ranges)}, value: {ranges}")

        super().__init__(si)
        self.ranges = ranges

        # Create a proper state object and set its values
        goal_state = ob.State(si.getStateSpace())
        goal_state().setX(goal[0])
        goal_state().setY(goal[1])
        goal_state().setYaw(goal[2])
        self.setState(goal_state)
        self.setThreshold(0.01)

        print(f"✅ SE2GoalState initialized successfully")

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
    startState: np.ndarray,
    objectShape: np.ndarray,
    ss: oc.SimpleSetup,
):
    print(f"🔍 DEBUG: pickGoalState called")
    print(f"  - system: {system}")
    print(f"  - goalState: {goalState}")
    print(f"  - startState: {startState}")
    print(f"  - objectShape: {objectShape}")
    print(f"  - ss type: {type(ss)}")

    if system == "pushing":
        print(f"🔍 Creating SE2GoalState for pushing system...")
        try:
            goal_state = SE2GoalState(
                ss.getSpaceInformation(),
                np.array([goalState[0], goalState[1], goalState[2]]),
                np.array([[-0.05, 0.05], [-0.05, 0.05], [-0.1, 0.1]]),
            )
            print(f"✅ SE2GoalState created successfully")
            return goal_state
        except Exception as e:
            print(f"❌ ERROR creating SE2GoalState: {e}")
            import traceback

            traceback.print_exc()
            raise e
        # edge = 0.725
        # goal_state = GraspableRegion(
        #     ss.getSpaceInformation(),
        #     np.array([edge, startState[1], 0.0]),
        #     objectShape,
        #     edge,
        # )
        # return goal_state
    else:
        raise ValueError(f"Unknown system for goal state: {system}")


def pickPlanner(planner_name: str, ss: oc.SimpleSetup):
    if planner_name == "fusion":
        planner = oc.Fusion(ss.getSpaceInformation())
        return planner
    elif planner_name == "sst":
        planner = oc.SST(ss.getSpaceInformation())
        return planner
    else:
        raise ValueError(f"Unknown planner: {planner_name}")


def pickControlSampler(system: str, obj_shape: np.ndarray):
    if system == "pushing":

        def ControlSamplerAllocator(space):
            return ControlSampler(space, obj_shape)

        return ControlSamplerAllocator
    else:
        raise ValueError(f"Unknown system for control sampler: {system}")


class ControlSampler(oc.ControlSampler):
    def __init__(self, space, obj_shape):
        super().__init__(space)
        self.control_space = space
        self.obj_shape = obj_shape

    def sample(self, control):
        control[0] = np.random.uniform(
            self.control_space.getBounds().low[0],
            self.control_space.getBounds().high[0],
        )
        control[1] = np.random.uniform(
            self.control_space.getBounds().low[1],
            self.control_space.getBounds().high[1],
        )
        control[2] = np.random.uniform(
            self.control_space.getBounds().low[2],
            self.control_space.getBounds().high[2],
        )

        # Convert to absolute control values
        control[0] = int(control[0]) * np.pi / 2
        mask = (control[0] % np.pi) == 0
        control[1] *= np.where(mask, self.obj_shape[1], self.obj_shape[0])


# Removed duplicate SE2GoalState class definition


def plan(
    system: str,
    objectName: str,
    startState: np.ndarray,
    goalState: np.ndarray,
    planningTime: float = 20.0,
    replanningTime: float = 2.0,
    plannerName: str = "fusion",
    visualize: bool = False,
):
    space, cspace = configurationSpace(system)

    # Define a simple setup class
    ss = oc.SimpleSetup(cspace)
    ss.setStateValidityChecker(
        ob.StateValidityCheckerFn(
            partial(isStateValid, ss.getSpaceInformation())
        )
    )

    # Choose propagator based on parameter
    objectShape = pickObjectShape(objectName)
    propagator = pickPropagator(system, objectShape)
    ss.setStatePropagator(oc.StatePropagatorFn(propagator))

    # Choose control sampler based on parameter
    controlSampler = pickControlSampler(system, objectShape)
    cspace.setControlSamplerAllocator(
        oc.ControlSamplerAllocator(controlSampler)
    )

    # Create a start state
    start = pickStartState(system, space, startState)
    ss.setStartState(start)

    # Create a goal state
    goal = pickGoalState(system, goalState, startState, objectShape, ss)
    goal.setThreshold(0.02)
    ss.setGoal(goal)

    # Choose planner based on parameter
    planner = pickPlanner(plannerName, ss)
    planner.setPruningRadius(0.1)
    ss.setPlanner(planner)

    # Attempt to solve the problem
    solved = ss.solve(planningTime)

    # Show 3D visualization of the tree
    if visualize:
        visualize_tree_3d(planner, filename=f"fusion_3d_{planningTime}s.png")

    if solved:
        # Print the path to screen
        print("Initial solution found")
        print("Found solution:\n%s" % ss.getSolutionPath().printAsMatrix())

        # Interactive resolve loop - continue until goal reached or user stops
        print(
            f"\nStarting interactive resolve with {replanningTime}s time limit per iteration..."
        )

        # Goal coordinates for distance checking
        goal_x, goal_y = 0.0, 0.5  # From goal state definition above

        resolve_iter = 0
        if replanningTime > 0:
            while True:
                resolve_iter += 1
                print(f"\n{'='*60}")
                print(f"RESOLVE ITERATION {resolve_iter}")
                print(f"{'='*60}")

                # Check current state before resolve
                try:
                    current_solution = ss.getSolutionPath()
                    if (
                        current_solution
                        and current_solution.getStateCount() > 0
                    ):
                        # Get the last state in current solution (where we are now)
                        last_state = current_solution.getState(
                            current_solution.getStateCount() - 1
                        )
                        try:
                            # Try to extract current position
                            current_x = last_state.getX()
                            current_y = last_state.getY()
                            current_yaw = last_state.getYaw()
                        except AttributeError:
                            try:
                                # Try compound state access
                                se2_comp = last_state[0]
                                current_x = se2_comp.getX()
                                current_y = se2_comp.getY()
                                current_yaw = se2_comp.getYaw()
                            except:
                                current_x, current_y, current_yaw = (
                                    0.0,
                                    0.0,
                                    0.0,
                                )

                        # Calculate distance to goal
                        distance_to_goal = (
                            (current_x - goal_x) ** 2
                            + (current_y - goal_y) ** 2
                        ) ** 0.5

                        print(
                            f"📍 Current position: x={current_x:.3f}, y={current_y:.3f}, yaw={current_yaw:.3f}"
                        )
                        print(
                            f"🎯 Goal position:    x={goal_x:.3f}, y={goal_y:.3f}"
                        )
                        print(f"📏 Distance to goal: {distance_to_goal:.3f}")

                        # Check if we've reached the goal
                        if (
                            distance_to_goal < 0.05
                        ):  # Same tolerance as setStartAndGoalStates
                            print(
                                f"🎉 SUCCESS! Reached goal (distance {distance_to_goal:.3f} < 0.05)"
                            )
                            print("No more resolve iterations needed!")
                            break

                    else:
                        print("❌ No current solution available")
                        break

                except Exception as e:
                    print(f"⚠️  Error checking current state: {e}")

                # Get tree size before resolve
                planner_data = ob.PlannerData(ss.getSpaceInformation())
                planner.getPlannerData(planner_data)
                tree_size_before = planner_data.numVertices()
                print(
                    f"🌳 Tree size before resolve: {tree_size_before} vertices"
                )

                # Ask user if they want to continue
                user_input = (
                    input(
                        f"\n🤔 Run resolve iteration {resolve_iter}? (y/n/q to quit): "
                    )
                    .lower()
                    .strip()
                )
                if user_input in ["n", "no", "q", "quit"]:
                    print("👋 Stopping resolve iterations by user request")
                    break
                elif user_input not in ["y", "yes", ""]:
                    print("Invalid input, assuming 'no'")
                    break

                # Run resolve iteration
                print(f"\n🚀 Running resolve iteration {resolve_iter}...")

                # Print current start and goal states
                try:
                    pdef = ss.getProblemDefinition()

                    # Print start state
                    if pdef.getStartStateCount() > 0:
                        start_state = pdef.getStartState(0)
                        try:
                            start_x = start_state.getX()
                            start_y = start_state.getY()
                            start_yaw = start_state.getYaw()
                            print(
                                f"🏁 Start state: x={start_x:.3f}, y={start_y:.3f}, yaw={start_yaw:.3f}"
                            )
                        except AttributeError:
                            print(f"🏁 Start state: {start_state}")
                    else:
                        print("🏁 No start state set")

                    # Print goal state
                    goal = pdef.getGoal()
                    if goal:
                        # Try to get goal state if it's a sampleable region
                        try:
                            goal_sampleable = goal.as_GoalSampleableRegion()
                            if goal_sampleable and goal_sampleable.hasStates():
                                goal_state = goal_sampleable.getState(0)
                                goal_x = goal_state.getX()
                                goal_y = goal_state.getY()
                                goal_yaw = goal_state.getYaw()
                                print(
                                    f"🎯 Goal state: x={goal_x:.3f}, y={goal_y:.3f}, yaw={goal_yaw:.3f}"
                                )
                            else:
                                print(f"🎯 Goal: {goal}")
                        except:
                            # If we can't extract specific goal state, just print the goal coordinates we know
                            print(
                                f"🎯 Goal region: x={goal_x:.3f}, y={goal_y:.3f}"
                            )
                    else:
                        print("🎯 No goal set")

                except Exception as e:
                    print(f"⚠️  Error getting start/goal states: {e}")

                try:
                    result = planner.resolve(replanningTime)
                    visualize_tree_3d(
                        planner,
                        filename=f"fusion_3d_{planningTime}s_resolve_{resolve_iter}.png",
                    )
                    print(f"✅ Resolve result: {result}")

                    # Check tree size after resolve
                    planner_data_after = ob.PlannerData(
                        ss.getSpaceInformation()
                    )
                    planner.getPlannerData(planner_data_after)
                    tree_size_after = planner_data_after.numVertices()
                    print(
                        f"🌳 Tree size after resolve: {tree_size_after} vertices"
                    )
                    tree_change = tree_size_after - tree_size_before
                    if tree_change > 0:
                        print(f"📈 Tree growth: {tree_change} vertices added")
                    elif tree_change < 0:
                        print(
                            f"📉 Tree reduction: {-tree_change} vertices removed"
                        )
                    else:
                        print(
                            f"📊 Tree size unchanged: {tree_size_after} vertices"
                        )

                    # Show the updated solution
                    try:
                        updated_solution = ss.getSolutionPath()
                        if updated_solution:
                            print("\n📋 Updated solution path:")
                            print(updated_solution.printAsMatrix())
                        else:
                            print("❌ No solution available after resolve")
                            print(
                                "🛑 Cannot continue - no path to goal from current state"
                            )
                            break
                    except Exception as e:
                        print(f"⚠️  Error getting solution after resolve: {e}")

                except Exception as e:
                    print(
                        f"💥 Error during resolve iteration {resolve_iter}: {e}"
                    )
                    print("🛑 Stopping resolve iterations due to error")
                    break

        print(f"\n🏁 Resolve process completed!")
        input("Press Enter to continue...")
