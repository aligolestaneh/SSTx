import sys
import yaml
import torch
import argparse
import numpy as np
import threading

# Fix HTMLParser compatibility issue
# import html

# if not hasattr(html, "unescape"):
#     import html.parser

#     html.unescape = html.parser.HTMLParser().unescape

from ik import IK
from functools import partial
from sim_network import SimClient
from geometry.pose import Pose, SE2Pose
from utils.utils import visualize_tree_3d
from train_model import load_model, load_opt_model_2
from geometry.random_push import generate_path_form_params
from planning.planning_utils import isStateValid, BoxPropagator


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
        "sampling_position_std": sampling_position_std,
        "sampling_rotation_std": sampling_rotation_std,
    }


def plan(
    system: str,
    objectShape: np.ndarray,
    startState: np.ndarray,
    goalState: np.ndarray,
    propagator: oc.StatePropagatorFn,
    planningTime: float = 20.0,
    replanningTime: float = 2.0,
    plannerName: str = "fusion",
    visualize: bool = False,
):
    print(f"🔍 DEBUG: plan function started")
    print(f"  - system: {system}")
    print(f"  - objectShape: {objectShape}")
    print(f"  - startState: {startState}")
    print(f"  - goalState: {goalState}")
    print(f"  - planningTime: {planningTime}")
    print(f"  - plannerName: {plannerName}")

    try:
        space, cspace = configurationSpace(system)
        # print(f"✅ Configuration space created")

        # Define a simple setup class
        ss = oc.SimpleSetup(cspace)
        # print(f"✅ SimpleSetup created")

        ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(
                partial(isStateValid, ss.getSpaceInformation())
            )
        )
        # print(f"✅ State validity checker set")

        # Set the propagator
        ss.setStatePropagator(oc.StatePropagatorFn(propagator))
        ss.getSpaceInformation().setMinMaxControlDuration(1, 1)
        # print(f"✅ State propagator set")

        # Set the control sampler
        controlSampler = pickControlSampler(system, objectShape)
        cspace.setControlSamplerAllocator(
            oc.ControlSamplerAllocator(controlSampler)
        )
        # print(f"✅ Control sampler set")

        # Create a start state
        # print(f"🔍 Creating start state...")
        start = pickStartState(system, space, startState)
        ss.setStartState(start)
        # print(f"✅ Start state set")

        # Create a goal state
        # print(f"🔍 Creating goal state...")
        goal = pickGoalState(system, goalState, startState, objectShape, ss)
        goal.setThreshold(0.02)
        ss.setGoal(goal)
        # print(f"✅ Goal state set")
    except Exception as e:
        print(f"❌ ERROR in plan function setup: {e}")
        import traceback

        traceback.print_exc()
        raise e

    # Choose planner based on parameter
    # print(f"🔍 Creating planner...")
    planner = pickPlanner(plannerName, ss)
    planner.setPruningRadius(0.1)
    ss.setPlanner(planner)
    # print(f"✅ Planner set")

    # Set the optimization objective to path length
    # print(f"🔍 Setting optimization objective...")
    ss.setOptimizationObjective(
        ob.PathLengthOptimizationObjective(ss.getSpaceInformation())
    )
    # print(f"✅ Optimization objective set")

    # Attempt to solve the problem
    # print(f"🔍 Attempting to solve the problem...")
    try:
        solved = ss.solve(planningTime)
        # print(f"✅ Solve completed, result: {solved}")
    except Exception as e:
        print(f"❌ ERROR during solve: {e}")
        import traceback

        traceback.print_exc()
        raise e

    # Show 3D visualization of the tree
    if visualize:
        visualize_tree_3d(planner, filename=f"fusion_3d_{planningTime}s.png")

    if solved:
        # Print the path to screen
        print("Initial solution found")
        return getSolutionsInfo(ss), ss
    else:
        print("No solution found")
        return None, None


def getPathInfo(solution_path, ss):
    solution_info = {}
    solution_info["state_count"] = solution_path.getStateCount()
    solution_info["control_count"] = solution_path.getControlCount()

    control_space = ss.getControlSpace()
    control_dimension = control_space.getDimension()

    # Extract all controls while objects are valid
    controls_list = []
    for i in range(solution_info["control_count"]):
        control = solution_path.getControl(i)
        control_values = [control[j] for j in range(control_dimension)]
        controls_list.append(control_values)
        solution_info["controls"] = controls_list

    # Extract all states while objects are valid
    states_list = []
    for i in range(solution_info["state_count"]):
        state = solution_path.getState(i)
        state_values = [state.getX(), state.getY(), state.getYaw()]
        states_list.append(state_values)
    solution_info["states"] = states_list

    return solution_info


def getAllPlannerSolutionsInfo(planner, ss):
    """Extract detailed information from all solutions stored in the planner."""
    print("🔍 Extracting detailed information from all planner solutions...")

    try:
        all_solutions = planner.getAllSolutions()
        all_solution_infos = []

        for i, solution in enumerate(all_solutions):
            try:
                # Create PathControl from the solution path
                path_control = oc.PathControl(solution.path_)

                # Extract path information
                info = getPathInfo(path_control, ss)
                info["cost"] = solution.cost_.value()
                info["solution_index"] = i

                all_solution_infos.append(info)

            except Exception as e:
                print(f"❌ Error processing solution {i+1}: {e}")
                # Create a minimal info structure
                info = {
                    "state_count": 0,
                    "control_count": 0,
                    "states": [],
                    "controls": [],
                    "cost": solution.cost_.value(),
                    "solution_index": i,
                    "error": str(e),
                }
                all_solution_infos.append(info)

        # Sort by cost (best first)
        all_solution_infos.sort(key=lambda x: x["cost"])

        print(
            f"✅ Successfully extracted information from {len(all_solution_infos)} solutions"
        )
        return all_solution_infos

    except Exception as e:
        print(f"❌ Error getting solutions from planner: {e}")
        return []


def printAllPlannerSolutions(planner, title="All Planner Solutions"):
    """Print a summary of all solutions tracked by the planner."""
    print(f"\n📊 {title}")
    print("=" * 60)
    try:
        all_solutions = planner.getAllSolutions()
        print(f"Number of solutions found: {len(all_solutions)}")
        for i, solution in enumerate(all_solutions):
            print(f"  Solution {i+1}:")
            print(f"    Cost: {solution.cost_.value():.3f}")
            try:
                path_control = oc.PathControl(solution.path_)
                if path_control:
                    print(f"    States: {path_control.getStateCount()}")
                    print(f"    Controls: {path_control.getControlCount()}")
                else:
                    print(f"    Path: Unable to cast to PathControl")
            except Exception as e:
                print(f"    Path: Error accessing path ({e})")
    except Exception as e:
        print(f"❌ Error getting solutions from planner: {e}")
    print("=" * 60)


def getSolutionsInfo(ss):
    """
    Enhanced function to get solution information using the new planner solution tracking feature.
    Falls back to the original method if the new feature is not available.
    """
    planner = ss.getPlanner()

    # Try to use the new solution tracking feature first
    try:
        if hasattr(planner, "getAllSolutions"):
            print("🔄 Using enhanced solution tracking from planner...")
            all_solution_infos = getAllPlannerSolutionsInfo(planner, ss)
            if len(all_solution_infos) > 0:
                print(
                    f"✅ Successfully extracted {len(all_solution_infos)} solutions using enhanced tracking."
                )
                return all_solution_infos
            else:
                print(
                    "⚠️ Enhanced tracking returned no solutions, falling back to original method..."
                )
        else:
            print(
                "⚠️ Planner doesn't support solution tracking, using original method..."
            )
    except Exception as e:
        print(
            f"⚠️ Error with enhanced solution tracking: {e}, falling back to original method..."
        )

    # Fallback to original method
    print("🔄 Using original solution extraction method...")
    solutions = ss.getProblemDefinition().getSolutions()
    allSolutionInfos = []

    # First try to get solutions from problem definition
    for solution in solutions:
        info = getPathInfo(solution.path_, ss)
        info["cost"] = solution.cost_.value()
        allSolutionInfos.append(info)

    # If no solutions found in problem definition, try to get the solution path directly
    if len(allSolutionInfos) == 0:
        try:
            solution_path = ss.getSolutionPath()
            if solution_path and solution_path.getStateCount() > 0:
                info = getPathInfo(solution_path, ss)
                # Try to get cost from optimization objective
                try:
                    opt = ss.getProblemDefinition().getOptimizationObjective()
                    if opt:
                        info["cost"] = opt.cost(solution_path).value()
                    else:
                        info["cost"] = (
                            0.0  # Default cost for approximate solutions
                        )
                except:
                    info["cost"] = (
                        0.0  # Default cost for approximate solutions
                    )
                allSolutionInfos.append(info)
        except Exception as e:
            print(f"❌ Error getting solution path directly: {e}")

    # Sort by cost (best first)
    allSolutionInfos.sort(key=lambda x: x["cost"])

    print(
        f"✅ Successfully extracted and sorted {len(allSolutionInfos)} solutions by cost."
    )
    return allSolutionInfos


def state2list(state, state_type: str) -> list:

    if state_type.upper() == "SE2":
        # SE2 state: x, y, theta
        return [state.getX(), state.getY(), state.getYaw()]

    elif state_type.upper() == "SE3":
        # SE3 state: x, y, z, qw, qx, qy, qz (position + quaternion)
        return [
            state.getX(),
            state.getY(),
            state.getZ(),
            state.getRotation().w,
            state.getRotation().x,
            state.getRotation().y,
            state.getRotation().z,
        ]

    else:
        print(
            f"Warning: Unknown state type '{state_type}'. Returning empty list."
        )
        return []


def isSE2Equal(state1, state2, tolerance=1e-6):
    diff_x = abs(state1[0] - state2[0])
    diff_y = abs(state1[1] - state2[1])
    diff_yaw = abs(state1[2] - state2[2])

    return diff_x < tolerance and diff_y < tolerance and diff_yaw < tolerance


def arrayDistance(array1, array2, system: str):
    # Convert SE2Pose objects to arrays if needed
    if hasattr(array1, "flat"):  # SE2Pose object
        array1 = array1.flat
    if hasattr(array2, "flat"):  # SE2Pose object
        array2 = array2.flat

    if system == "SE2":
        # Convert array1 and array2 to SE2State
        space = ob.SE2StateSpace()
        state1 = space.allocState()
        state2 = space.allocState()
        state1.setX(array1[0])
        state1.setY(array1[1])
        state1.setYaw(array1[2])
        state2.setX(array2[0])
        state2.setY(array2[1])
        state2.setYaw(array2[2])
        # Calculate using the functions in ompl SE2StateSpace
        return space.distance(state1, state2)
    elif system == "SE2Position":
        return np.sqrt(
            (array1[0] - array2[0]) ** 2 + (array1[1] - array2[1]) ** 2
        )
    else:
        raise ValueError(f"Invalid system: {system}")


def getChildrenStates(ss, targetState, tolerance=1e-6):
    print(
        f"\n🔍 DEBUG: getChildrenStates called with targetState: {targetState}"
    )

    planner_data = ob.PlannerData(ss.getSpaceInformation())
    ss.getPlanner().getPlannerData(planner_data)

    num_vertices = planner_data.numVertices()
    # print(f"📊 Total vertices in planner tree: {num_vertices}")

    targetVertexIdx = None
    # print(f"🔍 Searching for target state in planner tree...")

    # Search for the target state
    for i in range(num_vertices):
        state = planner_data.getVertex(i).getState()
        state_list = state2list(state, "SE2")

        if isSE2Equal(state_list, targetState, tolerance):
            targetVertexIdx = i
            print(f"✅ Found target state at vertex index: {i}")
            # print(f"   Target: {targetState}")
            # print(f"   Found:  {state_list}")
            break

    if targetVertexIdx is None:
        print(f"❌ State {targetState} not found in planner tree")
        # print(f"🔍 Checking first few vertices for debugging:")
        for i in range(min(5, num_vertices)):
            state = planner_data.getVertex(i).getState()
            state_list = state2list(state, "SE2")
            # print(f"   Vertex {i}: {state_list}")

        # Also check if any vertex is close to the target
        # print(f"🔍 Checking for close matches (tolerance: {tolerance}):")
        min_distance = float("inf")
        closest_vertex = None
        for i in range(min(10, num_vertices)):
            state = planner_data.getVertex(i).getState()
            state_list = state2list(state, "SE2")
            distance = arrayDistance(targetState, state_list, "SE2")
            if distance < min_distance:
                min_distance = distance
                closest_vertex = (i, state_list)
            # print(f"   Vertex {i}: {state_list} (distance: {distance:.6f})")

        if closest_vertex:
            print(
                f"🔍 Closest vertex: {closest_vertex[1]} (distance: {min_distance:.6f})"
            )

        return []

    # print(f"🔍 Getting edges for vertex {targetVertexIdx}...")
    childVertexIndices = ou.vectorUint()
    planner_data.getEdges(targetVertexIdx, childVertexIndices)

    print(f"📊 Found {len(childVertexIndices)} child vertices")

    children_states = []
    for childVertexIdx in childVertexIndices:
        childState = planner_data.getVertex(childVertexIdx).getState()
        child_state_list = state2list(childState, "SE2")
        children_states.append(child_state_list)
        # print(f"   Child {childVertexIdx}: {child_state_list}")

    print(f"✅ Returning {len(children_states)} children states")
    return children_states


def sampleRandomState(state, numStates=1000, posSTD=0.003, rotSTD=0.05):
    sampledStates = []
    # Convert state to list if it's not already
    if hasattr(state, "getX"):  # It's an OMPL state object
        stateList = state2list(state, "SE2")
    else:  # It's already a list
        stateList = state
    for _ in range(numStates):
        noisyX = stateList[0] + np.random.normal(0, posSTD)
        noisyY = stateList[1] + np.random.normal(0, posSTD)
        noisyYaw = stateList[2] + np.random.normal(0, rotSTD)
        while noisyYaw > np.pi:
            noisyYaw -= 2 * np.pi
        while noisyYaw < -np.pi:
            noisyYaw += 2 * np.pi
        sampledStates.append([noisyX, noisyY, noisyYaw])
    return sampledStates


def executeWaypoints(client, pos_waypoints, resultContainer):
    result = client.execute("execute_waypoints", pos_waypoints)
    resultContainer["result"] = result
    resultContainer["completed"] = True


def createExecuteThread(client, pos_waypoints):
    resultContainer = {"result": None, "completed": False}
    thread = threading.Thread(
        target=executeWaypoints,
        args=(client, pos_waypoints, resultContainer),
    )
    thread.daemon = True
    thread.resultContainer = resultContainer
    return thread


def runResolver(ss, replanningTime, resultContainer):
    ss.getPlanner().replan(replanningTime)
    result = getSolutionsInfo(ss)
    resultContainer["result"] = result
    resultContainer["completed"] = True


def createResolverThread(ss, replanningTime):
    resultContainer = {"result": None, "completed": False}
    thread = threading.Thread(
        target=runResolver, args=(ss, replanningTime, resultContainer)
    )
    thread.daemon = True
    thread.resultContainer = resultContainer
    return thread


def runOptimizer(
    nextState,
    childrenStatesArray,
    initialGuessControl,
    optModel,  # Changed from model to optModel
    numStates=1000,
    maxDistance=0.025,
    posSTD=0.003,
    rotSTD=0.05,
):
    print(f"🔍 DEBUG: runOptimizer function called")
    print(f"🔍 DEBUG: runOptimizer - nextState: {nextState}")
    print(
        f"🔍 DEBUG: runOptimizer - childrenStatesArray type: {type(childrenStatesArray)}"
    )
    print(
        f"🔍 DEBUG: runOptimizer - childrenStatesArray length: {len(childrenStatesArray)}"
    )
    print(
        f"🔍 DEBUG: runOptimizer - initialGuessControl: {initialGuessControl}"
    )
    print(f"🔍 DEBUG: runOptimizer - optModel type: {type(optModel)}")
    print(f"🔍 DEBUG: runOptimizer - numStates: {numStates}")

    try:
        print(f"🔍 DEBUG: runOptimizer started")
        print(f"  - nextState: {nextState}")
        print(
            f"  - childrenStatesArray shape: {np.array(childrenStatesArray).shape}"
        )
        print(f"  - initialGuessControl: {initialGuessControl}")
        print(f"  - numStates: {numStates}")
        print(f"  - maxDistance: {maxDistance}")

        # ✅ 1. Sample states
        sampledStates_raw = np.array(
            sampleRandomState(nextState, numStates, posSTD, rotSTD)
        )
        sampledStates = [
            SE2Pose(state[:2], state[2]) for state in sampledStates_raw
        ]
        print(f"✅ Created {len(sampledStates)} sampled SE2Pose objects")

        # ✅ 2. Children states
        childrenStates_raw = np.array(childrenStatesArray)
        optimizer_childrenStates = [
            SE2Pose(state[:2], state[2]) for state in childrenStates_raw
        ]
        numChildren = len(optimizer_childrenStates)
        print(
            f"✅ Created {len(optimizer_childrenStates)} children SE2Pose objects"
        )

        # ✅ 3. Compute relative poses using broadcasting (no nested loops)
        sampled_inverts = [s.invert for s in sampledStates]

        # Broadcast: shape will be (numStates, numChildren)
        relativePoses_matrix = [
            [inv @ c for c in optimizer_childrenStates]
            for inv in sampled_inverts
        ]

        # Flatten into (numStates*numChildren, 3)
        relativePosesFlat = np.array(
            [
                [pose.position[0], pose.position[1], pose.euler[2]]
                for row in relativePoses_matrix
                for pose in row
            ]
        )
        print(f"✅ Relative poses shape: {relativePosesFlat.shape}")

        # ✅ 4. Broadcast initial guess controls
        initialGuessControlsFlat = np.tile(
            initialGuessControl, (numStates * numChildren, 1)
        )
        print(
            f"✅ Initial guess controls shape: {initialGuessControlsFlat.shape}"
        )

        # ✅ 5. Predict optimized controls
        print(f"🔍 Calling optModel.predict...")

        # Define clamping bounds for the 3D controls [rotation, side, distance]
        # Adjust these bounds based on your physics model requirements
        x_min = np.array(
            [0.0, -0.5, 0.0]
        )  # [min_rotation, min_side, min_distance]
        x_max = np.array(
            [2 * np.pi, 0.5, 0.3]
        )  # [max_rotation, max_side, max_distance]

        optimizedControlsFlat, loss = optModel.predict(
            relativePosesFlat,
            initialGuessControlsFlat,
            x_min=x_min,
            x_max=x_max,
        )
        print(f"✅ Optimized controls shape: {optimizedControlsFlat.shape}")
        print(f"✅ Loss: {loss}")

        control_dim = (
            optimizedControlsFlat.shape[1]
            if len(optimizedControlsFlat.shape) > 1
            else 1
        )
        optimizedControls = optimizedControlsFlat.reshape(
            numStates, numChildren, control_dim
        )
        print(
            f"✅ Reshaped optimized controls shape: {optimizedControls.shape}"
        )

        # ✅ 6. Vectorized dictionary creation (no nested loops)
        flat_controls = optimizedControls.reshape(
            numStates * numChildren, control_dim
        )
        tiled_samples = np.repeat(sampledStates, numChildren)
        child_indices = np.tile(np.arange(numChildren), numStates)

        controls = {i: [] for i in range(numChildren)}  # Use indices as keys
        for idx in range(numChildren):
            mask = child_indices == idx
            controls[idx] = [
                [tiled_samples[i], flat_controls[i]] for i in np.where(mask)[0]
            ]

        print(f"✅ Created controls dictionary for {len(controls)} children")
        print(f"🔍 DEBUG: runOptimizer completed")
        print(f"✅ Controls: {len(controls.keys())}")
        print(f"✅ Sampled states: {len(sampledStates)}")

        # ✅ 7. Modify the first variable of each optimized control to the closest value in [0, np.pi/2, np.pi, 3*np.pi/2]
        target_rotations = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])

        # Convert the dictionary structure to a flat array for vectorized processing
        # Since controls is {child_idx: [[sampled_state, control], ...], ...}
        # We can extract all controls at once
        all_controls = np.array(
            [
                control_pair[1]
                for child_controls in controls.values()
                for control_pair in child_controls
            ]
        )

        if len(all_controls) > 0:
            # Vectorized computation of closest target rotations
            current_rotations = all_controls[:, 0]
            diffs = np.abs(current_rotations[:, np.newaxis] - target_rotations)
            closest_indices = np.argmin(diffs, axis=1)

            # Update all rotations at once
            all_controls[:, 0] = target_rotations[closest_indices]

            # Put the modified controls back - create a new dictionary structure
            control_idx = 0
            new_controls = {}
            for child_idx in controls:
                new_controls[child_idx] = []
                for control_pair in controls[child_idx]:
                    new_controls[child_idx].append(
                        [control_pair[0], all_controls[control_idx]]
                    )
                    control_idx += 1
            controls = new_controls

        return [
            controls,
            sampledStates,
            optimizer_childrenStates,
        ]  # Also return childrenStates for reference

    except Exception as e:
        print(f"❌ ERROR in runOptimizer: {e}")
        import traceback

        traceback.print_exc()
        return None


def createOptimizerThread(
    nextState,
    childrenStates,
    initialGuessControl,
    optModel,
    numStates=1000,
    posSTD=0.003,
    rotSTD=0.05,
):
    resultContainer = {"result": None, "completed": False}

    def optimizer_wrapper():
        try:
            print(f"🔍 DEBUG: optimizer_wrapper started")
            print(f"🔍 DEBUG: nextState: {nextState}")
            print(
                f"🔍 DEBUG: childrenStates parameter type: {type(childrenStates)}"
            )
            print(
                f"🔍 DEBUG: childrenStates parameter length: {len(childrenStates)}"
            )
            print(f"🔍 DEBUG: initialGuessControl: {initialGuessControl}")
            print(f"🔍 DEBUG: optModel type: {type(optModel)}")

            result = runOptimizer(
                nextState,
                childrenStates,
                initialGuessControl,
                optModel,
                numStates,
                posSTD,
                rotSTD,
            )
            print(f"🔍 DEBUG: optimizer_wrapper got result: {type(result)}")
            if result is not None:
                controls, sampledStates, optimizer_childrenStates = result
                print(f"🔍 DEBUG: controls type: {type(controls)}")
                print(
                    f"🔍 DEBUG: controls keys: {len(controls.keys()) if controls else 0}"
                )
                if controls:
                    total_controls = sum(
                        len(control_list) for control_list in controls.values()
                    )
                    print(
                        f"🔍 DEBUG: total controls in dictionary: {total_controls}"
                    )
            else:
                print(f"❌ ERROR: runOptimizer returned None")
            resultContainer["result"] = result
            resultContainer["completed"] = True
            print(f"🔍 DEBUG: optimizer_wrapper completed successfully")
        except Exception as e:
            print(f"❌ ERROR in optimizer_wrapper: {e}")
            import traceback

            traceback.print_exc()
            resultContainer["result"] = None
            resultContainer["completed"] = True

    thread = threading.Thread(target=optimizer_wrapper)
    thread.daemon = True
    thread.resultContainer = resultContainer
    return thread


def main(
    system: str,
    objectName: str,
    startState: np.ndarray,
    goalState: np.ndarray,
    planningTime: float = 20.0,
    replanningTime: float = 2.0,
    plannerName: str = "fusion",
    visualize: bool = False,
    learningRate: float = 0.001,
    numEpochs: int = 1000,
    sampling_num_states: int = 1000,
    sampling_position_std: float = 0.003,
    sampling_rotation_std: float = 0.05,
):
    # Set up the connection to the simulation
    client = SimClient()
    ik = IK("ur10_rod")
    tool_offset = Pose([0, 0, -0.02])
    _, dt, _ = client.execute("get_sim_info")
    client.execute(
        "set_obj_init_poses",
        [0, Pose((startState[0], startState[1], 0.73), (0, 0, startState[2]))],
    )

    # Get the object shape
    objectShape = pickObjectShape(objectName)

    # Pick the propagator and load the model for the optimizer
    propagator = pickPropagator(system, objectShape)

    try:
        # Load the optimization model
        torch_model = load_model("residual", 3, 3)  # 3D input, 3D output
        # Get the actual PyTorch model from the TorchModel wrapper
        actual_model = torch_model.model

        if actual_model is None:
            # If model hasn't been initialized yet, we need to create it
            actual_model = torch_model.model_class()
            actual_model = actual_model.to(torch_model.device)

        # Create the optimization model
        optModel = load_opt_model_2(
            actual_model, lr=learningRate, epochs=numEpochs
        )
    except Exception as e:
        print(f"❌ ERROR loading optimization model: {e}")
        import traceback

        traceback.print_exc()
        print("❌ Cannot continue without optimization model")
        return

    # Plan the initial solution
    solutionsInfo, ss = plan(
        system=system,
        objectShape=objectShape,
        startState=startState,
        goalState=goalState,
        propagator=propagator,
        planningTime=planningTime,
        replanningTime=replanningTime,
        plannerName=plannerName,
        visualize=visualize,
    )

    if solutionsInfo is None or len(solutionsInfo) == 0:
        print("❌ No solutions found after initial planning")
        return

    print(f"Initial solution found!")
    print(f"Found {len(solutionsInfo)} solutions")

    # Demonstrate the new solution tracking feature
    planner = ss.getPlanner()
    if hasattr(planner, "getAllSolutions"):
        printAllPlannerSolutions(
            planner, "Solutions Found During Initial Planning"
        )

    # Print initial planned states
    print(f"\n📋 Initial planned states:")
    for i, state in enumerate(solutionsInfo[0]["states"]):
        print(
            f"  State {i}: x={state[0]:.5f}, y={state[1]:.5f}, yaw={state[2]:.5f}"
        )

    print("\n🎮 Initial planned controls:")
    for i, control in enumerate(solutionsInfo[0]["controls"]):
        print(f"  Control {i}: {control}")

    doPhase1 = True
    if doPhase1:
        user_input = input("Execute initial plan without optimization? (y/n)")
        if user_input.lower() == "y":
            print("Executing initial plan without optimization...")
            doPhase1 = True
        else:
            doPhase1 = False

    ####################################################
    ########## PHASE 1: Execute initial plan without optimization ###########
    ####################################################
    if doPhase1:
        print("\n" + "=" * 80)
        print("🚀 PHASE 1: Executing initial plan WITHOUT optimization")
        print("=" * 80)

        # Execute all controls from the initial plan
        for i, control in enumerate(solutionsInfo[0]["controls"]):
            print(f"\n🎮 Executing control {i}: {control}")

            # Get current object state
            _, _, obj_rob_pos, obj_rob_quat, _ = client.execute(
                "get_obj_info", 0
            )
            obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
            currentState = np.array(
                [obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]]
            )
            print(
                f"📊 Current state: x={currentState[0]:.3f}, y={currentState[1]:.3f}, yaw={currentState[2]:.3f}"
            )

            # Compare with planned state
            if i < len(solutionsInfo[0]["states"]):
                planned_state = solutionsInfo[0]["states"][i]
                print(
                    f"📋 Planned state {i}: x={planned_state[0]:.3f}, y={planned_state[1]:.3f}, yaw={planned_state[2]:.3f}"
                )
                state_diff = np.array(planned_state) - currentState
                print(
                    f"📏 State difference: dx={state_diff[0]:.3f}, dy={state_diff[1]:.3f}, dyaw={state_diff[2]:.3f}"
                )

            # Execute the control
            print(f"🔄 Converting control to trajectory...")
            print(f"🔧 EXECUTING CONTROL (Phase 1): {control}")
            client.execute("rotate_scene", params=[None, -obj_pose.euler[2]])
            times, ws_path = generate_path_form_params(
                obj_pose, objectShape, control, tool_offset=tool_offset
            )
            traj = ik.ws_path_to_traj(Pose(), times, ws_path)
            waypoints = traj.to_step_waypoints(dt)
            pos_waypoints = np.stack([waypoints[0]], axis=1)
            print(
                f"✅ Trajectory generated with {len(pos_waypoints)} waypoints"
            )

            # Debug: Print trajectory details
            print(f"🔍 Trajectory debug info:")
            print(
                f"  - Current object pose: x={obj_pose.position[0]:.3f}, y={obj_pose.position[1]:.3f}, yaw={obj_pose.euler[2]:.3f}"
            )
            print(f"  - Control: {control}")
            print(f"  - Trajectory waypoints: {len(pos_waypoints)} points")
            if len(pos_waypoints) > 0:
                print(f"  - First waypoint: {pos_waypoints[0]}")
                if len(pos_waypoints) > 1:
                    print(f"  - Last waypoint: {pos_waypoints[-1]}")

            # Execute waypoints
            print("🚀 Starting execution...")
            executeThread = createExecuteThread(client, pos_waypoints)
            executeThread.start()
            executeThread.join()
            print(f"✅ Control {i} executed")

            # Get state after execution
            _, _, obj_rob_pos, obj_rob_quat, _ = client.execute(
                "get_obj_info", 0
            )
            obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
            newState = np.array(
                [obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]]
            )
            print(
                f"📍 State after execution: x={newState[0]:.3f}, y={newState[1]:.3f}, yaw={newState[2]:.3f}"
            )

            # Compare with expected next state
            if i + 1 < len(solutionsInfo[0]["states"]):
                expected_next_state = solutionsInfo[0]["states"][i + 1]
                print(
                    f"🎯 Expected next state: x={expected_next_state[0]:.3f}, y={expected_next_state[1]:.3f}, yaw={expected_next_state[2]:.3f}"
                )
                execution_diff = np.array(expected_next_state) - newState
                print(
                    f"📏 Execution deviation: dx={execution_diff[0]:.3f}, dy={execution_diff[1]:.3f}, dyaw={execution_diff[2]:.3f}"
                )

            # Input prompt for Phase 1
            # user_input = input(
            #     f"Press Enter to continue to control {i+1} (or 'q' to quit Phase 1): "
            # )
            # if user_input.lower() == "q":
            #     print("👋 Stopping Phase 1 by user request")
            #     break

        # Get final state from initial plan execution
        _, _, obj_rob_pos, obj_rob_quat, _ = client.execute("get_obj_info", 0)
        obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
        initial_final_state = np.array(
            [obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]]
        )

        # Calculate distance to goal for initial plan
        initial_distance_to_goal = arrayDistance(
            initial_final_state,
            goalState,
            system="SE2",
        )

        print(f"\n📊 PHASE 1 RESULTS:")
        print(
            f"  Final state: x={initial_final_state[0]:.3f}, y={initial_final_state[1]:.3f}, yaw={initial_final_state[2]:.3f}"
        )
        print(
            f"  Goal state: x={goalState[0]:.3f}, y={goalState[1]:.3f}, yaw={goalState[2]:.3f}"
        )
        print(f"  Distance to goal: {initial_distance_to_goal:.3f}")

    # input("Press Enter to continue to Phase 2...")
    ####################################################
    ########## PHASE 2: Reset scene and run with optimization ###########
    ####################################################
    print("\n" + "=" * 80)
    print("🔄 PHASE 2: Resetting scene and running WITH optimization")
    print("=" * 80)

    # Reset the scene
    print("🔄 Resetting scene...")
    client.execute(
        "set_obj_init_poses",
        [0, Pose((startState[0], startState[1], 0.73), (0, 0, startState[2]))],
    )
    print("✅ Scene reset complete")

    # Start the execution loop with optimization
    index = 0
    doOptimization = True
    nextControl = solutionsInfo[0]["controls"][0]
    print("\n🚀 Starting the execution loop WITH optimization")

    while len(solutionsInfo[0]["controls"]) > 1:

        if len(solutionsInfo[0]["states"]) <= 2:
            doOptimization = False

        print(f"\n{'='*80}")
        print(f"🚀 EXECUTING ITERATION {index}")
        print(f"{'='*80}")

        ####################################################
        ############# Execute the nextControl ##############
        ####################################################
        _, _, obj_rob_pos, obj_rob_quat, _ = client.execute("get_obj_info", 0)
        obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
        currentState = np.array(
            [obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]]
        )
        print(
            f"✅ Current object state: x={currentState[0]:.3f}, y={currentState[1]:.3f}, yaw={currentState[2]:.3f}"
        )

        # 1.1 Compare planned vs actual state
        # print("\n🔍 STEP 1.1: Comparing planned vs actual state...")
        planned_state = (
            solutionsInfo[0]["states"][0]
            if solutionsInfo[0].get("states")
            else None
        )
        if planned_state:
            print(
                f"📋 Planned state {index}: x={planned_state[0]:.3f}, y={planned_state[1]:.3f}, yaw={planned_state[2]:.3f}"
            )
            print(
                f"🎯 Actual object state: x={currentState[0]:.3f}, y={currentState[1]:.3f}, yaw={currentState[2]:.3f}"
            )
            state_diff = currentState - planned_state
            print(
                f"📏 State difference: dx={state_diff[0]:.3f}, dy={state_diff[1]:.3f}, dyaw={state_diff[2]:.3f}"
            )

            # # Debug: Check if this is the expected behavior
            # if index > 0 and index < len(solutionsInfo[0]["states"]):
            #     previous_planned_state = solutionsInfo[0]["states"][index - 1]
            #     planned_transition = np.array(planned_state) - np.array(
            #         previous_planned_state
            #     )
            #     actual_transition = currentState - np.array(
            #         previous_planned_state
            #     )
            #     print(f"🔍 Transition comparison:")
            #     print(
            #         f"  - Planned transition: dx={planned_transition[0]:.3f}, dy={planned_transition[1]:.3f}, dyaw={planned_transition[2]:.3f}"
            #     )
            #     print(
            #         f"  - Actual transition: dx={actual_transition[0]:.3f}, dy={actual_transition[1]:.3f}, dyaw={actual_transition[2]:.3f}"
            #     )
            #     transition_diff = planned_transition - actual_transition
            #     print(
            #         f"  - Transition difference: dx={transition_diff[0]:.3f}, dy={transition_diff[1]:.3f}, dyaw={transition_diff[2]:.3f}"
            #     )
            # elif index == 0:
            #     print(
            #         f"🔍 First iteration - no previous state to compare with"
            #     )
            # else:
            #     print(f"🔍 No more planned states available for comparison")
        else:
            print("❌ No planned state available.")
            break

        ####################################################
        ############# STEP 2: Execute current control ##############
        ####################################################
        print(f"\n🎮 STEP 2: Executing current control...")
        print(f"🔧 ORIGINAL nextControl (before optimization): {nextControl}")
        original_control = (
            nextControl.copy() if hasattr(nextControl, "copy") else nextControl
        )

        # Debug: Test the physics model prediction
        # print(f"🔍 Testing physics model prediction...")
        # try:
        #     # Use the same propagator that was used in planning
        #     test_state = np.array(
        #         [currentState[0], currentState[1], currentState[2]]
        #     )
        #     predicted_state = propagator(
        #         test_state, nextControl, 1.0
        #     )  # 1 second duration
        #     print(f"  - Current state: {test_state}")
        #     print(f"  - Control: {nextControl}")
        #     print(f"  - Physics model prediction: {predicted_state}")
        #     print(
        #         f"  - Predicted change: dx={predicted_state[0]-test_state[0]:.3f}, dy={predicted_state[1]-test_state[1]:.3f}, dyaw={predicted_state[2]-test_state[2]:.3f}"
        #     )
        # except Exception as e:
        #     print(f"  - Physics model test failed: {e}")

        print("🔄 Converting control to trajectory...")
        print(f"🔧 EXECUTING CONTROL (Phase 2): {nextControl}")
        client.execute("rotate_scene", params=[None, -obj_pose.euler[2]])
        times, ws_path = generate_path_form_params(
            obj_pose, objectShape, nextControl, tool_offset=tool_offset
        )
        traj = ik.ws_path_to_traj(Pose(), times, ws_path)
        waypoints = traj.to_step_waypoints(dt)
        pos_waypoints = np.stack([waypoints[0]], axis=1)
        print(f"✅ Trajectory generated with {len(pos_waypoints)} waypoints")

        # Debug: Print trajectory details
        # print(f"🔍 Trajectory debug info:")
        # print(
        #     f"  - Current object pose: x={obj_pose.position[0]:.3f}, y={obj_pose.position[1]:.3f}, yaw={obj_pose.euler[2]:.3f}"
        # )
        # print(f"  - Control: {nextControl}")
        # print(f"  - Trajectory waypoints: {len(pos_waypoints)} points")
        # if len(pos_waypoints) > 0:
        #     print(f"  - First waypoint: {pos_waypoints[0]}")
        #     if len(pos_waypoints) > 1:
        #         print(f"  - Last waypoint: {pos_waypoints[-1]}")

        # Execute waypoints in parallel thread
        print("🚀 Starting execution thread...")
        executeThread = createExecuteThread(client, pos_waypoints)
        executeThread.start()
        print("✅ Execution thread started")

        ####################################################
        ########## STEP 3: Run optimizer in parallel (COMMENTED OUT) ###########
        ####################################################
        print("\n🧠 STEP 3: Optimizer temporarily disabled")

        # # COMMENTED OUT: Optimizer logic (can be re-enabled later)
        # print("Starting optimizer thread...")
        # print(
        #     f"🎯 Using current actual state for optimization: {currentState}"
        # )

        # # Use current state as the starting point for optimization
        # nextState = solutionsInfo[0]["states"][1]
        # print(f"🔍 Finding children states for current state...")
        # print(f"🎯 Target state for children search: {nextState}")
        # childrenStates = getChildrenStates(ss, nextState)
        # print(f"✅ Found {len(childrenStates)} children states")

        # # Debug: Check if this state exists in the current tree
        # if len(childrenStates) == 0:
        #     print(f"⚠️  WARNING: No children found for state {nextState}")
        #     print(
        #         f"🔍 This might mean the state doesn't exist in the current tree"
        #     )
        #     print(
        #         f"🔍 This could happen if replanning modified the tree structure"
        #     )

        # print(
        #     f"🔧 Using nextControl as initial guess: {solutionsInfo[0]['controls'][0]}"
        # )
        # optimizerThread = createOptimizerThread(
        #     nextState,
        #     childrenStates,
        #     solutionsInfo[0]["controls"][
        #         0
        #     ],  # Use the first control from the replanned path
        #     optModel,
        #     sampling_num_states,
        #     sampling_position_std,
        #     sampling_rotation_std,
        # )
        # optimizerThread.start()
        # print("✅ Optimizer thread started")

        optimizerThread = None  # No optimizer for now

        ####################################################
        ########### STEP 4: Run replanning in parallel ###########
        ####################################################
        print("\n🔄 STEP 4: Starting replanning thread...")

        # Print states and controls BEFORE replanning
        print("=" * 60)
        print("📊 BEFORE REPLANNING:")
        print(f"  States count: {len(solutionsInfo[0]['states'])}")
        print(f"  Controls count: {len(solutionsInfo[0]['controls'])}")
        print(f"  States:")
        for i, state in enumerate(solutionsInfo[0]["states"]):
            print(
                f"    [{i}]: [{state[0]:.5f}, {state[1]:.5f}, {state[2]:.5f}]"
            )
        print(f"  Controls:")
        for i, control in enumerate(solutionsInfo[0]["controls"]):
            print(
                f"    [{i}]: [{control[0]:.5f}, {control[1]:.5f}, {control[2]:.5f}]"
            )
        print("=" * 60)

        replanThread = createResolverThread(ss, replanningTime)
        replanThread.start()
        print("✅ Replanner thread started")

        ####################################################
        ########## STEP 5: Wait for all threads ###########
        ####################################################
        print("\n⏳ STEP 5: Waiting for all threads to complete...")
        executeThread.join()
        replanThread.join()
        if optimizerThread:  # Only join if optimizer thread exists
            optimizerThread.join()
        # input("✅ All threads completed, press Enter to continue...")

        ####################################################
        ########## STEP 6: Collect results ###########
        ####################################################
        print("\n📊 STEP 6: Collecting thread results...")

        # Get execution result
        executeResult = executeThread.resultContainer["result"]
        executeCompleted = executeThread.resultContainer["completed"]
        # input(
        #     f"✅ Execution completed: {executeCompleted}, Result: {executeResult}"
        # )

        # Get the final object pose
        _, _, obj_rob_pos, obj_rob_quat, _ = client.execute("get_obj_info", 0)
        obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
        currentState = np.array(
            [obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]]
        )

        # Get replan result and update solutionsInfo
        newSolutionsInfo = replanThread.resultContainer["result"]
        replanCompleted = replanThread.resultContainer["completed"]

        # Update solutionsInfo with the new result from replanning
        # print(
        #     f"🔍 DEBUG: replanThread.resultContainer['result'] type: {type(newSolutionsInfo)}"
        # )
        # print(
        #     f"🔍 DEBUG: replanThread.resultContainer['result'] value: {newSolutionsInfo}"
        # )
        print(
            f"🔍 DEBUG: replanThread.resultContainer['completed']: {replanCompleted}"
        )

        if newSolutionsInfo and len(newSolutionsInfo) > 0:
            solutionsInfo = newSolutionsInfo
            print(
                f"🔄 Updated solutionsInfo from replanning, now has {len(solutionsInfo[0]['controls'])} controls"
            )
            print(f"🔄 NEW SOLUTION COST: {solutionsInfo[0]['cost']:.3f}")

            # Print states and controls AFTER replanning
            print("=" * 60)
            print("📊 AFTER REPLANNING:")
            print(f"  States count: {len(solutionsInfo[0]['states'])}")
            print(f"  Controls count: {len(solutionsInfo[0]['controls'])}")
            print(f"  States:")
            for i, state in enumerate(solutionsInfo[0]["states"]):
                print(
                    f"    [{i}]: [{state[0]:.5f}, {state[1]:.5f}, {state[2]:.5f}]"
                )
            print(f"  Controls:")
            for i, control in enumerate(solutionsInfo[0]["controls"]):
                print(
                    f"    [{i}]: [{control[0]:.5f}, {control[1]:.5f}, {control[2]:.5f}]"
                )
            print("=" * 60)

            # Demonstrate solution tracking after replanning
            if hasattr(planner, "getAllSolutions"):
                printAllPlannerSolutions(
                    planner, "All Solutions After Replanning"
                )
        else:
            print(
                f"⚠️  Replanning didn't return new solutions, keeping current solutionsInfo"
            )
        # input(
        #     f"✅ Replan completed: {replanCompleted}, Found {len(solutionsInfo) if solutionsInfo else 0} solutions"
        # )

        # # COMMENTED OUT: Optimizer result processing (can be re-enabled later)
        # optimizerResult = optimizerThread.resultContainer["result"]
        # optimizerCompleted = optimizerThread.resultContainer["completed"]
        # ... (optimizer logic here) ...

        # Simple logic: use the first control from the replanned solution
        if len(solutionsInfo[0]["controls"]) > 0:
            nextControl = solutionsInfo[0]["controls"][0]
            print(
                f"🔄 Using first control from replanned solution: {nextControl}"
            )
            print(
                f"📊 Remaining controls in replanned solution: {len(solutionsInfo[0]['controls'])}"
            )
        else:
            print(f"❌ ERROR: No controls available in replanned solution")
            break

        print(f"✅ Next control: {nextControl}")
        print(
            f"🔍 DEBUG: Current solutionsInfo[0]['controls'][0]: {solutionsInfo[0]['controls'][0]}"
        )
        print(
            f"🔍 DEBUG: Current solutionsInfo[0]['states'][0]: {solutionsInfo[0]['states'][0]}"
        )
        input(
            f"Length of solutionsInfo[0]['controls']: {len(solutionsInfo[0]['controls'])}"
        )
        index += 1

        # Check if we should break out of the loop
        if len(solutionsInfo[0]["controls"]) <= 1:
            print(
                f"🛑 Only {len(solutionsInfo[0]['controls'])} control(s) remaining, breaking loop"
            )
            break

    # Execure the nextControl
    print(f"🚀 EXECUTING NEXT CONTROL: {nextControl}")
    client.execute("rotate_scene", params=[None, -obj_pose.euler[2]])
    times, ws_path = generate_path_form_params(
        obj_pose, objectShape, nextControl, tool_offset=tool_offset
    )
    traj = ik.ws_path_to_traj(Pose(), times, ws_path)
    waypoints = traj.to_step_waypoints(dt)
    pos_waypoints = np.stack([waypoints[0]], axis=1)
    print(f"✅ Trajectory generated with {len(pos_waypoints)} waypoints")
    executeThread = createExecuteThread(client, pos_waypoints)
    executeThread.start()
    print("✅ Execution thread started")
    executeThread.join()

    print(f"🧠 PLANNING: Current controls: {solutionsInfo[0]['controls']}")
    print(f"🧠 PLANNING: Current states: {solutionsInfo[0]['states']}")

    print(f"\n🏁 PHASE 2 completed!")

    # Get final object pose from optimized execution

    optimized_final_state = np.array(
        [obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]]
    )

    # Calculate distance to goal for optimized execution
    optimized_distance_to_goal = arrayDistance(
        optimized_final_state,
        goalState,
        system="SE2",
    )

    ####################################################
    ########## FINAL COMPARISON ###########
    ####################################################
    print("\n" + "=" * 80)
    print("📊 FINAL COMPARISON: Initial Plan vs Optimized Execution")
    print("=" * 80)

    # Calculate original path cost (from the initial planning)
    original_path_cost = (
        solutionsInfo[0]["cost"]
        if solutionsInfo and len(solutionsInfo) > 0
        else 0.0
    )

    # Calculate executed path cost (sum of all controls executed)
    # We need to track the controls that were actually executed
    # For now, we'll estimate based on the number of controls executed
    executed_controls_count = index  # Number of iterations completed

    # Calculate a more accurate cost based on the original solution cost and number of controls
    if (
        solutionsInfo
        and len(solutionsInfo) > 0
        and len(solutionsInfo[0]["controls"]) > 0
    ):
        original_controls_count = len(solutionsInfo[0]["controls"])
        cost_per_control = (
            original_path_cost / original_controls_count
            if original_controls_count > 0
            else 1.0
        )
        estimated_executed_cost = executed_controls_count * cost_per_control
    else:
        estimated_executed_cost = executed_controls_count * 1.0  # Fallback

    print(f"\n📋 PHASE 1 (Initial Plan - NO optimization):")
    print(
        f"  Final state: x={initial_final_state[0]:.3f}, y={initial_final_state[1]:.3f}, yaw={initial_final_state[2]:.3f}"
    )
    print(f"  Distance to goal: {initial_distance_to_goal:.3f}")
    print(f"  Original path cost: {original_path_cost:.3f}")

    print(f"\n📋 PHASE 2 (Optimized Execution - WITH optimization):")
    print(
        f"  Final state: x={optimized_final_state[0]:.3f}, y={optimized_final_state[1]:.3f}, yaw={optimized_final_state[2]:.3f}"
    )
    print(f"  Distance to goal: {optimized_distance_to_goal:.3f}")
    print(
        f"  Executed path cost: {estimated_executed_cost:.3f} (estimated from {executed_controls_count} controls)"
    )

    print(
        f"\n🎯 Goal state: x={goalState[0]:.3f}, y={goalState[1]:.3f}, yaw={goalState[2]:.3f}"
    )

    # Calculate improvement
    improvement = initial_distance_to_goal - optimized_distance_to_goal
    improvement_percentage = (
        (improvement / initial_distance_to_goal) * 100
        if initial_distance_to_goal > 0
        else 0
    )

    print(f"\n📊 RESULTS:")
    print(f"  Initial plan distance: {initial_distance_to_goal:.3f}")
    print(f"  Optimized plan distance: {optimized_distance_to_goal:.3f}")
    print(f"  Improvement: {improvement:.3f} ({improvement_percentage:.1f}%)")
    print(f"  Original path cost: {original_path_cost:.3f}")
    print(f"  Executed path cost: {estimated_executed_cost:.3f}")
    print(
        f"  Cost difference: {estimated_executed_cost - original_path_cost:.3f}"
    )

    if optimized_distance_to_goal < initial_distance_to_goal:
        print(f"✅ OPTIMIZATION SUCCESSFUL: Better performance achieved!")
    elif optimized_distance_to_goal > initial_distance_to_goal:
        print(f"❌ OPTIMIZATION FAILED: Worse performance than initial plan")
    else:
        print(f"➖ OPTIMIZATION NEUTRAL: Same performance as initial plan")


if __name__ == "__main__":
    # Parse arguments and load configuration
    config = parse_args_and_config()
    main(
        **config,
    )
