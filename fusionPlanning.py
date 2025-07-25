import sys
import yaml
import torch
import argparse
import numpy as np
import threading

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
    ss.getSpaceInformation().setMinMaxControlDuration(1, 1)

    # Set the control sampler
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

    # Set the optimization objective to path length
    ss.setOptimizationObjective(
        ob.PathLengthOptimizationObjective(ss.getSpaceInformation())
    )

    # Attempt to solve the problem
    solved = ss.solve(planningTime)

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


def getSolutionsInfo(ss):
    solutions = ss.getProblemDefinition().getSolutions()
    allSolutionInfos = []
    for solution in solutions:
        info = getPathInfo(solution.path_, ss)
        info["cost"] = solution.cost_.value()
        allSolutionInfos.append(info)

    # Sort by cost (best first)
    allSolutionInfos.sort(key=lambda x: x["cost"])

    print(
        f"Successfully extracted and sorted {len(allSolutionInfos)} solutions by cost."
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
    print(f"📊 Total vertices in planner tree: {num_vertices}")

    targetVertexIdx = None
    print(f"🔍 Searching for target state in planner tree...")

    # Search for the target state
    for i in range(num_vertices):
        state = planner_data.getVertex(i).getState()
        state_list = state2list(state, "SE2")

        if isSE2Equal(state_list, targetState, tolerance):
            targetVertexIdx = i
            print(f"✅ Found target state at vertex index: {i}")
            print(f"   Target: {targetState}")
            print(f"   Found:  {state_list}")
            break

    if targetVertexIdx is None:
        print(f"❌ State {targetState} not found in planner tree")
        print(f"🔍 Checking first few vertices for debugging:")
        for i in range(min(5, num_vertices)):
            state = planner_data.getVertex(i).getState()
            state_list = state2list(state, "SE2")
            print(f"   Vertex {i}: {state_list}")

        # Also check if any vertex is close to the target
        print(f"🔍 Checking for close matches (tolerance: {tolerance}):")
        min_distance = float("inf")
        closest_vertex = None
        for i in range(min(10, num_vertices)):
            state = planner_data.getVertex(i).getState()
            state_list = state2list(state, "SE2")
            distance = arrayDistance(targetState, state_list, "SE2")
            if distance < min_distance:
                min_distance = distance
                closest_vertex = (i, state_list)
            print(f"   Vertex {i}: {state_list} (distance: {distance:.6f})")

        if closest_vertex:
            print(
                f"🔍 Closest vertex: {closest_vertex[1]} (distance: {min_distance:.6f})"
            )

        return []

    print(f"🔍 Getting edges for vertex {targetVertexIdx}...")
    childVertexIndices = ou.vectorUint()
    planner_data.getEdges(targetVertexIdx, childVertexIndices)

    print(f"📊 Found {len(childVertexIndices)} child vertices")

    children_states = []
    for childVertexIdx in childVertexIndices:
        childState = planner_data.getVertex(childVertexIdx).getState()
        child_state_list = state2list(childState, "SE2")
        children_states.append(child_state_list)
        print(f"   Child {childVertexIdx}: {child_state_list}")

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
    childrenStates,
    initialGuessControl,
    propagator,
    optModel,
    numStates=1000,
    maxDistance=0.025,
):
    print(f"\n🧠 DEBUG: runOptimizer started")
    print(f"📊 Input parameters:")
    print(f"  - nextState: {nextState}")
    print(f"  - childrenStates count: {len(childrenStates)}")
    print(f"  - initialGuessControl: {initialGuessControl}")
    print(f"  - numStates: {numStates}")
    print(f"  - maxDistance: {maxDistance}")

    if len(childrenStates) == 0:
        print("❌ ERROR: No children states provided!")
        print("🔍 This could be due to:")
        print("  1. The target state doesn't exist in the planner tree")
        print("  2. The target state exists but has no children")
        print("  3. The replanning modified the tree structure")
        return {}

    print(f"\n🔍 Step 1: Sampling random states...")
    sampledStates = sampleRandomState(nextState, numStates=numStates)
    print(f"✅ Sampled {len(sampledStates)} random states")

    print(f"\n🔍 Step 2: Converting to SE2Pose objects...")
    closestStates = [SE2Pose(state[:2], state[2]) for state in sampledStates]
    print(f"✅ Converted {len(closestStates)} states to SE2Pose")

    # Pre-allocate arrays to avoid for loops
    numChildren = len(childrenStates)
    totalPairs = numChildren * numStates
    print(f"\n🔍 Step 3: Setting up optimization arrays...")
    print(f"  - numChildren: {numChildren}")
    print(f"  - numStates: {numStates}")
    print(f"  - totalPairs: {totalPairs}")

    # Create startGuessArray using tensor operation instead of list comprehension
    startGuessArray = np.full(
        (totalPairs, len(initialGuessControl)), initialGuessControl
    )
    print(f"✅ Created startGuessArray with shape: {startGuessArray.shape}")

    # Pre-allocate relativePoses list
    relativePoses = [None] * totalPairs
    print(
        f"✅ Pre-allocated relativePoses list with {len(relativePoses)} elements"
    )

    print(f"\n🔍 Step 4: Computing relative poses...")
    # Convert childrenStates to SE2Pose objects if they're lists
    childrenStatesSE2 = []
    for childState in childrenStates:
        if isinstance(childState, list):
            childrenStatesSE2.append(SE2Pose(childState[:2], childState[2]))
        else:
            childrenStatesSE2.append(childState)  # Already SE2Pose

    print(f"✅ Converted {len(childrenStatesSE2)} children states to SE2Pose")

    # Use single loop with index calculation instead of nested loops
    for i in range(totalPairs):
        childIdx = i // numStates
        sampledIdx = i % numStates
        relativePoses[i] = (
            closestStates[sampledIdx].invert @ childrenStatesSE2[childIdx]
        )
    print(f"✅ Computed {len(relativePoses)} relative poses")

    print(f"\n🔍 Step 5: Converting poses to array...")
    # Convert to array using vectorized operation
    relativePosesArray = np.array(
        [
            [pose.position[0], pose.position[1], pose.euler[2]]
            for pose in relativePoses
        ]
    )
    print(f"✅ Converted to array with shape: {relativePosesArray.shape}")

    print(f"\n🔍 Step 6: Running model prediction...")
    try:
        print(f"🔍 Input shapes:")
        print(f"  - relativePosesArray shape: {relativePosesArray.shape}")
        print(f"  - startGuessArray shape: {startGuessArray.shape}")
        print(f"  - relativePosesArray dtype: {relativePosesArray.dtype}")
        print(f"  - startGuessArray dtype: {startGuessArray.dtype}")

        relativeControls, loss = optModel.predict(
            relativePosesArray, startGuessArray
        )
        print(f"✅ Model prediction successful")
        print(f"  - relativeControls shape: {relativeControls.shape}")
        print(f"  - loss: {loss}")
    except Exception as e:
        print(f"❌ ERROR in model prediction: {e}")
        import traceback

        traceback.print_exc()
        print(f"🔍 Model type: {type(optModel)}")
        print(f"🔍 Model attributes: {dir(optModel)}")
        return {}

    print(f"\n🔍 Step 7: Creating controls tensor...")
    try:
        controlsTensor = torch.tensor(
            relativeControls, device=optModel.device, dtype=torch.float32
        )
        print(f"✅ Created controls tensor with shape: {controlsTensor.shape}")
    except Exception as e:
        print(f"❌ ERROR creating controls tensor: {e}")
        return {}

    print(f"\n🔍 Step 8: Running propagator...")
    try:
        stateDelta = propagator(controlsTensor)
        print(f"✅ Propagator successful")
        print(f"  - stateDelta shape: {stateDelta.shape}")
    except Exception as e:
        print(f"❌ ERROR in propagator: {e}")
        return {}

    print(f"\n🔍 Step 9: Converting stateDelta to SE2Pose...")
    try:
        stateDelta = SE2Pose(
            np.array(
                [
                    stateDelta[0, 0].detach().cpu().numpy(),
                    stateDelta[0, 1].detach().cpu().numpy(),
                ]
            ),
            stateDelta[0, 2].detach().cpu().numpy(),
        )
        print(f"✅ Converted stateDelta to SE2Pose")
        print(f"  - position: {stateDelta.position}")
        print(f"  - euler: {stateDelta.euler}")
    except Exception as e:
        print(f"❌ ERROR converting stateDelta: {e}")
        return {}

    print(f"\n🔍 Step 10: Calculating distances...")
    try:
        # Calculate distances using vectorized operation
        relativeDistances = np.array(
            [
                stateDelta.distance(relativePose)
                for relativePose in relativePoses
            ]
        )
        print(f"✅ Calculated {len(relativeDistances)} distances")
        print(f"  - min distance: {np.min(relativeDistances):.6f}")
        print(f"  - max distance: {np.max(relativeDistances):.6f}")
        print(f"  - mean distance: {np.mean(relativeDistances):.6f}")
    except Exception as e:
        print(f"❌ ERROR calculating distances: {e}")
        return {}

    print(f"\n🔍 Step 11: Filtering by distance...")
    # Find indices to remove using vectorized boolean indexing
    keepMask = relativeDistances <= maxDistance
    keepIndices = np.where(keepMask)[0]
    print(f"✅ Filtering results:")
    print(f"  - total pairs: {len(relativeDistances)}")
    print(f"  - pairs within maxDistance: {len(keepIndices)}")
    print(f"  - maxDistance threshold: {maxDistance}")

    if len(keepIndices) < len(relativeControls):
        relativeControls = relativeControls[keepIndices]
        relativePoses = [relativePoses[i] for i in keepIndices]
        startGuessArray = startGuessArray[keepIndices]
        numStates = len(keepIndices)
        print(f"✅ Filtered arrays to {len(keepIndices)} pairs")
    else:
        print(
            f"✅ No filtering needed, keeping all {len(relativeControls)} pairs"
        )

    print(f"\n🔍 Step 12: Creating pose2control dictionary...")
    # A dictionary mapping relative poses to their corresponding relative controls
    pose2control = {}
    for i, (pose, control) in enumerate(zip(relativePoses, relativeControls)):
        pose_key = (pose.position[0], pose.position[1], pose.euler[2])
        pose2control[pose_key] = control
        if i < 5:  # Print first 5 entries for debugging
            print(f"  Entry {i}: pose={pose_key} -> control={control}")

    print(
        f"✅ Created pose2control dictionary with {len(pose2control)} entries"
    )
    print(f"🧠 DEBUG: runOptimizer completed successfully")

    return pose2control


def createOptimizerThread(
    nextState,
    childrenStates,
    initialGuessControl,
    optModel,
    propagator,
):
    resultContainer = {"result": None, "completed": False}

    def optimizer_wrapper():
        print(f"\n🧠 DEBUG: optimizer_wrapper started")
        try:
            print(f"🔍 Calling runOptimizer...")
            result = runOptimizer(
                nextState,
                childrenStates,
                initialGuessControl,
                propagator,
                optModel,
            )
            print(f"✅ runOptimizer returned: {type(result)}")
            if isinstance(result, dict):
                print(f"  - Dictionary has {len(result)} entries")
            else:
                print(f"  - Result is not a dictionary: {result}")

            resultContainer["result"] = result
            resultContainer["completed"] = True
            print(f"✅ optimizer_wrapper completed successfully")
        except Exception as e:
            print(f"❌ ERROR in optimizer_wrapper: {e}")
            import traceback

            traceback.print_exc()
            resultContainer["error"] = str(e)
            resultContainer["completed"] = True
            print(f"❌ optimizer_wrapper failed with error")

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
):
    # Set up the connection to the simulation
    print("Setting up the connection to the simulation")
    client = SimClient()
    ik = IK("ur10_rod")
    tool_offset = Pose([0, 0, -0.02])
    _, dt, _ = client.execute("get_sim_info")
    client.execute(
        "set_obj_init_poses",
        [0, Pose((startState[0], startState[1], 0.73), (0, 0, startState[2]))],
    )

    # Get the object shape
    print("Getting the object shape")
    objectShape = pickObjectShape(objectName)

    # Pick the propagator and load the model for the optimizer
    print("Picking the propagator and loading the model for the optimizer")
    propagator = pickPropagator(system, objectShape)
    print(f"✅ Propagator loaded: {type(propagator)}")

    print(f"🔍 Loading optimization model...")
    try:
        optModel = load_opt_model_2(
            propagator, lr=learningRate, epochs=numEpochs
        )
        print(f"✅ Optimization model loaded successfully: {type(optModel)}")
        print(
            f"🔍 Model attributes: {[attr for attr in dir(optModel) if not attr.startswith('_')]}"
        )
    except Exception as e:
        print(f"❌ ERROR loading optimization model: {e}")
        import traceback

        traceback.print_exc()
        print("❌ Cannot continue without optimization model")
        return

    # Plan the initial solution
    print("Planning the initial solution")
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

    if solutionsInfo is None:
        print("No initial solution found. Exiting.")
        return

    print("Initial solution found!")
    # print the number of solutions found
    print(f"Found {len(solutionsInfo)} solutions")

    # Print initial planned states
    print("\n📋 Initial planned states:")
    for i, state in enumerate(solutionsInfo[0]["states"]):
        print(
            f"  State {i}: x={state[0]:.3f}, y={state[1]:.3f}, yaw={state[2]:.3f}"
        )

    print("\n🎮 Initial planned controls:")
    for i, control in enumerate(solutionsInfo[0]["controls"]):
        print(f"  Control {i}: {control}")

    # Start the execution loop
    index = 0
    nextControl = solutionsInfo[0]["controls"][0]
    print("\n🚀 Starting the execution loop")

    while True:
        print(f"\n{'='*80}")
        print(f"🚀 EXECUTING ITERATION {index}")
        print(f"{'='*80}")

        ####################################################
        ############# Execute the nextControl ##############
        ####################################################
        print("\n📊 STEP 1: Getting current object state...")
        _, _, obj_rob_pos, obj_rob_quat, _ = client.execute("get_obj_info", 0)
        obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
        currentState = np.array(
            [obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]]
        )
        print(
            f"✅ Current object state: x={currentState[0]:.3f}, y={currentState[1]:.3f}, yaw={currentState[2]:.3f}"
        )

        # 1.1 Compare planned vs actual state
        print("\n🔍 STEP 1.1: Comparing planned vs actual state...")
        planned_state = (
            solutionsInfo[0]["states"][0]
            if solutionsInfo[0].get("states")
            else None
        )
        if planned_state:
            print(
                f"📋 Planned first state: x={planned_state[0]:.3f}, y={planned_state[1]:.3f}, yaw={planned_state[2]:.3f}"
            )
            print(
                f"🎯 Actual object state: x={currentState[0]:.3f}, y={currentState[1]:.3f}, yaw={currentState[2]:.3f}"
            )
            state_diff = np.array(planned_state) - currentState
            print(
                f"📏 State difference: dx={state_diff[0]:.3f}, dy={state_diff[1]:.3f}, dyaw={state_diff[2]:.3f}"
            )
        else:
            print("❌ No planned state available.")
            break

        ####################################################
        ############# STEP 2: Execute current control ##############
        ####################################################
        print(f"\n🎮 STEP 2: Executing current control...")
        print(f"🔧 ORIGINAL nextControl (before optimization): {nextControl}")

        print("🔄 Converting control to trajectory...")
        client.execute("rotate_scene", params=[None, -obj_pose.euler[2]])
        times, ws_path = generate_path_form_params(
            obj_pose, objectShape, nextControl, tool_offset=tool_offset
        )
        traj = ik.ws_path_to_traj(Pose(), times, ws_path)
        waypoints = traj.to_step_waypoints(dt)
        pos_waypoints = np.stack([waypoints[0]], axis=1)
        print(f"✅ Trajectory generated with {len(pos_waypoints)} waypoints")

        # Execute waypoints in parallel thread
        print("🚀 Starting execution thread...")
        executeThread = createExecuteThread(client, pos_waypoints)
        executeThread.start()
        print("✅ Execution thread started")

        ####################################################
        ########## STEP 3: Run optimizer in parallel ###########
        ####################################################
        print("\n🧠 STEP 3: Starting optimizer thread...")
        print(
            f"🎯 Getting next state from solution: {solutionsInfo[0]['states'][1]}"
        )
        nextState = solutionsInfo[0]["states"][1]
        print(f"🔍 Finding children states for next state...")
        print(f"🎯 Target state for children search: {nextState}")
        childrenStates = getChildrenStates(ss, nextState)
        print(f"✅ Found {len(childrenStates)} children states")

        # Debug: Check if this state exists in the current tree
        if len(childrenStates) == 0:
            print(f"⚠️  WARNING: No children found for state {nextState}")
            print(
                f"🔍 This might mean the state doesn't exist in the current tree"
            )
            print(
                f"🔍 This could happen if replanning modified the tree structure"
            )

        print(f"🔧 Using nextControl as initial guess: {nextControl}")
        optimizerThread = createOptimizerThread(
            nextState,
            childrenStates,
            nextControl,
            optModel,
            propagator,
        )
        optimizerThread.start()
        print("✅ Optimizer thread started")

        ####################################################
        ########### STEP 4: Run replanning in parallel ###########
        ####################################################
        print("\n🔄 STEP 4: Starting replanning thread...")
        replanThread = createResolverThread(ss, replanningTime)
        replanThread.start()
        print("✅ Replanner thread started")

        ####################################################
        ########## STEP 5: Wait for all threads ###########
        ####################################################
        print("\n⏳ STEP 5: Waiting for all threads to complete...")
        executeThread.join()
        replanThread.join()
        optimizerThread.join()
        print("✅ All threads completed")

        ####################################################
        ########## STEP 6: Collect results ###########
        ####################################################
        print("\n📊 STEP 6: Collecting thread results...")

        # Get execution result
        executeResult = executeThread.resultContainer["result"]
        executeCompleted = executeThread.resultContainer["completed"]
        print(
            f"✅ Execution completed: {executeCompleted}, Result: {executeResult}"
        )

        # Get replan result
        newSolutionsInfo = replanThread.resultContainer["result"]
        replanCompleted = replanThread.resultContainer["completed"]
        print(
            f"✅ Replan completed: {replanCompleted}, Found {len(newSolutionsInfo) if newSolutionsInfo else 0} solutions"
        )

        # Get optimizer result
        optimizerResult = optimizerThread.resultContainer["result"]
        optimizerCompleted = optimizerThread.resultContainer["completed"]

        if optimizerResult is None:
            print(f"❌ Optimizer failed: result is None")
            if "error" in optimizerThread.resultContainer:
                print(
                    f"❌ Optimizer error: {optimizerThread.resultContainer['error']}"
                )
            print(f"🔄 Using original control without optimization")
            # Continue with the original control instead of breaking
            optimizerResult = {}  # Empty dict to continue
        elif isinstance(optimizerResult, dict):
            print(
                f"✅ Optimizer completed: {optimizerCompleted}, Generated {len(optimizerResult)} pose-control pairs"
            )
        else:
            print(
                f"❌ Optimizer returned unexpected type: {type(optimizerResult)}"
            )
            print(f"🔄 Using original control without optimization")
            optimizerResult = {}  # Empty dict to continue

        ####################################################
        ########## STEP 7: Get updated state ###########
        ####################################################
        print("\n📊 STEP 7: Getting updated object state...")
        _, _, obj_rob_pos, obj_rob_quat, _ = client.execute("get_obj_info", 0)
        obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
        currentState = np.array(
            [obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]]
        )
        print(
            f"✅ Updated object state: x={currentState[0]:.3f}, y={currentState[1]:.3f}, yaw={currentState[2]:.3f}"
        )

        ####################################################
        ########## STEP 8: Check goal condition ###########
        ####################################################
        print("\n🎯 STEP 8: Checking goal condition...")
        goal_x, goal_y = goalState[0], goalState[1]
        distance_to_goal = arrayDistance(
            currentState[:2], [goal_x, goal_y], system="SE2Position"
        )
        print(f"📏 Distance to goal: {distance_to_goal:.3f}")
        if distance_to_goal < 0.05:
            print("🎉 SUCCESS! Reached goal!")
            break

        ####################################################
        ########## STEP 9: Update solution costs ###########
        ####################################################
        print("\n💰 STEP 9: Updating solution costs...")
        print(f"📊 Updating costs for {len(newSolutionsInfo)} solutions...")
        for i, solution in enumerate(newSolutionsInfo):
            # Get the distance between the first two state in the solution
            currentFirstCost = arrayDistance(
                solution["states"][0], solution["states"][1], system="SE2"
            )
            nextCandidateState = solution["states"][1]
            actualFirstCost = arrayDistance(
                currentState, nextCandidateState, system="SE2"
            )
            old_cost = solution["cost"]
            # Modify the new cost of the solution
            solution["cost"] = (
                solution["cost"] - currentFirstCost + actualFirstCost
            )
            print(
                f"  Solution {i}: Cost updated from {old_cost:.5f} to {solution['cost']:.5f}"
            )

        ####################################################
        ########## STEP 10: Find best solution ###########
        ####################################################
        print("\n🏆 STEP 10: Finding best solution...")
        bestSolution = min(newSolutionsInfo, key=lambda x: x["cost"])
        print(f"✅ Best solution cost: {bestSolution['cost']:.5f}")
        print(
            f"📋 Best solution first state: x={bestSolution['states'][0][0]:.3f}, y={bestSolution['states'][0][1]:.3f}, yaw={bestSolution['states'][0][2]:.3f}"
        )
        print(
            f"🔍 Current state: x={currentState[0]:.3f}, y={currentState[1]:.3f}, yaw={currentState[2]:.3f}"
        )

        ####################################################
        ########## STEP 11: Calculate relative pose ###########
        ####################################################
        print("\n🧮 STEP 11: Calculating relative pose...")
        currentStateSE2 = SE2Pose(currentState[:2], currentState[2])
        bestNextStateSE2 = SE2Pose(
            bestSolution["states"][1][:2], bestSolution["states"][1][2]
        )
        relativePose = currentStateSE2.invert @ bestNextStateSE2
        print(
            f"✅ Relative pose: x={relativePose.position[0]:.3f}, y={relativePose.position[1]:.3f}, yaw={relativePose.euler[2]:.3f}"
        )

        ####################################################
        ########## STEP 12: Find optimal control ###########
        ####################################################
        print("\n🎯 STEP 12: Finding optimal control from optimizer...")
        # Convert the relative pose to a tuple key to look up in the optimizer result
        actualRelativePose = (
            relativePose.position[0],
            relativePose.position[1],
            relativePose.euler[2],
        )
        print(
            f"🔍 Looking for control matching relative pose: {actualRelativePose}"
        )

        # Find the closest matching pose in the optimizer result
        closestControl = None
        minDistance = float("inf")
        print(
            f"🔍 Searching through {len(optimizerResult) if optimizerResult else 0} optimizer results..."
        )

        if len(optimizerResult) == 0:
            print("⚠️  No optimizer results available, using original control")
            closestControl = None
        else:
            for pose_key in optimizerResult.keys():
                # Calculate distance between the target relative pose and this pose key based on ompl
                pose_distance = arrayDistance(
                    actualRelativePose, pose_key, system="SE2"
                )
                if pose_distance < minDistance:
                    minDistance = pose_distance
                    closestControl = optimizerResult[pose_key]
                    print(
                        f"  🎯 New closest: distance={pose_distance:.5f}, pose={pose_key}, control={closestControl}"
                    )

        if closestControl is not None:
            # Use the closest matching control
            old_control = (
                nextControl.copy()
                if hasattr(nextControl, "copy")
                else nextControl
            )
            nextControl = closestControl
            print(
                f"✅ OPTIMIZED nextControl (after optimization): {nextControl}"
            )
            print(f"🔄 Control change: {old_control} → {nextControl}")
            print(f"📏 Closest pose distance: {minDistance:.5f}")
        else:
            print("⚠️  Using original control (no optimization applied)")
            # Continue with the original control

        print(f"\n✅ ITERATION {index} COMPLETED")
        index += 1

    print(f"\n🏁 Execution process completed!")

    # Get final object pose
    _, _, obj_rob_pos, obj_rob_quat, _ = client.execute("get_obj_info", 0)
    obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
    finalState = np.array(
        [obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]]
    )

    print(
        f"Final object pose: x={finalState[0]:.3f}, y={finalState[1]:.3f}, yaw={finalState[2]:.3f}"
    )
    print(
        f"Goal state: x={goalState[0]:.3f}, y={goalState[1]:.3f}, yaw={goalState[2]:.3f}"
    )
    print(
        f"Distance to goal: {((finalState[0] - goalState[0])**2 + (finalState[1] - goalState[1])**2)**0.5:.3f}"
    )


if __name__ == "__main__":
    # Parse arguments and load configuration
    config = parse_args_and_config()
    main(
        **config,
    )
