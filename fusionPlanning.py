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

    # Attempt to solve the problem
    solved = ss.solve(planningTime)

    # Show 3D visualization of the tree
    if visualize:
        visualize_tree_3d(planner, filename=f"fusion_3d_{planningTime}s.png")

    if solved:
        # Print the path to screen
        print("Initial solution found")
        return getSolutionInfo(ss.getSolutionPath(), ss), ss
    else:
        print("No solution found")
        return None


def getSolutionInfo(solution_path, ss):
    solution_info = {}
    solution_info["state_count"] = solution_path.getStateCount()
    solution_info["control_count"] = solution_path.getControlCount()

    # Extract all solution data while objects are still valid
    solution_info = {}
    solution_info["state_count"] = solution_path.getStateCount()
    solution_info["control_count"] = solution_path.getControlCount()

    control_space = ss.getControlSpace()
    control_dimension = control_space.getDimension()

    # Extract all controls while objects are valid
    controls_list = []
    for i in range(solution_info["control_count"]):
        control = solution_path.getControl(i)

        control_values = []
        for j in range(control_dimension):
            control_values.append(control[j])

        controls_list.append(control_values)

        solution_info["controls"] = controls_list

    # Extract all states while objects are valid
    states_list = []
    for i in range(solution_info["state_count"]):
        state = solution_path.getState(i)
        state_values = [state.getX(), state.getY(), state.getYaw()]
        states_list.append(state_values)

    solution_info["states"] = states_list

    print(
        f"Successfully extracted {len(controls_list)} controls and {len(states_list)} states"
    )

    return solution_info


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


def getChildrenStates(ss, targetState, tolerance=1e-6):
    # targetState = state2list(targetState, "SE2")

    planner_data = ob.PlannerData(ss.getSpaceInformation())
    ss.getPlanner().getPlannerData(planner_data)

    num_vertices = planner_data.numVertices()
    targetVertexIdx = None
    for i in range(num_vertices):
        state = planner_data.getVertex(i).getState()

        if isSE2Equal(state2list(state, "SE2"), targetState, tolerance):
            targetVertexIdx = i
            break

    if targetVertexIdx is None:
        print(f"State {targetState} not found in planner tree")
        return []

    childVertexIndices = ou.vectorUint()
    planner_data.getEdges(targetVertexIdx, childVertexIndices)

    children_states = []
    for childVertexIdx in childVertexIndices:
        childState = planner_data.getVertex(childVertexIdx).getState()

        children_states.append(state2list(childState, "SE2"))

    return children_states


def sampleRandomState(state, numStates=1000, posSTD=0.003, rotSTD=0.05):
    sampledStates = []
    stateList = state2list(state, "SE2")
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


def createWaypointThread(client, pos_waypoints):
    resultContainer = {"result": None, "completed": False}
    thread = threading.Thread(
        target=executeWaypoints,
        args=(client, pos_waypoints, resultContainer),
    )
    thread.daemon = True
    thread.resultContainer = resultContainer
    return thread


def runResolver(planner, replanningTime, resultContainer):
    result = planner.resolve(replanningTime)
    resultContainer["result"] = result
    resultContainer["completed"] = True


def createResolverThread(planner, replanningTime):
    resultContainer = {"result": None, "completed": False}
    thread = threading.Thread(
        target=runResolver, args=(planner, replanningTime, resultContainer)
    )
    thread.daemon = True
    thread.resultContainer = (
        resultContainer  # Attach result container to thread
    )
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
    sampledStates = sampleRandomState(nextState, numStates=numStates)
    closestStates = [SE2Pose(state[:2], state[2]) for state in sampledStates]

    # Pre-allocate arrays to avoid for loops
    numChildren = len(childrenStates)
    totalPairs = numChildren * numStates

    # Create startGuessArray using tensor operation instead of list comprehension
    startGuessArray = np.full(
        (totalPairs, len(initialGuessControl)), initialGuessControl
    )

    # Pre-allocate relativePoses list
    relativePoses = [None] * totalPairs

    # Use single loop with index calculation instead of nested loops
    for i in range(totalPairs):
        childIdx = i // numStates
        sampledIdx = i % numStates
        relativePoses[i] = (
            closestStates[sampledIdx].invert @ childrenStates[childIdx]
        )

    # Convert to array using vectorized operation
    relativePosesArray = np.array(
        [
            [pose.position[0], pose.position[1], pose.euler[2]]
            for pose in relativePoses
        ]
    )

    relativeControls, loss = optModel.predict(
        relativePosesArray, startGuessArray
    )

    controlsTensor = torch.tensor(
        relativeControls, device=optModel.device, dtype=torch.float32
    )

    stateDelta = propagator(controlsTensor)
    stateDelta = SE2Pose(
        np.array(
            [
                stateDelta[0, 0].detach().cpu().numpy(),
                stateDelta[0, 1].detach().cpu().numpy(),
            ]
        ),
        stateDelta[0, 2].detach().cpu().numpy(),
    )

    # Calculate distances using vectorized operation
    relativeDistances = np.array(
        [stateDelta.distance(relativePose) for relativePose in relativePoses]
    )

    # Find indices to remove using vectorized boolean indexing
    keepMask = relativeDistances <= maxDistance
    keepIndices = np.where(keepMask)[0]

    if len(keepIndices) < len(relativeControls):
        relativeControls = relativeControls[keepIndices]
        relativePoses = [relativePoses[i] for i in keepIndices]
        startGuessArray = startGuessArray[keepIndices]
        numStates = len(keepIndices)

    return relativeControls, relativePoses


def createOptimizerThread(
    nextState,
    childrenStates,
    initialGuessControl,
    optModel,
    propagator,
):
    resultContainer = {"result": None, "completed": False}

    def optimizer_wrapper():
        try:
            result = runOptimizer(
                nextState,
                childrenStates,
                initialGuessControl,
                propagator,
                optModel,
            )
            resultContainer["result"] = result
            resultContainer["completed"] = True
        except Exception as e:
            resultContainer["error"] = str(e)
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
    optModel = load_opt_model_2(propagator, lr=learningRate, epochs=numEpochs)

    # Plan the initial solution
    print("Planning the initial solution")
    solutionInfo, ss = plan(
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

    nextControl = solutionInfo["controls"][0]

    # Start the execution loop
    index = 0
    print("Starting the execution loop")
    while True:
        print(f"Executing the {index}th iteration")
        ####################################################
        ############# Execute the nextControl ##############
        ####################################################
        print("Getting the object information")
        _, _, obj_rob_pos, obj_rob_quat, _ = client.execute("get_obj_info", 0)
        obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])

        print("Generating the path")
        times, ws_path = generate_path_form_params(
            obj_pose,
            objectShape,
            nextControl,
            tool_offset=tool_offset,
        )

        print("Converting the path to waypoints")
        traj = ik.ws_path_to_traj(Pose(), times, ws_path)
        waypoints = traj.to_step_waypoints(dt)
        pos_waypoints = np.stack(
            [waypoints[0]], axis=1
        )  # Wrap in list to create proper shape

        print("Executing the waypoints")
        print(nextControl)

        # Execute waypoints without waiting for response
        waypointThread = createWaypointThread(client, pos_waypoints)
        waypointThread.start()

        print("Waypoint execution sent (continuing immediately)")

        ####################################################
        ########### Run the resolver in parallel ###########
        ####################################################
        # Create and start resolver thread (clean one-liner)
        print("Creating and starting resolver thread")
        resolverThread = createResolverThread(ss.getPlanner(), replanningTime)
        resolverThread.start()

        ####################################################
        ########## Run the optimizer in parallel ###########
        ####################################################
        # print("Getting the next state")
        # nextState = solutionInfo["states"][1]
        # childrenStates = getChildrenStates(ss, nextState)

        # print("Creating and starting optimizer thread")
        # optimizerThread = createOptimizerThread(
        #     nextState,
        #     childrenStates,
        #     nextControl,
        #     optModel,
        #     propagator,
        # )
        # optimizerThread.start()

        # Optimize the contol from the nextState in a parallel thread
        # Resolve the path for the rest of the path
        # Get the state estimation
        # Choose the nextControl based on the current state and the current plan

        # Wait for the resolver to complete
        waypointThread.join()
        resolverThread.join()
        # optimizerThread.join()

        # Get the waypoint execution result
        waypoint_result = waypointThread.resultContainer["result"]
        waypoint_completed = waypointThread.resultContainer["completed"]

        print(f"Waypoint execution completed: {waypoint_completed}")
        if waypoint_completed:
            print(f"Waypoint result: {waypoint_result}")

        # Get the resolver result
        resolver_result = resolverThread.resultContainer["result"]
        resolver_completed = resolverThread.resultContainer["completed"]

        # optimizer_result = optimizerThread.resultContainer["result"]
        # optimizer_completed = optimizerThread.resultContainer["completed"]

        # print(f"Optimizer completed: {optimizer_completed}")
        # if optimizer_completed:
        #     print(f"Optimizer result: {optimizer_result}")
        print(f"Resolver completed: {resolver_completed}")
        if resolver_completed:
            print(f"Resolver result: {resolver_result}")

            # Check if resolve was successful
            if resolver_result:
                print("✅ Resolve successful - path updated")
                # You can now get the updated solution path
                updated_solution = ss.getSolutionPath()
                if updated_solution and updated_solution.getStateCount() > 0:
                    print(
                        f"Updated solution has {updated_solution.getStateCount()} states"
                    )
                    # Extract updated solution info if needed
                    # updated_solution_info = getSolutionInfo(updated_solution, ss)
            else:
                print("❌ Resolve failed - no path improvement found")

        # Break the loop if we reached the goal


if __name__ == "__main__":
    # Parse arguments and load configuration
    config = parse_args_and_config()
    main(
        **config,
    )
