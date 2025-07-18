import sys
import yaml
import argparse
import numpy as np
import torch

from ik import IK
from functools import partial
from sim_network import SimClient
from geometry.pose import Pose, SE2Pose
from geometry.random_push import generate_path_form_params
from utils.utils import visualize_tree_3d
from planning.planning_utils import isStateValid

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
        description="Run fusion planning with YAML configuration (planning only)"
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

    return {
        "system": system,
        "objectName": objectName,
        "startState": startState,
        "goalState": goalState,
        "planningTime": planningTime,
        "replanningTime": replanningTime,
        "plannerName": plannerName,
        "visualize": visualize,
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

    # Set control duration to 1 (both min and max)
    si = ss.getSpaceInformation()
    si.setMinMaxControlDuration(1, 1)
    print("✓ Set control duration to 1 time step")

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


def main(
    system: str,
    objectName: str,
    startState: np.ndarray,
    goalState: np.ndarray,
    planningTime: float = 20.0,
    replanningTime: float = 2.0,
    plannerName: str = "fusion",
    visualize: bool = False,
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

    # Pick the propagator
    print("Picking the propagator")
    propagator = pickPropagator(system, objectShape)

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

    if solutionInfo is None:
        print("No initial solution found. Exiting.")
        return

    print("Initial solution found!")
    print(
        f"Solution has {solutionInfo['state_count']} states and {solutionInfo['control_count']} controls"
    )

    # Print initial planned states
    print("\n📋 Initial planned states:")
    for i, state in enumerate(solutionInfo["states"]):
        print(
            f"  State {i}: x={state[0]:.3f}, y={state[1]:.3f}, yaw={state[2]:.3f}"
        )

    print("\n🎮 Initial planned controls:")
    for i, control in enumerate(solutionInfo["controls"]):
        print(f"  Control {i}: {control}")

    # Start the execution loop
    index = 0
    print("\n🚀 Starting the execution loop")

    while True:
        print(f"\n{'='*60}")
        print(f"EXECUTION ITERATION {index}")
        print(f"{'='*60}")

        # 1. Get current object pose from simulator
        _, _, obj_rob_pos, obj_rob_quat, _ = client.execute("get_obj_info", 0)
        obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
        currentState = np.array(
            [obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]]
        )

        # 1.1 Get the first state from solutionInfo and compare
        planned_state = (
            solutionInfo["states"][0] if solutionInfo.get("states") else None
        )
        if planned_state:
            print(
                f"Planned first state: x={planned_state[0]:.3f}, y={planned_state[1]:.3f}, yaw={planned_state[2]:.3f}"
            )
            print(
                f"Actual object state: x={currentState[0]:.3f}, y={currentState[1]:.3f}, yaw={currentState[2]:.3f}"
            )
            print(
                f"Difference: dx={abs(planned_state[0]-currentState[0]):.3f}, dy={abs(planned_state[1]-currentState[1]):.3f}, dyaw={abs(planned_state[2]-currentState[2]):.3f}"
            )
        else:
            print("No planned state available.")
            break

        # 2. Execute the first control in solutionInfo
        if solutionInfo.get("controls") and len(solutionInfo["controls"]) > 0:
            nextControl = solutionInfo["controls"][0]
            print(f"Executing control: {nextControl}")
        else:
            print("No controls available in solution.")
            break

        # Convert control to end-effector trajectory and execute
        input("Press Enter to continue...")
        client.execute("rotate_scene", params=[None, -obj_pose.euler[2]])
        times, ws_path = generate_path_form_params(
            obj_pose, objectShape, nextControl, tool_offset=tool_offset
        )
        traj = ik.ws_path_to_traj(Pose(), times, ws_path)
        waypoints = traj.to_step_waypoints(dt)
        pos_waypoints = np.stack([waypoints[0]], axis=1)
        result = client.execute("execute_waypoints", pos_waypoints)
        print(f"Waypoint execution completed: {result}")

        # 3. Run the replan function of the planner
        print("Running replan...")
        input("Press Enter to continue...")
        replan_result = ss.getPlanner().replan(replanningTime)
        print(f"Replan result: {replan_result}")

        # 3.1 Get updated solutionInfo and print updated states
        updated_solution = ss.getSolutionPath()
        if updated_solution and updated_solution.getStateCount() > 0:
            solutionInfo = getSolutionInfo(updated_solution, ss)
            print("Updated planned states after replan:")
            for i, state in enumerate(solutionInfo["states"]):
                print(
                    f"  State {i}: x={state[0]:.3f}, y={state[1]:.3f}, yaw={state[2]:.3f}"
                )
        else:
            print("No updated solution found after replan.")
            break

        # 3.2 Get and print the current object state after execution
        _, _, obj_rob_pos, obj_rob_quat, _ = client.execute("get_obj_info", 0)
        obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
        currentState = np.array(
            [obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]]
        )
        print(
            f"Current object state after execution: x={currentState[0]:.3f}, y={currentState[1]:.3f}, yaw={currentState[2]:.3f}"
        )

        # Check for goal
        goal_x, goal_y = goalState[0], goalState[1]
        distance_to_goal = (
            (currentState[0] - goal_x) ** 2 + (currentState[1] - goal_y) ** 2
        ) ** 0.5
        print(f"Distance to goal: {distance_to_goal:.3f}")
        if distance_to_goal < 0.05:
            print("🎉 SUCCESS! Reached goal!")
            break

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
        f"Distance to goal: {((finalState[0] - goalState[0])**2 + (finalState[1] - goalState[1])**2)**0.5:.3f}"
    )


if __name__ == "__main__":
    # Parse arguments and load configuration
    config = parse_args_and_config()
    main(**config)
