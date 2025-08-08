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

from utils.configHandler import parse_args_and_config

from utils.solutionsHandler import (
    getSolutionsInfo,
    printAllPlannerSolutions,
    printBestSolution,
    getBestSolutionFromPlanner,
    getAllPlannerSolutionsInfo,
    getPathInfo,
)
from utils.utils import (
    visualize_tree_3d,
    state2list,
    isSE2Equal,
    arrayDistance,
    log,
)

from utils.threadHandler import (
    createExecuteThread,
    createResolverThread,
    createOptimizerThread,
)

from utils.childrenHandler import getChildrenStates, sampleRandomState

from ompl import base as ob
from ompl import util as ou
from ompl import control as oc


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
    print(f"[INFO] Plan function started:")
    print(f"     - system: {system}")
    print(f"     - objectShape: {objectShape}")
    print(f"     - startState: {startState}")
    print(f"     - goalState: {goalState}")
    print(f"     - planningTime: {planningTime}")
    print(f"     - replanningTime: {replanningTime}")
    print(f"     - plannerName: {plannerName}")
    print(f"     - visualize: {visualize}")

    try:
        space, cspace = configurationSpace(system)

        # Define a simple setup class
        ss = oc.SimpleSetup(cspace)

        ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(partial(isStateValid, ss.getSpaceInformation()))
        )

        # Set the propagator
        ss.setStatePropagator(oc.StatePropagatorFn(propagator))
        ss.getSpaceInformation().setMinMaxControlDuration(1, 1)

        # Set the control sampler
        controlSampler = pickControlSampler(system, objectShape)
        cspace.setControlSamplerAllocator(oc.ControlSamplerAllocator(controlSampler))

        # Create a start state
        start = pickStartState(system, space, startState)
        ss.setStartState(start)

        # Create a goal state
        goal = pickGoalState(system, goalState, startState, objectShape, ss)
        goal.setThreshold(0.02)
        ss.setGoal(goal)
    except Exception as e:
        log(f"[ERROR] in plan function setup: {e}", "error")
        import traceback

        traceback.print_exc()
        raise e

    # Choose planner based on parameter
    planner = pickPlanner(plannerName, ss)
    planner.setPruningRadius(0.1)
    ss.setPlanner(planner)

    # Set the optimization objective to path length
    ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(ss.getSpaceInformation()))

    try:
        solved = ss.solve(planningTime)
    except Exception as e:
        log(f"[ERROR] during solve: {e}", "error")
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


def pickBestControl(
    optimizerResult,
    actualCurrentState,
    actualNextState,
    childrenStates,
    childrenControls,
    optModel,
    maxDistance,
):
    if not optimizerResult:
        log("[ERROR] No optimizer result available", "error")
        return None

    controls, sampledStates, childrenStates = optimizerResult

    print(f"[INFO] Choosing best control from optimizer results:")
    print(f"       - Actual current state: {actualCurrentState}")
    print(f"       - Actual next state: {actualNextState}")
    print(f"       - Controls dictionary has {len(controls)} children")
    print(f"       - Number of children states: {len(childrenStates)}")

    actualCurrentPose = SE2Pose(actualCurrentState[:2], actualCurrentState[2])
    actualNextPose = SE2Pose(actualNextState[:2], actualNextState[2])

    allSampledStates = []
    allControls = []
    allChildIndices = []

    # Flatten the controls dictionary into arrays
    for child_idx, control_list in controls.items():
        for sampled_state, control in control_list:
            allSampledStates.append(sampled_state)
            allControls.append(control)
            allChildIndices.append(child_idx)

    if len(allSampledStates) == 0:
        log("[ERROR] No controls available in optimizer results", "error")
        return None

    # Convert to numpy arrays for vectorized operations
    allChildIndices = np.array(allChildIndices)
    allControls = np.array(allControls)

    print(f"  - Total flattened controls: {len(allSampledStates)}")

    sampledDistances = np.array(
        [actualCurrentPose.distance(sampled_state) for sampled_state in allSampledStates]
    )

    sorted_indices = np.argsort(sampledDistances)

    for i, idx in enumerate(sorted_indices):
        control = allControls[idx]
        sampledState = allSampledStates[idx]
        childIdx = allChildIndices[idx]
        distance = sampledDistances[idx]

        control_tensor = torch.tensor(control, dtype=torch.float32).unsqueeze(0)

        if hasattr(optModel.model, "device"):
            device = optModel.model.device
        elif next(optModel.model.parameters()).is_cuda:
            device = next(optModel.model.parameters()).device
        else:
            device = torch.device("cpu")

        control_tensor = control_tensor.to(device)

        with torch.no_grad():
            model_output_tensor = optModel.model(control_tensor)

        # print(f"    DEBUG - Raw model output tensor: {model_output_tensor}")

        rel_x = model_output_tensor[0, 0].detach().cpu().numpy()
        rel_y = model_output_tensor[0, 1].detach().cpu().numpy()
        rel_yaw = model_output_tensor[0, 2].detach().cpu().numpy()

        predicted_relative_distance = SE2Pose(
            np.array([rel_x, rel_y]),
            rel_yaw,
        )

        predicted_next_state = actualCurrentPose @ predicted_relative_distance

        predicted_next_state_list = [
            predicted_next_state.position[0],
            predicted_next_state.position[1],
            predicted_next_state.euler[2],
        ]
        predicted_distance = arrayDistance(
            predicted_next_state_list,
            actualNextState,
            "SE2",
        )

        # print(
        #     f"    Control {i+1}/{len(sorted_indices)}: distance to current={distance:.4f}, predicted distance to next={predicted_distance:.4f}"
        # )
        # print(f"    Actual current state: {actualCurrentState}")
        # print(f"    Sampled state: {sampledState}")
        # print(f"    Predicted next state: {predicted_next_state}")
        # print(f"    Actual next state: {actualNextState}")
        # input("Press Enter to continue...")

        if childIdx < len(childrenStates) and childIdx < len(childrenControls):
            original_child_state = childrenStates[childIdx]
            original_child_control = childrenControls[childIdx]
        #     print(
        #         f"  - Original child {childIdx}: state=[{original_child_state.position[0]:.5f}, {original_child_state.position[1]:.5f}, {original_child_state.euler[2]:.5f}], control={original_child_control}"
        #     )

        # print(
        #     f"    DEBUG - About to check: predicted_distance ({predicted_distance:.6f}) < maxDistance ({maxDistance})"
        # )
        if predicted_distance < maxDistance:
            print(f"    DEBUG - Distance check PASSED, about to return control")
            print(f"\n[INFO] BEST CONTROL SELECTED (model-based):")
            print(f"  - Child index: {childIdx}")
            print(
                f"  - Sampled state: [{sampledState.position[0]:.5f}, {sampledState.position[1]:.5f}, {sampledState.euler[2]:.5f}]"
            )
            print(f"  - Distance to current: {distance:.5f}")
            print(f"  - Predicted distance to next: {predicted_distance:.5f}")
            print(f"  - Selected control: {control}")
            print(f"  - Predicted relative distance: {predicted_relative_distance}")

            print(f"    DEBUG - About to return control: {control}")
            print(f"    DEBUG - Control type: {type(control)}")
            print(f"    DEBUG - Control hasattr tolist: {hasattr(control, 'tolist')}")
            return control.tolist() if hasattr(control, "tolist") else list(control)
        # else:
        #     print(f"    DEBUG - Distance check FAILED, continuing to next control")

    print(f"\n[WARNING] No control found within maxDistance ({maxDistance}), returning None")
    return None


def simpleExecution(
    client,
    solutionsInfo,
    goalState,
    objectShape,
    tool_offset,
    ik,
):
    """
    Execute the initial plan without optimization (Phase 1).

    Args:
        client: SimClient instance
        solutionsInfo: List of solution dictionaries
        goalState: Target goal state
        objectShape: Object shape for trajectory generation
        tool_offset: Tool offset for trajectory generation
        ik: IK solver instance

    Returns:
        tuple: (initial_final_state, initial_distance_to_goal)
    """
    print("\n" + "=" * 80)
    print("🚀 PHASE 1: Executing initial plan WITHOUT optimization")
    print("=" * 80)

    for i, control in enumerate(solutionsInfo[0]["controls"]):
        print(
            f"\n       - Executing control {i}: {control[0]:.5f}, {control[1]:.5f}, {control[2]:.5f}"
        )

        # Get current object state
        _, _, obj_rob_pos, obj_rob_quat, _ = client.execute("get_obj_info", 0)
        obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
        currentState = np.array([obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]])
        print(
            f"       - Current state: x={currentState[0]:.3f}, y={currentState[1]:.3f}, yaw={currentState[2]:.3f}"
        )

        # Compare with planned state
        if i < len(solutionsInfo[0]["states"]):
            planned_state = solutionsInfo[0]["states"][i]
            print(
                f"       - Planned state {i}: x={planned_state[0]:.3f}, y={planned_state[1]:.3f}, yaw={planned_state[2]:.3f}"
            )
            state_diff = np.array(planned_state) - currentState
            print(
                f"       - State difference: dx={state_diff[0]:.3f}, dy={state_diff[1]:.3f}, dyaw={state_diff[2]:.3f}"
            )

        # Execute the control
        client.execute("rotate_scene", params=[None, -obj_pose.euler[2]])
        times, ws_path = generate_path_form_params(
            obj_pose, objectShape, control, tool_offset=tool_offset
        )
        traj = ik.ws_path_to_traj(Pose(), times, ws_path)
        waypoints = traj.to_step_waypoints(client.execute("get_sim_info")[1])
        pos_waypoints = np.stack([waypoints[0]], axis=1)

        # Execute waypoints
        executeThread = createExecuteThread(client, pos_waypoints)
        executeThread.start()
        executeThread.join()

        # Get state after execution
        _, _, obj_rob_pos, obj_rob_quat, _ = client.execute("get_obj_info", 0)
        obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
        newState = np.array([obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]])
        print(
            f"       - State after execution: x={newState[0]:.3f}, y={newState[1]:.3f}, yaw={newState[2]:.3f}"
        )

        # Compare with expected next state
        if i + 1 < len(solutionsInfo[0]["states"]):
            expected_next_state = solutionsInfo[0]["states"][i + 1]
            print(
                f"       - Expected next state: x={expected_next_state[0]:.3f}, y={expected_next_state[1]:.3f}, yaw={expected_next_state[2]:.3f}"
            )
            execution_diff = np.array(expected_next_state) - newState
            print(
                f"       - Execution deviation: dx={execution_diff[0]:.3f}, dy={execution_diff[1]:.3f}, dyaw={execution_diff[2]:.3f}"
            )

    # Get final state from initial plan execution
    _, _, obj_rob_pos, obj_rob_quat, _ = client.execute("get_obj_info", 0)
    obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
    initial_final_state = np.array([obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]])

    # Get the final planned state (last state in the solution)
    final_planned_state = solutionsInfo[0]["states"][-1]

    # Calculate distance to final planned state for initial plan
    initial_distance_to_planned = arrayDistance(
        initial_final_state,
        final_planned_state,
        system="SE2",
    )

    # Also calculate distance to original goal for reference
    initial_distance_to_goal = arrayDistance(
        initial_final_state,
        goalState,
        system="SE2",
    )

    initial_cost = solutionsInfo[0]["cost"]

    print(f"\n📊 PHASE 1 RESULTS:")
    print(
        f"  Actual final state: x={initial_final_state[0]:.3f}, y={initial_final_state[1]:.3f}, yaw={initial_final_state[2]:.3f}"
    )
    print(
        f"  Planned final state: x={final_planned_state[0]:.3f}, y={final_planned_state[1]:.3f}, yaw={final_planned_state[2]:.3f}"
    )
    print(f"  Distance to planned final: {initial_distance_to_planned:.3f}")
    print(f"  Distance to original goal: {initial_distance_to_goal:.3f}")

    return initial_final_state, initial_distance_to_planned, initial_cost


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
    sampling_max_distance: float = 0.025,
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

    objectShape = pickObjectShape(objectName)
    propagator = pickPropagator(system, objectShape)

    try:
        # Load the SAME model that OMPL BoxPropagator uses for consistency
        torch_model = load_model("residual", 3, 3)
        torch_model.load(f"saved_models/crackerBoxRandom9000.pth")  # Load trained weights
        actual_model = torch_model.model
        actual_model.eval()  # Set to evaluation mode

        print(f"[INFO] Loaded SAME trained model (crackerBoxRandom9000.pth) for optimization")

        optModel = load_opt_model_2(actual_model, lr=learningRate, epochs=numEpochs)
    except Exception as e:
        log(f"[ERROR] loading optimization model: {e}", "error")
        import traceback

        traceback.print_exc()
        log("[ERROR] Cannot continue without optimization model", "error")
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
        log("[ERROR] No solutions found after initial planning", "error")
        return

    print(f"Initial solution found!")
    print(f"Found {len(solutionsInfo)} solutions")

    # Demonstrate the new solution tracking feature
    planner = ss.getPlanner()
    if hasattr(planner, "getAllSolutions"):
        printAllPlannerSolutions(planner, "Solutions Found During Initial Planning")

    # Print best solution details using the new function
    printBestSolution(solutionsInfo[0], "INITIAL PLANNING")

    doPhase1 = True
    if doPhase1:
        user_input = input("Execute initial plan without optimization? (y/n)")
        if user_input.lower() == "y":
            print("Executing initial plan without optimization...")
        else:
            doPhase1 = False

    ####################################################
    ########## PHASE 1: Execute initial plan without optimization ###########
    ####################################################
    if doPhase1:
        # Call the simpleExecution function to execute Phase 1
        initial_final_state, initial_distance_to_planned, initial_cost = simpleExecution(
            client=client,
            solutionsInfo=solutionsInfo,
            goalState=goalState,
            objectShape=objectShape,
            tool_offset=tool_offset,
            ik=ik,
        )

    # input("Press Enter to continue to Phase 2...")
    ####################################################
    ########## PHASE 2: Reset scene and run with optimization ###########
    ####################################################
    print("\n" + "=" * 80)
    print("🚀 PHASE 2: Running WITH optimization")
    print("=" * 80)

    # Reset the scene
    client.execute(
        "set_obj_init_poses",
        [0, Pose((startState[0], startState[1], 0.73), (0, 0, startState[2]))],
    )

    # Start the execution loop with optimization
    index = 0
    nextControl = solutionsInfo[0]["controls"][0]

    while len(solutionsInfo[0]["controls"]) > 1:

        print(f"\n{'='*20} EXECUTING ITERATION {index} {'='*20}")

        ################### Get object state info ####################
        ##############################################################
        _, _, obj_rob_pos, obj_rob_quat, _ = client.execute("get_obj_info", 0)
        obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
        currentState = np.array([obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]])

        # Compare planned vs actual state
        planned_state = solutionsInfo[0]["states"][0]
        print(
            f"[INFO] Planned state {index}: x={planned_state[0]:.3f}, y={planned_state[1]:.3f}, yaw={planned_state[2]:.3f}"
        )
        print(
            f"[INFO] Actual object state: x={currentState[0]:.3f}, y={currentState[1]:.3f}, yaw={currentState[2]:.3f}"
        )
        state_diff = currentState - planned_state
        print(
            f"[INFO] State difference: dx={state_diff[0]:.3f}, dy={state_diff[1]:.3f}, dyaw={state_diff[2]:.3f}"
        )

        ################## Execute the nextControl ###################
        ##############################################################
        print(
            f"[INFO] EXECUTING CONTROL: {nextControl[0]:.5f}, {nextControl[1]:.5f}, {nextControl[2]:.5f}"
        )
        client.execute("rotate_scene", params=[None, -obj_pose.euler[2]])
        times, ws_path = generate_path_form_params(
            obj_pose, objectShape, nextControl, tool_offset=tool_offset
        )
        traj = ik.ws_path_to_traj(Pose(), times, ws_path)
        waypoints = traj.to_step_waypoints(dt)
        pos_waypoints = np.stack([waypoints[0]], axis=1)

        ################## Run the execution thread ##################
        ##############################################################
        executeThread = createExecuteThread(client, pos_waypoints)
        executeThread.start()

        ############### Get children of the next state ###############
        ##############################################################
        nextState = solutionsInfo[0]["states"][1]
        childrenStates, childrenControls = getChildrenStates(ss, nextState)
        print(f"[INFO] Found {len(childrenStates)} children states")

        # Verify if the extracted controls can actually produce the children states
        print(f"\n[INFO] VERIFYING CHILDREN CONTROLS:")
        print(f"[INFO] Using the SAME propagator model as OMPL planning tree...")
        nextStatePose = SE2Pose(nextState[:2], nextState[2])

        # Load the exact same model used by the BoxPropagator in OMPL
        try:
            verification_model = load_model("residual", 3, 3)
            verification_model.load(f"saved_models/crackerBoxRandom9000.pth")
            verification_model = verification_model.model
            verification_model.eval()
            print(f"[INFO] Loaded OMPL BoxPropagator model for verification")
        except Exception as e:
            print(f"[WARNING] Could not load BoxPropagator model for verification: {e}")
            print(f"[WARNING] Using optModel instead (may show discrepancies)")
            verification_model = optModel.model

        for i, (child_state, child_control) in enumerate(zip(childrenStates, childrenControls)):
            try:
                # Apply the child control to the verification model
                import torch
                import matplotlib

                matplotlib.use("Agg")  # Use non-interactive backend

                control_tensor = torch.tensor(child_control, dtype=torch.float32).unsqueeze(0)

                # Move tensor to the same device as the model
                if (
                    hasattr(verification_model, "parameters")
                    and len(list(verification_model.parameters())) > 0
                ):
                    device = next(verification_model.parameters()).device
                else:
                    device = torch.device("cpu")

                control_tensor = control_tensor.to(device)

                with torch.no_grad():
                    predicted_delta = verification_model(control_tensor)
                    predicted_delta = predicted_delta.squeeze(0).cpu().numpy()

                # Apply the predicted delta to the current state using SE2 composition
                predicted_delta_pose = SE2Pose(predicted_delta[:2], predicted_delta[2])
                predicted_child_state = nextStatePose @ predicted_delta_pose

                # Compare with actual child state
                predicted_child_list = [
                    predicted_child_state.position[0],
                    predicted_child_state.position[1],
                    predicted_child_state.euler[2],
                ]

                distance_error = arrayDistance(predicted_child_list, child_state, "SE2")

                # print(f"  Child {i}: control={child_control}")
                # print(
                #     f"    Actual child:    [{child_state[0]:.5f}, {child_state[1]:.5f}, {child_state[2]:.5f}]"
                # )
                # print(
                #     f"    Predicted child: [{predicted_child_list[0]:.5f}, {predicted_child_list[1]:.5f}, {predicted_child_list[2]:.5f}]"
                # )
                # print(f"    Distance error: {distance_error:.6f}")

                # if distance_error > 0.01:  # Flag significant errors
                # print(f"    ⚠️  WARNING: Large prediction error!")
                # else:
                # print(f"    ✅ Good prediction!")

            except Exception as e:
                print(f"  Child {i}: Error verifying control - {e}")

        print(f"[INFO] Children verification completed\n")

        # Debug: Check if this state exists in the current tree
        if len(childrenStates) == 0:
            log(f"[WARNING] No children found for state {nextState}", "warning")
            log(
                f"[WARNING] This might mean the state doesn't exist in the current tree",
                "warning",
            )
            log(
                f"[WARNING] This could happen if replanning modified the tree structure",
                "warning",
            )

        ################## Run the optimizer thread ##################
        ##############################################################
        optimizerThread = createOptimizerThread(
            nextState,
            childrenStates,
            childrenControls,
            optModel,
            sampling_num_states,
            sampling_max_distance,
            sampling_position_std,
            sampling_rotation_std,
        )
        optimizerThread.start()

        ################## Run the replanning thread #################
        ##############################################################
        replanThread = createResolverThread(ss, replanningTime)
        replanThread.start()
        print("[INFO] Replanner thread started")

        #################### Wait for the threads ####################
        ##############################################################
        print("\n[INFO] Waiting for all threads to complete...")
        executeThread.join()
        replanThread.join()
        if optimizerThread:
            optimizerThread.join()
        print("[INFO] All threads completed")

        ################## Collect the results ##################
        ##############################################################
        # Get execution result
        executeResult = executeThread.resultContainer["result"]
        executeCompleted = executeThread.resultContainer["completed"]
        print(f"[INFO] Execution completed: {executeCompleted}")

        # Get the final object pose
        _, _, obj_rob_pos, obj_rob_quat, _ = client.execute("get_obj_info", 0)
        obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
        currentState = np.array([obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]])

        # Get replan result and update solutionsInfo
        solutionsInfo = replanThread.resultContainer["result"]
        replanCompleted = replanThread.resultContainer["completed"]
        print(f"[INFO] Replan completed: {replanCompleted}")

        # Get optimizer result
        optimizerResult = optimizerThread.resultContainer["result"]
        optimizerCompleted = optimizerThread.resultContainer["completed"]
        print(f"[INFO] Optimizer completed: {optimizerCompleted}")

        # Choose the best control from optimizer results
        if optimizerResult:
            nextState = solutionsInfo[0]["states"][1]
            best_control = pickBestControl(
                optimizerResult,
                currentState,
                nextState,
                childrenStates,
                childrenControls,
                optModel,
                sampling_max_distance,
            )
            if best_control is not None:
                nextControl = best_control
                print(f"[INFO] Using optimized control: {nextControl}")
            else:
                log(
                    "[WARNING] No control found within maxDistance, using next replanned control.",
                    "warning",
                )
                nextControl = solutionsInfo[0]["controls"][0]
                print(f"[INFO] Automatically selected next replanned control: {nextControl}")
        else:
            log(
                "[WARNING] No optimizer result available, using next replanned control.",
                "warning",
            )
            nextControl = solutionsInfo[0]["controls"][0]
            print(f"[INFO] Automatically selected next replanned control: {nextControl}")

        index += 1
        # input("Press Enter to continue...")

        # Check if we should break out of the loop
        if len(solutionsInfo[0]["controls"]) == 1:
            print(f"[INFO] Only 1 control remaining, breaking loop...")
            break

    # Execure the nextControl
    print(
        f"[INFO] EXECUTING THE FINAL CONTROL: {nextControl[0]:.5f}, {nextControl[1]:.5f}, {nextControl[2]:.5f}"
    )
    client.execute("rotate_scene", params=[None, -obj_pose.euler[2]])
    times, ws_path = generate_path_form_params(
        obj_pose, objectShape, nextControl, tool_offset=tool_offset
    )
    traj = ik.ws_path_to_traj(Pose(), times, ws_path)
    waypoints = traj.to_step_waypoints(dt)
    pos_waypoints = np.stack([waypoints[0]], axis=1)
    executeThread = createExecuteThread(client, pos_waypoints)
    executeThread.start()
    executeThread.join()

    ##################### Print the results ######################
    ##############################################################
    # Get the FINAL object pose after the last execution
    _, _, obj_rob_pos, obj_rob_quat, _ = client.execute("get_obj_info", 0)
    obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
    optimized_final_state = np.array(
        [obj_pose.position[0], obj_pose.position[1], obj_pose.euler[2]]
    )

    # Get the final planned state (last state in the solution at this point)
    final_planned_state = solutionsInfo[0]["states"][-1]

    # Calculate distance to final planned state for optimized execution
    optimized_distance_to_planned = arrayDistance(
        optimized_final_state,
        final_planned_state,
        system="SE2",
    )

    # Also calculate distance to original goal for reference
    optimized_distance_to_goal = arrayDistance(
        optimized_final_state,
        goalState,
        system="SE2",
    )

    print(f"\n📊 PHASE 2 RESULTS:")
    print(
        f"  Actual final state: x={optimized_final_state[0]:.3f}, y={optimized_final_state[1]:.3f}, yaw={optimized_final_state[2]:.3f}"
    )
    print(
        f"  Planned final state: x={final_planned_state[0]:.3f}, y={final_planned_state[1]:.3f}, yaw={final_planned_state[2]:.3f}"
    )
    print(f"  Distance to planned final: {optimized_distance_to_planned:.3f}")
    print(f"  Distance to original goal: {optimized_distance_to_goal:.3f}")

    ####################################################
    ########## FINAL COMPARISON ###########
    ####################################################
    print("\n" + "=" * 80)
    print("📊 FINAL COMPARISON: Initial Plan vs Optimized Execution")
    print("=" * 80)

    optimized_cost = solutionsInfo[0]["cost"]

    # Calculate distance improvement (to planned final state)
    distance_improvement = initial_distance_to_planned - optimized_distance_to_planned
    distance_improvement_percentage = (
        (distance_improvement / initial_distance_to_planned) * 100
        if initial_distance_to_planned > 0
        else 0
    )

    # Calculate cost improvement
    cost_improvement = initial_cost - optimized_cost
    cost_improvement_percentage = (cost_improvement / initial_cost) * 100 if initial_cost > 0 else 0

    print(f"\n📊 RESULTS:")
    print(f"  Initial plan distance to planned final: {initial_distance_to_planned:.3f}")
    print(f"  Optimized plan distance to planned final: {optimized_distance_to_planned:.3f}")
    print(
        f"  Distance improvement: {distance_improvement:.3f} ({distance_improvement_percentage:.1f}%)"
    )
    print(f"  Original path cost: {initial_cost:.3f}")
    print(f"  Executed path cost: {optimized_cost:.3f}")
    print(f"  Cost improvement: {cost_improvement:.3f} ({cost_improvement_percentage:.1f}%)")

    if optimized_distance_to_planned < initial_distance_to_planned:
        print(f"✅ OPTIMIZATION SUCCESSFUL: Better execution accuracy achieved!")
    elif optimized_distance_to_planned > initial_distance_to_planned:
        print(f"❌ OPTIMIZATION FAILED: Worse execution accuracy than initial plan")
    else:
        print(f"➖ OPTIMIZATION NEUTRAL: Same execution accuracy as initial plan")


if __name__ == "__main__":
    # Parse arguments and load configuration
    config = parse_args_and_config()
    main(
        **config,
    )
