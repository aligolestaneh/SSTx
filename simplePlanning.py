import torch
import numpy as np

from functools import partial
from scipy.spatial.transform import Rotation as SciRot

from factories import (
    configurationSpace,
    pickControlSampler,
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

from utils.threadHandler import (
    createExecuteThread,
    createResolverThread,
    createOptimizerThread,
    createSimulatedExecutionThread,
)

from utils.childrenHandler import getChildrenStates

from utils.utils import (
    arrayDistance,
    isStateValid,
    printState,
    addNoise,
    log,
)

from planning.propagators import DublinsAirplaneDynamics, carDynamics, carDynamicsTorch

from ompl import base as ob
from ompl import control as oc


def plan(
    system: str,
    startState: np.ndarray,
    goalState: np.ndarray,
    propagator: oc.StatePropagatorFn,
    minControlDuration: int,
    maxControlDuration: int,
    propagationStepSize: float,
    planningTime: float = 20.0,
    replanningTime: float = 2.0,
    plannerName: str = "fusion",
    pruningRadius: float = 0.1,
    visualize: bool = False,
):
    print(f"[INFO] Plan function started:")
    print(f"     - system: {system}")
    print(f"     - startState: {startState}")
    print(f"     - goalState: {goalState}")
    print(f"     - planningTime: {planningTime}")
    print(f"     - replanningTime: {replanningTime}")
    print(f"     - plannerName: {plannerName}")
    print(f"     - minControlDuration: {minControlDuration}")
    print(f"     - maxControlDuration: {maxControlDuration}")
    print(f"     - propagationStepSize: {propagationStepSize}")
    print(f"     - visualize: {visualize}")

    space, cspace = configurationSpace(system)

    # Define a simple setup class
    ss = oc.SimpleSetup(cspace)

    ss.setStateValidityChecker(
        ob.StateValidityCheckerFn(partial(isStateValid, ss.getSpaceInformation()))
    )

    # Set the propagator
    ss.setStatePropagator(oc.StatePropagatorFn(propagator))
    ss.getSpaceInformation().setMinMaxControlDuration(minControlDuration, maxControlDuration)
    ss.getSpaceInformation().setPropagationStepSize(propagationStepSize)

    # Create a start state
    start = pickStartState(system, space, startState)
    ss.setStartState(start)

    # Create a goal state
    goal = pickGoalState(system, goalState, ss, threshold=0.1)
    ss.setGoal(goal)

    # Choose planner based on parameter
    planner = pickPlanner(plannerName, ss, pruningRadius=pruningRadius)
    ss.setPlanner(planner)

    # Set the optimization objective to path length
    ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(ss.getSpaceInformation()))

    try:
        if planningTime < 0:
            ptc = ob.exactSolnPlannerTerminationCondition(ss.getProblemDefinition())
        else:
            ptc = ob.timedPlannerTerminationCondition(planningTime)

        solved = ss.solve(ptc)
    except Exception as e:
        log(f"[ERROR] during solve: {e}", "error")
        import traceback

        traceback.print_exc()
        raise e

    # Show 3D visualization of the tree
    # if visualize:
    #     visualize_tree_3d(planner, filename=f"fusion_3d_{planningTime}s.png")

    if solved:
        # Print the path to screen
        print("Initial solution found")
        return getSolutionsInfo(ss), ss
    else:
        print("No solution found")
        return None, None


def pickBestControl(
    system,
    optimizerResult,
    actualCurrentState,
    actualNextState,
    childrenStates,
    childrenControls,
    samplingNumStates,
    propagationStepSize,
    maxDistance,
):
    print(f"[DEBUG] pickBestControl called with maxDistance: {maxDistance}")

    # Extract optimizer result
    optimizedControls = optimizerResult["optimized_controls"]
    startStates = optimizerResult["start_states"]
    targetStates = optimizerResult["target_states"]

    print(f"optimizedControls: {optimizedControls.shape}")
    print(f"startStates: {startStates.shape}")
    print(f"targetStates: {targetStates.shape}")

    # Reshape the shape to match the number of children
    optimizedControls = optimizedControls.reshape(len(childrenStates), samplingNumStates, -1)
    startStates = startStates.reshape(len(childrenStates), samplingNumStates, -1)
    targetStates = targetStates.reshape(len(childrenStates), samplingNumStates, -1)

    print(f"optimizedControls: {optimizedControls.shape}")
    print(f"startStates: {startStates.shape}")
    print(f"targetStates: {targetStates.shape}")

    childrenSamples = targetStates[:, 0, :]
    print(f"Children states from optimizer: {childrenSamples}")

    # Find the index of the children state that is closest to the actual next state
    print(f"Actual next state: {actualNextState}")
    print(f"Shape of childrenSamples: {childrenSamples.shape}")

    closestIndex = np.argmin(
        [
            arrayDistance(actualNextState, child.cpu().numpy(), system=system)
            for child in childrenSamples
        ]
    )

    print(f"Closest index: {closestIndex}")

    # Get the optimized control for the closest state
    optimizedControl = optimizedControls[closestIndex, :, :]
    startState = startStates[closestIndex, :, :]
    targetState = targetStates[closestIndex, :, :]

    print(f"Optimized control: {optimizedControl.shape}")
    print(f"Start state: {startState.shape}")
    print(f"Target state: {targetState.shape}")

    print(f"Controls are identical: {torch.allclose(optimizedControl[0], optimizedControl[-1])}")

    # Repeat the actual current state (which has shape (1, 3)) to be (1000, 3)
    # Convert to numpy first to avoid CUDA tensor issues
    actualCurrentState_np = (
        actualCurrentState.cpu().numpy()
        if hasattr(actualCurrentState, "cpu")
        else actualCurrentState
    )
    actualCurrentState = np.tile(actualCurrentState_np, (optimizedControl.shape[0], 1))

    # Make it a torch tensor
    actualCurrentState = torch.tensor(actualCurrentState, dtype=torch.float64).to(
        optimizedControl.device
    )

    # Apply all the optimized controls to the actual next state using the propagator
    if system == "simple_car":
        predictedStates = carDynamicsTorch(
            actualCurrentState, optimizedControl, duration=propagationStepSize
        )

        # Find the index of the predicted state that is closest to the actual next state
        distances = [
            arrayDistance(actualNextState, predictedState.detach().cpu().numpy(), system=system)
            for predictedState in predictedStates
        ]

        # Sort the distances along their indices
        sorted_indices = np.argsort(distances)

        print(f"Min distance: {np.min(distances)}")
        print(f"Closest Predicted State: {predictedStates[sorted_indices[0]]}")

        closestIndex = None
        for i in range(len(sorted_indices)):
            current_distance = distances[sorted_indices[i]]
            print(
                f"[DEBUG] Checking distance {current_distance:.6f} at index {sorted_indices[i]} vs maxDistance {maxDistance:.6f}"
            )
            if current_distance < maxDistance:
                closestIndex = sorted_indices[i]
                print(f"[DEBUG] ✅ Accepted distance {current_distance:.6f} < {maxDistance:.6f}")
                break
            else:
                print(f"[DEBUG] ❌ Rejected distance {current_distance:.6f} >= {maxDistance:.6f}")

        if closestIndex is None:
            return None

        print(f"Closest index: {closestIndex}")
        print(f"Distance at closest index: {distances[closestIndex]:.6f}")
        print(f"Closest predicted state: {predictedStates[closestIndex]}")

    input("Press Enter to continue...")

    return optimizedControl[closestIndex, :]


def simpleExecution(
    system,
    solutionsInfo,
    goalState,
    propagation_step_size,  # Fixed parameter name to match the call
    sampling_position_std,
    sampling_rotation_std,
):
    print(f"       - DEBUG: solutionsInfo keys: {list(solutionsInfo.keys())}")
    print(f"       - DEBUG: solutionsInfo['states'] type: {type(solutionsInfo['states'])}")
    print(f"       - DEBUG: solutionsInfo['states'] length: {len(solutionsInfo['states'])}")
    print(f"       - DEBUG: solutionsInfo['states'][0] type: {type(solutionsInfo['states'][0])}")
    print(f"       - DEBUG: solutionsInfo['states'][0] content: {solutionsInfo['states'][0]}")

    currentState = solutionsInfo["states"][0]
    printState(currentState, system, "Current")

    for i, control in enumerate(solutionsInfo["controls"]):
        printState(control, "control", f"Executing control {i}")

        printState(currentState, system, "Current")

        plannedState = solutionsInfo["states"][i]
        printState(plannedState, system, f"Planned state {i}")

        # Calculate state difference using arrayDistance for efficiency
        stateDiff = arrayDistance(plannedState, currentState, system=system)
        print(f"       - State difference (distance): {stateDiff:.6f}")

        # Get the time duration of the control from the solutionsInfo
        controlDuration = solutionsInfo["time"][i]
        print(f"       - Control duration: {controlDuration:.3f}s")

        executeThread = createSimulatedExecutionThread(
            system, currentState, control, controlDuration, propagation_step_size
        )
        executeThread.start()
        executeThread.join()

        # Get state after execution from execution thread
        currentState = addNoise(
            system,
            executeThread.resultContainer["result"],
            sampling_position_std / 2,
            sampling_rotation_std / 2,
        )

    initialDistanceToPlanned = arrayDistance(
        currentState,
        solutionsInfo["states"][-1],
        system=system,
    )
    initialDistanceToGoal = arrayDistance(
        currentState,
        goalState,
        system=system,
    )
    initialCost = solutionsInfo["cost"]

    print(f"\n📊 PHASE 1 RESULTS:")
    printState(currentState, system, "Actual final")
    printState(solutionsInfo["states"][-1], system, "Planned final")
    print(f"  Distance to planned final: {initialDistanceToPlanned:.3f}")
    print(f"  Distance to original goal: {initialDistanceToGoal:.3f}")

    return currentState, initialDistanceToPlanned, initialCost


def main(
    system: str,
    objectName: str,
    startState: np.ndarray,
    goalState: np.ndarray,
    planningTime: float = 10.0,
    replanningTime: float = 2.0,
    plannerName: str = "fusion",
    pruningRadius: float = 0.1,
    visualize: bool = False,
    learningRate: float = 0.001,  # Matches configHandler.py
    numEpochs: int = 1000,  # Matches configHandler.py
    sampling_num_states: int = 1000,  # Matches configHandler.py
    sampling_max_distance: float = 0.05,  # Matches configHandler.py
    sampling_position_std: float = 0.003,  # Matches configHandler.py
    sampling_rotation_std: float = 0.05,  # Matches configHandler.py
    min_control_duration: int = 1,  # Matches configHandler.py
    max_control_duration: int = 5,  # Matches configHandler.py
    propagation_step_size: float = 1.0,  # Matches configHandler.py
    plateau_factor: float = 0.5,  # Matches configHandler.py
    plateau_patience: int = 2,  # Matches configHandler.py
    plateau_min_lr: float = 1e-6,  # Matches configHandler.py
):
    print(f"system: {system}")
    propagator = pickPropagator(system, None)

    # Plan the initial solution
    solutionsInfo, ss = plan(
        system=system,
        startState=startState,
        goalState=goalState,
        propagator=propagator,
        minControlDuration=min_control_duration,
        maxControlDuration=max_control_duration,
        propagationStepSize=propagation_step_size,
        planningTime=planningTime,
        replanningTime=replanningTime,
        plannerName=plannerName,
        pruningRadius=pruningRadius,
        visualize=visualize,
    )

    if solutionsInfo is None or len(solutionsInfo) == 0:
        log("[ERROR] No solutions found after initial planning", "error")
        return

    print(f"Found {len(solutionsInfo)} solutions")

    # Print best solution details using the new function
    printBestSolution(solutionsInfo[0], "INITIAL PLANNING")

    doSimpleExecution = True
    if doSimpleExecution:
        user_input = input("Execute initial plan without optimization? (y/n)")
        if user_input.lower() == "n":
            doSimpleExecution = False
        else:
            print("Executing initial plan without optimization...")

    # PHASE 1: Execute initial plan without optimization
    ####################################################
    if doSimpleExecution:
        # Call the simpleExecution function to execute Phase 1
        initialFinalState, initialDistanceToPlanned, initialCost = simpleExecution(
            system=system,
            solutionsInfo=solutionsInfo[0],
            goalState=goalState,
            propagation_step_size=propagation_step_size,  # Fixed parameter name
            sampling_position_std=sampling_position_std,
            sampling_rotation_std=sampling_rotation_std,
        )

    # PHASE 2: Optimize the plan
    ####################################################
    index = 0
    nextControl = solutionsInfo[0]["controls"][0]
    nextControlDuration = solutionsInfo[0]["time"][0]
    currentState = solutionsInfo[0]["states"][0]

    while len(solutionsInfo[0]["controls"]) > 1:
        # Report the next control
        print(f"Exectuing control {index}: {nextControl}")

        # Compare the planned and actual state
        plannedState = solutionsInfo[0]["states"][0]
        printState(currentState, system, "Current")
        printState(plannedState, system, "Planned")

        state_diff = arrayDistance(currentState, plannedState, system=system)
        print(f"State difference: {state_diff:.3f}")

        # Run the execution thread
        executeThread = createSimulatedExecutionThread(
            system, currentState, nextControl, nextControlDuration, propagation_step_size
        )
        executeThread.start()

        # Get children states (not needed for simplified replanning test)
        # Get the children of the planned state (not the execution result state)
        # The execution result state is not in the planner tree, but the planned state should be
        nextState = solutionsInfo[0]["states"][1]
        childrenStates, childrenControls = getChildrenStates(ss, nextState, system=system)
        print(f"[INFO] Found {len(childrenStates)} children states")

        if len(childrenStates) == 0:
            log(
                f"[WARNING] No children found for planned state {nextState}, continuing with replanning",
                "warning",
            )

        # Run the optimizer thread
        optimizerThread = createOptimizerThread(
            system,
            nextState,
            childrenStates,
            childrenControls,
            solutionsInfo[0]["time"][0],
            None,  # optModel for learned models
            sampling_num_states,
            sampling_max_distance,
            sampling_position_std,
            sampling_rotation_std,
            numEpochs,  # Changed from optimizer_num_steps
            learningRate,  # Changed from optimizer_learning_rate
            plateau_factor,
            plateau_patience,
            plateau_min_lr,
        )
        optimizerThread.start()

        # Run the replanning thread
        replanThread = createResolverThread(ss, solutionsInfo[0]["time"][0])
        replanThread.start()

        # Wait for the threads to complete
        print("\n[INFO] Waiting for all threads to complete...")
        executeThread.join()
        optimizerThread.join()
        replanThread.join()
        print("[INFO] All threads completed")

        # Get execution result
        currentState = executeThread.resultContainer["result"]
        executeCompleted = executeThread.resultContainer["completed"]
        print(f"[INFO] Execution completed: {executeCompleted}")

        # Check if execution was successful
        if currentState is None:
            print(f"[ERROR] Execution failed - result is None")
            # Use the planned state as fallback
            currentState = solutionsInfo[0]["states"][0]
            print(f"[INFO] Using planned state as fallback: {currentState}")
        else:
            # Add noise to the execution result
            currentState = addNoise(
                system, currentState, sampling_position_std / 2, sampling_rotation_std / 2
            )

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
            print(f"[DEBUG] Optimizer using:")
            print(f"  - actualCurrentState (planned next): {currentState}")
            print(f"  - actualNextState (planned next): {nextState}")
            print(f"  - childrenStates: {len(childrenStates)} states")

            best_control = pickBestControl(
                system,
                optimizerResult,
                currentState,
                nextState,
                childrenStates,
                childrenControls,
                sampling_num_states,
                propagation_step_size,
                sampling_max_distance,
            )
            print(f"[DEBUG] pickBestControl called with maxDistance: {sampling_max_distance}")
            if best_control is not None:
                nextControl = best_control
                print(f"[INFO] Using optimized control: {nextControl}")
            else:
                log(
                    f"[WARNING] No control found within maxDistance {sampling_max_distance}, using next replanned control.",
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
        input("Press Enter to continue...")

        # Check if we should break out of the loop
        if len(solutionsInfo[0]["controls"]) == 1:
            print(f"[INFO] Only 1 control remaining, breaking loop...")

    # Execute the final control
    print(f"Executing final control: {nextControl}")
    print(f"Final control type: {type(nextControl)}")
    if hasattr(nextControl, "device"):
        nextControl_np = nextControl.detach().cpu().numpy()
        print(f"[DEBUG] Converted final control to numpy: {nextControl_np}")
        # Use the numpy version for execution
        executeThread = createSimulatedExecutionThread(
            system, currentState, nextControl_np, nextControlDuration, propagation_step_size
        )
    else:
        # Control is already numpy, use as is
        print(f"[DEBUG] Final control is already numpy: {nextControl}")
        executeThread = createSimulatedExecutionThread(
            system, currentState, nextControl, nextControlDuration, propagation_step_size
        )

    executeThread.start()
    executeThread.join()

    # Check if final execution was successful
    final_execution_result = executeThread.resultContainer["result"]
    if final_execution_result is None:
        print(f"[ERROR] Final execution failed - result is None")
        # Use the current state as fallback
        finalState = currentState
        print(f"[INFO] Using current state as fallback for final state: {finalState}")
    else:
        # Add noise to the final execution result
        finalState = addNoise(
            system,
            final_execution_result,
            sampling_position_std / 2,
            sampling_rotation_std / 2,
        )

    print(f"Final state: {finalState}")

    print(f"\n📊 PHASE 2 RESULTS:")
    printState(finalState, system, "Final")
    printState(solutionsInfo[0]["states"][-1], system, "Planned final")
    optimizedDistanceToPlanned = arrayDistance(
        finalState, solutionsInfo[0]["states"][-1], system=system
    )
    print(f"  Distance to planned final: {optimizedDistanceToPlanned:.3f}")
    print(f"  Distance to original goal: {arrayDistance(finalState, goalState, system=system):.3f}")

    print("\n" + "=" * 80)
    print("📊 FINAL COMPARISON: Initial Plan vs Optimized Execution")
    print("=" * 80)

    optimized_cost = solutionsInfo[0]["cost"]

    # Calculate distance improvement (to planned final state)
    distance_improvement = initialDistanceToPlanned - optimizedDistanceToPlanned
    distance_improvement_percentage = (
        (distance_improvement / initialDistanceToPlanned) * 100
        if initialDistanceToPlanned > 0
        else 0
    )

    # Calculate cost improvement
    cost_improvement = initialCost - optimized_cost
    cost_improvement_percentage = (cost_improvement / initialCost) * 100 if initialCost > 0 else 0

    print(f"\n📊 RESULTS:")
    print(f"  Initial plan distance to planned final: {initialDistanceToPlanned:.3f}")
    print(f"  Optimized plan distance to planned final: {optimizedDistanceToPlanned:.3f}")
    print(
        f"  Distance improvement: {distance_improvement:.3f} ({distance_improvement_percentage:.1f}%)"
    )
    print(f"  Original path cost: {initialCost:.3f}")
    print(f"  Executed path cost: {optimized_cost:.3f}")
    print(f"  Cost improvement: {cost_improvement:.3f} ({cost_improvement_percentage:.1f}%)")

    if optimizedDistanceToPlanned < initialDistanceToPlanned:
        print(f"✅ OPTIMIZATION SUCCESSFUL: Better execution accuracy achieved!")
    elif optimizedDistanceToPlanned > initialDistanceToPlanned:
        print(f"❌ OPTIMIZATION FAILED: Worse execution accuracy than initial plan")
    else:
        print(f"➖ OPTIMIZATION NEUTRAL: Same execution accuracy as initial plan")


if __name__ == "__main__":
    # Parse arguments and load configuration
    config = parse_args_and_config()
    main(**config)
