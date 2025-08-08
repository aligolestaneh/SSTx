import threading
import numpy as np

from geometry.pose import SE2Pose

from utils.utils import log
from utils.solutionsHandler import getSolutionsInfo
from utils.childrenHandler import sampleRandomState


def executeWaypoints(client, pos_waypoints, resultContainer):
    result = client.execute("execute_waypoints", pos_waypoints)
    resultContainer["result"] = result
    resultContainer["completed"] = True


def createExecuteThread(client, pos_waypoints):
    resultContainer = {"result": None, "completed": False}
    print(f"[INFO] Execute thread created")
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
    print(f"[INFO] Resolver thread created")
    thread = threading.Thread(
        target=runResolver, args=(ss, replanningTime, resultContainer)
    )
    thread.daemon = True
    thread.resultContainer = resultContainer
    return thread


def runOptimizer(
    nextState,
    childrenStatesArray,
    childrenControlsArray,
    optModel,  # Changed from model to optModel
    numStates=1000,
    maxDistance=0.025,
    posSTD=0.003,
    rotSTD=0.05,
):
    print(f"[INFO] Optimizer thread created")
    print(
        f"     - Next state: {nextState[0]:.5f}, {nextState[1]:.5f}, {nextState[2]:.5f}"
    )
    print(f"     - Number of children states: {len(childrenStatesArray)}")
    print(f"     - Number of children controls: {len(childrenControlsArray)}")
    print(
        f"     - Number of sampled states: {numStates} (maxDistance: {maxDistance})"
    )
    print(f"     - Max distance: {maxDistance}")
    print(f"     - Position standard deviation: {posSTD}")
    print(f"     - Rotation standard deviation: {rotSTD}")

    try:
        sampledStates_raw = np.array(
            sampleRandomState(nextState, numStates, posSTD, rotSTD)
        )
        sampledStates = [
            SE2Pose(state[:2], state[2]) for state in sampledStates_raw
        ]

        childrenStates_raw = np.array(childrenStatesArray)
        optimizer_childrenStates = [
            SE2Pose(state[:2], state[2]) for state in childrenStates_raw
        ]
        numChildren = len(optimizer_childrenStates)

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

        # Create initial guess controls using the correct control for each child
        # For each sampled state, use the appropriate control for each child
        initialGuessControlsFlat = []
        for i in range(numStates):
            for j in range(numChildren):
                if j < len(childrenControlsArray):
                    initialGuessControlsFlat.append(childrenControlsArray[j])
                else:
                    # Fallback to the first control if we don't have enough controls
                    initialGuessControlsFlat.append(
                        childrenControlsArray[0]
                        if len(childrenControlsArray) > 0
                        else [0.0, 0.0, 0.0]
                    )

        initialGuessControlsFlat = np.array(initialGuessControlsFlat)

        x_min = np.array([0.0, -0.1, 0.0])
        x_max = np.array([2 * np.pi, 0.1, 0.3])

        optimizedControlsFlat, loss = optModel.predict(
            relativePosesFlat,
            initialGuessControlsFlat,
            x_min=x_min,
            x_max=x_max,
            plot=False,
        )
        print(f"     - Loss: {loss}")

        control_dim = (
            optimizedControlsFlat.shape[1]
            if len(optimizedControlsFlat.shape) > 1
            else 1
        )
        optimizedControls = optimizedControlsFlat.reshape(
            numStates, numChildren, control_dim
        )

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

        print(f"     - Number of controls: {len(controls.keys())}")
        print(f"     - Number of sampled states: {len(sampledStates)}")

        target_rotations = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])

        all_controls = np.array(
            [
                control_pair[1]
                for child_controls in controls.values()
                for control_pair in child_controls
            ]
        )

        if len(all_controls) > 0:
            current_rotations = all_controls[:, 0]
            diffs = np.abs(current_rotations[:, np.newaxis] - target_rotations)
            closest_indices = np.argmin(diffs, axis=1)

            # Update all rotations at once
            all_controls[:, 0] = target_rotations[closest_indices]

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
        log(f"[ERROR] in runOptimizer: {e}", "error")
        import traceback

        traceback.print_exc()
        return None


def createOptimizerThread(
    nextState,
    childrenStates,
    childrenControls,
    optModel,
    numStates=1000,
    maxDistance=0.025,
    posSTD=0.003,
    rotSTD=0.05,
):
    resultContainer = {"result": None, "completed": False}

    def optimizer_wrapper():
        try:
            result = runOptimizer(
                nextState,
                childrenStates,
                childrenControls,
                optModel,
                numStates,
                maxDistance,
                posSTD,
                rotSTD,
            )
            if result is not None:
                controls, sampledStates, optimizer_childrenStates = result
                if controls:
                    total_controls = sum(
                        len(control_list) for control_list in controls.values()
                    )
            else:
                log(f"[ERROR] runOptimizer returned None", "error")
            resultContainer["result"] = result
            resultContainer["completed"] = True
        except Exception as e:
            log(f"[ERROR] in optimizer_wrapper: {e}", "error")
            import traceback

            traceback.print_exc()
            resultContainer["result"] = None
            resultContainer["completed"] = True

    thread = threading.Thread(target=optimizer_wrapper)
    thread.daemon = True
    thread.resultContainer = resultContainer
    return thread
