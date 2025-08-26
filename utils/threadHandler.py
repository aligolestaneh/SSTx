import torch
import threading
import numpy as np
from typing import Callable, Dict, Any, Optional, Tuple

from geometry.pose import SE2Pose

from utils.utils import log
from utils.solutionsHandler import getSolutionsInfo
from utils.childrenHandler import sampleRandomState

from planning.propagators import DublinsAirplaneDynamics, carDynamics, carDynamicsTorch


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


def simulatedExecution(system, state, control, totalTime, dt, resultContainer):
    try:
        if system == "dublin_airplane":
            result = DublinsAirplaneDynamics(state, control, totalTime)

        elif system == "simple_car":
            # Create a local copy to avoid modifying the original parameter
            current_state = state.copy() if hasattr(state, "copy") else np.array(state)

            # Convert control to numpy if it's a tensor
            if hasattr(control, "cpu"):
                control_np = control.detach().cpu().numpy()
            else:
                control_np = control

            # Validate inputs
            if totalTime <= 0 or dt <= 0:
                raise ValueError(f"Invalid time parameters: totalTime={totalTime}, dt={dt}")

            if current_state is None or control_np is None:
                raise ValueError(
                    f"Invalid state or control: state={current_state}, control={control_np}"
                )

            num_steps = int(totalTime / dt)

            for i in range(num_steps):
                result = carDynamics(current_state, control_np, dt)

                if result is None:
                    raise RuntimeError(f"carDynamics returned None at step {i}")

                # Validate result
                if not isinstance(result, (np.ndarray, list)) or len(result) != 3:
                    raise RuntimeError(
                        f"carDynamics returned invalid result at step {i}: {result} (type: {type(result)})"
                    )

                current_state = result  # Update local copy, not parameter

            result = current_state  # Return final result

        else:
            result = None

        resultContainer["result"] = result
        resultContainer["completed"] = True

    except Exception as e:
        print(f"[ERROR] simulatedExecution failed: {e}")
        import traceback

        traceback.print_exc()
        resultContainer["result"] = None
        resultContainer["completed"] = False


def simulatedExecutionWrapper(system, state, control, totalTime, dt, resultContainer):
    """
    Wrapper function for simulatedExecution with additional error handling
    """
    try:
        # Input validation
        if system not in ["dublin_airplane", "simple_car"]:
            raise ValueError(f"Invalid system: {system}")

        if state is None:
            raise ValueError("State is None")

        if control is None:
            raise ValueError("Control is None")

        if totalTime <= 0:
            raise ValueError(f"totalTime must be positive, got: {totalTime}")

        if dt <= 0:
            raise ValueError(f"dt must be positive, got: {dt}")

        if totalTime < dt:
            raise ValueError(f"totalTime ({totalTime}) must be >= dt ({dt})")

        # Call the actual execution function
        simulatedExecution(system, state, control, totalTime, dt, resultContainer)

    except Exception as e:
        print(f"[ERROR] simulatedExecutionWrapper failed: {e}")
        import traceback

        traceback.print_exc()
        resultContainer["result"] = None
        resultContainer["completed"] = False


def createSimulatedExecutionThread(system, state, control, totalTime, dt):
    resultContainer = {"result": None, "completed": False}
    thread = threading.Thread(
        target=simulatedExecutionWrapper,  # Use the wrapper instead
        args=(system, state, control, totalTime, dt, resultContainer),
    )
    thread.daemon = True
    thread.resultContainer = resultContainer
    return thread


def runResolver(ss, replanningTime, resultContainer):
    ss.getPlanner().resolve(replanningTime)
    result = getSolutionsInfo(ss)
    resultContainer["result"] = result
    resultContainer["completed"] = True


def createResolverThread(ss, replanningTime):
    resultContainer = {"result": None, "completed": False}
    print(f"[INFO] Resolver thread created for replanning time {replanningTime}")
    thread = threading.Thread(target=runResolver, args=(ss, replanningTime, resultContainer))
    thread.daemon = True
    thread.resultContainer = resultContainer
    return thread


def runOptimizer(
    nextState,
    childrenStatesArray,
    childrenControlsArray,
    optModel,
    numStates=1000,
    maxDistance=0.025,
    posSTD=0.003,
    rotSTD=0.05,
):
    print(f"[INFO] Optimizer thread created")
    print(f"     - Next state: {nextState[0]:.5f}, {nextState[1]:.5f}, {nextState[2]:.5f}")
    print(f"     - Number of children states: {len(childrenStatesArray)}")
    print(f"     - Number of children controls: {len(childrenControlsArray)}")
    print(f"     - Number of sampled states: {numStates} (maxDistance: {maxDistance})")
    print(f"     - Max distance: {maxDistance}")
    print(f"     - Position standard deviation: {posSTD}")
    print(f"     - Rotation standard deviation: {rotSTD}")

    try:
        sampledStates_raw = np.array(sampleRandomState(nextState, numStates, posSTD, rotSTD))
        sampledStates = [SE2Pose(state[:2], state[2]) for state in sampledStates_raw]

        childrenStates_raw = np.array(childrenStatesArray)
        optimizer_childrenStates = [SE2Pose(state[:2], state[2]) for state in childrenStates_raw]
        numChildren = len(optimizer_childrenStates)

        sampled_inverts = [s.invert for s in sampledStates]

        # Broadcast: shape will be (numStates, numChildren)
        relativePoses_matrix = [
            [inv @ c for c in optimizer_childrenStates] for inv in sampled_inverts
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

        control_dim = optimizedControlsFlat.shape[1] if len(optimizedControlsFlat.shape) > 1 else 1
        optimizedControls = optimizedControlsFlat.reshape(numStates, numChildren, control_dim)

        flat_controls = optimizedControls.reshape(numStates * numChildren, control_dim)
        tiled_samples = np.repeat(sampledStates, numChildren)
        child_indices = np.tile(np.arange(numChildren), numStates)

        controls = {i: [] for i in range(numChildren)}  # Use indices as keys
        for idx in range(numChildren):
            mask = child_indices == idx
            controls[idx] = [[tiled_samples[i], flat_controls[i]] for i in np.where(mask)[0]]

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
                    new_controls[child_idx].append([control_pair[0], all_controls[control_idx]])
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


def controlJacobian(
    dynamics: Callable[..., torch.Tensor],
    x: torch.Tensor,
    u: torch.Tensor,
    *,
    dynamics_kwargs: Optional[Dict[str, Any]] = None,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (y, df/du) with shapes:
      y:    (..., state_dim)
      dfdu: (..., state_dim, control_dim)
    """
    if dynamics_kwargs is None:
        dynamics_kwargs = {}

    u = u.clone().requires_grad_(True)

    with torch.enable_grad():
        y = dynamics(x.detach(), u, **dynamics_kwargs)

    # flatten leading batch dims for per-item jacobian clarity
    def _flatten(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1, t.shape[-1]) if t.ndim > 1 else t.unsqueeze(0)

    x_flat = _flatten(x.detach())
    u_flat = _flatten(u)
    y_flat = _flatten(y)

    B = x_flat.shape[0]
    state_dim = y_flat.shape[-1]
    from torch.autograd.functional import jacobian

    def per_item(i: int) -> torch.Tensor:
        xi = x_flat[i]
        ui = u_flat[i]

        def g(u_local: torch.Tensor) -> torch.Tensor:
            return dynamics(xi, u_local, **dynamics_kwargs)

        J = jacobian(g, ui, vectorize=True, create_graph=create_graph, strict=True)
        return J.reshape(state_dim, -1)

    Js = [per_item(i) for i in range(B)]
    J_flat = torch.stack(Js, dim=0)  # (B, state_dim, control_dim)

    batch_shape = y.shape[:-1]
    control_dim = J_flat.shape[-1]
    dfdu = J_flat.reshape(*batch_shape, state_dim, control_dim)
    return y, dfdu


def clampControls(controls, controlBounds):
    controls[:, 0] = torch.clamp(controls[:, 0], controlBounds[0], controlBounds[1])
    controls[:, 1] = torch.clamp(controls[:, 1], controlBounds[2], controlBounds[3])


def runDynamicOptimizer(
    system,
    nextState,
    childrenStatesArray,
    childrenControlsArray,
    controlDuration,
    numStates=1000,
    maxDistance=0.025,
    posSTD=0.003,
    rotSTD=0.05,
    numSteps=10,
    learningRate=0.1,
    plateau_factor=0.5,
    plateau_patience=2,
    plateau_min_lr=1e-6,
):
    """
    Optimizer specifically for dynamic systems like Dublin airplane using PyTorch-based optimization.
    """
    print(f"[INFO] Dynamic Optimizer thread created")
    print(f"     - Next state: {nextState[0]:.5f}, {nextState[1]:.5f}, {nextState[2]:.5f}")
    print(f"     - Number of children states: {len(childrenStatesArray)}")
    print(f"     - Number of children controls: {len(childrenControlsArray)}")
    print(f"     - Number of sampled states: {numStates} (maxDistance: {maxDistance})")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sampledStates = torch.tensor(
            np.array(sampleRandomState(system, nextState, numStates, posSTD, rotSTD)),
            dtype=torch.float64,
            device=device,
        )

        targetStates = torch.tensor(
            np.array(childrenStatesArray), dtype=torch.float64, device=device
        ).repeat_interleave(numStates, dim=0)

        base_controls = np.array(childrenControlsArray)
        repeated_controls = np.repeat(base_controls, numStates, axis=0)

        initialControls = torch.tensor(
            repeated_controls, requires_grad=True, dtype=torch.float64, device=device
        )

        sampled_states_np = sampleRandomState(system, nextState, numStates, posSTD, rotSTD)
        repeated_states_np = np.repeat(sampled_states_np, len(childrenStatesArray), axis=0)
        startStates = torch.tensor(repeated_states_np, dtype=torch.float64, device=device)

        test_loss = torch.sum(initialControls**2)
        test_loss.backward()

        initialControls.grad.zero_()

        def compute_loss(pred, target):
            return torch.nn.functional.mse_loss(pred, target)

        optimizer = torch.optim.Adam([initialControls], lr=learningRate)

        # Add plateau regulator to automatically reduce learning rate when loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # Monitor loss (minimize)
            factor=plateau_factor,  # Reduce LR by half when plateauing
            patience=plateau_patience,  # Wait 2 steps before reducing LR
            min_lr=plateau_min_lr,  # Minimum learning rate
        )

        # Track loss history for plateau detection
        loss_history = []
        lr_history = []

        for step in range(numSteps):
            optimizer.zero_grad(set_to_none=True)

            predictedStates = carDynamicsTorch(
                startStates, initialControls, duration=controlDuration
            )

            current_loss = compute_loss(predictedStates, targetStates)
            current_loss.backward()
            optimizer.step()

            # Update scheduler with current loss
            scheduler.step(current_loss)
            loss_history.append(current_loss.item())
            lr_history.append(optimizer.param_groups[0]["lr"])

            # Print progress every few steps
            if step % max(1, numSteps // 5) == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"     - Step {step}: Loss = {current_loss.item():.6f}, LR = {current_lr:.6f}"
                )

        print(f"     - Final Loss: {current_loss}")
        print(f"     - Final Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(
            f"     - Loss history: {[f'{l:.6f}' for l in loss_history[:5]]}..."
        )  # Show first 5 losses

        # Create result dictionary
        result = {
            "optimized_controls": initialControls,
            "start_states": startStates,
            "target_states": targetStates,
            "optimization_success": True,
            "loss_history": loss_history,
            "learning_rate_history": lr_history,
            "step_history": list(range(numSteps)),
        }

        return result

    except Exception as e:
        print(f"[ERROR] Error in dynamic optimizer: {e}")
        import traceback

        traceback.print_exc()
        return None


def createOptimizerThread(
    system,
    nextState,
    childrenStates,
    childrenControls,
    controlDuration,
    optModel,
    numStates=1000,
    maxDistance=0.025,
    posSTD=0.003,
    rotSTD=0.05,
    numSteps=10,
    learningRate=0.1,
    plateau_factor=0.5,
    plateau_patience=2,
    plateau_min_lr=1e-6,
):
    resultContainer = {"result": None, "completed": False}

    # Choose optimizer based on system type
    if system == "simple_car":

        def optimizer_wrapper():
            try:
                result = runDynamicOptimizer(
                    system,
                    nextState,
                    childrenStates,
                    childrenControls,
                    controlDuration,
                    numStates,
                    maxDistance,
                    posSTD,
                    rotSTD,
                    numSteps,
                    learningRate,
                    plateau_factor,
                    plateau_patience,
                    plateau_min_lr,
                )
                resultContainer["result"] = result
                resultContainer["completed"] = True
                print(f"[INFO] Simple car optimizer completed with result: {result is not None}")
            except Exception as e:
                print(f"[ERROR] Error in simple car optimizer: {e}")
                import traceback

                traceback.print_exc()
                resultContainer["result"] = None
                resultContainer["completed"] = True

        thread = threading.Thread(target=optimizer_wrapper)

    elif system == "dublin_airplane":

        def optimizer_wrapper():
            try:
                result = runDynamicOptimizer(
                    system,
                    nextState,
                    childrenStates,
                    childrenControls,
                    controlDuration,
                    numStates,
                    maxDistance,
                    posSTD,
                    rotSTD,
                    numSteps,
                    learningRate,
                    plateau_factor,
                    plateau_patience,
                    plateau_min_lr,
                )
                resultContainer["result"] = result
                resultContainer["completed"] = True
                print(
                    f"[INFO] Dublin airplane optimizer completed with result: {result is not None}"
                )
            except Exception as e:
                print(f"[ERROR] Error in dublin airplane optimizer: {e}")
                import traceback

                traceback.print_exc()
                resultContainer["result"] = result
                resultContainer["completed"] = True

        thread = threading.Thread(target=optimizer_wrapper)

    elif system == "pushing":

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
                resultContainer["result"] = result
                resultContainer["completed"] = True
                print(f"[INFO] Pushing optimizer completed with result: {result is not None}")
            except Exception as e:
                print(f"[ERROR] Error in pushing optimizer: {e}")
                import traceback

                traceback.print_exc()
                resultContainer["result"] = None
                resultContainer["completed"] = True

        thread = threading.Thread(target=optimizer_wrapper)

    else:
        print(f"[ERROR] Unknown system type: {system}")
        resultContainer["result"] = None
        resultContainer["completed"] = True

    thread.daemon = True
    thread.resultContainer = resultContainer

    # Debug: Print initial state
    print(f"[DEBUG] Created optimizer thread for system: {system}")
    print(f"[DEBUG] Initial resultContainer: {resultContainer}")

    return thread
