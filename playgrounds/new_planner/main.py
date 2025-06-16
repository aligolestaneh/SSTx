import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Add the root project directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import torch
import FUSION as fusion
from train_model import load_model, load_opt_model, load_opt_model_2
from geometry.pose import SE2Pose


class Colors:
    INFO = "\033[92m"  # Green
    WARNING = "\033[93m"  # Yellow
    ERROR = "\033[91m"  # Red
    DEBUG = "\033[94m"  # Blue
    HEADER = "\033[95m"  # Magenta
    BOLD = "\033[1m"  # Bold
    ENDC = "\033[0m"  # End color


def main(
    startPos,
    goalPos,
    objShape,
    solveTime,
    goalTolerance,
    spaceBounds,
    controlBounds,
    propagatorName,
    pruningRadius,
    optimizerName,
    numTrials,
    learningRate,
    numEpochs,
):
    # Load the model
    if propagatorName == "forward_model":
        propagator = load_model("residual", 3, 3)
        propagator.load(f"../../saved_models/crackerBoxBait9000.pth")
        propagator = propagator.model
        propagator.eval()
        print("[INFO] Forward model loaded.")
    else:
        raise ValueError(f"Propagator {propagatorName} not supported")

    numSuccess = 0
    fusionErrorList = []
    naiveErrorList = []
    for i in range(numTrials):
        ss = plan(
            startPos,
            goalPos,
            objShape,
            solveTime,
            goalTolerance,
            spaceBounds,
            controlBounds,
            propagator,
            pruningRadius,
        )
        fusionError, naiveError = execute(
            ss, propagator, optimizerName, objShape, learningRate, numEpochs
        )
        fusionErrorList.append(fusionError)
        naiveErrorList.append(naiveError)
        if fusionError < naiveError:
            numSuccess += 1

    success_rate = numSuccess / numTrials
    print(f"{Colors.INFO}[INFO] Success rate: {success_rate:.2%}")

    if len(fusionErrorList) > 0:
        print(
            f"{Colors.INFO}[INFO] Fusion error: {np.mean(fusionErrorList):.4f}"
        )
        print(
            f"{Colors.INFO}[INFO] Naive error: {np.mean(naiveErrorList):.4f}"
        )
    else:
        print(
            f"{Colors.WARNING}[WARNING] No successful trials! Could not calculate mean errors. This might indicate that the planning or execution is failing."
        )


def plan(
    startPos,
    goalPos,
    objShape,
    solveTime,
    goalTolerance,
    spaceBounds,
    controlBounds,
    propagator,
    pruningRadius,
):

    # Run the planning process
    start_time = time.time()
    ss = fusion.run_planner(
        start_pos=startPos,
        goal_pos=goalPos,
        obj_shape=objShape,
        solve_time=solveTime,
        goal_tolerance=goalTolerance,
        space_bounds=spaceBounds,
        control_bounds=controlBounds,
        python_propagator=propagator,
        pruning_radius=pruningRadius,
    )
    end_time = time.time()
    print(f"[INFO] Planning time: {end_time - start_time} seconds")

    # Get the solution path
    start_time = time.time()
    soluationPath = fusion.get_path(ss)
    soluationControls = fusion.get_controls(ss)
    end_time = time.time()
    print(f"[INFO] Get path time: {end_time - start_time} seconds")

    # Approve the final state
    # valid = input(
    #     f"[INFO] Is final state {SE2Pose(soluationPath[-1])} valid? "
    # )
    # if valid == "y":
    #     print("[INFO] Final state is valid\n")
    #     # Print the solution path
    #     print(f"[INFO] Solution path: \n{np.round(soluationPath, 4)}")
    #     print(f"--------------------------------\n")
    # else:
    #     print("[INFO] Final state is invalid\n")
    #     exit(0)

    # print the solution path and control
    print(f"[INFO] Solution path:")
    for i in range(len(soluationPath)):
        print(f"  {i}: {np.round(soluationPath[i], 4)}")
    print(f"[INFO] Solution control:")
    for i in range(len(soluationControls)):
        print(f"  {i}: {np.round(soluationControls[i], 4)}")
    print("--------------------------------")
    return ss


def execute(ss, propagator, optimizerName, objShape, learningRate, numEpochs):
    # Start executing the fusion model
    # input("Press Enter to start executing the fusion model...")

    if optimizerName == "forward_backward":
        # Load the backward model
        backwardModel = load_model("mlp", 3, 3)
        backwardModel.load(f"../../saved_models/crackerBoxMLP_backward.pth")
        backwardModel = backwardModel.model
        backwardModel.eval()
        optModel = load_opt_model(propagator, backwardModel, epochs=1000)
    elif optimizerName == "forward_optimizer":
        optModel = load_opt_model_2(
            propagator, lr=learningRate, epochs=numEpochs
        )
    else:
        raise ValueError(f"Optimizer {optimizerName} not supported")

    soluationPath = fusion.get_path(ss)
    soluationControls = fusion.get_controls(ss)

    # Helpers for storing the data
    relativePoseList = []
    closestNodeList = []
    numClosestNodeList = []
    startGuessList = []
    device = next(propagator.parameters()).device

    # Get the nearest neighbor of the solution path
    start_time = time.time()
    for i in range(len(soluationPath) - 1):
        node = soluationPath[i]
        nextNode = soluationPath[i + 1]
        nextPose = SE2Pose(
            nextNode[:2], nextNode[2]
        )  # position=[x,y], rotation=yaw
        initialGuess = soluationControls[i]

        closestNodes = fusion.get_nearest_nodes(ss, node)

        # Generate additional nodes by sampling Gaussians around the current node
        num_additional_nodes = 10000
        additional_nodes = sample_random_nodes(node, num_additional_nodes)

        # Combine original closest nodes with additional sampled nodes
        all_closest_nodes = list(closestNodes) + additional_nodes
        numClosestNodeList.append(len(all_closest_nodes))

        # Convert all nodes to SE2Pose objects at once
        closest_poses = [
            SE2Pose(node[:2], node[2]) for node in all_closest_nodes
        ]

        # Calculate all relative poses at once
        relative_poses = [pose.invert @ nextPose for pose in closest_poses]

        # Convert relative poses to numpy arrays
        relative_poses_array = np.array(
            [
                [pose.position[0], pose.position[1], pose.euler[2]]
                for pose in relative_poses
            ]
        )

        # Add to lists
        closestNodeList.extend(all_closest_nodes)
        relativePoseList.extend(relative_poses_array)
        startGuessList.extend([initialGuess] * len(all_closest_nodes))

    end_time = time.time()
    print(
        f"[INFO] Generate nearest node and relative pose time: {end_time - start_time} seconds"
    )

    # Convert list of numpy arrays to single numpy array
    relativePoseArray = np.array(relativePoseList)
    closestNodeArray = np.array(closestNodeList)
    startGuessArray = np.array(startGuessList)

    # Solve the optimization problem
    start_time = time.time()
    if optimizerName == "forward_backward":
        relativeControls = optModel.predict(
            relativePoseArray,
            x_min=[0.0, -0.4, 0.0],
            x_max=[4.0, 0.4, 0.3],
        )
    elif optimizerName == "forward_optimizer":
        relativeControls, loss = optModel.predict(
            relativePoseArray,
            startGuessArray,
            # x_min=[0.0, -0.4, 0.0],
            # x_max=[4.0, 0.4, 0.3],
        )
    else:
        raise ValueError(f"Optimizer {optimizerName} not supported")

        # Plot the loss
        # plt.figure(figsize=(10, 5))
        # plt.plot(loss, 1000)
        # plt.title("Optimization Loss Over Time")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.grid(True)
    end_time = time.time()
    print(f"[INFO] Optimization time: {end_time - start_time} seconds")

    # Print the type of relativeControls and relativePoseArray
    print(f"[INFO] Relative controls: {relativeControls.shape}")
    print(f"[INFO] Relative poses: {relativePoseArray.shape}")
    print(f"[INFO] Start guess: {startGuessArray.shape}")
    print(f"[INFO] Relative poses numbers: {numClosestNodeList}\n")

    # Split arrays into batches based on numClosestNodeList
    splitIndices = np.cumsum(numClosestNodeList[:-1])
    controlBatches = np.split(relativeControls, splitIndices)
    closestNodeBatches = np.split(closestNodeArray, splitIndices)
    relativePoseBatches = np.split(relativePoseArray, splitIndices)
    startGuessBatches = np.split(startGuessArray, splitIndices)

    # Validate controls by checking if they result in states close to next states
    valid_controls = []
    valid_poses = []
    valid_start_guesses = []
    valid_closest_nodes = []
    current_idx = 0

    for i in range(len(soluationPath) - 1):
        # Get the i-th batch of control, pass through the propogator and get the delta, compare with relativePoseBatches[i]
        batch_controls = controlBatches[i]
        batch_start_guesses = startGuessBatches[i]
        # Convert to tensor before passing to propagator
        control_tensor = torch.tensor(
            batch_controls, device=device, dtype=torch.float32
        )
        batch_delta = propagator(control_tensor)
        batch_delta = SE2Pose(
            np.array(
                [
                    batch_delta[0, 0].detach().cpu().numpy(),
                    batch_delta[0, 1].detach().cpu().numpy(),
                ]
            ),
            batch_delta[0, 2].detach().cpu().numpy(),
        )
        batch_relativePose = [
            SE2Pose(relativePose[:2], relativePose[2])
            for relativePose in relativePoseBatches[i]
        ]

        # Calculate the distance between the batch_delta and the batch_relativePose using the distance function
        batch_distance = [
            batch_delta.distance(_relativePose)
            for _relativePose in batch_relativePose
        ]

        # Collect all indices to remove
        indices_to_remove = [
            j for j, dist in enumerate(batch_distance) if dist > 0.05
        ]

        # Remove all invalid controls at once
        if indices_to_remove:
            controlBatches[i] = np.delete(
                controlBatches[i], indices_to_remove, axis=0
            )
            relativePoseBatches[i] = np.delete(
                relativePoseBatches[i], indices_to_remove, axis=0
            )
            startGuessBatches[i] = np.delete(
                startGuessBatches[i], indices_to_remove, axis=0
            )
            closestNodeBatches[i] = np.delete(
                closestNodeBatches[i], indices_to_remove, axis=0
            )
            numClosestNodeList[i] -= len(indices_to_remove)

        print(
            f"[INFO] Removed {len(indices_to_remove)} controls for step {i + 1}"
        )

    # Convert lists back to numpy arrays
    # relativeControls = np.array(controlBatches)
    # closestNodeArray = np.array(closestNodeBatches)
    # startGuessArray = np.array(startGuessBatches)

    # Fix the first element of relativeControls to be close to one of the values in rotations for all the batches
    rotations = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    for i in range(len(controlBatches)):
        current_rotations = controlBatches[i][:, 0]
        diffs = np.abs(current_rotations[:, np.newaxis] - rotations)
        closest_indices = np.argmin(diffs, axis=1)
        controlBatches[i][:, 0] = rotations[closest_indices]

        # if relativeControls[i, 0] % np.pi == 0:
        #     relativeControls[i, 1] *= objShape[1]
        # else:
        #     relativeControls[i, 1] *= objShape[0]

    simulatedState = SE2Pose(soluationPath[0])
    simulatedFusion = SE2Pose(soluationPath[0])

    for i in range(len(soluationPath) - 1):
        if i == 0:
            control_tensor = torch.tensor(
                soluationControls[i], device=device
            ).unsqueeze(0)
            output = propagator(control_tensor)
            delta = SE2Pose(
                np.array(
                    [
                        output[0, 0].detach().cpu().numpy(),
                        output[0, 1].detach().cpu().numpy(),
                    ]
                ),
                output[0, 2].detach().cpu().numpy(),
            )
            simulatedFusion = simulatedFusion @ delta
            simulatedState = simulatedState @ delta
            print(
                f"[STEP {i + 1}] Simulated state (FUSION): {simulatedFusion}"
            )
            print(f"[STEP {i + 1}] Simulated state (NAIVE): {simulatedState}")
            noiseX = np.clip(np.random.normal(0, 0.01), -0.01, 0.01)
            noiseY = np.clip(np.random.normal(0, 0.01), -0.01, 0.01)
            noiseTheta = np.clip(np.random.normal(0, 0.05), -0.05, 0.05)
            simulatedFusion.position[0] += noiseX
            simulatedFusion.position[1] += noiseY
            simulatedFusion.euler[2] += noiseTheta
            simulatedState.position[0] += noiseX
            simulatedState.position[1] += noiseY
            simulatedState.euler[2] += noiseTheta
            print(
                f"[STEP {i + 1}] Simulated state (FUSION): {simulatedFusion}"
            )
            print(f"[STEP {i + 1}] Simulated state (NAIVE): {simulatedState}")
            print(f"----------------------------------------------------")
            continue

        # Get the current batch of validated controls
        current_controls = controlBatches[i]
        current_nodes = closestNodeBatches[i]
        current_start_guesses = startGuessBatches[i]

        if len(current_controls) == 0:
            print(f"[WARNING] No valid controls found for step {i + 1}")
            continue

        # Try each validated control and find the one that gets closest to the next node
        best_control_idx = 0
        min_next_distance = float("inf")

        for idx, control in enumerate(current_controls):
            # Apply the control to current state
            control_tensor = torch.tensor(
                control, device=device, dtype=torch.float32
            ).unsqueeze(0)
            output = propagator(control_tensor)
            delta = SE2Pose(
                np.array(
                    [
                        output[0, 0].detach().cpu().numpy(),
                        output[0, 1].detach().cpu().numpy(),
                    ]
                ),
                output[0, 2].detach().cpu().numpy(),
            )
            # Calculate resulting state
            resulting_state = simulatedFusion @ delta
            # Calculate distance to next node in solution path
            next_node = SE2Pose(
                soluationPath[i + 1][:2], soluationPath[i + 1][2]
            )
            distance_to_next = resulting_state.distance(
                next_node, angular_weight=0.5
            )

            if distance_to_next < min_next_distance:
                min_next_distance = distance_to_next
                best_control_idx = idx

        # Use the best control found
        best_control = current_controls[best_control_idx]
        best_node = current_nodes[best_control_idx]

        print(
            f"For node {simulatedFusion}, best node is {SE2Pose(best_node[:2], best_node[2])} with distance {min_next_distance:.4f} "
            f"and control {np.round(best_control, 4)} -- initial guess: {np.round(current_start_guesses[best_control_idx], 4)}"
        )

        # Get the optimal control with FUSION method
        control_tensor = torch.tensor(
            best_control, device=device, dtype=torch.float32
        ).unsqueeze(0)
        outputFusion = propagator(control_tensor)
        deltaFusion = SE2Pose(
            np.array(
                [
                    outputFusion[0, 0].detach().cpu().numpy(),
                    outputFusion[0, 1].detach().cpu().numpy(),
                ]
            ),
            outputFusion[0, 2].detach().cpu().numpy(),
        )
        simulatedFusion = simulatedFusion @ deltaFusion
        print(f"[STEP {i + 1}] Simulated state (FUSION): {simulatedFusion}")

        # Get the regular control
        control_tensor = torch.tensor(
            soluationControls[i], device=device, dtype=torch.float32
        ).unsqueeze(0)
        outputNoise = propagator(control_tensor)
        deltaNoise = SE2Pose(
            np.array(
                [
                    outputNoise[0, 0].detach().cpu().numpy(),
                    outputNoise[0, 1].detach().cpu().numpy(),
                ]
            ),
            outputNoise[0, 2].detach().cpu().numpy(),
        )
        simulatedState = simulatedState @ deltaNoise
        print(f"[STEP {i + 1}] Simulated state (NAIVE): {simulatedState}")

        # Add noise to the simulated state
        noiseX = np.clip(np.random.normal(0, 0.01), -0.01, 0.01)
        noiseY = np.clip(np.random.normal(0, 0.01), -0.01, 0.01)
        noiseTheta = np.clip(np.random.normal(0, 0.05), -0.05, 0.05)
        simulatedFusion.position[0] += noiseX
        simulatedFusion.position[1] += noiseY
        simulatedFusion.euler[2] += noiseTheta
        simulatedState.position[0] += noiseX
        simulatedState.position[1] += noiseY
        simulatedState.euler[2] += noiseTheta
        print(f"[STEP {i + 1}] Simulated state (FUSION): {simulatedFusion}")
        print(f"[STEP {i + 1}] Simulated state (NAIVE): {simulatedState}")
        print(f"----------------------------------------------------")

    print(
        f"{Colors.INFO}[INFO] Expected final state: {SE2Pose(soluationPath[-1])}"
    )
    print(
        f"{Colors.INFO}[INFO] Actual final state (FUSION):   {simulatedFusion}"
    )
    print(
        f"{Colors.INFO}[INFO] Actual final state (NAIVE):   {simulatedState}"
    )
    fusionError = simulatedFusion.distance(SE2Pose(soluationPath[-1]))
    naiveError = simulatedState.distance(SE2Pose(soluationPath[-1]))
    print(f"{Colors.INFO}[INFO] Total error (FUSION): {fusionError:.4f}")
    print(f"{Colors.INFO}[INFO] Total error (NAIVE): {naiveError:.4f}")
    print(f"{Colors.ENDC}")

    return fusionError, naiveError


def sample_random_nodes(node, num_nodes):
    # Generate additional nodes by sampling Gaussians around the current node
    additional_nodes = []

    # Gaussian sampling parameters
    pos_std = 0.003  # Standard deviation for position (x, y)
    angle_std = 0.05  # Standard deviation for angle (theta)

    for _ in range(num_nodes):
        # Sample position around current node with Gaussian noise
        noisy_x = node[0] + np.random.normal(0, pos_std)
        noisy_y = node[1] + np.random.normal(0, pos_std)
        noisy_theta = node[2] + np.random.normal(0, angle_std)

        # Normalize angle to [-π, π]
        while noisy_theta > np.pi:
            noisy_theta -= 2.0 * np.pi
        while noisy_theta < -np.pi:
            noisy_theta += 2.0 * np.pi

        additional_nodes.append([noisy_x, noisy_y, noisy_theta])

    return additional_nodes


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run the planner with specified parameters"
    )
    parser.add_argument(
        "--solve-time",
        type=float,
        default=5.0,
        help="Time limit for solving the planning problem (default: 5.0)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=20,
        help="Number of trials to run (default: 20)",
    )

    # Parse arguments
    args = parser.parse_args()

    _startPos = [0, -0.7, 0]
    _goalPos = [0.3, -0.7, 0]
    _objShape = [0.1628, 0.2139, 0.0676]
    _solveTime = args.solve_time
    _goalTolerance = 0.03
    _spaceBounds = [-0.9, 0.76, -0.9, -0.3]
    _controlBounds = [0, 4, -0.4, 0.4, 0, 0.3]
    _pruningRadius = 0.1
    _propagatorName = "forward_model"
    _optimizerName = "forward_optimizer"
    _learningRate = 1e-3
    _numEpochs = 200
    _numTrials = args.num_trials

    _stateSpace = main(
        startPos=_startPos,
        goalPos=_goalPos,
        objShape=_objShape,
        solveTime=_solveTime,
        goalTolerance=_goalTolerance,
        spaceBounds=_spaceBounds,
        controlBounds=_controlBounds,
        propagatorName=_propagatorName,
        pruningRadius=_pruningRadius,
        optimizerName=_optimizerName,
        numTrials=_numTrials,
        learningRate=_learningRate,
        numEpochs=_numEpochs,
    )
