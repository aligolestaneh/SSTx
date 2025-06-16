import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

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
        if fusionError < naiveError:
            numSuccess += 1
            fusionErrorList.append(fusionError)
            naiveErrorList.append(naiveError)
    print(f"{Colors.INFO}[INFO] Success rate: {numSuccess / numTrials}")
    print(f"{Colors.INFO}[INFO] Fusion error: {np.mean(fusionErrorList)}")
    print(f"{Colors.INFO}[INFO] Naive error: {np.mean(naiveErrorList)}")


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


# def executeNoise(ss, propagator):
#     # Start simulating the final pose using an gaussian noise
#     # input("Press Enter to start executing the noise model...")

#     soluationPath = fusion.get_path(ss)
#     soluationControls = fusion.get_controls(ss)
#     start_time = time.time()
#     simulatedState = SE2Pose(soluationPath[0])
#     print("\nTesting the execution error:")
#     for i, control in enumerate(soluationControls):
#         # Get the device of the model
#         device = next(propagator.parameters()).device

#         # Create tensor and move to same device as model
#         control_tensor = torch.tensor(control, device=device).unsqueeze(0)
#         output = propagator(control_tensor)
#         delta = SE2Pose(
#             np.array(
#                 [
#                     output[0, 0].detach().cpu().numpy(),
#                     output[0, 1].detach().cpu().numpy(),
#                 ]
#             ),
#             output[0, 2].detach().cpu().numpy(),
#         )
#         simulatedState = simulatedState @ delta
#         # add noise to the simulated state
#         print(f"[STEP {i}] Simulated state: {simulatedState}")
#         simulatedState.position[0] += np.clip(
#             np.random.normal(0, 0.003), -0.01, 0.01
#         )
#         simulatedState.position[1] += np.clip(
#             np.random.normal(0, 0.003), -0.01, 0.01
#         )
#         simulatedState.euler[2] += np.clip(
#             np.random.normal(0, 0.05), -0.05, 0.05
#         )
#         print(f"[STEP {i}] Simulated state: {simulatedState}")
#         print(f"[STEP {i}] Control: {np.round(control, 4)}")

#     print(
#         f"{Colors.INFO}\n[INFO] Expected final state: {SE2Pose(soluationPath[-1])}"
#     )
#     print(f"{Colors.INFO}[INFO] Actual final state:   {simulatedState}")
#     print(
#         f"{Colors.INFO}[INFO] Total error: {simulatedState.distance(SE2Pose(soluationPath[-1])):.4f}"
#     )
#     print(f"{Colors.ENDC}")
#     end_time = time.time()
#     print(f"[INFO] Testing time: {end_time - start_time} seconds")


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

        # Generate 100 additional nodes by sampling Gaussians around the current node
        num_additional_nodes = 15000
        additional_nodes = []
        additional_controls = []  # Store controls for additional nodes

        # Gaussian sampling parameters
        pos_std = 0.003  # Standard deviation for position (x, y)
        angle_std = 0.05  # Standard deviation for angle (theta)

        for _ in range(num_additional_nodes):
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

        # Combine original closest nodes with additional sampled nodes
        all_closest_nodes = list(closestNodes) + additional_nodes
        numClosestNodeList.append(len(all_closest_nodes))

        # print(f"[INFO] For node: {np.round(node, 4)}")
        for closestNode in all_closest_nodes:
            closestNodeList.append(closestNode)

            # Get the relative se2pose between the closest node and the next node
            closestPose = SE2Pose(
                closestNode[:2], closestNode[2]
            )  # position=[x,y], rotation=yaw

            relativePose = closestPose.invert @ nextPose
            relativePoseList.append(
                np.array(
                    [
                        relativePose.position[0],
                        relativePose.position[1],
                        relativePose.euler[2],
                    ]
                )
            )
            startGuessList.append(initialGuess)

            # print(f"      relative_pose: {np.round(relativePose, 4)}")

    end_time = time.time()
    print(f"[INFO] Get nearest node time: {end_time - start_time} seconds")

    # Solve the optimization problem
    start_time = time.time()
    # Convert list of numpy arrays to single numpy array to avoid PyTorch warning
    relativePoseArray = np.array(relativePoseList)
    closestNodeArray = np.array(closestNodeList)
    startGuessArray = np.array(startGuessList)
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

        # plot the loss
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
    print(f"[INFO] Relative controls: {relativeControls.shape}")
    print(f"[INFO] Start guess: {startGuessArray.shape}")
    print(f"[INFO] Relative poses numbers: {numClosestNodeList}\n")

    # Fix the first element of relativeControls to be close to one of the values in rotations
    rotations = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    for i in range(relativeControls.shape[0]):
        diff = np.abs(relativeControls[i, 0] - rotations)
        relativeControls[i, 0] = rotations[np.argmin(diff)]
        # if relativeControls[i, 0] % np.pi == 0:
        #     relativeControls[i, 1] *= objShape[1]
        # else:
        #     relativeControls[i, 1] *= objShape[0]

    # Split arrays into batches based on numClosestNodeList
    splitIndices = np.cumsum(numClosestNodeList[:-1])
    controlBatches = np.split(relativeControls, splitIndices)
    closestNodeBatches = np.split(closestNodeArray, splitIndices)
    relativePoseBatches = np.split(relativePoseArray, splitIndices)
    startGuessBatches = np.split(startGuessArray, splitIndices)

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

        distanceList = [
            simulatedFusion.distance(
                SE2Pose(pose[:2], pose[2]), angular_weight=0.5
            )
            for pose in closestNodeBatches[i]
        ]

        # Get indices of 10 closest nodes
        closest_indices = np.argsort(distanceList)[:5]
        closest_nodes = closestNodeBatches[i][closest_indices]
        closest_controls = controlBatches[i][closest_indices]

        # Try each control and find the one that gets closest to the next node
        best_control_idx = 0
        min_next_distance = float("inf")

        for idx, control in enumerate(closest_controls):
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
        best_control = closest_controls[best_control_idx]
        best_node = closest_nodes[best_control_idx]
        best_pose = SE2Pose(best_node[:2], best_node[2])

        print(
            f"For node {simulatedFusion}, best node is {best_pose} with distance {distanceList[closest_indices[best_control_idx]]:.4f} "
            f"and control {np.round(best_control, 4)} -- initial guess: {np.round(startGuessBatches[i][closest_indices[best_control_idx]], 4)}"
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


if __name__ == "__main__":

    _startPos = [0, -0.7, 0]
    _goalPos = [0.3, -0.7, 0]
    _objShape = [0.1628, 0.2139, 0.0676]
    _solveTime = 5
    _goalTolerance = 0.03
    _spaceBounds = [-0.9, 0.76, -0.9, -0.3]
    _controlBounds = [0, 4, -0.4, 0.4, 0, 0.3]
    _pruningRadius = 0.1
    _propagatorName = "forward_model"
    _optimizerName = "forward_optimizer"
    _learningRate = 1e-3
    _numEpochs = 200
    _numTrials = 20

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
