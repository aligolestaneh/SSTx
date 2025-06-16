import time
import torch
import numpy as np
from tqdm import tqdm

from ik import IK
from geometry.pose import Pose
from sim_network import SimClient
from train_model import load_model, load_opt_model, load_opt_model_2
from geometry.random_push import (
    get_random_push,
    generate_push_params,
    generate_path_form_params,
)


def main(obj_name):
    # Need to run sim_network.py first
    # Sim class
    client = SimClient()
    # IK solver - Expansion GRR
    ik = IK("ur10_rod")

    # Initial state parameters
    n_envs, dt, robot_base_pose = client.execute("get_sim_info")
    # assume all objects start the same
    init_state = Pose([0, -0.7, 0.73])
    client.execute("set_obj_init_poses", [0, [init_state]])
    _, _, obj_rob_pos, obj_rob_quat, obj_shape = client.execute(
        "get_obj_info", 0
    )
    obj_pose = Pose(obj_rob_pos[0], obj_rob_quat[0])
    tool_offset = Pose([0, 0, -0.02])

    # Generate random push waypoints
    n_data = 100
    trials_per_round = n_envs  # this should not be larger than n_envs
    assert n_data < n_envs or n_data % n_envs == 0
    data_x = np.load(f"data/x_{obj_name}.npy")[:n_data]
    data_y = np.load(f"data/y_{obj_name}.npy")[:n_data]
    backward_y = np.zeros((n_data, 3))

    # Load the forward model
    forward_model = load_model("residual", 3, 3)
    forward_model.load(f"saved_models/crackerBoxBait9000.pth")
    forward_model = forward_model.model
    # forward_model.eval()

    # Load the backward model
    backward_model = load_model("mlp", 3, 3)
    backward_model.load(f"saved_models/crackerBoxMLP_backward.pth")
    backward_model = backward_model.model
    backward_model.eval()

    model = load_opt_model(forward_model, backward_model, epochs=1000)

    for i in tqdm(range(int(n_data // n_envs))):
        params = []
        pos_waypoints = []
        client.execute("reset")

        for k in range(trials_per_round):

            target_y = data_y[
                i * trials_per_round : (i + 1) * trials_per_round
            ]
            target_y[:, 0] += np.clip(
                np.random.normal(0, 0.005), -0.005, 0.005
            )
            target_y[:, 1] += np.clip(
                np.random.normal(0, 0.005), -0.005, 0.005
            )
            target_y[:, 2] += np.clip(np.random.normal(0, 0.1), -0.1, 0.1)

            start_guess = torch.tensor(
                data_x[i * trials_per_round : (i + 1) * trials_per_round],
                dtype=torch.float32,
            )
            x_pred = model.predict(
                target_y,
                # start_guess,
                x_min=[0.0, -0.4, 0.0],
                x_max=[4.0, 0.4, 0.3],
            )

            times, ws_path = generate_path_form_params(
                obj_pose, obj_shape, x_pred[k], tool_offset
            )

            traj = ik.ws_path_to_traj(Pose(), times, ws_path)
            waypoints = traj.to_step_waypoints(dt)

            params.append(x_pred[k])
            pos_waypoints.append(waypoints[0])

        # Stack (num_time_step, trials, robot_dof) and send it to Sim to execute
        params = np.array(params)
        pos_waypoints = np.stack(pos_waypoints, axis=1)
        pose = client.execute("execute_waypoints", pos_waypoints)
        # input(pose)

        # Predict
        # Convert params to tensor
        device = model.device
        params_tensor = torch.tensor(params, dtype=torch.float32).to(device)

        # Get the predictions

        # Save data
        backward_y[i * trials_per_round : (i + 1) * trials_per_round] = pose

    np.save(f"data/backward_y_{obj_name}.npy", backward_y)

    # Report the difference between data_y and pred_y in x, y and theta
    diff = data_y - backward_y
    print(f"Difference in x: {np.mean(np.abs(diff[:, 0]))}")
    print(f"Difference in y: {np.mean(np.abs(diff[:, 1]))}")
    print(f"Difference in theta: {np.mean(np.abs(diff[:, 2]))}")
    client.close()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    obj_name = "cracker_box"
    main(obj_name)
