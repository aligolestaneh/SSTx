import time
import torch
import numpy as np
from tqdm import tqdm

from ik import IK
from geometry.pose import Pose
from sim_network import SimClient
from train_model import load_model
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
    n_data = 10000
    trials_per_round = n_envs  # this should not be larger than n_envs
    assert n_data < n_envs or n_data % n_envs == 0
    data_x = np.zeros((n_data, 3))
    data_y = np.zeros((n_data, 3))
    pred_y = np.zeros((n_data, 3))

    # Load model
    model = load_model("residual", 3, 3)
    model.load(f"saved_models/crackerBoxBait9000.pth")
    model = model.model
    model.eval()

    for i in tqdm(range(int(n_data // n_envs))):
        params = []
        pos_waypoints = []
        client.execute("reset")

        for _ in range(trials_per_round):
            random_push_params, times, ws_path = get_random_push(
                obj_pose, obj_shape, tool_offset
            )

            traj = ik.ws_path_to_traj(Pose(), times, ws_path)
            waypoints = traj.to_step_waypoints(dt)

            params.append(random_push_params)
            pos_waypoints.append(waypoints[0])

        # Stack (num_time_step, trials, robot_dof) and send it to Sim to execute
        params = np.array(params)
        pos_waypoints = np.stack(pos_waypoints, axis=1)
        pose = client.execute("execute_waypoints", pos_waypoints)

        # Predict
        # Convert params to tensor
        device = next(model.parameters()).device
        params_tensor = torch.tensor(params, dtype=torch.float32).to(device)

        # Check the time it takes to predict
        start_time = time.time()
        pred_pose = model(params_tensor)
        end_time = time.time()
        print(f"Time taken to predict: {end_time - start_time} seconds")

        # Save data
        data_x[i * trials_per_round : (i + 1) * trials_per_round] = params
        data_y[i * trials_per_round : (i + 1) * trials_per_round] = pose
        pred_y[i * trials_per_round : (i + 1) * trials_per_round] = (
            pred_pose.detach().cpu().numpy()
        )
    np.save(f"data/x_{obj_name}.npy", data_x)
    np.save(f"data/y_{obj_name}.npy", data_y)
    np.save(f"data/pred_y_{obj_name}.npy", pred_y)

    # Report the difference between data_y and pred_y in x, y and theta
    diff = data_y - pred_y
    print(f"Difference in x: {np.mean(diff[:, 0])}")
    print(f"Difference in y: {np.mean(diff[:, 1])}")
    print(f"Difference in theta: {np.mean(diff[:, 2])}")
    client.close()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    obj_name = "cracker_box"
    main(obj_name)
