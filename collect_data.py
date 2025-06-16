import numpy as np
from tqdm import tqdm
from geometry.pose import Pose
from ik import IK
from geometry.random_push import get_random_push
from sim_network import SimClient


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

    for i in tqdm(range(int(n_data // n_envs))):
        params = []
        pos_waypoints = []
        client.execute("reset")

        for _ in range(trials_per_round):
            push_params, times, ws_path = get_random_push(
                obj_pose, obj_shape, tool_offset
            )

            traj = ik.ws_path_to_traj(Pose(), times, ws_path)
            waypoints = traj.to_step_waypoints(dt)

            params.append(push_params)
            pos_waypoints.append(waypoints[0])

        # Stack (num_time_step, trials, robot_dof) and send it to Sim to execute
        params = np.array(params)
        pos_waypoints = np.stack(pos_waypoints, axis=1)
        pose = client.execute("execute_waypoints", pos_waypoints)

        # Save data
        data_x[i * trials_per_round : (i + 1) * trials_per_round] = params
        data_y[i * trials_per_round : (i + 1) * trials_per_round] = pose

    np.save(f"data/x_{obj_name}.npy", data_x)
    np.save(f"data/y_{obj_name}.npy", data_y)
    client.close()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    obj_name = "cracker_box"
    main(obj_name)
