import time
import torch
import numpy as np
from tqdm import tqdm

from ik import IK
from geometry.pose import Pose, SE2Pose
from sim_network import SimClient
from train_model import load_model, load_opt_model
from geometry.random_push import (
    get_random_push,
    generate_push_params,
    generate_path_form_params,
)


def main(obj_name):
    # Load data
    n_data = 100
    data_x = np.load(f"data/x_{obj_name}.npy")[:n_data]
    data_y = np.load(f"data/y_{obj_name}.npy")[:n_data]

    # Create a hardcoded SE2Pose
    target_pose = SE2Pose(
        np.array([0.079, -0.733]), -0.472
    )  # x=0.079, y=-0.733, theta=-0.472
    print(f"Target pose: {target_pose}")

    # Calculate distances to all poses in data_y
    distances = []
    for pose_data in data_y:
        pose = SE2Pose(pose_data[:2], pose_data[2])
        dist = target_pose.distance(pose, angular_weight=0.5)
        distances.append((dist, pose_data))

    # Sort by distance and get top 10
    distances.sort(key=lambda x: x[0])
    print("\nTop 10 closest poses:")
    for i, (dist, pose_data) in enumerate(distances[:10]):
        print(f"{i+1}. Distance: {dist:.4f}, Pose: {pose_data}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    obj_name = "cracker_box"
    main(obj_name)
