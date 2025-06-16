import numpy as np
from expansion_grr.bullet_api.loader import load_grr
from geometry.pose import Pose
from geometry.trajectory import SplineTrajectory


class IK:
    """A IK package that uses Expansion GRR that
    converts workspace path to robot joint trajectory
    """

    def __init__(self, robot: str):
        """Initialize with a robot name"""
        self.grr = load_grr(robot, "rot_variable_yaw")

    def solve(self, target: np.ndarray, current: np.ndarray | None = None):
        """Solve IK for target point"""
        return self.grr.solve(target, current)

    def ws_path_to_traj(
        self, robot_base_pose: Pose, t_path: np.ndarray, ws_path: np.ndarray
    ):
        """Convert a workspace path with time stamps to a trajectory"""
        # Convert the work space path to robot frame
        ws_local = np.array(
            [
                (robot_base_pose.invert @ Pose(p[:3], p[3:])).flat
                for p in ws_path
            ]
        )
        # Pose uses quaternion in wxyz format while GRR uses xyzw format
        ws_local = ws_local[:, [0, 1, 2, 4, 5, 6, 3]]

        # Generate Configuration Path with GRR
        # keep Giving the Last Solution as a Reference
        c_path = [self.solve(ws_local[0])] * len(ws_local)
        for c_i in range(1, len(ws_local)):
            c_path[c_i] = self.solve(ws_local[c_i], c_path[c_i - 1])

        # Convert to Spline Trajectory
        trajectory = SplineTrajectory(np.array(c_path), np.array(t_path))
        return trajectory
