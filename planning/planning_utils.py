import torch
import numpy as np

from ompl import base as ob
from ompl import util as ou
from ompl import control as oc

from geometry.pose import SE2Pose
from active_learning.kernel import get_posteriors

ou.setLogLevel(ou.LOG_NONE)


class ControlCountObjective(ob.OptimizationObjective):
    def __init__(self, si, cost_per_control=1.0):
        super().__init__(si)
        self.cost_per_control = cost_per_control

    def stateCost(self, s):
        return ob.Cost(0.0)

    def motionCost(self, s1, s2):
        # penalize each control application
        return ob.Cost(self.cost_per_control)


class BoxPropagator:
    """Class to handle box state propagation with physics model."""

    def __init__(self, model, obj_shape):
        """Initialize propagator with object dimensions."""
        self.model = model
        self.obj_shape = obj_shape

    def propagate(self, start, control, duration, state):
        """Propagate the box state given control inputs."""
        x = start.getX()
        y = start.getY()
        theta = start.getYaw()
        initial_pose = SE2Pose(np.array([x, y]), theta)

        # Move tensor to the same device as the model
        device = next(self.model.parameters()).device
        control_tensor = torch.tensor(
            [[float(control[0]), float(control[1]), float(control[2])]],
            dtype=torch.float32,
        ).to(device)

        # Get the predicted output from the model
        output = self.model(control_tensor)
        delta = SE2Pose(
            np.array(
                [
                    output[0, 0].detach().cpu().numpy(),
                    output[0, 1].detach().cpu().numpy(),
                ]
            ),
            output[0, 2].detach().cpu().numpy(),
        )

        # Get the final pose by applying the delta to the initial pose
        final_pose = initial_pose @ delta
        state.setX(final_pose.position[0])
        state.setY(final_pose.position[1])
        state.setYaw(final_pose.euler[2])


class ControlSampler(oc.ControlSampler):
    def __init__(self, space, obj_shape, control_list=None):
        super().__init__(space)
        self.control_space = space
        self.control_list = control_list
        self.obj_shape = obj_shape

    def sample(self, control):
        # Pick a control randomly from the predefined list if given
        if self.control_list is not None and len(self.control_list) > 0:
            idx = np.random.choice(range(len(self.control_list)))
            chosen = self.control_list[idx]
            control[0] = chosen[0]
            control[1] = chosen[1]
            control[2] = chosen[2]

        # Regular sampling
        else:
            control[0] = np.random.uniform(
                self.control_space.getBounds().low[0],
                self.control_space.getBounds().high[0],
            )
            control[1] = np.random.uniform(
                self.control_space.getBounds().low[1],
                self.control_space.getBounds().high[1],
            )
            control[2] = np.random.uniform(
                self.control_space.getBounds().low[2],
                self.control_space.getBounds().high[2],
            )

            # Convert to absolute control values
            control[0] = int(control[0]) * np.pi / 2
            # mask = (control[0] % np.pi) == 0
            # control[1] *= np.where(mask, self.obj_shape[1], self.obj_shape[0])


class ActiveControlSampler(oc.ControlSampler):
    def __init__(self, space, model, x_train, obj_shape, control_list=None):
        super().__init__(space)
        self.control_space = space
        self.model = model
        self.x_train = x_train
        self.obj_shape = obj_shape
        self.control_list = control_list
        self.rng = ou.RNG()

    def sample(self, control):
        pool_size = 5

        # Pick a pool of controls randomly from the predefined list if given
        if self.control_list is not None and len(self.control_list) > 0:
            indices = np.random.choice(
                range(len(self.control_list)), pool_size
            )
            x_pool = np.array(self.control_list)[indices]

        # Regular active sampling
        else:
            pool = np.hstack(
                (
                    np.random.uniform(
                        self.control_space.getBounds().low[0],
                        self.control_space.getBounds().high[0],
                        (pool_size, 1),
                    ),
                    np.random.uniform(
                        self.control_space.getBounds().low[1],
                        self.control_space.getBounds().high[1],
                        (pool_size, 1),
                    ),
                    np.random.uniform(
                        self.control_space.getBounds().low[2],
                        self.control_space.getBounds().high[2],
                        (pool_size, 1),
                    ),
                )
            )
            # Convert to absolute control values
            x_pool = pool.copy()
            x_pool[:, 0] = x_pool[:, 0].astype(int) * np.pi / 2
            mask = (x_pool[:, 0] % np.pi) == 0
            x_pool[:, 1] *= np.where(
                mask, self.obj_shape[1], self.obj_shape[0]
            )

        # Active selection
        variances = get_posteriors(
            self.model, self.x_train, x_pool, sigma=1e-2
        )

        # Given the variances, assign weights to each control
        # Bias towards controls with lower variances
        weights = 1.0 / (variances + 1e-6)
        weights = weights / np.sum(weights)
        # Sample from the pool with the weights
        best_controls = x_pool[np.random.choice(range(len(x_pool)), p=weights)]

        # Get the best control
        # best_controls = x_pool[np.argmin(variances)]
        control[0] = best_controls[0]
        control[1] = best_controls[1]
        control[2] = best_controls[2]


class GraspableRegion(ob.GoalSampleableRegion):

    def __init__(self, si: ob.SpaceInformation, goal, obj_shape, edge):
        super().__init__(si)
        self.goal_point = goal
        self.obj_shape = obj_shape
        self.edge = edge

    def setThreshold(self, threshold):
        self.threshold = threshold

    def distanceGoal(self, state: ob.State) -> float:
        x = state.getX()
        y = state.getY()
        yaw = state.getYaw()
        corners = np.array(
            [
                [-self.obj_shape[0] / 2, -self.obj_shape[1] / 2, 1],
                [-self.obj_shape[0] / 2, +self.obj_shape[1] / 2, 1],
                [+self.obj_shape[0] / 2, -self.obj_shape[1] / 2, 1],
                [+self.obj_shape[0] / 2, +self.obj_shape[1] / 2, 1],
            ]
        )
        homogenous_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), x],
                [np.sin(yaw), np.cos(yaw), y],
                [0, 0, 1],
            ]
        )
        rotated_corners = np.dot(homogenous_matrix, corners.T)
        max_x = np.max(rotated_corners[0, :])
        if max_x - self.edge > 0.05 and x < self.edge:
            return 0
        else:
            x_dist = x - self.goal_point[0]
            y_dist = y - self.goal_point[1]
            yaw_dist = yaw - self.goal_point[2]
            yaw_dist = (yaw_dist + np.pi) % (2 * np.pi) - np.pi

            dist = np.sqrt(x_dist**2 + y_dist**2 + yaw_dist**2)
            # dist = np.sqrt(x_dist**2 + y_dist**2)

            return dist

    def sampleGoal(self, state: ob.State):
        state.setX(self.goal_point[0])
        state.setY(self.goal_point[1])
        state.setYaw(self.goal_point[2])

    def maxSampleCount(self) -> int:
        return 1

    def couldSample(self) -> bool:
        return True


def isStateValid(spaceInformation, state):
    """Check if state is valid (within bounds and collision-free)."""
    return spaceInformation.satisfiesBounds(state)


def get_combined_objective(
    si, cost_per_control=1.0, weight_path_length=1.0, weight_control_count=1.0
):
    """
    Return a combined multi optimization objective that considers
    both path length and control count
    """
    obj = ob.MultiOptimizationObjective(si)

    # Path length
    path_length_obj = ob.PathLengthOptimizationObjective(si)
    obj.addObjective(path_length_obj, weight_path_length)

    # Control count
    control_count_obj = ControlCountObjective(si, cost_per_control)
    obj.addObjective(control_count_obj, weight_control_count)

    return obj


def isSuccess(pose, obj_shape, edge):
    x = pose.position[0]
    y = pose.position[1]
    yaw = pose.euler[2]
    corners = np.array(
        [
            [-obj_shape[0] / 2, -obj_shape[1] / 2, 1],
            [-obj_shape[0] / 2, +obj_shape[1] / 2, 1],
            [+obj_shape[0] / 2, -obj_shape[1] / 2, 1],
            [+obj_shape[0] / 2, +obj_shape[1] / 2, 1],
        ]
    )
    homogenous_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), x],
            [np.sin(yaw), np.cos(yaw), y],
            [0, 0, 1],
        ]
    )
    rotated_corners = np.dot(homogenous_matrix, corners.T)
    # print(rotated_corners[0, :])
    max_x = np.max(rotated_corners[0, :])
    if max_x - edge >= 0.025 and x < edge:
        return 1
    else:
        return 0
