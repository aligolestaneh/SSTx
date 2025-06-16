import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


class DataLoader:
    """Class for loading data and splitting them"""

    def __init__(
        self,
        object_name: str,
        folder: str = "data",
        val_size: int = 1000,
        invert_xy: bool = False,
        shuffle: bool = False,
    ):
        """Initialize with data files, split sizes, and some options"""
        self.data_x_file = folder + "/x_" + object_name + ".npy"
        self.data_y_file = folder + "/y_" + object_name + ".npy"
        self.val_size = val_size

        # Load data
        x = np.load(self.data_x_file).astype(np.float32)
        y = np.load(self.data_y_file).astype(np.float32)
        # Pre-process data
        if invert_xy:
            x, y = y, x
        if shuffle:
            idx = np.random.permutation(len(x))
            x, y = x[idx], y[idx]
        self.x = x
        self.y = y

        # Check
        self.pool_size = len(self.x) - self.val_size
        if self.pool_size <= 0:
            raise ValueError("Pool size is 0")

    def load_data(self, verbose=1):
        """Load all data as a dictionary"""
        # Split data

        x_pool = self.x[: self.pool_size]
        y_pool = self.y[: self.pool_size]
        x_val = self.x[self.pool_size : self.pool_size + self.val_size]
        y_val = self.y[self.pool_size : self.pool_size + self.val_size]

        if verbose:
            print("Loading data")
            print(f"Pool data points: {x_pool.shape[0]}")
            print(f"Validation data points: {x_val.shape[0]}")

        datasets = dict()
        datasets["x_pool"] = x_pool
        datasets["y_pool"] = y_pool
        datasets["x_val"] = x_val
        datasets["y_val"] = y_val
        return datasets


def plot_states(states, planned_states=None, obj_shape=None):
    """Plot the states of the object."""
    states = np.array(states)
    if planned_states is not None:
        planned_states = np.array(planned_states)

    plt.figure(figsize=(8, 6))
    # Plot the table as the background
    draw_rectangle(0, -0.505, 1.524, 1.524, 0, "k", alpha=0.1)
    # Plot a robot
    draw_rectangle(0, 0, 0.2, 0.2, 0, "gray", alpha=1.0, label="Robot")

    # Plot the states path
    if planned_states is not None:
        plt.plot(
            planned_states[:, 0],
            planned_states[:, 1],
            "o-",
            color="b",
            label="Planned Path",
        )
    plt.plot(
        states[:, 0],
        states[:, 1],
        "o-",
        color="g",
        label="Actual Path",
    )
    # If object shape is provided, draw rectangles for start and goal
    if obj_shape is not None:
        w, l = obj_shape[0], obj_shape[1]

        # Draw rectangles
        if planned_states is not None:
            for state in planned_states:
                state_x, state_y, state_theta = state
                draw_rectangle(
                    state_x, state_y, w, l, state_theta, "b", alpha=0.3
                )
        for state in states:
            state_x, state_y, state_theta = state
            draw_rectangle(state_x, state_y, w, l, state_theta, "g", alpha=0.3)

    # Plot start positions
    plt.plot(states[0, 0], states[0, 1], "ro", label="Start")

    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Push Path")
    plt.legend()
    plt.show()


def draw_rectangle(
    x, y, width, length, theta, color="b", alpha=0.5, label=None
):
    """Draw a rectangle at the given position with the given orientation."""
    # Calculate the four corners of the rectangle
    corners = np.array(
        [
            [-width / 2, -length / 2],
            [width / 2, -length / 2],
            [width / 2, length / 2],
            [-width / 2, length / 2],
            [-width / 2, -length / 2],  # Close the rectangle
        ]
    )

    # Rotate the corners
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    rotated_corners = np.dot(corners, rot_matrix.T)
    # Translate the corners
    translated_corners = rotated_corners + np.array([x, y])

    # Plot the rectangle
    plt.plot(
        translated_corners[:, 0],
        translated_corners[:, 1],
        color=color,
        alpha=alpha,
    )
    plt.fill(
        translated_corners[:, 0],
        translated_corners[:, 1],
        color=color,
        alpha=alpha,
    )

    # Add text label
    if label is not None:
        plt.text(x, y, label, ha="center", va="center", color="black")
