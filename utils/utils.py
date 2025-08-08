import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


try:
    from ompl import base as ob
    from ompl import control as oc
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys

    sys.path.insert(
        0, join(dirname(dirname(abspath(__file__))), "py-bindings")
    )
    from ompl import base as ob
    from ompl import control as oc


def parse_command_line_args():
    """Parse command line arguments in key=value format"""
    args = {}

    # Default values
    defaults = {
        "plan_time": 10.0,
        "replan_time": 3.0,
        "dynamics_type": "model",
        "planner": "fusion",
    }

    # Parse command line arguments
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Convert to appropriate type
            if key in ["plan_time", "replan_time"]:
                try:
                    args[key] = float(value)
                except ValueError:
                    print(
                        f"Warning: Invalid float value for {key}: {value}. Using default."
                    )
                    args[key] = defaults[key]
            else:
                args[key] = value
        else:
            print(f"Warning: Ignoring invalid argument format: {arg}")

    # Fill in defaults for missing arguments
    for key, default_value in defaults.items():
        if key not in args:
            args[key] = default_value

    return args


def visualize_tree_3d(planner, filename="fusion_tree_3d.png", show_plot=True):
    """Visualize the tree structure in 3D (x, y, theta) using matplotlib

    Args:
        planner: The OMPL planner instance
        filename: Name of the file to save the plot (default: "fusion_tree_3d.png")
        show_plot: Whether to display the plot interactively (default: True)
    """
    from mpl_toolkits.mplot3d import Axes3D

    # Get planner data
    planner_data = ob.PlannerData(planner.getSpaceInformation())
    print("Getting planner data for 3D visualization...")
    planner.getPlannerData(planner_data)
    print("Planner data obtained")

    # Extract all vertices (x, y, theta)
    all_vertices = []
    print("Collecting vertices...")
    for i in range(planner_data.numVertices()):
        vertex = planner_data.getVertex(i)
        state = vertex.getState()
        # Extract SE2 state components
        x = state.getX()
        y = state.getY()
        theta = state.getYaw()
        all_vertices.append((x, y, theta))

    # Get solution path states
    solution_states = []
    try:
        solution_path = planner.getProblemDefinition().getSolutionPath()
        if solution_path:
            print("Extracting solution path...")
            for i in range(solution_path.getStateCount()):
                state = solution_path.getState(i)
                x = state.getX()
                y = state.getY()
                theta = state.getYaw()
                solution_states.append((x, y, theta))
    except:
        print("No solution path available")
        pass

    print("Creating 3D plot...")
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if all_vertices:
        # Convert to numpy arrays for easier plotting
        all_vertices_array = np.array(all_vertices)

        # Plot all tree vertices as small blue dots
        ax.scatter(
            all_vertices_array[:, 0],  # x
            all_vertices_array[:, 1],  # y
            all_vertices_array[:, 2],  # theta
            c="steelblue",
            s=20,
            alpha=0.3,  # More transparent to show density patterns
            label="Tree nodes",
        )

    # Plot solution path if available
    if solution_states:
        solution_array = np.array(solution_states)

        # Plot solution path vertices as larger red dots
        ax.scatter(
            solution_array[:, 0],  # x
            solution_array[:, 1],  # y
            solution_array[:, 2],  # theta
            c="red",
            s=60,
            alpha=0.9,
            label="Solution path nodes",
            marker="o",
            edgecolors="darkred",
            linewidth=2,
        )

        # Connect solution path states with lines
        ax.plot(
            solution_array[:, 0],  # x
            solution_array[:, 1],  # y
            solution_array[:, 2],  # theta
            color="orange",
            linewidth=3,
            alpha=0.8,
            label="Solution path",
        )

        # Mark start and goal specially
        if len(solution_states) > 0:
            # Start state (green square)
            start = solution_array[0]
            ax.scatter(
                start[0],
                start[1],
                start[2],
                c="green",
                s=100,
                marker="s",
                edgecolors="darkgreen",
                linewidth=2,
                label="Start state",
            )

            # Goal state (red star)
            goal = solution_array[-1]
            ax.scatter(
                goal[0],
                goal[1],
                goal[2],
                c="red",
                s=120,
                marker="*",
                edgecolors="darkred",
                linewidth=2,
                label="Goal state",
            )

    # Set labels and title
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Theta (radians)", fontsize=12)
    ax.set_title(
        "Fusion Planner Tree Visualization (3D: x, y, θ)",
        fontsize=14,
        fontweight="bold",
    )

    # Add legend
    ax.legend(loc="upper right", fontsize=10)

    # Set fixed axis limits for x, y, and theta to match state space bounds
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-np.pi, np.pi)  # Theta typically ranges from -π to π

    # Set equal aspect ratio for x and y
    ax.set_box_aspect([1, 1, 0.5])  # Make theta axis shorter for better view

    plt.tight_layout()

    print(f"Saving 3D tree to {filename}")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"3D tree saved to {filename}")

    if show_plot:
        plt.show(block=False)
    else:
        plt.close()


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


def state2list(state, state_type: str) -> list:

    if state_type.upper() == "SE2":
        # SE2 state: x, y, theta
        return [state.getX(), state.getY(), state.getYaw()]

    elif state_type.upper() == "SE3":
        # SE3 state: x, y, z, qw, qx, qy, qz (position + quaternion)
        return [
            state.getX(),
            state.getY(),
            state.getZ(),
            state.getRotation().w,
            state.getRotation().x,
            state.getRotation().y,
            state.getRotation().z,
        ]

    else:
        print(
            f"Warning: Unknown state type '{state_type}'. Returning empty list."
        )
        return []


def isSE2Equal(state1, state2, tolerance=1e-6):
    diff_x = abs(state1[0] - state2[0])
    diff_y = abs(state1[1] - state2[1])
    diff_yaw = abs(state1[2] - state2[2])

    return diff_x < tolerance and diff_y < tolerance and diff_yaw < tolerance


def arrayDistance(array1, array2, system: str):
    # Convert SE2Pose objects to arrays if needed
    if hasattr(array1, "flat"):  # SE2Pose object
        array1 = array1.flat
    if hasattr(array2, "flat"):  # SE2Pose object
        array2 = array2.flat

    if system == "SE2":
        # Calculate position distance
        pos_distance = np.sqrt(
            (array1[0] - array2[0]) ** 2 + (array1[1] - array2[1]) ** 2
        )

        # Calculate angular distance with proper wrapping
        # Compute the shortest angular difference using the standard formula
        yaw_diff = abs((array1[2] - array2[2] + np.pi) % (2 * np.pi) - np.pi)

        # Combine position and angular distances
        # Using a weighted combination similar to OMPL's SE2StateSpace
        return np.sqrt(pos_distance**2 + 0.5 * yaw_diff**2)
    elif system == "SE2Position":
        return np.sqrt(
            (array1[0] - array2[0]) ** 2 + (array1[1] - array2[1]) ** 2
        )
    else:
        raise ValueError(f"Invalid system: {system}")


def log(message, log_type="info"):
    colors = {
        "error": "\033[91m",  # Red
        "warning": "\033[93m",  # Yellow
        "info": "\033[0m",  # Default
        "success": "\033[92m",  # Green
    }

    reset = "\033[0m"
    color_code = colors.get(log_type.lower(), colors["info"])

    if log_type.lower() == "info":
        print(message)
    else:
        print(f"{color_code}{message}{reset}")
