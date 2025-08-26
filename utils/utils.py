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

    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), "py-bindings"))
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
                    print(f"Warning: Invalid float value for {key}: {value}. Using default.")
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
                draw_rectangle(state_x, state_y, w, l, state_theta, "b", alpha=0.3)
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


def draw_rectangle(x, y, width, length, theta, color="b", alpha=0.5, label=None):
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
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
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


def isStateValid(spaceInformation, state):
    """Check if state is valid (within bounds and collision-free)."""
    return spaceInformation.satisfiesBounds(state)


def state2list(state, state_type: str) -> list:

    if state_type == "simple_car":
        # SE2 state: x, y, theta
        return [state.getX(), state.getY(), state.getYaw()]

    elif state_type == "dublin_airplane":
        # SE3 state: x, y, z, qw, qx, qy, qz (position + quaternion)
        # For OMPL SE3 states, access the SO3 part using the rotation method
        try:
            # Try the standard OMPL SE3 state access
            return [
                state.getX(),
                state.getY(),
                state.getZ(),
                state.rotation().w,  # quaternion w component
                state.rotation().x,  # quaternion x component
                state.rotation().y,  # quaternion y component
                state.rotation().z,  # quaternion z component
            ]
        except AttributeError:
            # Fallback: try to access as compound state
            try:
                return [
                    state.getX(),
                    state.getY(),
                    state.getZ(),
                    state.rotation.w,  # quaternion w component
                    state.rotation.x,  # quaternion x component
                    state.rotation.y,  # quaternion y component
                    state.rotation.z,  # quaternion z component
                ]
            except AttributeError:
                print(f"Warning: Could not access SE3 state components for type {type(state)}")
                return []

    else:
        print(f"Warning: Unknown state type '{state_type}'. Returning empty list.")
        return []


def isSE2Equal(state1, state2, tolerance=1e-6):
    diff_x = abs(state1[0] - state2[0])
    diff_y = abs(state1[1] - state2[1])
    diff_yaw = abs(state1[2] - state2[2])

    return diff_x < tolerance and diff_y < tolerance and diff_yaw < tolerance


def isSE3Equal(state1, state2, tolerance=1e-6):
    """Compare two SE3 states for equality within tolerance."""
    if len(state1) < 7 or len(state2) < 7:
        return False

    # Compare position components
    pos_diff = np.sqrt(
        (state1[0] - state2[0]) ** 2 + (state1[1] - state2[1]) ** 2 + (state1[2] - state2[2]) ** 2
    )

    # Compare quaternion components (normalize first to handle sign ambiguity)
    quat1_norm = normalize_quaternion([state1[3], state1[4], state1[5], state1[6]])
    quat2_norm = normalize_quaternion([state2[3], state2[4], state2[5], state2[6]])

    quat_diff = np.sqrt(
        (quat1_norm[0] - quat2_norm[0]) ** 2
        + (quat1_norm[1] - quat2_norm[1]) ** 2
        + (quat1_norm[2] - quat2_norm[2]) ** 2
        + (quat1_norm[3] - quat2_norm[3]) ** 2
    )

    return pos_diff < tolerance and quat_diff < tolerance


def isStateEqual(state1, state2, system, tolerance=1e-6):
    """Generic state comparison function that handles different systems."""
    if system == "simple_car":
        return isSE2Equal(state1, state2, tolerance)
    elif system == "dublin_airplane":
        return isSE3Equal(state1, state2, tolerance)
    else:
        # Fallback to simple element-wise comparison
        if len(state1) != len(state2):
            return False
        return all(abs(a - b) < tolerance for a, b in zip(state1, state2))


def normalize_quaternion(quat):
    """Normalize quaternion to handle sign ambiguity."""
    quat = np.array(quat)
    # Normalize to unit length
    norm = np.linalg.norm(quat)
    if norm > 0:
        quat = quat / norm
    # Ensure consistent sign (make first non-zero component positive)
    for i in range(4):
        if abs(quat[i]) > 1e-10:
            if quat[i] < 0:
                quat = -quat
            break
    return quat


def arrayDistance(array1, array2, system: str):
    from ompl import base as ob

    if hasattr(array1, "flat"):
        array1 = array1.flat
    if hasattr(array2, "flat"):
        array2 = array2.flat

    if system == "dublin_airplane":
        # Check if arrays have enough elements for SE3
        if len(array1) < 7 or len(array2) < 7:
            raise ValueError(
                f"SE3 states need at least 7 elements, got {len(array1)} and {len(array2)}"
            )

        pos_distance = np.sqrt(
            (array1[0] - array2[0]) ** 2
            + (array1[1] - array2[1]) ** 2
            + (array1[2] - array2[2]) ** 2
        )

        # Input format is [qx, qy, qz, qw] at indices [3, 4, 5, 6]
        # Convert to [w, x, y, z] for OMPL
        quat1_norm = normalize_quaternion(
            [array1[6], array1[3], array1[4], array1[5]]
        )  # [w, x, y, z]
        quat2_norm = normalize_quaternion(
            [array2[6], array2[3], array2[4], array2[5]]
        )  # [w, x, y, z]

        # Create SO3 states and set quaternions
        so3_space = ob.SO3StateSpace()
        quat1 = so3_space.allocState()
        quat2 = so3_space.allocState()

        # quat1_norm is [w, x, y, z], map to OMPL SO3State
        quat1.w = quat1_norm[0]  # w component
        quat1.x = quat1_norm[1]  # x component
        quat1.y = quat1_norm[2]  # y component
        quat1.z = quat1_norm[3]  # z component

        quat2.w = quat2_norm[0]  # w component
        quat2.x = quat2_norm[1]  # x component
        quat2.y = quat2_norm[2]  # y component
        quat2.z = quat2_norm[3]  # z component

        rot_distance = 2.0 * so3_space.distance(quat1, quat2)
        return np.sqrt(pos_distance**2 + 0.5 * rot_distance**2)

    elif system == "simple_car" or system == "pushing":
        # Check if arrays have enough elements for SE2
        if len(array1) < 3 or len(array2) < 3:
            raise ValueError(
                f"SE2 states need at least 3 elements, got {len(array1)} and {len(array2)}"
            )

        posDistance = np.sqrt((array1[0] - array2[0]) ** 2 + (array1[1] - array2[1]) ** 2)
        yawDistance = abs((array1[2] - array2[2] + np.pi) % (2 * np.pi) - np.pi)
        return np.sqrt(posDistance**2 + 0.5 * yawDistance**2)

    elif system == "position":
        # Check if arrays have enough elements for SE2Position
        if len(array1) < 2 or len(array2) < 2:
            raise ValueError(
                f"SE2Position states need at least 2 elements, got {len(array1)} and {len(array2)}"
            )

        posDistance = np.sqrt((array1[0] - array2[0]) ** 2 + (array1[1] - array2[1]) ** 2)
        return posDistance

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


def printState(state, system, situation):
    """Print the state in a readable format."""
    if system == "simple_car" or system == "pushing":
        print(
            f"       - {situation} State: x={state[0]:.3f}, y={state[1]:.3f}, theta={state[2]:.3f}"
        )

    elif system == "dublin_airplane":
        if len(state) == 7:
            print(
                f"       - {situation} State: x={state[0]:.3f}, y={state[1]:.3f}, z={state[2]:.3f}, "
                f"quat=[{state[3]:.3f}, {state[4]:.3f}, {state[5]:.3f}, {state[6]:.3f}]"
            )
        else:
            print(
                f"       - {situation} State: x={state[0]:.3f}, y={state[1]:.3f}, z={state[2]:.3f}, "
                f"psi={state[3]:.3f}, gamma={state[4]:.3f}, phi={state[5]:.3f}"
            )

    elif system == "control":
        print(f"       - {situation} Control: {state}")

    else:
        print(f"       - {situation} State: {state}")


def addNoise(system, state, pos_std, rot_std):
    if system == "simple_car":
        state[0] = state[0] + np.clip(np.random.normal(0, pos_std), -pos_std, pos_std)
        state[1] = state[1] + np.clip(np.random.normal(0, pos_std), -pos_std, pos_std)
        state[2] = state[2] + np.clip(np.random.normal(0, rot_std), -rot_std, rot_std)

    elif system == "dublin_airplane":
        state[0] = state[0] + np.clip(np.random.normal(0, pos_std), -pos_std, pos_std)
        state[1] = state[1] + np.clip(np.random.normal(0, pos_std), -pos_std, pos_std)
        state[2] = state[2] + np.clip(np.random.normal(0, pos_std), -pos_std, pos_std)
        state[3] = state[3] + np.clip(np.random.normal(0, rot_std), -rot_std, rot_std)
        state[4] = state[4] + np.clip(np.random.normal(0, rot_std), -rot_std, rot_std)
        state[5] = state[5] + np.clip(np.random.normal(0, rot_std), -rot_std, rot_std)

    return state
