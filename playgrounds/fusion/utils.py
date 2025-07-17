import matplotlib.pyplot as plt
import numpy as np

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


def verify_resolve_correctness(planner):
    """Verify that all remaining motions are descendants of the second motion in solution path

    Args:
        planner: The OMPL planner instance

    Returns:
        bool: True if verification passes, False otherwise
    """
    print("\n=== VERIFYING RESOLVE CORRECTNESS ===")

    # Get planner data
    planner_data = ob.PlannerData(planner.getSpaceInformation())
    planner.getPlannerData(planner_data)

    # Get solution path
    try:
        solution_path = planner.getProblemDefinition().getSolutionPath()
        if not solution_path or solution_path.getStateCount() < 2:
            print("❌ No valid solution path available")
            return False
    except:
        print("❌ Could not get solution path")
        return False

    # Find the second state in solution path
    second_state = solution_path.getState(1)
    print(
        f"🎯 Second state: x={second_state.getX():.3f}, y={second_state.getY():.3f}, θ={second_state.getYaw():.3f}"
    )

    # Get all vertices from planner data
    all_vertices = []
    all_states = []
    for i in range(planner_data.numVertices()):
        vertex = planner_data.getVertex(i)
        state = vertex.getState()
        all_vertices.append(i)
        all_states.append((state.getX(), state.getY(), state.getYaw()))

    print(f"📊 Total remaining vertices: {len(all_vertices)}")

    # Find the vertex corresponding to the second motion
    second_vertex_idx = None
    min_distance = float("inf")
    for i, (x, y, theta) in enumerate(all_states):
        dist = (
            (x - second_state.getX()) ** 2
            + (y - second_state.getY()) ** 2
            + (theta - second_state.getYaw()) ** 2
        ) ** 0.5
        if dist < min_distance:
            min_distance = dist
            second_vertex_idx = i

    if second_vertex_idx is None or min_distance > 1e-6:
        print(
            f"❌ Could not find second motion in tree (min_dist: {min_distance})"
        )
        return False

    print(f"✅ Found second motion at vertex index {second_vertex_idx}")

    # Build parent-child relationships from edges
    children_map = {}  # vertex_id -> list of children
    parent_map = {}  # vertex_id -> parent

    for i in range(planner_data.numVertices()):
        children_map[i] = []

    # Extract edges to build the tree structure
    for i in range(planner_data.numVertices()):
        try:
            child_indices = []
            planner_data.getEdges(i, child_indices)
            for child in child_indices:
                children_map[i].append(child)
                parent_map[child] = i
        except:
            # Try alternative edge extraction method
            try:
                for j in range(planner_data.numVertices()):
                    if planner_data.edgeExists(i, j):
                        children_map[i].append(j)
                        parent_map[j] = i
            except:
                pass

    # Find all descendants of the second vertex using BFS
    descendants = set()
    queue = [second_vertex_idx]
    descendants.add(second_vertex_idx)

    while queue:
        current = queue.pop(0)
        for child in children_map.get(current, []):
            if child not in descendants:
                descendants.add(child)
                queue.append(child)

    print(f"🌳 Descendants of second motion: {len(descendants)}")

    # Check if all remaining vertices are descendants of the second motion
    all_vertices_set = set(all_vertices)
    unexpected_vertices = all_vertices_set - descendants

    print(f"📈 Verification Results:")
    print(f"   - Total remaining vertices: {len(all_vertices_set)}")
    print(f"   - Expected descendants: {len(descendants)}")
    print(f"   - Unexpected vertices: {len(unexpected_vertices)}")

    if len(unexpected_vertices) == 0:
        print(
            "✅ VERIFICATION PASSED: All remaining motions are descendants of second motion!"
        )
        return True
    else:
        print("❌ VERIFICATION FAILED: Found unexpected vertices:")
        for vertex_idx in list(unexpected_vertices)[:5]:  # Show first 5
            x, y, theta = all_states[vertex_idx]
            print(
                f"   - Vertex {vertex_idx}: x={x:.3f}, y={y:.3f}, θ={theta:.3f}"
            )
        if len(unexpected_vertices) > 5:
            print(f"   - ... and {len(unexpected_vertices) - 5} more")
        return False
