#!/usr/bin/env python3

import numpy as np
from functools import partial

# Import OMPL modules
from ompl import base as ob
from ompl import control as oc
from ompl import util as ou

# Set OMPL log level
ou.setLogLevel(ou.LogLevel.LOG_INFO)


def get_children_states(planner, ss, target_state, tolerance=1e-6):
    """
    Get children states of a given target state in the planner's search tree.

    Args:
        planner: OMPL planner object (e.g., Fusion, SST, etc.)
        ss: SimpleSetup object
        target_state: List or numpy array [x, y, yaw] representing the target state
        tolerance: Tolerance for state comparison (default: 1e-6)

    Returns:
        List of child states as [[x1, y1, yaw1], [x2, y2, yaw2], ...]
        Returns empty list if target state not found or no children exist
    """
    # Convert target_state to list if it's a numpy array
    if isinstance(target_state, np.ndarray):
        target_state = target_state.tolist()

    # Create PlannerData object to extract tree information
    planner_data = ob.PlannerData(ss.getSpaceInformation())
    planner.getPlannerData(planner_data)

    num_vertices = planner_data.numVertices()

    # Find the vertex index that matches our target state
    target_vertex_idx = None
    for i in range(num_vertices):
        vertex = planner_data.getVertex(i)
        state = vertex.getState()

        state_coords = [state.getX(), state.getY(), state.getYaw()]
        if _states_are_equal(state_coords, target_state, tolerance):
            target_vertex_idx = i
            break

    if target_vertex_idx is None:
        print(f"State {target_state} not found in planner tree")
        return []

    # Get children of the target vertex
    child_vertex_indices = ou.vectorUint()
    planner_data.getEdges(target_vertex_idx, child_vertex_indices)

    children_states = []
    for child_vertex_idx in child_vertex_indices:
        child_vertex = planner_data.getVertex(child_vertex_idx)
        child_state = child_vertex.getState()
        children_states.append(child_state.tolist())

    return children_states


def _states_are_equal(state1, state2, tolerance=1e-6):
    """Helper function to check if two states are approximately equal."""
    diff_x = abs(state1[0] - state2[0])
    diff_y = abs(state1[1] - state2[1])
    diff_yaw = abs(state1[2] - state2[2])

    return diff_x < tolerance and diff_y < tolerance and diff_yaw < tolerance


def get_all_states(planner, ss):
    """
    Get all states in the planner's search tree.

    Args:
        planner: OMPL planner object
        ss: SimpleSetup object

    Returns:
        List of all states as [[x1, y1, yaw1], [x2, y2, yaw2], ...]
    """
    planner_data = ob.PlannerData(ss.getSpaceInformation())
    planner.getPlannerData(planner_data)

    all_states = []
    for i in range(planner_data.numVertices()):
        vertex = planner_data.getVertex(i)
        state = vertex.getState()

        state_coords = [state.getX(), state.getY(), state.getYaw()]
        all_states.append(state_coords)

    return all_states


def get_tree_info(planner, ss):
    """
    Get basic information about the planner's search tree.

    Args:
        planner: OMPL planner object
        ss: SimpleSetup object

    Returns:
        Dictionary with tree information: {'num_vertices', 'num_edges'}
    """
    planner_data = ob.PlannerData(ss.getSpaceInformation())
    planner.getPlannerData(planner_data)

    return {
        "num_vertices": planner_data.numVertices(),
        "num_edges": planner_data.numEdges(),
    }


def simple_propagate(start, control, duration, result):
    """Simple propagation function for testing."""
    # Get current state values
    x = start.getX()
    y = start.getY()
    yaw = start.getYaw()

    # Simple motion model: move forward with some rotation
    # control[0] = forward velocity, control[1] = angular velocity
    dt = duration

    new_x = x + control[0] * np.cos(yaw) * dt
    new_y = y + control[0] * np.sin(yaw) * dt
    new_yaw = yaw + control[1] * dt

    # Set result state
    result.setX(new_x)
    result.setY(new_y)
    result.setYaw(new_yaw)


def is_state_valid(state):
    """Simple state validity checker."""
    x = state.getX()
    y = state.getY()

    # Keep states within bounds
    return -2.0 <= x <= 2.0 and -2.0 <= y <= 2.0


def setup_and_solve_simple_problem():
    """Set up and solve a simple planning problem."""
    print("Setting up simple planning problem...")

    # Create state space (SE2 - x, y, theta)
    space = ob.SE2StateSpace()

    # Set bounds for the state space
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, -2.0)  # x bounds
    bounds.setHigh(0, 2.0)
    bounds.setLow(1, -2.0)  # y bounds
    bounds.setHigh(1, 2.0)
    space.setBounds(bounds)

    # Create control space (2D - forward velocity, angular velocity)
    cspace = oc.RealVectorControlSpace(space, 2)

    # Set bounds for controls
    cbounds = ob.RealVectorBounds(2)
    cbounds.setLow(0, -1.0)  # forward velocity
    cbounds.setHigh(0, 1.0)
    cbounds.setLow(1, -1.0)  # angular velocity
    cbounds.setHigh(1, 1.0)
    cspace.setBounds(cbounds)

    # Create simple setup
    ss = oc.SimpleSetup(cspace)

    # Set state validity checker
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))

    # Set state propagator
    ss.setStatePropagator(oc.StatePropagatorFn(simple_propagate))

    # Create start state
    start = ob.State(space)
    start().setX(-1.0)
    start().setY(-1.0)
    start().setYaw(0.0)
    ss.setStartState(start)

    # Create goal state
    goal = ob.State(space)
    goal().setX(1.0)
    goal().setY(1.0)
    goal().setYaw(0.0)

    # Set goal with some tolerance
    ss.setGoalState(goal, 0.1)

    # Use Fusion planner
    planner = oc.Fusion(ss.getSpaceInformation())
    planner.setPruningRadius(0.2)
    ss.setPlanner(planner)

    print("Solving planning problem...")

    # Solve the problem
    solved = ss.solve(5.0)  # 5 seconds planning time

    if solved:
        print("✓ Solution found!")
        solution_path = ss.getSolutionPath()
        print(f"Solution has {solution_path.getStateCount()} states")

        # Print first few states of the solution
        print("\nSolution path (first 3 states):")
        for i in range(min(3, solution_path.getStateCount())):
            state = solution_path.getState(i)
            print(
                f"  State {i}: x={state.getX():.3f}, y={state.getY():.3f}, yaw={state.getYaw():.3f}"
            )

        return ss, planner, True
    else:
        print("✗ No solution found")
        return ss, planner, False


def test_get_children():
    """Test the getChildrenOfState function."""
    print("=" * 60)
    print("TESTING GET CHILDREN FUNCTION")
    print("=" * 60)

    # Set up and solve problem
    ss, planner, solved = setup_and_solve_simple_problem()

    if not solved:
        print("Cannot test children function - no solution found")
        return

    print("\n" + "=" * 40)
    print("TESTING CHILDREN FUNCTION")
    print("=" * 40)

    # Get the solution path to have some known states
    solution_path = ss.getSolutionPath()

    # Test with the start state (should have children)
    if solution_path.getStateCount() > 0:
        start_state = solution_path.getState(0)
        start_coords = [
            start_state.getX(),
            start_state.getY(),
            start_state.getYaw(),
        ]

        print(f"\nTesting children of start state: {start_coords}")
        children = get_children_states(planner, ss, start_coords)

        if children:
            print(f"Success! Found {len(children)} children:")
            for i, child in enumerate(children):
                print(f"  Child {i+1}: {child}")
        else:
            print("No children found for start state")

    # Test with a middle state if available
    if solution_path.getStateCount() > 2:
        mid_idx = solution_path.getStateCount() // 2
        mid_state = solution_path.getState(mid_idx)
        mid_coords = [mid_state.getX(), mid_state.getY(), mid_state.getYaw()]

        print(f"\nTesting children of middle state: {mid_coords}")
        children = get_children_states(planner, ss, mid_coords)

        if children:
            print(f"Success! Found {len(children)} children:")
            for i, child in enumerate(children):
                print(f"  Child {i+1}: {child}")
        else:
            print("No children found for middle state")

    # Test with a non-existent state
    fake_state = [99.0, 99.0, 0.0]
    print(f"\nTesting children of non-existent state: {fake_state}")
    children = get_children_states(planner, ss, fake_state)


if __name__ == "__main__":
    test_get_children()
