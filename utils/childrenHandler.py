import numpy as np
from utils.utils import state2list, isStateEqual, arrayDistance, log

from ompl import base as ob
from ompl import util as ou
from ompl import control as oc


def getChildrenStates(ss, targetState, system="simple_car", tolerance=1e-6):
    """
    Extract children states and their corresponding controls from the OMPL planner tree.

    Args:
        ss: OMPL SimpleSetup object
        targetState: The target state to find children for
        system: The system type ("SE2" or "SE3") to determine state format
        tolerance: Tolerance for state comparison

    Returns:
        tuple: (children_states, children_controls)
    """
    print(f"[INFO] Getting children states for system: {system}")

    # Get planner data
    planner_data = oc.PlannerData(ss.getSpaceInformation())
    planner = ss.getPlanner()
    planner.getPlannerData(planner_data)

    num_vertices = planner_data.numVertices()
    print(f"[INFO] Planner tree has {num_vertices} vertices")

    if num_vertices == 0:
        log("[WARNING] Planner tree is empty", "warning")
        return [], []

    # Search for the target state
    targetVertexIdx = None
    print(f"[DEBUG] Searching for targetState: {targetState}")
    print(f"[DEBUG] First 5 planner vertices:")
    for i in range(min(5, num_vertices)):
        state = planner_data.getVertex(i).getState()
        state_list = state2list(state, system)
        # print(f"[DEBUG]   Vertex {i}: {state_list}")

    for i in range(num_vertices):
        state = planner_data.getVertex(i).getState()
        state_list = state2list(state, system)

        if isStateEqual(state_list, targetState, system, tolerance):
            targetVertexIdx = i
            print(f"[INFO] Found target state at vertex index: {i}")
            # print(f"   Target: {targetState}")
            # print(f"   Found:  {state_list}")
            break

    if targetVertexIdx is None:
        log(
            f"[WARNING] State {targetState} not found in planner tree",
            "warning",
        )
        # print(f"🔍 Checking first few vertices for debugging:")
        for i in range(min(5, num_vertices)):
            state = planner_data.getVertex(i).getState()
            state_list = state2list(state, system)
            # print(f"   Vertex {i}: {state_list}")

        # Also check if any vertex is close to the target
        # print(f"🔍 Checking for close matches (tolerance: {tolerance}):")
        min_distance = float("inf")
        closest_vertex = None
        for i in range(min(10, num_vertices)):
            state = planner_data.getVertex(i).getState()
            state_list = state2list(state, system)
            distance = arrayDistance(targetState, state_list, system)
            if distance < min_distance:
                min_distance = distance
                closest_vertex = (i, state_list)
            # print(f"   Vertex {i}: {state_list} (distance: {distance:.6f})")

        if closest_vertex:
            print(f"[INFO] Closest vertex: {closest_vertex[1]} (distance: {min_distance:.6f})")

        return [], []

    # print(f"🔍 Getting edges for vertex {targetVertexIdx}...")
    childVertexIndices = ou.vectorUint()
    planner_data.getEdges(targetVertexIdx, childVertexIndices)

    print(f"[INFO] Found {len(childVertexIndices)} child vertices")

    children_states = []
    children_controls = []
    control_space = ss.getControlSpace()
    control_dimension = control_space.getDimension()

    for childVertexIdx in childVertexIndices:
        childState = planner_data.getVertex(childVertexIdx).getState()
        child_state_list = state2list(childState, system)
        children_states.append(child_state_list)

        # Get the control that takes us from parent (targetState) to this child
        try:
            # Get the edge from parent to child using indices
            edge = planner_data.getEdge(targetVertexIdx, childVertexIdx)

            # Get control directly from the edge
            control = edge.getControl()
            control_values = [control[j] for j in range(control_dimension)]
            children_controls.append(control_values)
            # print(f"   Child {childVertexIdx}: Control: {control_values}")

        except Exception as e:
            print(f"   [WARNING] Could not get control for edge to child {childVertexIdx}: {e}")
            # Use a fallback control if the direct method fails
            fallback_control = [1.0, 0.0, 0.1]
            children_controls.append(fallback_control)

    print(
        f"[INFO] Returning {len(children_states)} children states and {len(children_controls)} controls"
    )
    return children_states, children_controls


def sampleRandomState(system, state, numStates=1000, posSTD=0.003, rotSTD=0.05):
    sampledStates = []

    if system == "simple_car" or system == "pushing":
        # Convert state to list if it's not already
        if hasattr(state, "getX"):  # It's an OMPL state object
            stateList = state2list(state, "SE2")
        else:  # It's already a list
            stateList = state
        for _ in range(numStates):
            noisyX = stateList[0] + np.random.normal(0, posSTD)
            noisyY = stateList[1] + np.random.normal(0, posSTD)
            noisyYaw = stateList[2] + np.random.normal(0, rotSTD)
            while noisyYaw > np.pi:
                noisyYaw -= 2 * np.pi
            while noisyYaw < -np.pi:
                noisyYaw += 2 * np.pi
            sampledStates.append([noisyX, noisyY, noisyYaw])

    elif system == "dublin_airplane":
        for _ in range(numStates):
            noisyX = stateList[0] + np.random.normal(0, posSTD)
            noisyY = stateList[1] + np.random.normal(0, posSTD)
            noisyZ = stateList[2] + np.random.normal(0, posSTD)
            noisyYaw = stateList[3] + np.random.normal(0, rotSTD)
            while noisyYaw > np.pi:
                noisyYaw -= 2 * np.pi
            while noisyYaw < -np.pi:
                noisyYaw += 2 * np.pi
            sampledStates.append([noisyX, noisyY, noisyZ, noisyYaw])
    return sampledStates
