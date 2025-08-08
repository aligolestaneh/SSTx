import numpy as np
from utils.utils import state2list, isSE2Equal, arrayDistance, log

from ompl import base as ob
from ompl import util as ou
from ompl import control as oc


def getChildrenStates(ss, targetState, tolerance=1e-6):
    print(
        f"\n[INFO] getChildrenStates called with targetState: {targetState[0]:.5f}, {targetState[1]:.5f}, {targetState[2]:.5f}"
    )

    # Get planner data using control-specific PlannerData
    from ompl import control as oc

    si = ss.getSpaceInformation()
    planner_data = oc.PlannerData(si)
    ss.getPlanner().getPlannerData(planner_data)

    # Check if planner data has control information
    has_controls = planner_data.hasControls()
    print(f"[INFO] Planner data has controls: {has_controls}")

    num_vertices = planner_data.numVertices()
    # print(f"📊 Total vertices in planner tree: {num_vertices}")

    targetVertexIdx = None
    # print(f"🔍 Searching for target state in planner tree...")

    # Search for the target state
    for i in range(num_vertices):
        state = planner_data.getVertex(i).getState()
        state_list = state2list(state, "SE2")

        if isSE2Equal(state_list, targetState, tolerance):
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
            state_list = state2list(state, "SE2")
            # print(f"   Vertex {i}: {state_list}")

        # Also check if any vertex is close to the target
        # print(f"🔍 Checking for close matches (tolerance: {tolerance}):")
        min_distance = float("inf")
        closest_vertex = None
        for i in range(min(10, num_vertices)):
            state = planner_data.getVertex(i).getState()
            state_list = state2list(state, "SE2")
            distance = arrayDistance(targetState, state_list, "SE2")
            if distance < min_distance:
                min_distance = distance
                closest_vertex = (i, state_list)
            # print(f"   Vertex {i}: {state_list} (distance: {distance:.6f})")

        if closest_vertex:
            print(
                f"[INFO] Closest vertex: {closest_vertex[1]} (distance: {min_distance:.6f})"
            )

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
        child_state_list = state2list(childState, "SE2")
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
            print(
                f"   [WARNING] Could not get control for edge to child {childVertexIdx}: {e}"
            )
            # Use a fallback control if the direct method fails
            fallback_control = [1.0, 0.0, 0.1]
            children_controls.append(fallback_control)

    print(
        f"[INFO] Returning {len(children_states)} children states and {len(children_controls)} controls"
    )
    return children_states, children_controls


def sampleRandomState(state, numStates=1000, posSTD=0.003, rotSTD=0.05):
    sampledStates = []
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
    return sampledStates
