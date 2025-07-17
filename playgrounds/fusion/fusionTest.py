from functools import partial
import sys

from utils import visualize_tree_3d, verify_resolve_correctness
from train_model import load_model
from light_planning_utils import (
    isStateValid,
    propagate_simple,
    propagate_complex,
    propagate_unstable,
    propagate_pendulum,
    BoxPropagator,
)

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


def plan(planning_time=20.0, dynamics_type="complex", replanning_time=2.0):
    # construct the state space we are planning in
    space = ob.SE2StateSpace()

    # set the bounds for the R^2 part of SE(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, -0.9)
    bounds.setHigh(0, 0.76)
    bounds.setLow(1, -0.9)
    bounds.setHigh(1, -0.3)
    space.setBounds(bounds)

    # create a control space
    cspace = oc.RealVectorControlSpace(space, 2)

    # set the bounds for the control space
    cspace = oc.RealVectorControlSpace(space, 3)
    cbounds = ob.RealVectorBounds(3)
    cbounds.setLow(0, 0)  # minimum rotation
    cbounds.setHigh(0, 4)  # maximum rotation
    cbounds.setLow(1, -0.4)  # minimum side offset
    cbounds.setHigh(1, 0.4)  # maximum side offset
    cbounds.setLow(2, 0.0)  # minimum push distance
    cbounds.setHigh(2, 0.3)  # maximum push distance
    cspace.setBounds(cbounds)

    # define a simple setup class
    ss = oc.SimpleSetup(cspace)
    ss.setStateValidityChecker(
        ob.StateValidityCheckerFn(
            partial(isStateValid, ss.getSpaceInformation())
        )
    )

    # Choose propagate function based on parameter
    if dynamics_type == "simple":
        ss.setStatePropagator(oc.StatePropagatorFn(propagate_simple))
    elif dynamics_type == "complex":
        ss.setStatePropagator(oc.StatePropagatorFn(propagate_complex))
    elif dynamics_type == "unstable":
        ss.setStatePropagator(oc.StatePropagatorFn(propagate_unstable))
    elif dynamics_type == "pendulum":
        ss.setStatePropagator(oc.StatePropagatorFn(propagate_pendulum))
    elif dynamics_type == "model":
        model = load_model("residual", 3, 3)
        model.load(f"results/models/model.pth")
        model = model.model  # use torch model directly
        model.eval()
        propagator = BoxPropagator(model, (0.1, 0.1, 0.05))
        ss.setStatePropagator(oc.StatePropagatorFn(propagator.propagate))
    else:
        print(f"Unknown dynamics type: {dynamics_type}. Using complex.")
        ss.setStatePropagator(oc.StatePropagatorFn(propagate_complex))
        dynamics_type = "complex"

    print(f"Using {dynamics_type} dynamics model")
    print(f"Replanning time: {replanning_time} seconds")

    # Set control duration properly
    # si = ss.getSpaceInformation()

    # Set control duration to 1-5 time steps (proper integer values)
    # si.setMinMaxControlDuration(1, 5)
    # print(f"✓ Set control duration to 1 time step")

    # Verify current settings
    # min_dur = si.getMinControlDuration()
    # max_dur = si.getMaxControlDuration()
    # print(f"Control duration range: {min_dur}-{max_dur} steps")

    # create a start state
    start = ob.State(space)
    start().setX(-0.5)
    start().setY(0.0)
    start().setYaw(0.0)

    # create a goal state
    goal = ob.State(space)
    goal().setX(0.0)
    goal().setY(0.5)
    goal().setYaw(0.0)

    # set the start and goal states
    ss.setStartAndGoalStates(start, goal, 0.1)

    # Set propagation step size before setting up planner
    # si.setPropagationStepSize(0.1)
    # print(f"✓ Set propagation step size to 0.1")

    # (optionally) set planner
    # planner = oc.RRT(si)
    # planner = oc.EST(si)
    # planner = oc.KPIECE1(si) # this is the default
    # SyclopEST and SyclopRRT require a decomposition to guide the search
    # decomp = MyDecomposition(32, bounds)
    # planner = oc.Fusion(ss.getSpaceInformation())
    planner = oc.SST(ss.getSpaceInformation())
    # Set larger pruning radius for sparser tree
    planner.setPruningRadius(0.1)
    # planner = oc.SyclopRRT(si, decomp)
    ss.setPlanner(planner)

    # attempt to solve the problem
    solved = ss.solve(planning_time)

    # Show 3D visualization of the tree
    # print("Creating 3D visualization...")
    visualize_tree_3d(planner, filename=f"fusion_3d_{planning_time}s.png")
    # input("Press Enter to continue...")

    if solved:
        # print the path to screen
        print("Found solution:\n%s" % ss.getSolutionPath().printAsMatrix())

        print("Initial solution found")

        # Interactive resolve loop - continue until goal reached or user stops
        print(
            f"\nStarting interactive resolve with {replanning_time}s time limit per iteration..."
        )

        # Goal coordinates for distance checking
        goal_x, goal_y = 0.0, 0.5  # From goal state definition above

        resolve_iter = 0
        if replanning_time > 0:
            while True:
                resolve_iter += 1
                print(f"\n{'='*60}")
                print(f"RESOLVE ITERATION {resolve_iter}")
                print(f"{'='*60}")

                # Check current state before resolve
                try:
                    current_solution = ss.getSolutionPath()
                    if (
                        current_solution
                        and current_solution.getStateCount() > 0
                    ):
                        # Get the last state in current solution (where we are now)
                        last_state = current_solution.getState(
                            current_solution.getStateCount() - 1
                        )
                        try:
                            # Try to extract current position
                            current_x = last_state.getX()
                            current_y = last_state.getY()
                            current_yaw = last_state.getYaw()
                        except AttributeError:
                            try:
                                # Try compound state access
                                se2_comp = last_state[0]
                                current_x = se2_comp.getX()
                                current_y = se2_comp.getY()
                                current_yaw = se2_comp.getYaw()
                            except:
                                current_x, current_y, current_yaw = (
                                    0.0,
                                    0.0,
                                    0.0,
                                )

                        # Calculate distance to goal
                        distance_to_goal = (
                            (current_x - goal_x) ** 2
                            + (current_y - goal_y) ** 2
                        ) ** 0.5

                        print(
                            f"📍 Current position: x={current_x:.3f}, y={current_y:.3f}, yaw={current_yaw:.3f}"
                        )
                        print(
                            f"🎯 Goal position:    x={goal_x:.3f}, y={goal_y:.3f}"
                        )
                        print(f"📏 Distance to goal: {distance_to_goal:.3f}")

                        # Check if we've reached the goal
                        if (
                            distance_to_goal < 0.05
                        ):  # Same tolerance as setStartAndGoalStates
                            print(
                                f"🎉 SUCCESS! Reached goal (distance {distance_to_goal:.3f} < 0.05)"
                            )
                            print("No more resolve iterations needed!")
                            break

                    else:
                        print("❌ No current solution available")
                        break

                except Exception as e:
                    print(f"⚠️  Error checking current state: {e}")

                # Get tree size before resolve
                planner_data = ob.PlannerData(ss.getSpaceInformation())
                planner.getPlannerData(planner_data)
                tree_size_before = planner_data.numVertices()
                print(
                    f"🌳 Tree size before resolve: {tree_size_before} vertices"
                )

                # Ask user if they want to continue
                user_input = (
                    input(
                        f"\n🤔 Run resolve iteration {resolve_iter}? (y/n/q to quit): "
                    )
                    .lower()
                    .strip()
                )
                if user_input in ["n", "no", "q", "quit"]:
                    print("👋 Stopping resolve iterations by user request")
                    break
                elif user_input not in ["y", "yes", ""]:
                    print("Invalid input, assuming 'no'")
                    break

                # Run resolve iteration
                print(f"\n🚀 Running resolve iteration {resolve_iter}...")

                # Print current start and goal states
                try:
                    pdef = ss.getProblemDefinition()

                    # Print start state
                    if pdef.getStartStateCount() > 0:
                        start_state = pdef.getStartState(0)
                        try:
                            start_x = start_state.getX()
                            start_y = start_state.getY()
                            start_yaw = start_state.getYaw()
                            print(
                                f"🏁 Start state: x={start_x:.3f}, y={start_y:.3f}, yaw={start_yaw:.3f}"
                            )
                        except AttributeError:
                            print(f"🏁 Start state: {start_state}")
                    else:
                        print("🏁 No start state set")

                    # Print goal state
                    goal = pdef.getGoal()
                    if goal:
                        # Try to get goal state if it's a sampleable region
                        try:
                            goal_sampleable = goal.as_GoalSampleableRegion()
                            if goal_sampleable and goal_sampleable.hasStates():
                                goal_state = goal_sampleable.getState(0)
                                goal_x = goal_state.getX()
                                goal_y = goal_state.getY()
                                goal_yaw = goal_state.getYaw()
                                print(
                                    f"🎯 Goal state: x={goal_x:.3f}, y={goal_y:.3f}, yaw={goal_yaw:.3f}"
                                )
                            else:
                                print(f"🎯 Goal: {goal}")
                        except:
                            # If we can't extract specific goal state, just print the goal coordinates we know
                            print(
                                f"🎯 Goal region: x={goal_x:.3f}, y={goal_y:.3f}"
                            )
                    else:
                        print("🎯 No goal set")

                except Exception as e:
                    print(f"⚠️  Error getting start/goal states: {e}")

                try:
                    result = planner.resolve(replanning_time)
                    visualize_tree_3d(
                        planner,
                        filename=f"fusion_3d_{planning_time}s_resolve_{resolve_iter}.png",
                    )
                    print(f"✅ Resolve result: {result}")

                    # Check tree size after resolve
                    planner_data_after = ob.PlannerData(
                        ss.getSpaceInformation()
                    )
                    planner.getPlannerData(planner_data_after)
                    tree_size_after = planner_data_after.numVertices()
                    print(
                        f"🌳 Tree size after resolve: {tree_size_after} vertices"
                    )
                    tree_change = tree_size_after - tree_size_before
                    if tree_change > 0:
                        print(f"📈 Tree growth: {tree_change} vertices added")
                    elif tree_change < 0:
                        print(
                            f"📉 Tree reduction: {-tree_change} vertices removed"
                        )
                    else:
                        print(
                            f"📊 Tree size unchanged: {tree_size_after} vertices"
                        )

                    # Show the updated solution
                    try:
                        updated_solution = ss.getSolutionPath()
                        if updated_solution:
                            print("\n📋 Updated solution path:")
                            print(updated_solution.printAsMatrix())
                        else:
                            print("❌ No solution available after resolve")
                            print(
                                "🛑 Cannot continue - no path to goal from current state"
                            )
                            break
                    except Exception as e:
                        print(f"⚠️  Error getting solution after resolve: {e}")

                except Exception as e:
                    print(
                        f"💥 Error during resolve iteration {resolve_iter}: {e}"
                    )
                    print("🛑 Stopping resolve iterations due to error")
                    break

        print(f"\n🏁 Resolve process completed!")
        input("Press Enter to continue...")


if __name__ == "__main__":
    # Default values
    planning_time = 20.0
    dynamics_type = "unstable"  # Default to the new challenging dynamics
    replanning_time = 2.0

    if len(sys.argv) > 1:
        try:
            planning_time = float(sys.argv[1])
        except ValueError:
            print("Error: Planning time must be a number")
            print(
                "Usage: python3 fusionTest.py [planning_time] [replanning_time] [dynamics_type]"
            )
            print(
                "       dynamics_type: 'simple', 'complex', 'unstable', or 'pendulum' (default: unstable)"
            )
            print("Example: python3 fusionTest.py 5 3 unstable")
            sys.exit(1)

    if len(sys.argv) > 2:
        try:
            replanning_time = float(sys.argv[2])
        except ValueError:
            print("Error: Replanning time must be a number")
            print(
                "Usage: python3 fusionTest.py [planning_time] [replanning_time] [dynamics_type]"
            )
            print(
                "       dynamics_type: 'simple', 'complex', 'unstable', or 'pendulum' (default: unstable)"
            )
            print("Example: python3 fusionTest.py 5 3 unstable")
            sys.exit(1)

    if len(sys.argv) > 3:
        dynamics_arg = sys.argv[3].lower()
        if dynamics_arg in ["simple", "complex", "unstable", "pendulum"]:
            dynamics_type = dynamics_arg
        else:
            print(
                "Error: dynamics_type must be 'simple', 'complex', 'unstable', or 'pendulum'"
            )
            print(
                "Usage: python3 fusionTest.py [planning_time] [replanning_time] [dynamics_type]"
            )
            print("Example: python3 fusionTest.py 5 3 unstable")
            sys.exit(1)

    print(
        f"Starting planning with: {planning_time}s planning, {dynamics_type} dynamics, {replanning_time}s replanning"
    )
    plan(planning_time, dynamics_type, replanning_time)
