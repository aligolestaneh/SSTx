import re
import sys
import torch
import pickle
import numpy as np

from ompl import base as ob
from ompl import control as oc
from planning.planning_utils import (
    GraspableRegion,
    BoxPropagator,
    ControlSampler,
    ActiveControlSampler,
    get_combined_objective,
)

from utils.utils import DataLoader
from models.physics import push_physics
from models.torch_model import ResidualPhysics, MLP
from train_model import load_model


def generate_problems(
    n_problems,
    ranges=((0.3, 0.5), (-0.4, -0.8), (-np.pi / 2, np.pi / 2)),
    save_name="data/planning_initial_states.npy",
):
    """Generate random initial states"""
    random_x = np.random.uniform(ranges[0][0], ranges[0][1], n_problems)
    random_y = np.random.uniform(ranges[1][0], ranges[1][1], n_problems)
    random_theta = np.random.uniform(ranges[2][0], ranges[2][1], n_problems)
    initial_state = np.stack([random_x, random_y, random_theta], axis=1)
    np.save(save_name, initial_state)
    return initial_state


def plan_to_edge(
    start,
    goal,
    obj_shape,
    model,
    active_sampling=False,
    x_train=None,
    planner="SST",
    planning_time=10,
    control_list=None,
):
    """Plan to edge"""
    # Set the bounds for the state space
    space = ob.SE2StateSpace()
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, -0.9)
    bounds.setHigh(0, 0.76)
    bounds.setLow(1, -0.9)
    bounds.setHigh(1, -0.3)
    space.setBounds(bounds)

    # Set the bounds for the control space
    cspace = oc.RealVectorControlSpace(space, 3)
    cbounds = ob.RealVectorBounds(3)
    cbounds.setLow(0, 0)  # minimum rotation
    cbounds.setHigh(0, 4)  # maximum rotation
    cbounds.setLow(1, -0.4)  # minimum side offset
    cbounds.setHigh(1, 0.4)  # maximum side offset
    cbounds.setLow(2, 0.0)  # minimum push distance
    cbounds.setHigh(2, 0.3)  # maximum push distance
    cspace.setBounds(cbounds)

    # Setup
    ss = oc.SimpleSetup(cspace)
    si = ss.getSpaceInformation()

    # Set the state validity checker
    # No need to check collision in the side grasp planning
    def isStateValid(state):
        return si.satisfiesBounds(state)

    ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))

    # Set the state propagator
    propagator = BoxPropagator(model, obj_shape)
    ss.setStatePropagator(oc.StatePropagatorFn(propagator.propagate))

    # Set the control sampler
    if not active_sampling:
        # Regular control sampler
        def ControlSamplerAllocator(space):
            return ControlSampler(space, obj_shape, control_list)

    else:
        # Active control sampler
        def ControlSamplerAllocator(space):
            return ActiveControlSampler(space, model, x_train, obj_shape, control_list)

    cspace.setControlSamplerAllocator(oc.ControlSamplerAllocator(ControlSamplerAllocator))

    # Set the start and goal
    start_state = ob.State(space)
    start_state().setX(start[0])
    start_state().setY(start[1])
    start_state().setYaw(start[2])
    ss.setStartState(start_state)

    goal_state = GraspableRegion(ss.getSpaceInformation(), goal, obj_shape, 0.76)
    goal_state.setThreshold(0.01)
    ss.setGoal(goal_state)

    # Set the planners
    if planner == "SST":
        planner = oc.SST(si)
    elif planner == "RRT":
        planner = oc.RRT(si)
    else:
        raise ValueError(f"Invalid planner: {planner}")
    ss.setPlanner(planner)

    # Set the optimization objective
    # objective = get_combined_objective(
    #     si,
    #     cost_per_control=1.0,
    #     weight_path_length=1.0,
    #     weight_control_count=1.0,
    # )
    objective = ob.PathLengthOptimizationObjective(si)
    pdef = ss.getProblemDefinition()
    pdef.setOptimizationObjective(objective)

    # Control duration in steps
    si.setPropagationStepSize(3.0)
    si.setMinMaxControlDuration(1, 1)

    # Solve the problem
    solved = ss.solve(planning_time)
    #  Use exact solution only
    while solved.asString() == "Approximate solution":
        print(f"Approximate solution, re-planning!")
        solved = ss.solve(planning_time)

    # Extract the path
    if solved:
        print(f"Planning to edge successful!")
        path = ss.getSolutionPath()

        states = []
        for i in range(path.getStateCount()):
            state = path.getState(i)
            states.append([state.getX(), state.getY(), state.getYaw()])

        controls = []
        durations = []
        for i in range(path.getControlCount()):
            control = path.getControl(i)
            controls.append([control[0], control[1], control[2]])
            durations.append(path.getControlDuration(i))
        return states, controls, durations

    else:
        print(f"Planning to push edge failed!")
        return None, None, None


def main(
    model_class,
    obj_shape,
    initial_states,
    sampling,
    x_train,
    model_name,
    iteration,
    num_planning=100,
    planning_time=10.0,
    save_name="planning",
    control_list=None,
):
    """Main function to plan to edge"""
    # Load dynamics model
    model = load_model(model_class, 3, 3)
    model.load(f"saved_models/{model_name}.pth")
    model = model.model  # use torch model directly
    model.eval()

    # Split the problems
    itr = int(iteration) - 1
    initial_states = initial_states[itr * num_planning : (itr + 1) * num_planning]

    # Define the start and goal
    controls_list = []  # [n, steps]
    states_list = []  # [n, steps]

    for initial_state in initial_states:
        x, y, theta = initial_state

        # Select the goal rotation, choose the closest one
        # rotation_goals = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        rotation_goals = [np.pi / 2, 3 * np.pi / 2]
        rotation_diff = [(theta - goal) % np.pi for goal in rotation_goals]
        goal_idx = np.argmin(rotation_diff)

        # Goal state (edge, initial_y, closest_rotation)
        desired_state = np.array([0.725, y, rotation_goals[goal_idx]])

        # Plan to edge
        controls = [0] * 5
        max_control_length = 5
        # avoid infinite loop
        max_trials = 50
        trial_count = 0
        while len(controls) >= max_control_length and trial_count < max_trials:
            states, controls, durations = plan_to_edge(
                initial_state,
                desired_state,
                obj_shape,
                model,
                sampling,
                x_train,
                planner="SST",
                planning_time=planning_time,
                control_list=control_list,
            )
            if controls is None:
                controls = [[0, 0, -0.05]]

            trial_count += 1
            if len(controls) >= max_control_length:
                print(f"[INFO]: Path too long, replan. ({trial_count} times)")

        # Save all the data
        states_list.append(states)
        controls_list.append(controls)

    sampling = "_active" if sampling else "_regular"
    np.save(
        f"data/planning/{model_name}{sampling}/{model_name}{sampling}_initial_states_{iteration}.npy",
        initial_states,
    )
    states_list = np.array(states_list, dtype=object)
    np.save(
        f"data/planning/{model_name}{sampling}/{model_name}{sampling}_states_{iteration}.npy",
        states_list,
    )
    controls_list = np.array(controls_list, dtype=object)
    np.save(
        f"data/planning/{model_name}{sampling}/{model_name}{sampling}_controls_{iteration}.npy",
        controls_list,
    )


if __name__ == "__main__":
    # Define the object and model
    iteration = sys.argv[1]
    model_name = sys.argv[2]
    sampling = sys.argv[3]

    sampling = True if sampling == "True" else False

    if "residual" in model_name:
        model_class = "residual"
    elif "mlp" in model_name:
        model_class = "mlp"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    num_planning = 10
    planning_time = 10

    obj_shape = np.array([0.1628, 0.2139, 0.0676])
    save_name = "cracker_box"

    # For active sampling, we need to know what are the training data
    exp_idx = 3
    residual_indices = pickle.load(
        open(f"results/learning/idx_used_{save_name}_{model_class}.pkl", "rb")
    )
    active_residual_indices = residual_indices["bait"][exp_idx]
    data_loader = DataLoader(save_name, "data", val_size=200)
    datasets = data_loader.load_data()
    x_pool = datasets["x_pool"]
    num_training = int(re.findall(r"[-+]?\d*\.\d+|\d+", sys.argv[2])[-1])
    x_train = x_pool[active_residual_indices[:num_training]]

    # Load problem set
    initial_states = np.load("data/planning_initial_states_fixed.npy")

    # No pre-defined control list
    control_list = None
    main(
        model_class,
        obj_shape,
        initial_states,
        sampling,
        x_train,
        model_name,
        iteration,
        num_planning,
        planning_time,
        save_name,
        control_list=control_list,
    )
