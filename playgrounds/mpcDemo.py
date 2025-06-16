import numpy as np
import matplotlib.pyplot as plt
import do_mpc

# --- System and MPC Parameters ---
MODEL_TYPE = "continuous"
MASS = 1.0
# Force limits per component
F_COMPONENT_MAX = (
    7.0  # Adjusted for 2D, total force magnitude will be sqrt(fx^2+fy^2)
)
F_COMPONENT_MIN = -7.0

# Simulation Time
TOTAL_SIM_TIME = 20.0
DT_CONTROL_INTERVAL = 0.1

# MPC Horizon
N_HORIZON_MPC = 20
T_STEP_MPC = DT_CONTROL_INTERVAL

# MPC Tuning Weights (applied to sum of squared errors/efforts in x and y)
Q_POS = 100.0  # Weight for position error ((pos_x-sp_x)^2 + (pos_y-sp_y)^2)
R_FORCE = 0.05  # Weight for control effort (force_x^2 + force_y^2) - adjusted
R_DFORCE = 0.1  # Weight for change in control effort (delta_force component^2) - adjusted


# --- Desired Path (Setpoint Trajectory) - Now 2D ---
def get_setpoint_trajectory(time_points):
    """
    Defines the desired 2D position (x, y) of the box over time.
    Returns:
        np.array: Shape (N, 2) where N is len(time_points), columns are [sp_x, sp_y].
    """
    setpoints_xy = np.zeros((len(time_points), 2))
    for i, t_val in enumerate(time_points):
        sp_x, sp_y = 0, 0
        if t_val < 5.0:  # Move in x
            sp_x = 0.5 * t_val
            sp_y = 0.0
        elif t_val < 7.0:  # Hold
            sp_x = 2.5
            sp_y = 0.0
        elif t_val < 12.0:  # Move in y, while x is held
            sp_x = 2.5
            sp_y = 0.4 * (t_val - 7.0)  # Ramp y to 0.4 * 5 = 2.0
        elif t_val < 14.0:  # Hold
            sp_x = 2.5
            sp_y = 2.0
        elif t_val < 19.0:  # Move in x back
            sp_x = 2.5 - 0.5 * (t_val - 14.0)  # Ramp x back to 0
            sp_y = 2.0
        else:  # Hold
            sp_x = 0.0
            sp_y = 2.0
        setpoints_xy[i, 0] = sp_x
        setpoints_xy[i, 1] = sp_y
    return setpoints_xy


# --- 1. Define the Model (2D) ---
def setup_do_mpc_model(model_type_str):
    model = do_mpc.model.Model(model_type_str)

    # State variables (order matters for x0 initialization)
    pos_x = model.set_variable(var_type="_x", var_name="pos_x")
    pos_y = model.set_variable(var_type="_x", var_name="pos_y")
    vel_x = model.set_variable(var_type="_x", var_name="vel_x")
    vel_y = model.set_variable(var_type="_x", var_name="vel_y")

    # Control inputs
    force_x = model.set_variable(var_type="_u", var_name="force_x")
    force_y = model.set_variable(var_type="_u", var_name="force_y")

    # Time-varying parameters for setpoints
    pos_setpoint_x = model.set_variable(
        var_type="_tvp", var_name="pos_setpoint_x"
    )
    pos_setpoint_y = model.set_variable(
        var_type="_tvp", var_name="pos_setpoint_y"
    )

    # Differential equations
    model.set_rhs("pos_x", vel_x)
    model.set_rhs("pos_y", vel_y)
    model.set_rhs("vel_x", force_x / MASS)
    model.set_rhs("vel_y", force_y / MASS)

    model.setup()
    return model


# --- 2. Define the MPC Controller (2D) ---
def setup_do_mpc_controller(model_in):
    mpc = do_mpc.controller.MPC(model_in)
    setup_mpc_params = {
        "n_horizon": N_HORIZON_MPC,
        "t_step": T_STEP_MPC,
        "n_robust": 0,
        "store_full_solution": True,
    }
    mpc.set_param(**setup_mpc_params)

    # Objective function
    mterm = Q_POS * (
        (model_in.x["pos_x"] - model_in.tvp["pos_setpoint_x"]) ** 2
        + (model_in.x["pos_y"] - model_in.tvp["pos_setpoint_y"]) ** 2
    )
    lterm = Q_POS * (
        (model_in.x["pos_x"] - model_in.tvp["pos_setpoint_x"]) ** 2
        + (model_in.x["pos_y"] - model_in.tvp["pos_setpoint_y"]) ** 2
    ) + R_FORCE * (model_in.u["force_x"] ** 2 + model_in.u["force_y"] ** 2)

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(
        force_x=R_DFORCE, force_y=R_DFORCE
    )  # Penalty on change for each force component

    # Constraints
    mpc.bounds["lower", "_u", "force_x"] = F_COMPONENT_MIN
    mpc.bounds["upper", "_u", "force_x"] = F_COMPONENT_MAX
    mpc.bounds["lower", "_u", "force_y"] = F_COMPONENT_MIN
    mpc.bounds["upper", "_u", "force_y"] = F_COMPONENT_MAX

    # Time-varying parameters function for MPC
    tvp_template_mpc = mpc.get_tvp_template()

    def tvp_fun_mpc(t_now_val):
        for k in range(N_HORIZON_MPC + 1):
            t_pred = t_now_val + k * T_STEP_MPC
            setpoint_k = get_setpoint_trajectory(np.array([t_pred]))[
                0
            ]  # Returns [sp_x, sp_y]
            tvp_template_mpc["_tvp", k, "pos_setpoint_x"] = setpoint_k[0]
            tvp_template_mpc["_tvp", k, "pos_setpoint_y"] = setpoint_k[1]
        return tvp_template_mpc

    mpc.set_tvp_fun(tvp_fun_mpc)

    mpc.setup()
    return mpc


# --- 3. Define the Simulator (Plant) (2D) ---
def setup_do_mpc_simulator(model_in):
    simulator = do_mpc.simulator.Simulator(model_in)
    simulator.set_param(t_step=T_STEP_MPC)

    tvp_template_sim = simulator.get_tvp_template()

    def tvp_fun_sim(t_now_val):
        current_setpoint_xy = get_setpoint_trajectory(np.array([t_now_val]))[0]
        tvp_template_sim["pos_setpoint_x"] = current_setpoint_xy[0]
        tvp_template_sim["pos_setpoint_y"] = current_setpoint_xy[1]
        return tvp_template_sim

    simulator.set_tvp_fun(tvp_fun_sim)

    simulator.setup()
    return simulator


# --- 4. Run the Simulation Loop and Plot (2D) ---
def run_simulation_and_plot(model_obj, mpc_obj, simulator_obj):
    # Initial state: [pos_x, pos_y, vel_x, vel_y]
    x0 = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
    mpc_obj.x0 = x0
    simulator_obj.x0 = x0
    mpc_obj.set_initial_guess()

    graphics_sim = do_mpc.graphics.Graphics(simulator_obj.data)
    graphics_mpc = do_mpc.graphics.Graphics(mpc_obj.data)

    plt.ion()
    # Window 1: Position Components (X and Y vs. Time)
    fig_pos_t, ax_pos_t = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    fig_pos_t.suptitle("Position Components vs. Time")
    ax_pos_t[0].set_ylabel("Position X (m)")
    graphics_sim.add_line(
        var_type="_x",
        var_name="pos_x",
        axis=ax_pos_t[0],
        color="blue",
        label="Actual Pos X (Sim)",
    )
    ax_pos_t[1].set_ylabel("Position Y (m)")
    ax_pos_t[1].set_xlabel("Time (s)")
    graphics_sim.add_line(
        var_type="_x",
        var_name="pos_y",
        axis=ax_pos_t[1],
        color="dodgerblue",
        label="Actual Pos Y (Sim)",
    )

    # Window 2: Velocity Components (X and Y vs. Time)
    fig_vel_t, ax_vel_t = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    fig_vel_t.suptitle("Velocity Components vs. Time")
    ax_vel_t[0].set_ylabel("Velocity X (m/s)")
    graphics_sim.add_line(
        var_type="_x",
        var_name="vel_x",
        axis=ax_vel_t[0],
        color="green",
        label="Actual Vel X (Sim)",
    )
    ax_vel_t[1].set_ylabel("Velocity Y (m/s)")
    ax_vel_t[1].set_xlabel("Time (s)")
    graphics_sim.add_line(
        var_type="_x",
        var_name="vel_y",
        axis=ax_vel_t[1],
        color="limegreen",
        label="Actual Vel Y (Sim)",
    )

    # Window 3: Force Components (X and Y vs. Time)
    fig_force_t, ax_force_t = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    fig_force_t.suptitle("Force Components vs. Time")
    ax_force_t[0].set_ylabel("Force X (N)")
    graphics_mpc.add_line(
        var_type="_u",
        var_name="force_x",
        axis=ax_force_t[0],
        color="red",
        label="Force X (MPC)",
    )
    ax_force_t[1].set_ylabel("Force Y (N)")
    ax_force_t[1].set_xlabel("Time (s)")
    graphics_mpc.add_line(
        var_type="_u",
        var_name="force_y",
        axis=ax_force_t[1],
        color="salmon",
        label="Force Y (MPC)",
    )

    # Window 4: 2D Path (Y vs. X)
    fig_path_2d, ax_path_2d = plt.subplots(1, 1, figsize=(8, 8))
    ax_path_2d.set_title("2D Path (Y vs. X)")
    ax_path_2d.set_xlabel("Position X (m)")
    ax_path_2d.set_ylabel("Position Y (m)")
    (line_actual_path,) = ax_path_2d.plot(
        [], [], "b-", label="Actual Path (Sim)"
    )  # For live update
    line_predicted_paths = []  # For MPC predictions

    # Plot overall desired trajectory
    n_simulation_steps = int(TOTAL_SIM_TIME / T_STEP_MPC)
    overall_time_points = (
        np.arange(n_simulation_steps + 2) * T_STEP_MPC
    )  # +2 to ensure full range for setpoints
    overall_setpoints_xy = get_setpoint_trajectory(overall_time_points)

    ax_pos_t[0].plot(
        overall_time_points,
        overall_setpoints_xy[:, 0],
        "k--",
        label="Desired Pos X",
    )
    ax_pos_t[1].plot(
        overall_time_points,
        overall_setpoints_xy[:, 1],
        "k--",
        label="Desired Pos Y",
    )
    ax_path_2d.plot(
        overall_setpoints_xy[:, 0],
        overall_setpoints_xy[:, 1],
        "k--",
        label="Desired Path",
    )

    all_axes = (
        list(ax_pos_t) + list(ax_vel_t) + list(ax_force_t) + [ax_path_2d]
    )
    for ax_item in all_axes:
        ax_item.legend(loc="best")
        ax_item.grid(True)
    ax_path_2d.axis(
        "equal"
    )  # Ensure X and Y scales are the same for path plot

    print("Starting 2D do-mpc simulation...")

    for k_step in range(n_simulation_steps):
        u0_optimal = mpc_obj.make_step(x0)
        x0 = simulator_obj.make_step(u0_optimal)

        if k_step % 5 == 0:
            graphics_sim.plot_results(t_ind=k_step)
            graphics_mpc.plot_results(t_ind=k_step)
            graphics_mpc.plot_predictions(
                t_ind=k_step
            )  # This also updates MPC predicted states

            # Update 2D Path Plot
            hist_pos_x = simulator_obj.data["_x", "pos_x"]
            hist_pos_y = simulator_obj.data["_x", "pos_y"]
            line_actual_path.set_data(
                hist_pos_x[: k_step + 1], hist_pos_y[: k_step + 1]
            )

            # Update MPC predicted paths on 2D plot
            for line in line_predicted_paths:
                line.remove()  # Clear old predicted paths
            line_predicted_paths.clear()

            # Corrected way to get prediction data
            prediction_values_x = mpc_obj.data.prediction(
                ("_x", "pos_x"), t_ind=k_step
            )
            prediction_values_y = mpc_obj.data.prediction(
                ("_x", "pos_y"), t_ind=k_step
            )

            if (
                prediction_values_x is not None
                and prediction_values_y is not None
            ):
                # Assuming prediction_values_x and _y are 2D arrays [scenario, time_step_in_horizon]
                # For non-robust MPC, scenario is 0.
                pred_pos_x = prediction_values_x[0, :]
                pred_pos_y = prediction_values_y[0, :]
                (line_pred,) = ax_path_2d.plot(
                    pred_pos_x,
                    pred_pos_y,
                    color="purple",
                    linestyle=":",
                    alpha=0.7,
                    label=(
                        "Predicted Path (MPC)" if k_step == 0 else "_nolegend_"
                    ),
                )
                line_predicted_paths.append(line_pred)

            # Autoscale all axes
            for fig_ax_pair in [
                (fig_pos_t, ax_pos_t),
                (fig_vel_t, ax_vel_t),
                (fig_force_t, ax_force_t),
            ]:
                for ax_item in fig_ax_pair[
                    1
                ]:  # If axes are arrays (like subplots(2,1))
                    ax_item.relim()
                    ax_item.autoscale_view(True, True, True)
                    ax_item.legend(loc="best")  # Re-apply legend
                fig_ax_pair[0].canvas.draw_idle()

            ax_path_2d.relim()
            ax_path_2d.autoscale_view(True, True, True)
            ax_path_2d.legend(loc="best")
            fig_path_2d.canvas.draw_idle()

            plt.pause(0.01)

        if k_step > 0 and k_step % (n_simulation_steps // 10) == 0:
            current_pos_x = simulator_obj.data["_x", "pos_x"][-1]
            current_pos_y = simulator_obj.data["_x", "pos_y"][-1]
            print(
                f"Sim step {k_step}/{n_simulation_steps}. Time: {k_step*T_STEP_MPC:.1f}s. Pos: ({current_pos_x[0]:.2f}, {current_pos_y[0]:.2f})"
            )

    print("do-mpc simulation finished.")

    # Final plot updates
    graphics_sim.plot_results()
    graphics_mpc.plot_results()
    graphics_mpc.plot_predictions()

    hist_pos_x_final = simulator_obj.data["_x", "pos_x"]
    hist_pos_y_final = simulator_obj.data["_x", "pos_y"]
    line_actual_path.set_data(
        hist_pos_x_final, hist_pos_y_final
    )  # Ensure full actual path is plotted

    # Corrected way to get final prediction data
    prediction_values_x_final = mpc_obj.data.prediction(
        ("_x", "pos_x"), t_ind=n_simulation_steps - 1
    )
    prediction_values_y_final = mpc_obj.data.prediction(
        ("_x", "pos_y"), t_ind=n_simulation_steps - 1
    )

    if (
        prediction_values_x_final is not None
        and prediction_values_y_final is not None
    ):
        for line in line_predicted_paths:
            line.remove()
        line_predicted_paths.clear()
        # Assuming prediction_values_x_final and _y_final are 2D arrays [scenario, time_step_in_horizon]
        pred_pos_x = prediction_values_x_final[0, :]
        pred_pos_y = prediction_values_y_final[0, :]
        ax_path_2d.plot(
            pred_pos_x,
            pred_pos_y,
            color="purple",
            linestyle=":",
            alpha=0.7,
            label="Final Predicted Path",
        )

    for fig_ax_pair in [
        (fig_pos_t, ax_pos_t),
        (fig_vel_t, ax_vel_t),
        (fig_force_t, ax_force_t),
    ]:
        for ax_item in fig_ax_pair[1]:
            ax_item.relim()
            ax_item.autoscale_view(True, True, True)
            ax_item.legend(loc="best")
    ax_path_2d.relim()
    ax_path_2d.autoscale_view(True, True, True)
    ax_path_2d.legend(loc="best")

    plt.ioff()
    plt.show()


# --- Main execution ---
if __name__ == "__main__":
    rod_box_model_2d = setup_do_mpc_model(MODEL_TYPE)
    mpc_controller_2d = setup_do_mpc_controller(rod_box_model_2d)
    plant_simulator_2d = setup_do_mpc_simulator(rod_box_model_2d)
    run_simulation_and_plot(
        rod_box_model_2d, mpc_controller_2d, plant_simulator_2d
    )
