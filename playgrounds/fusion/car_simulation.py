import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import threading
import time
from planning_utils import propagate_simple, propagate_complex


class CarSimulation:
    def __init__(self, use_complex_dynamics=True):
        self.use_complex_dynamics = use_complex_dynamics
        self.car_length = 0.08  # Length of car rectangle
        self.car_width = 0.04  # Width of car rectangle
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.car_patch = None
        self.path_line = None
        self.current_pos_marker = None
        self.animation = None
        self.is_running = False

    def extract_path_from_planner_data(self, planner_data):
        """Extract states and controls from planner data"""
        # This is kept for compatibility but will use sample data
        # In practice, the real path data will be passed directly
        sample_states = [
            (-0.5, 0.0, 0.0),  # start
            (-0.3, 0.1, 0.2),
            (-0.1, 0.2, 0.4),
            (0.0, 0.3, 0.3),
            (0.0, 0.5, 0.0),  # goal
        ]

        sample_controls = [
            (0.2, 0.1),  # forward and slight right
            (0.2, 0.0),  # forward
            (0.15, -0.1),  # forward and slight left
            (0.1, -0.05),  # slow forward and slight left
        ]

        sample_durations = [0.5, 0.5, 0.5, 0.5]  # seconds per control

        return sample_states, sample_controls, sample_durations

    def extract_goal_from_planner_data(self, planner_data):
        """Extract goal position from planner data if available"""
        try:
            if planner_data and planner_data.numGoalVertices() > 0:
                goal_vertex = planner_data.getGoalVertex(0)
                goal_state = goal_vertex.getState()

                try:
                    # Try direct attribute access
                    x = goal_state.getX()
                    y = goal_state.getY()
                    yaw = goal_state.getYaw()
                except AttributeError:
                    try:
                        # Try accessing via compound state
                        se2_component = goal_state[0]  # SE2 is first component
                        x = se2_component.getX()
                        y = se2_component.getY()
                        yaw = se2_component.getYaw()
                    except:
                        # If extraction fails, return None to use path end as goal
                        print(
                            "Could not extract goal state, using path end as goal"
                        )
                        return None

                return (x, y, yaw)
        except Exception as e:
            print(f"Could not extract goal from planner data: {e}")

        return None

    def setup_plot(self, states, planner_data=None):
        """Setup the plot with boundaries and path"""
        self.ax.clear()
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(
            f'Car Simulation - {"Complex" if self.use_complex_dynamics else "Simple"} Dynamics'
        )
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")

        # Plot planned path
        if states and len(states) > 0:
            path_x = [state[0] for state in states]
            path_y = [state[1] for state in states]
            (self.path_line,) = self.ax.plot(
                path_x,
                path_y,
                "b--",
                alpha=0.6,
                linewidth=2,
                label="Planned Path",
            )

            # Start is always the first state in the solution path
            start_state = states[0]
            self.ax.plot(
                start_state[0],
                start_state[1],
                "go",
                markersize=10,
                label="Start",
            )

            # Try to get goal from planner data first, otherwise use last state in path
            goal_from_planner = self.extract_goal_from_planner_data(
                planner_data
            )
            if goal_from_planner:
                goal_state = goal_from_planner
                goal_source = "Planner Data"
            else:
                goal_state = states[-1]
                goal_source = "Path End"

            self.ax.plot(
                goal_state[0],
                goal_state[1],
                "ro",
                markersize=10,
                label=f"Goal ({goal_source})",
            )

        # Create car rectangle (initially at origin)
        self.car_patch = patches.Rectangle(
            (0, 0),
            self.car_length,
            self.car_width,
            angle=0,
            facecolor="red",
            edgecolor="black",
            linewidth=2,
        )
        self.ax.add_patch(self.car_patch)

        # Current position marker
        (self.current_pos_marker,) = self.ax.plot(
            [], [], "ro", markersize=8, label="Current Position"
        )

        self.ax.legend()

    def update_car_position(self, x, y, theta):
        """Update car rectangle position and orientation"""
        # Convert theta to degrees for matplotlib
        angle_deg = np.degrees(theta)

        # Calculate rectangle corner (bottom-left) from center position
        corner_x = (
            x
            - self.car_length / 2 * np.cos(theta)
            + self.car_width / 2 * np.sin(theta)
        )
        corner_y = (
            y
            - self.car_length / 2 * np.sin(theta)
            - self.car_width / 2 * np.cos(theta)
        )

        # Update car patch
        self.car_patch.set_xy((corner_x, corner_y))
        self.car_patch.set_angle(angle_deg)

        # Update current position marker
        self.current_pos_marker.set_data([x], [y])

    def simulate_path(self, states, controls, durations):
        """Simulate the car following the control path"""
        if not states or not controls:
            print("No path data available for simulation")
            return

        print(f"Starting car simulation with {len(controls)} control segments")

        # Start from first state
        current_state = np.array(states[0])
        all_positions = [current_state.copy()]

        dt = 0.05  # 50ms time steps for smooth animation

        for i, (control, duration) in enumerate(zip(controls, durations)):
            print(
                f"Executing control {i+1}/{len(controls)}: u=({control[0]:.2f}, {control[1]:.2f}) for {duration}s"
            )

            # Simulate this control for the given duration
            steps = int(duration / dt)
            for step in range(steps):
                # Apply propagator function
                if self.use_complex_dynamics:
                    next_state = np.zeros(3)
                    propagate_complex(current_state, control, dt, next_state)
                else:
                    next_state = np.zeros(3)
                    propagate_simple(current_state, control, dt, next_state)

                current_state = next_state.copy()
                all_positions.append(current_state.copy())

                # Small delay for real-time visualization
                time.sleep(dt)

                if not self.is_running:
                    return

                # Update visualization
                self.update_car_position(
                    current_state[0], current_state[1], current_state[2]
                )
                plt.pause(0.001)  # Allow GUI to update

        print("Car simulation completed!")
        print(
            f"Final position: x={current_state[0]:.3f}, y={current_state[1]:.3f}, theta={current_state[2]:.3f}"
        )

    def run_simulation(
        self, planner_data=None, states=None, controls=None, durations=None
    ):
        """Main simulation function that runs in a separate thread"""
        try:
            self.is_running = True

            # Use provided path data if available, otherwise extract from planner_data
            if (
                states is not None
                and controls is not None
                and durations is not None
            ):
                sim_states, sim_controls, sim_durations = (
                    states,
                    controls,
                    durations,
                )
                print("Using provided path data for simulation")
            else:
                # Extract path data from planner_data or use sample data
                sim_states, sim_controls, sim_durations = (
                    self.extract_path_from_planner_data(planner_data)
                )
                print("Using extracted/sample path data for simulation")

            # Setup plot
            self.setup_plot(sim_states, planner_data)
            plt.show(block=False)
            plt.pause(0.1)  # Allow plot to initialize

            # Run simulation
            self.simulate_path(sim_states, sim_controls, sim_durations)

        except Exception as e:
            print(f"Error in car simulation: {e}")
        finally:
            self.is_running = False

    def start_simulation_thread(
        self, planner_data=None, states=None, controls=None, durations=None
    ):
        """Start the simulation in a separate thread"""
        if self.is_running:
            print("Simulation already running")
            return

        simulation_thread = threading.Thread(
            target=self.run_simulation,
            args=(planner_data, states, controls, durations),
        )
        simulation_thread.daemon = True  # Dies when main program exits
        simulation_thread.start()
        return simulation_thread

    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        if plt.get_fignums():  # Check if figure exists
            plt.close(self.fig)


def extract_path_from_solution(solution_path):
    """Extract states and controls from OMPL solution path"""
    states = []
    controls = []
    durations = []

    if solution_path is None:
        return states, controls, durations

    # Get path as matrix and parse it
    try:
        path_matrix = solution_path.printAsMatrix()
        lines = path_matrix.strip().split("\n")

        for line in lines:
            if line.strip():
                values = [float(x) for x in line.split()]
                if len(values) >= 3:  # x, y, theta
                    states.append((values[0], values[1], values[2]))
                if len(values) >= 6:  # x, y, theta, u1, u2, duration
                    controls.append((values[3], values[4]))
                    durations.append(values[5])
    except:
        # Fallback to sample data if parsing fails
        pass

    return states, controls, durations


def run_car_simulation_parallel(solution_path, use_complex_dynamics=True):
    """Convenience function to run car simulation in parallel"""
    simulation = CarSimulation(use_complex_dynamics)

    # Create dummy planner data for now
    # In practice, you'd pass the actual planner data
    planner_data = None

    # Start simulation thread
    return simulation.start_simulation_thread(planner_data)


if __name__ == "__main__":
    # Test the simulation
    sim = CarSimulation(use_complex_dynamics=True)
    sim.run_simulation(None)  # Using sample data
    plt.show()
