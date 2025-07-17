from math import sin, cos, tan
import numpy as np
import torch


def isStateValid(spaceInformation, state):
    """
    Check if a state is valid (collision checking or constraint satisfaction)
    """
    # perform collision checking or check if other constraints are
    # satisfied
    return spaceInformation.satisfiesBounds(state)


class BoxPropagator:
    """Class to handle box state propagation with physics model."""

    def __init__(self, model, obj_shape):
        """Initialize propagator with object dimensions."""
        self.model = model
        self.device = next(self.model.parameters()).device
        self.obj_shape = obj_shape

    def propagate(self, start, control, duration, state):
        """Propagate the box state given control inputs."""
        self.propagate_se2(start, control, duration, state)

    def propagate_se2(self, start, control, duration, state):
        """Propagate the SE2 state given control inputs."""
        x = start.getX()
        y = start.getY()
        theta = start.getYaw()
        initial_pose = self.to_matrix([x, y, theta])

        # Move tensor to the same device as the model
        device = next(self.model.parameters()).device
        control_tensor = torch.tensor(
            [[float(control[0]), float(control[1]), float(control[2])]],
            dtype=torch.float32,
        ).to(device)

        # Get the predicted output from the model
        output = self.model(control_tensor)[0].detach().cpu().numpy()
        delta = self.to_matrix(output[:3])

        # Get the final pose by applying the delta to the initial pose
        final_pose = initial_pose @ delta
        final_se2 = self.to_vector(final_pose)
        state.setX(final_se2[0])
        state.setY(final_se2[1])
        state.setYaw(final_se2[2])

    @staticmethod
    def to_matrix(vector):
        c = np.cos(vector[2])
        s = np.sin(vector[2])
        return np.array([[c, -s, vector[0]], [s, c, vector[1]], [0, 0, 1]])

    @staticmethod
    def to_vector(matrix):
        return np.array(
            [
                matrix[0, 2],
                matrix[1, 2],
                np.arctan2(matrix[1, 0], matrix[0, 0]),
            ]
        )


def propagate_simple(start, control, duration, state):
    """
    Simple car dynamics model (similar to original OMPL examples)

    Control inputs:
    - control[0]: velocity command
    - control[1]: steering angle command
    """
    x = start.getX()
    y = start.getY()
    yaw = start.getYaw()

    # Simple bicycle model
    velocity = control[0]
    steering_angle = control[1]

    # Basic integration
    state.setX(x + velocity * cos(yaw) * duration)
    state.setY(y + velocity * sin(yaw) * duration)
    state.setYaw(yaw + velocity * tan(steering_angle) * duration)


def propagate_complex(start, control, duration, state):
    """
    More complex car dynamics model with realistic constraints

    Control inputs:
    - control[0]: acceleration command [-1, 1] (normalized)
    - control[1]: steering angle command [-1, 1] (normalized)

    Car parameters and constraints:
    - Maximum velocity: 0.8 m/s
    - Maximum acceleration: 1.0 m/s²
    - Maximum steering angle: π/3 radians (60 degrees)
    - Wheelbase length: 0.2 m
    - Velocity-dependent steering (harder to steer at high speeds)
    - Rolling resistance and drag
    """

    # Car parameters
    MAX_VELOCITY = 0.8
    MAX_ACCELERATION = 1.0
    MAX_STEERING_ANGLE = np.pi / 3  # 60 degrees
    WHEELBASE = 0.2
    ROLLING_RESISTANCE = 0.1
    DRAG_COEFFICIENT = 0.05
    MIN_VELOCITY = 0.01  # Minimum velocity to avoid division by zero

    # Extract current state
    x = start.getX()
    y = start.getY()
    yaw = start.getYaw()

    # Estimate current velocity from previous motion (simplified)
    # In a real implementation, velocity would be part of the state
    # For this demo, we'll use a heuristic based on control history
    current_velocity = MAX_VELOCITY * 0.5  # Assume moderate speed

    # Process control inputs
    accel_cmd = np.clip(control[0], -1.0, 1.0) * MAX_ACCELERATION
    steering_cmd = np.clip(control[1], -1.0, 1.0) * MAX_STEERING_ANGLE

    # Apply velocity-dependent steering constraint (harder to steer at high speed)
    velocity_factor = max(0.3, 1.0 - current_velocity / MAX_VELOCITY)
    steering_angle = steering_cmd * velocity_factor

    # Apply rolling resistance and drag
    resistance_force = (
        ROLLING_RESISTANCE * current_velocity
        + DRAG_COEFFICIENT * current_velocity**2
    )
    net_acceleration = accel_cmd - resistance_force

    # Update velocity with constraints
    new_velocity = current_velocity + net_acceleration * duration
    new_velocity = np.clip(new_velocity, 0.0, MAX_VELOCITY)

    # Ensure minimum velocity for numerical stability
    if new_velocity < MIN_VELOCITY:
        new_velocity = MIN_VELOCITY

    # Bicycle model dynamics
    # Angular velocity depends on velocity and steering angle
    angular_velocity = (new_velocity / WHEELBASE) * np.tan(steering_angle)

    # Add some non-linear effects
    # 1. Steering becomes less effective at very low speeds
    low_speed_factor = min(1.0, new_velocity / (MAX_VELOCITY * 0.2))
    angular_velocity *= low_speed_factor

    # 2. Add slight understeer at high speeds (more realistic)
    if new_velocity > MAX_VELOCITY * 0.6:
        understeer_factor = 0.8
        angular_velocity *= understeer_factor

    # 3. Add some slip effect based on steering input magnitude
    slip_factor = 1.0 - 0.1 * abs(steering_angle) / MAX_STEERING_ANGLE
    effective_velocity = new_velocity * slip_factor

    # Integrate the motion using the bicycle model
    # Use multiple small steps for better numerical integration
    dt = duration / 5.0  # 5 substeps for smoother integration

    current_x, current_y, current_yaw = x, y, yaw
    current_vel = current_velocity

    for _ in range(5):
        # Update velocity
        current_vel += net_acceleration * dt
        current_vel = np.clip(current_vel, MIN_VELOCITY, MAX_VELOCITY)

        # Update angular velocity
        current_angular_vel = (current_vel / WHEELBASE) * np.tan(
            steering_angle
        )
        current_angular_vel *= low_speed_factor

        if current_vel > MAX_VELOCITY * 0.6:
            current_angular_vel *= understeer_factor

        # Update position and orientation
        current_x += current_vel * np.cos(current_yaw) * dt
        current_y += current_vel * np.sin(current_yaw) * dt
        current_yaw += current_angular_vel * dt

        # Add some noise/disturbance for realism (small amount)
        disturbance_scale = 0.005
        current_x += disturbance_scale * (np.random.random() - 0.5) * dt
        current_y += disturbance_scale * (np.random.random() - 0.5) * dt
        current_yaw += disturbance_scale * (np.random.random() - 0.5) * dt

    # Normalize yaw angle to [-π, π]
    while current_yaw > np.pi:
        current_yaw -= 2 * np.pi
    while current_yaw < -np.pi:
        current_yaw += 2 * np.pi

    # Set the final state
    state.setX(current_x)
    state.setY(current_y)
    state.setYaw(current_yaw)


def propagate_unstable(start, control, duration, state):
    """
    Unstable dynamics model with drift, wind effects, and non-linear instabilities.
    This model is designed to be challenging and require multiple replanning iterations.

    Control inputs:
    - control[0]: thrust in x-direction [-1, 1]
    - control[1]: thrust in y-direction [-1, 1]

    Features that make this challenging:
    - Momentum/inertia effects (velocity doesn't change instantly)
    - Environmental drift (constant wind/current)
    - Position-dependent disturbances (turbulence zones)
    - Velocity-dependent drag
    - Control delay/lag
    - Non-linear instabilities at certain regions
    """

    # Extract current state
    x = start.getX()
    y = start.getY()
    yaw = start.getYaw()

    # Control parameters
    MAX_THRUST = 0.5
    DRAG_COEFFICIENT = 0.3
    INERTIA_DAMPING = 0.7  # How quickly velocity responds to control

    # Environmental effects
    WIND_X = 0.15  # Constant wind/drift in x direction
    WIND_Y = 0.08  # Constant wind/drift in y direction

    # Estimate current velocity from yaw (simplified momentum model)
    # In a real system, velocity would be part of state space
    # For this demo, we derive it from orientation and add complexity
    base_velocity = 0.3
    vel_x = (
        base_velocity * cos(yaw) * 0.7
    )  # Not exactly aligned with orientation
    vel_y = base_velocity * sin(yaw) * 0.7

    # Process control inputs with saturation
    thrust_x = np.clip(control[0], -1.0, 1.0) * MAX_THRUST
    thrust_y = np.clip(control[1], -1.0, 1.0) * MAX_THRUST

    # Add position-dependent disturbances (turbulence zones)
    # Create disturbance zones that make certain areas harder to navigate
    disturbance_x = 0.0
    disturbance_y = 0.0

    # Turbulence zone 1: Around center-left region
    if -0.5 < x < 0.0 and -0.2 < y < 0.3:
        turb_strength = 0.25
        disturbance_x += turb_strength * sin(x * 20) * cos(y * 15)
        disturbance_y += turb_strength * cos(x * 15) * sin(y * 20)

    # Turbulence zone 2: Around goal region (makes final approach challenging)
    goal_x, goal_y = 0.0, 0.5
    dist_to_goal = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
    if dist_to_goal < 0.2:
        # Swirling pattern around goal
        angle_to_goal = np.arctan2(y - goal_y, x - goal_x)
        swirl_strength = (
            0.2 * (0.2 - dist_to_goal) / 0.2
        )  # Stronger closer to goal
        disturbance_x += swirl_strength * sin(angle_to_goal + np.pi / 2)
        disturbance_y += swirl_strength * cos(angle_to_goal + np.pi / 2)

    # Add velocity-dependent drag
    vel_magnitude = np.sqrt(vel_x**2 + vel_y**2)
    if vel_magnitude > 0:
        drag_x = -DRAG_COEFFICIENT * vel_x * vel_magnitude
        drag_y = -DRAG_COEFFICIENT * vel_y * vel_magnitude
    else:
        drag_x = drag_y = 0.0

    # Add instability based on position (creates drift in certain regions)
    if x > 0.2:  # Right side is more unstable
        instability_factor = 0.15
        disturbance_x += instability_factor * sin(y * 8)
        disturbance_y += instability_factor * cos(x * 8)

    # Combine all forces
    total_force_x = thrust_x + WIND_X + disturbance_x + drag_x
    total_force_y = thrust_y + WIND_Y + disturbance_y + drag_y

    # Apply inertial effects (velocity doesn't change instantly)
    # Current velocity influences response to new forces
    accel_x = total_force_x * INERTIA_DAMPING + vel_x * (1.0 - INERTIA_DAMPING)
    accel_y = total_force_y * INERTIA_DAMPING + vel_y * (1.0 - INERTIA_DAMPING)

    # Update velocity with acceleration
    new_vel_x = vel_x + accel_x * duration
    new_vel_y = vel_y + accel_y * duration

    # Limit maximum velocity to prevent numerical issues
    MAX_VELOCITY = 1.0
    vel_magnitude = np.sqrt(new_vel_x**2 + new_vel_y**2)
    if vel_magnitude > MAX_VELOCITY:
        new_vel_x = new_vel_x / vel_magnitude * MAX_VELOCITY
        new_vel_y = new_vel_y / vel_magnitude * MAX_VELOCITY

    # Use multiple integration steps for better numerical stability
    dt = duration / 4.0  # 4 substeps
    current_x, current_y = x, y
    current_vel_x, current_vel_y = new_vel_x, new_vel_y

    for step in range(4):
        # Update position
        current_x += current_vel_x * dt
        current_y += current_vel_y * dt

        # Add small random perturbations to make planning more challenging
        noise_scale = 0.008
        current_x += noise_scale * (np.random.random() - 0.5) * dt
        current_y += noise_scale * (np.random.random() - 0.5) * dt

        # Recompute disturbances based on new position
        step_disturbance_x = 0.0
        step_disturbance_y = 0.0

        # Turbulence zones (recalculated)
        if -0.5 < current_x < 0.0 and -0.2 < current_y < 0.3:
            turb_strength = 0.25
            step_disturbance_x += (
                turb_strength * sin(current_x * 20) * cos(current_y * 15)
            )
            step_disturbance_y += (
                turb_strength * cos(current_x * 15) * sin(current_y * 20)
            )

        # Goal region swirl
        dist_to_goal = np.sqrt(
            (current_x - goal_x) ** 2 + (current_y - goal_y) ** 2
        )
        if dist_to_goal < 0.2:
            angle_to_goal = np.arctan2(current_y - goal_y, current_x - goal_x)
            swirl_strength = 0.2 * (0.2 - dist_to_goal) / 0.2
            step_disturbance_x += swirl_strength * sin(
                angle_to_goal + np.pi / 2
            )
            step_disturbance_y += swirl_strength * cos(
                angle_to_goal + np.pi / 2
            )

        # Apply disturbances to velocity
        current_vel_x += step_disturbance_x * dt * 0.5
        current_vel_y += step_disturbance_y * dt * 0.5

    # Calculate orientation based on velocity direction (with some lag)
    if abs(current_vel_x) > 0.01 or abs(current_vel_y) > 0.01:
        desired_yaw = np.arctan2(current_vel_y, current_vel_x)
        # Add some orientation lag - yaw doesn't instantly align with velocity
        yaw_diff = desired_yaw - yaw
        # Normalize angle difference
        while yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        while yaw_diff < -np.pi:
            yaw_diff += 2 * np.pi

        # Gradual yaw change
        yaw_rate = 3.0  # How quickly orientation changes
        new_yaw = yaw + yaw_diff * yaw_rate * duration
    else:
        new_yaw = yaw

    # Normalize yaw to [-π, π]
    while new_yaw > np.pi:
        new_yaw -= 2 * np.pi
    while new_yaw < -np.pi:
        new_yaw += 2 * np.pi

    # Set final state
    state.setX(current_x)
    state.setY(current_y)
    state.setYaw(new_yaw)


def propagate_pendulum(start, control, duration, state):
    """
    Inverted pendulum dynamics - extremely challenging to control.
    This creates a highly unstable system that requires constant correction.

    Control inputs:
    - control[0]: force in x-direction
    - control[1]: force in y-direction

    The system behaves like an inverted pendulum trying to balance at each point,
    making it very difficult to maintain a straight path to the goal.
    """

    # Extract current state
    x = start.getX()
    y = start.getY()
    yaw = start.getYaw()

    # Pendulum parameters
    PENDULUM_LENGTH = 0.3
    GRAVITY = 9.81
    DAMPING = 0.8
    CONTROL_FORCE_SCALE = 2.0

    # Process controls
    force_x = control[0] * CONTROL_FORCE_SCALE
    force_y = control[1] * CONTROL_FORCE_SCALE

    # Treat yaw as the pendulum angle deviation from vertical
    # The system tries to fall over, and controls must prevent this

    # Pendulum equation: theta_ddot = (g/L) * sin(theta) + disturbances
    # where theta is the angle from vertical (stored in yaw)

    # Angular acceleration from gravity (destabilizing)
    gravity_accel = (GRAVITY / PENDULUM_LENGTH) * sin(yaw)

    # Control torque from forces (stabilizing if applied correctly)
    control_torque_x = force_x / PENDULUM_LENGTH
    control_torque_y = force_y / PENDULUM_LENGTH

    # Convert to angular acceleration
    # This is a simplified model - real pendulum would be more complex
    total_angular_accel = (
        gravity_accel
        - control_torque_x * cos(yaw)
        - control_torque_y * sin(yaw)
    )

    # Add damping
    # Estimate angular velocity from previous state changes
    angular_velocity = yaw * 2.0  # Simplified estimate
    total_angular_accel -= DAMPING * angular_velocity

    # Update angular velocity and position
    new_angular_vel = angular_velocity + total_angular_accel * duration
    new_yaw = yaw + new_angular_vel * duration

    # The pendulum movement affects x,y position
    # As it falls in one direction, the base moves in the opposite direction
    delta_x = -PENDULUM_LENGTH * (sin(new_yaw) - sin(yaw))
    delta_y = -PENDULUM_LENGTH * (cos(new_yaw) - cos(yaw))

    # Add direct control forces for translation
    translation_x = force_x * duration * 0.3
    translation_y = force_y * duration * 0.3

    # Add environmental disturbances
    disturbance_x = 0.02 * sin(x * 10 + y * 7) * duration
    disturbance_y = 0.02 * cos(x * 7 + y * 10) * duration

    # Combine all effects
    new_x = x + delta_x + translation_x + disturbance_x
    new_y = y + delta_y + translation_y + disturbance_y

    # Normalize yaw
    while new_yaw > np.pi:
        new_yaw -= 2 * np.pi
    while new_yaw < -np.pi:
        new_yaw += 2 * np.pi

    # Set final state
    state.setX(new_x)
    state.setY(new_y)
    state.setYaw(new_yaw)
