import time
import torch
import torch.nn.functional as F

import numpy as np
from math import cos, sin, tan
from scipy.spatial.transform import Rotation as SciRot


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
        ROLLING_RESISTANCE * current_velocity + DRAG_COEFFICIENT * current_velocity**2
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
        current_angular_vel = (current_vel / WHEELBASE) * np.tan(steering_angle)
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


def DublinsAirplaneDynamics(state, control, duration):
    """
    Calculate the dynamics for Dublin's airplane model over a given duration.

    Args:
        state: [x, y, z, psi, gamma, phi] - position and orientation (Euler angles)
        control: [phi_rate, gamma_rate, v] - control inputs
        duration: time duration for integration

    Returns:
        final_state: [x, y, z, psi, gamma, phi] after integration
    """
    print(f"DublinsAirplaneDynamics: state: {state}, control: {control}, duration: {duration}")

    # Handle both 7-element (quaternion) and 6-element (Euler) state formats
    if len(state) == 7:
        # Convert from quaternion to Euler angles
        x, y, z, qx, qy, qz, qw = state
        psi, gamma, phi = SciRot.from_quat([qx, qy, qz, qw]).as_euler("zyx", degrees=False)
    elif len(state) == 6:
        # Already in Euler format
        x, y, z, psi, gamma, phi = state
    else:
        raise ValueError(f"Expected state with 6 or 7 elements, got {len(state)}")

    phi_rate, gamma_rate, v = control[0], control[1], control[2]

    def f(sv):
        x, y, z, psi, gamma, phi = sv
        x_dot = v * np.cos(psi) * np.cos(gamma)
        y_dot = v * np.sin(psi) * np.cos(gamma)
        z_dot = v * np.sin(gamma)
        psi_dot = (9.81 / v) * np.tan(phi) if v > 1e-6 else 0.0
        gamma_dot = gamma_rate
        phi_dot = phi_rate
        return np.array([x_dot, y_dot, z_dot, psi_dot, gamma_dot, phi_dot], dtype=float)

    # RK4 integration over the duration
    s = np.array([x, y, z, psi, gamma, phi], dtype=float)
    total = float(duration)
    sub_dt = 0.01
    steps = max(1, int(round(total / sub_dt))) if total > 0 else 1
    dt = total / steps if steps > 0 else total

    for _ in range(steps):
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # wrap psi to [-pi, pi]
        s[3] = (s[3] + np.pi) % (2 * np.pi) - np.pi

    return s.tolist()


def propagateDublinsAirplane(start, control, duration, state):
    try:
        # print("Start:")
        # print(
        #     f"x: {start.getX()}, y: {start.getY()}, z: {start.getZ()}, quaternion: {start.rotation()}"
        # )
        # Extract start SE3 pose
        x0 = float(start.getX())
        y0 = float(start.getY())
        z0 = float(start.getZ())
        rot = start.rotation()
        # OMPL stores quaternion as w,x,y,z; scipy expects x,y,z,w
        q_xyzw = [float(rot.x), float(rot.y), float(rot.z), float(rot.w)]
        # Get Euler angles: psi (yaw), gamma (pitch), phi (roll)
        # Use ZYX order: returns [yaw(z), pitch(y), roll(x)]
        psi0, gamma0, phi0 = SciRot.from_quat(q_xyzw).as_euler("zyx", degrees=False)

        # Read control robustly: [phi_rate, gamma_rate, v]
        try:
            phi_rate = float(control[0])
            gamma_rate = float(control[1])
            v = float(control[2])
        except Exception:
            from ompl import control as oc

            phi_rate = float(oc.RealVectorControlSpace.getValue(control, 0))
            gamma_rate = float(oc.RealVectorControlSpace.getValue(control, 1))
            v = float(oc.RealVectorControlSpace.getValue(control, 2))

        # State vector s = [x,y,z, psi,yaw, gamma,pitch, phi,roll]
        s = np.array([x0, y0, z0, psi0, gamma0, phi0], dtype=float)

        def f(sv):
            x, y, z, psi, gamma, phi = sv
            x_dot = v * np.cos(psi) * np.cos(gamma)
            y_dot = v * np.sin(psi) * np.cos(gamma)
            z_dot = v * np.sin(gamma)
            psi_dot = (9.81 / v) * np.tan(phi) if v > 1e-6 else 0.0
            gamma_dot = gamma_rate
            phi_dot = phi_rate
            return np.array([x_dot, y_dot, z_dot, psi_dot, gamma_dot, phi_dot], dtype=float)

        # RK4 with substeps
        total = float(duration)
        sub_dt = 0.01
        steps = max(1, int(round(total / sub_dt))) if total > 0 else 1
        dt = total / steps if steps > 0 else total
        for _ in range(steps):
            k1 = f(s)
            k2 = f(s + 0.5 * dt * k1)
            k3 = f(s + 0.5 * dt * k2)
            k4 = f(s + dt * k3)
            s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            # wrap psi to [-pi, pi]
            s[3] = (s[3] + np.pi) % (2 * np.pi) - np.pi

        # Write back
        state.setX(float(s[0]))
        state.setY(float(s[1]))
        state.setZ(float(s[2]))
        # Build quaternion from Euler (ZYX -> yaw, pitch, roll)
        r_new = SciRot.from_euler("zyx", [s[3], s[4], s[5]], degrees=False)
        q_new_xyzw = r_new.as_quat()  # [x,y,z,w]
        r_out = state.rotation()
        r_out.x = float(q_new_xyzw[0])
        r_out.y = float(q_new_xyzw[1])
        r_out.z = float(q_new_xyzw[2])
        r_out.w = float(q_new_xyzw[3])
        # print("State:")
        # print(
        #     f"x: {state.getX()}, y: {state.getY()}, z: {state.getZ()}, quaternion: {state.rotation().x}, {state.rotation().y}, {state.rotation().z}, {state.rotation().w}"
        # )
        # input("Press Enter to continue...")
        return True
    except Exception as e:
        print(f"DublinsAirplane propagate error: {e}")
        return False


def carDynamics(start, control, duration):
    x, y, yaw = start
    v, w = control
    d_x = v * cos(yaw) * duration
    d_y = v * sin(yaw) * duration
    d_yaw = w * duration
    return np.array([x + d_x, y + d_y, yaw + d_yaw])


def carDynamicsTorch(
    start: torch.Tensor, control: torch.Tensor, *, duration: float
) -> torch.Tensor:
    # Ensure we have the right shapes and preserve gradients
    # start: [batch_size, 3] -> [x, y, yaw]
    # control: [batch_size, 2] -> [v, w]

    # Debug: Print input tensor properties
    if start.numel() > 0 and start.numel() <= 10:  # Only debug for small tensors
        print(f"[DEBUG] carDynamicsTorch inputs:")
        print(f"  - start shape: {start.shape}, requires_grad: {start.requires_grad}")
        print(f"  - control shape: {control.shape}, requires_grad: {control.requires_grad}")
        print(f"  - duration: {duration}")

    # Extract components using proper indexing to preserve gradients
    x = start[:, 0]  # [batch_size]
    y = start[:, 1]  # [batch_size]
    yaw = start[:, 2]  # [batch_size]

    v = control[:, 0]  # [batch_size]
    w = control[:, 1]  # [batch_size]

    # Debug: Check extracted components
    if start.numel() > 0 and start.numel() <= 10:
        print(f"  - x requires_grad: {x.requires_grad}, shape: {x.shape}")
        print(f"  - v requires_grad: {v.requires_grad}, shape: {v.shape}")

    # Compute state changes (all operations preserve gradients)
    d_x = v * torch.cos(yaw) * duration
    d_y = v * torch.sin(yaw) * duration
    d_yaw = w * duration

    # Debug: Check computed changes
    if start.numel() > 0 and start.numel() <= 10:
        print(f"  - d_x requires_grad: {d_x.requires_grad}, shape: {d_x.shape}")

    # Compute new states
    new_x = x + d_x
    new_y = y + d_y
    new_yaw = yaw + d_yaw

    # Debug: Check new states
    if start.numel() > 0 and start.numel() <= 10:
        print(f"  - new_x requires_grad: {new_x.requires_grad}, shape: {new_x.shape}")

    # Stack results to create [batch_size, 3] output
    # This preserves the computational graph and gradients
    result = torch.stack([new_x, new_y, new_yaw], dim=1)

    # Debug: Check final result
    if start.numel() > 0 and start.numel() <= 10:
        print(f"  - result requires_grad: {result.requires_grad}, shape: {result.shape}")
        print(f"  - result grad_fn: {result.grad_fn}")

    return result


def propagateCar(start, control, duration, state):
    start = np.array([start.getX(), start.getY(), start.getYaw()])
    control = np.array([control[0], control[1]])
    result = carDynamics(start, control, duration)
    time.sleep(0.0002)
    state.setX(result[0])
    state.setY(result[1])
    state.setYaw(result[2])
