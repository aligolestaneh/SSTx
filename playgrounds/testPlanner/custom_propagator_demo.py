#!/usr/bin/env python3

import my_custom_planner_module as mcp
import numpy as np
import math


def demo_custom_propagators():
    """Demonstrate using custom Python propagators with C++ SST planner"""

    print("=== Custom Python Propagator Demo ===\n")

    # 1. Simple kinematic car model (same as default)
    def simple_car_propagator(start_state, control_input, duration):
        """
        Simple kinematic car model
        Args:
            start_state: [x, y, yaw] - current position and orientation
            control_input: [v, omega] - linear velocity and angular velocity
            duration: float - time step
        Returns:
            [x_new, y_new, yaw_new] - new state after propagation
        """
        x, y, yaw = start_state
        v, omega = control_input

        # Simple kinematic model: x' = v*cos(yaw), y' = v*sin(yaw), yaw' = omega
        x_new = x + v * duration * math.cos(yaw)
        y_new = y + v * duration * math.sin(yaw)
        yaw_new = yaw + omega * duration

        return [x_new, y_new, yaw_new]

    # 2. Car with dynamics (acceleration/deceleration)
    def dynamic_car_propagator(start_state, control_input, duration):
        """
        Car model with dynamics - control input affects acceleration
        Args:
            start_state: [x, y, yaw] - position and orientation
            control_input: [acceleration, angular_velocity] - controls
            duration: float - time step
        Returns:
            [x_new, y_new, yaw_new] - new state after propagation
        """
        x, y, yaw = start_state
        acceleration, omega = control_input

        # Assume current velocity is small (could be extended to include velocity in state)
        v_current = 0.1  # Small base velocity
        v_new = v_current + acceleration * duration

        # Limit velocity
        v_new = max(-0.5, min(0.5, v_new))

        # Propagate using average velocity
        v_avg = (v_current + v_new) / 2
        x_new = x + v_avg * duration * math.cos(yaw)
        y_new = y + v_avg * duration * math.sin(yaw)
        yaw_new = yaw + omega * duration

        return [x_new, y_new, yaw_new]

    # 3. Dubins car model with minimum turning radius
    def dubins_car_propagator(start_state, control_input, duration):
        """
        Dubins car model with minimum turning radius constraint
        Args:
            start_state: [x, y, yaw] - position and orientation
            control_input: [v, steering_angle] - velocity and steering
            duration: float - time step
        Returns:
            [x_new, y_new, yaw_new] - new state after propagation
        """
        x, y, yaw = start_state
        v, steering_angle = control_input

        # Convert steering angle to angular velocity (with wheelbase)
        wheelbase = 0.2  # meters
        max_steering = 0.5  # max steering angle
        steering_angle = max(-max_steering, min(max_steering, steering_angle))

        if abs(steering_angle) > 1e-6:
            omega = v * math.tan(steering_angle) / wheelbase
        else:
            omega = 0.0

        # Propagate
        x_new = x + v * duration * math.cos(yaw)
        y_new = y + v * duration * math.sin(yaw)
        yaw_new = yaw + omega * duration

        return [x_new, y_new, yaw_new]

    # 4. Noisy propagator (adds uncertainty)
    def noisy_propagator(start_state, control_input, duration):
        """
        Propagator with noise/uncertainty
        """
        x, y, yaw = start_state
        v, omega = control_input

        # Add small random noise
        noise_x = np.random.normal(0, 0.01)
        noise_y = np.random.normal(0, 0.01)
        noise_yaw = np.random.normal(0, 0.05)

        # Basic kinematic model with noise
        x_new = x + v * duration * math.cos(yaw) + noise_x
        y_new = y + v * duration * math.sin(yaw) + noise_y
        yaw_new = yaw + omega * duration + noise_yaw

        return [x_new, y_new, yaw_new]

    # Test different propagators
    propagators = [
        ("Simple Kinematic Car", simple_car_propagator),
        ("Dynamic Car (with acceleration)", dynamic_car_propagator),
        ("Dubins Car (steering constraints)", dubins_car_propagator),
        ("Noisy Propagator", noisy_propagator),
    ]

    start_pos = [-0.8, 0.0, 0.0]
    goal_pos = [0.8, 0.0, 0.0]

    for name, propagator in propagators:
        print(f"\n=== Testing: {name} ===")
        print("-" * 50)

        try:
            # Test planning with custom propagator
            result = mcp.run_sst_with_custom_propagator(
                propagator,
                start_pos=start_pos,
                goal_pos=goal_pos,
                solve_time=5.0,
            )
            print(result)

            # Get path if solution found
            if "found a solution" in result:
                path = mcp.run_sst_get_path_custom_propagator(
                    propagator,
                    start_pos=start_pos,
                    goal_pos=goal_pos,
                    solve_time=3.0,
                )
                print(f"Path length: {len(path)} waypoints")
                if path:
                    print(f"Start: {path[0]}")
                    print(f"End: {path[-1]}")

                    # Calculate path statistics
                    total_distance = 0
                    for i in range(len(path) - 1):
                        dx = path[i + 1][0] - path[i][0]
                        dy = path[i + 1][1] - path[i][1]
                        total_distance += math.sqrt(dx * dx + dy * dy)
                    print(f"Total path distance: {total_distance:.3f}")

        except Exception as e:
            print(f"Error with {name}: {e}")

        print()

    # Demo: Comparison with default propagator
    print("\n=== Comparison with Default Propagator ===")
    print("-" * 50)

    print("Default C++ propagator:")
    default_result = mcp.run_sst_planner(
        start_pos=start_pos, goal_pos=goal_pos, solve_time=5.0
    )
    print(default_result)

    print("\nCustom Python propagator (same logic):")
    custom_result = mcp.run_sst_with_custom_propagator(
        simple_car_propagator,
        start_pos=start_pos,
        goal_pos=goal_pos,
        solve_time=5.0,
    )
    print(custom_result)


def test_propagator_interface():
    """Test the propagator interface with different scenarios"""

    print("\n=== Testing Propagator Interface ===\n")

    def test_propagator(start_state, control_input, duration):
        """Test propagator that prints inputs for debugging"""
        print(
            f"  Propagator called: start={start_state}, control={control_input}, dt={duration}"
        )

        # Simple implementation
        x, y, yaw = start_state
        v, omega = control_input

        x_new = x + v * duration * math.cos(yaw)
        y_new = y + v * duration * math.sin(yaw)
        yaw_new = yaw + omega * duration

        result = [x_new, y_new, yaw_new]
        print(f"  Returning: {result}")
        return result

    print("Running with debug propagator (shows function calls):")
    result = mcp.run_sst_with_custom_propagator(
        test_propagator,
        start_pos=[0.0, 0.0, 0.0],
        goal_pos=[0.2, 0.0, 0.0],  # Short distance for quick solve
        solve_time=2.0,
    )
    print(f"Final result: {result}")


if __name__ == "__main__":
    demo_custom_propagators()
    test_propagator_interface()
