#!/usr/bin/env python3

import my_custom_planner_module as mcp


def test_configurable_planner():
    """Test the new configurable run_sst_planner function"""

    print("=== Testing Configurable SST Planner ===\n")

    # Test 1: Default parameters
    print("Test 1: Default Parameters")
    print("-" * 40)
    result = mcp.run_sst_planner()
    print(result)
    print()

    # Test 2: Custom start/goal positions
    print("Test 2: Custom Start/Goal Positions")
    print("-" * 40)
    result = mcp.run_sst_planner(
        start_pos=[0.0, -0.8, 0.0], goal_pos=[0.0, 0.8, 3.14159]
    )
    print(result)
    print()

    # Test 3: Faster solve time
    print("Test 3: Quick Solve (3 seconds)")
    print("-" * 40)
    result = mcp.run_sst_planner(solve_time=3.0)
    print(result)
    print()

    # Test 4: Different bounds
    print("Test 4: Larger Space Bounds")
    print("-" * 40)
    result = mcp.run_sst_planner(
        start_pos=[-1.5, -1.5, 0.0],
        goal_pos=[1.5, 1.5, 0.0],
        space_bounds=[-2.0, 2.0],
    )
    print(result)
    print()

    # Test 5: Tighter goal tolerance
    print("Test 5: Tighter Goal Tolerance")
    print("-" * 40)
    result = mcp.run_sst_planner(goal_tolerance=0.01, solve_time=5.0)
    print(result)
    print()

    # Test 6: Different control bounds
    print("Test 6: Different Control Bounds")
    print("-" * 40)
    result = mcp.run_sst_planner(
        control_bounds=[-0.5, 0.5], solve_time=5.0  # More aggressive controls
    )
    print(result)
    print()

    print("=== All Tests Complete ===")


if __name__ == "__main__":
    test_configurable_planner()
