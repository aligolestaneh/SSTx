#!/usr/bin/env python3
"""
Simple test to verify planning_utils can be imported and basic functions work
"""

import sys
import numpy as np

# Test the import
try:
    from planning_utils import (
        isStateValid,
        propagate_simple,
        propagate_complex,
    )

    print("✓ Successfully imported functions from planning_utils")
except ImportError as e:
    print(f"✗ Failed to import from planning_utils: {e}")
    sys.exit(1)


# Create mock state and control objects for testing
class MockState:
    def __init__(self, x=0, y=0, yaw=0):
        self.x = x
        self.y = y
        self.yaw = yaw

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getYaw(self):
        return self.yaw

    def setX(self, x):
        self.x = x

    def setY(self, y):
        self.y = y

    def setYaw(self, yaw):
        self.yaw = yaw


class MockSpaceInformation:
    def satisfiesBounds(self, state):
        # Simple bounds check: -2 to 2 for x and y
        return -2 <= state.getX() <= 2 and -2 <= state.getY() <= 2


class MockControl:
    def __init__(self, control_values):
        self.values = control_values

    def __getitem__(self, index):
        return self.values[index]


def test_isStateValid():
    """Test the isStateValid function"""
    print("\nTesting isStateValid:")

    si = MockSpaceInformation()

    # Test valid state
    valid_state = MockState(1.0, 0.5, 0.0)
    result = isStateValid(si, valid_state)
    print(f"  Valid state (1.0, 0.5, 0.0): {result}")

    # Test invalid state
    invalid_state = MockState(3.0, 0.5, 0.0)  # x > 2
    result = isStateValid(si, invalid_state)
    print(f"  Invalid state (3.0, 0.5, 0.0): {result}")


def test_propagate_simple():
    """Test the simple propagate function"""
    print("\nTesting propagate_simple:")

    start_state = MockState(0.0, 0.0, 0.0)
    control = MockControl([0.5, 0.1])  # velocity=0.5, steering=0.1
    duration = 1.0
    end_state = MockState()

    try:
        propagate_simple(start_state, control, duration, end_state)
        print(
            f"  Start: ({start_state.getX():.3f}, {start_state.getY():.3f}, {start_state.getYaw():.3f})"
        )
        print(
            f"  End:   ({end_state.getX():.3f}, {end_state.getY():.3f}, {end_state.getYaw():.3f})"
        )
        print("  ✓ propagate_simple executed successfully")
    except Exception as e:
        print(f"  ✗ propagate_simple failed: {e}")


def test_propagate_complex():
    """Test the complex propagate function"""
    print("\nTesting propagate_complex:")

    start_state = MockState(0.0, 0.0, 0.0)
    control = MockControl([0.5, 0.1])  # accel=0.5, steering=0.1
    duration = 0.1  # Short duration to avoid large changes
    end_state = MockState()

    try:
        propagate_complex(start_state, control, duration, end_state)
        print(
            f"  Start: ({start_state.getX():.3f}, {start_state.getY():.3f}, {start_state.getYaw():.3f})"
        )
        print(
            f"  End:   ({end_state.getX():.3f}, {end_state.getY():.3f}, {end_state.getYaw():.3f})"
        )
        print("  ✓ propagate_complex executed successfully")
    except Exception as e:
        print(f"  ✗ propagate_complex failed: {e}")


if __name__ == "__main__":
    print("Testing planning_utils functions...")

    test_isStateValid()
    test_propagate_simple()
    test_propagate_complex()

    print("\n✓ All tests completed!")
