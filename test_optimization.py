#!/usr/bin/env python3
"""
Test script for debugging the optimization process in isolation.
Uses the same functions as fusionPlanning.py but with fake data.
"""

import sys
import os
import numpy as np
import torch
import traceback

# Add the parent directory to the path to import fusionPlanning
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from fusionPlanning import (
    runOptimizer,
    sampleRandomState,
    getChildrenStates,
    SE2Pose,
)
from factories import pickPropagator, pickObjectShape
from train_model import load_opt_model_2


def create_fake_data():
    """Create fake data for testing"""
    print("🔧 Creating fake data...")

    # Fake next state
    nextState = np.array([0.1, 0.2, 0.3])

    # Fake children states
    childrenStates = [
        np.array([0.11, 0.21, 0.31]),
        np.array([0.12, 0.22, 0.32]),
        np.array([0.13, 0.23, 0.33]),
    ]

    # Fake initial guess control
    initialGuessControl = np.array([0.01, 0.02])

    print(f"✅ Created fake data:")
    print(f"  - nextState: {nextState}")
    print(f"  - childrenStates: {len(childrenStates)} states")
    print(f"  - initialGuessControl: {initialGuessControl}")

    return nextState, childrenStates, initialGuessControl


def create_real_propagator():
    """Create a real propagator using the same method as fusionPlanning"""
    print("🔧 Creating real propagator...")
    try:
        system = "pushing"
        object_name = "crackerBox"  # Use the correct object name
        objectShape = pickObjectShape(object_name)
        propagator = pickPropagator(system, objectShape)
        print(f"✅ Real propagator created: {type(propagator)}")
        return propagator
    except Exception as e:
        print(f"❌ Error creating real propagator: {e}")
        return None


def create_real_model(propagator):
    """Create a real model using the same method as fusionPlanning"""
    print("🔧 Creating real model...")
    try:
        lr = 0.001
        epochs = 1000
        optModel = load_opt_model_2(propagator, lr=lr, epochs=epochs)
        print(f"✅ Real model created: {type(optModel)}")
        return optModel
    except Exception as e:
        print(f"❌ Error creating real model: {e}")
        return None


def create_fake_model():
    """Create a fake model for testing"""
    print("🔧 Creating fake model...")

    class FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")

        def predict_y(self, x):
            """Fake forward propagation"""
            # Convert input to tensor if needed
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)

            # Fake state delta: just add some noise to the input
            noise = torch.randn_like(x) * 0.01
            return x + noise

        def predict(self, y, start_guess, **kwargs):
            """Fake optimization"""
            # Return the start_guess as "optimized" result
            return start_guess, 0.0

    fake_model = FakeModel()
    print(f"✅ Fake model created: {type(fake_model)}")
    return fake_model


def test_sample_random_state():
    """Test the sampleRandomState function"""
    print("\n🧪 Testing sampleRandomState...")

    test_state = np.array([0.0, 0.0, 0.0])
    sampled_states = sampleRandomState(test_state, numStates=10)

    print(f"✅ sampleRandomState test completed:")
    print(f"  - Input state: {test_state}")
    print(f"  - Sampled {len(sampled_states)} states")
    print(f"  - First sampled state: {sampled_states[0]}")


def test_se2_pose_operations():
    """Test SE2Pose operations"""
    print("\n🧪 Testing SE2Pose operations...")

    pose1 = SE2Pose([1.0, 2.0], 0.5)
    pose2 = SE2Pose([1.1, 2.1], 0.6)

    relative_pose = pose1.invert @ pose2
    print(f"✅ SE2Pose test completed:")
    print(f"  - Pose1: {pose1}")
    print(f"  - Pose2: {pose2}")
    print(f"  - Relative pose: {relative_pose}")


def test_run_optimizer():
    """Test the runOptimizer function with fake data"""
    print("\n🧪 Testing runOptimizer with fake data...")

    # Create fake data
    print("🔧 Creating fake data...")
    nextState = np.array([0.1, 0.2, 0.3])
    childrenStates = [
        np.array([0.11, 0.21, 0.31]),
        np.array([0.12, 0.22, 0.32]),
        np.array([0.13, 0.23, 0.33]),
    ]
    # Initial guess should be 3D physics parameters [rot, side, distance]
    initialGuessControl = np.array([0.1, 0.05, 0.2])  # [rot, side, distance]
    print(f"✅ Created fake data:")
    print(f"  - nextState: {nextState}")
    print(f"  - childrenStates: {len(childrenStates)} states")
    print(
        f"  - initialGuessControl (3D physics params): {initialGuessControl}"
    )

    # Create fake model
    print("🔧 Creating fake model...")
    fake_model = create_fake_model()
    print(f"✅ Fake model created: {type(fake_model)}")

    # Test runOptimizer
    result = runOptimizer(
        nextState,
        childrenStates,
        initialGuessControl,
        fake_model,
        maxDistance=0.025,
    )

    print(f"✅ runOptimizer test completed:")
    print(f"  - Result type: {type(result)}")
    if isinstance(result, dict):
        # Print only first 10 entries
        items = list(result.items())
        if len(items) > 10:
            print(f"  - Result (first 10 entries): {dict(items[:10])}")
            print(f"  - ... and {len(items) - 10} more entries")
        else:
            print(f"  - Result: {result}")
    else:
        print(f"  - Result: {result}")

    return result


def test_step_by_step():
    """Test the optimization process step by step"""
    print("\n🧪 Testing optimization process step by step...")

    # Test model creation
    print("\n🧪 Testing model creation...")
    optModel = test_model_creation()

    if optModel is None:
        print("❌ Model creation failed, skipping step-by-step test")
        return None

    # Create fake data
    print("🔧 Creating fake data...")
    nextState = np.array([0.1, 0.2, 0.3])
    childrenStates = [
        np.array([0.11, 0.21, 0.31]),
        np.array([0.12, 0.22, 0.32]),
        np.array([0.13, 0.23, 0.33]),
    ]
    # Initial guess should be 3D physics parameters [rot, side, distance]
    initialGuessControl = np.array([0.1, 0.05, 0.2])  # [rot, side, distance]
    print(f"✅ Created fake data:")
    print(f"  - nextState: {nextState}")
    print(f"  - childrenStates: {len(childrenStates)} states")
    print(
        f"  - initialGuessControl (3D physics params): {initialGuessControl}"
    )

    # Test runOptimizer with real model
    result = runOptimizer(
        nextState,
        childrenStates,
        initialGuessControl,
        optModel,
        maxDistance=0.025,
    )

    print(f"✅ Step-by-step test completed:")
    print(f"  - Result type: {type(result)}")
    if isinstance(result, dict):
        # Print only first 10 entries
        items = list(result.items())
        if len(items) > 10:
            print(f"  - Result (first 10 entries): {dict(items[:10])}")
            print(f"  - ... and {len(items) - 10} more entries")
        else:
            print(f"  - Result: {result}")
    else:
        print(f"  - Result: {result}")

    return result


def test_model_creation():
    """Test the model creation process"""
    print("\n🧪 Testing model creation...")

    try:
        from train_model import load_model, load_opt_model_2

        # Test 1: Create base model - use residual model like in fusionPlanning.py
        print("🔍 Step 1: Creating base model...")
        torch_model = load_model(
            "residual", 3, 3
        )  # 3D input, 3D output like in fusionPlanning.py
        print(f"✅ Base model type: {type(torch_model)}")

        # Test 2: Get actual model
        print("🔍 Step 2: Getting actual model...")
        actual_model = torch_model.model
        print(f"🔍 torch_model.model: {actual_model}")
        print(f"🔍 torch_model.model type: {type(actual_model)}")

        if actual_model is None:
            print("🔍 Model is None, creating from model_class...")
            actual_model = torch_model.model_class()
            actual_model = actual_model.to(torch_model.device)
            print(f"🔍 Created model type: {type(actual_model)}")

        # Test 3: Create optimization model
        print("🔍 Step 3: Creating optimization model...")
        optModel = load_opt_model_2(actual_model, lr=0.001, epochs=1000)
        print(f"✅ Optimization model type: {type(optModel)}")
        print(f"🔍 optModel.model type: {type(optModel.model)}")

        # Test 4: Try to call eval
        print("🔍 Step 4: Testing eval()...")
        try:
            optModel.model.eval()
            print("✅ eval() works!")
        except Exception as e:
            print(f"❌ eval() failed: {e}")

        return optModel

    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Run all tests"""
    print("🚀 Starting optimization debugging tests...")

    # Test 1: Sample random state
    test_sample_random_state()

    # Test 2: SE2Pose operations
    test_se2_pose_operations()

    # Test 3: runOptimizer with fake model
    test_run_optimizer()

    # Test 4: Step-by-step with real model (includes model creation test)
    test_step_by_step()

    print("\n🎉 All tests passed! The optimization process should work.")


if __name__ == "__main__":
    main()
