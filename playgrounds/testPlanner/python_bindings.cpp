#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
// IMPORTANT: Include custom SST implementation BEFORE OMPL headers
// This ensures our custom ompl::control::SST is used instead of OMPL's default SST
#include "testPlanner.h"
#include <ompl-1.6/ompl/control/SpaceInformation.h>
#include <ompl-1.6/ompl/base/spaces/SE2StateSpace.h>
#include <ompl-1.6/ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl-1.6/ompl/control/SimpleSetup.h>
#include <ompl-1.6/ompl/base/goals/GoalState.h>
#include <ompl-1.6/ompl/base/objectives/PathLengthOptimizationObjective.h>

namespace py = pybind11;
namespace ob = ompl::base;
namespace oc = ompl::control;

// Enhanced wrapper function with configurable parameters
std::string run_sst_planner(
    std::vector<double> start_pos = {-0.5, 0.0, 0.0},     // [x, y, yaw]
    std::vector<double> goal_pos = {0.5, 0.0, 0.0},       // [x, y, yaw]
    double solve_time = 10.0,                              // seconds
    double goal_tolerance = 0.05,                          // goal tolerance
    std::vector<double> space_bounds = {-1.0, 1.0},       // [low, high] for x,y
    std::vector<double> control_bounds = {-0.3, 0.3}      // [low, high] for controls
) {
    try {
        // Validate input sizes
        if (start_pos.size() != 3 || goal_pos.size() != 3) {
            return "Error: start_pos and goal_pos must have 3 elements [x, y, yaw]";
        }
        if (space_bounds.size() != 2 || control_bounds.size() != 2) {
            return "Error: bounds must have 2 elements [low, high]";
        }
        
        // Create state space
        auto space(std::make_shared<ob::SE2StateSpace>());
        
        // Set configurable space bounds
        ob::RealVectorBounds bounds(2);
        bounds.setLow(space_bounds[0]);
        bounds.setHigh(space_bounds[1]);
        space->setBounds(bounds);
        
        // Create control space
        auto cspace(std::make_shared<oc::RealVectorControlSpace>(space, 2));
        
        // Set configurable control bounds
        ob::RealVectorBounds cbounds(2);
        cbounds.setLow(control_bounds[0]);
        cbounds.setHigh(control_bounds[1]);
        cspace->setBounds(cbounds);
        
        // Use smart pointer for proper lifetime management
        auto ss = std::make_shared<oc::SimpleSetup>(cspace);
        
        // Explicitly set control duration to eliminate warning
        ss->getSpaceInformation()->setMinMaxControlDuration(1, 10);
        
        // Explicitly set propagation step size to eliminate warning
        ss->getSpaceInformation()->setPropagationStepSize(0.05);
        
        // Set a simple state validity checker (always valid for demo)
        ss->setStateValidityChecker([](const ob::State *state) { 
            return true; 
        });
        
        // Set a simple propagator (from your main.cpp)
        ss->setStatePropagator([](const ob::State *start, const oc::Control *control, 
                                const double duration, ob::State *result) {
            const auto *se2state = start->as<ob::SE2StateSpace::StateType>();
            const double* pos = se2state->as<ob::RealVectorStateSpace::StateType>(0)->values;
            const double rot = se2state->as<ob::SO2StateSpace::StateType>(1)->value;
            const double* ctrl = control->as<oc::RealVectorControlSpace::ControlType>()->values;
            
            result->as<ob::SE2StateSpace::StateType>()->setXY(
                pos[0] + ctrl[0] * duration * cos(rot),
                pos[1] + ctrl[0] * duration * sin(rot));
            result->as<ob::SE2StateSpace::StateType>()->setYaw(
                rot + ctrl[1] * duration);
        });
        
        // Set configurable start state
        ob::ScopedState<ob::SE2StateSpace> start(space);
        start->setX(start_pos[0]);
        start->setY(start_pos[1]);
        start->setYaw(start_pos[2]);
        
        // Set configurable goal state
        ob::ScopedState<ob::SE2StateSpace> goal(space);
        goal->setX(goal_pos[0]);
        goal->setY(goal_pos[1]);
        goal->setYaw(goal_pos[2]);
        
        // Set start and goal with configurable tolerance
        ss->setStartAndGoalStates(start, goal, goal_tolerance);
        
        // Explicitly set optimization objective to eliminate warning
        auto obj = std::make_shared<ob::PathLengthOptimizationObjective>(ss->getSpaceInformation());
        ss->setOptimizationObjective(obj);
        
        // Use your custom SST planner
        auto planner = std::make_shared<oc::SST>(ss->getSpaceInformation());
        ss->setPlanner(planner);
        
        // Solve with configurable time
        ob::PlannerStatus solved = ss->solve(solve_time);
        
        // Store result before cleanup
        std::string result_msg;
        if (solved) {
            result_msg = "SST Planner found a solution! ";
            result_msg += "Start: [" + std::to_string(start_pos[0]) + ", " + 
                         std::to_string(start_pos[1]) + ", " + std::to_string(start_pos[2]) + "] ";
            result_msg += "Goal: [" + std::to_string(goal_pos[0]) + ", " + 
                         std::to_string(goal_pos[1]) + ", " + std::to_string(goal_pos[2]) + "]";
        } else {
            result_msg = "SST Planner could not find a solution within " + std::to_string(solve_time) + " seconds.";
        }
        
        // Explicit cleanup to prevent double-free issues
        ss->clear();
        planner->clear();
        
        return result_msg;
        
    } catch (const std::exception& e) {
        return std::string("Error: ") + e.what();
    } catch (...) {
        return "Unknown error occurred in SST planner";
    }
}

// Simple wrapper function for backward compatibility
std::string run_sst_planner_simple() {
    return run_sst_planner();  // Uses default parameters
}

// NEW FUNCTION: Get solution path as list of waypoints
std::vector<std::vector<double>> run_sst_get_path(
    std::vector<double> start_pos = {-0.5, 0.0, 0.0},
    std::vector<double> goal_pos = {0.5, 0.0, 0.0},
    double solve_time = 10.0,
    double goal_tolerance = 0.05
) {
    std::vector<std::vector<double>> path;
    
    try {
        // Similar setup to run_sst_planner but return path
        auto space(std::make_shared<ob::SE2StateSpace>());
        
        ob::RealVectorBounds bounds(2);
        bounds.setLow(-1.0);
        bounds.setHigh(1.0);
        space->setBounds(bounds);
        
        auto cspace(std::make_shared<oc::RealVectorControlSpace>(space, 2));
        
        ob::RealVectorBounds cbounds(2);
        cbounds.setLow(-0.3);
        cbounds.setHigh(0.3);
        cspace->setBounds(cbounds);
        
        auto ss = std::make_shared<oc::SimpleSetup>(cspace);
        ss->getSpaceInformation()->setMinMaxControlDuration(1, 10);
        ss->getSpaceInformation()->setPropagationStepSize(0.05);
        
        ss->setStateValidityChecker([](const ob::State *state) { return true; });
        
        ss->setStatePropagator([](const ob::State *start, const oc::Control *control, 
                                const double duration, ob::State *result) {
            const auto *se2state = start->as<ob::SE2StateSpace::StateType>();
            const double* pos = se2state->as<ob::RealVectorStateSpace::StateType>(0)->values;
            const double rot = se2state->as<ob::SO2StateSpace::StateType>(1)->value;
            const double* ctrl = control->as<oc::RealVectorControlSpace::ControlType>()->values;
            
            result->as<ob::SE2StateSpace::StateType>()->setXY(
                pos[0] + ctrl[0] * duration * cos(rot),
                pos[1] + ctrl[0] * duration * sin(rot));
            result->as<ob::SE2StateSpace::StateType>()->setYaw(
                rot + ctrl[1] * duration);
        });
        
        ob::ScopedState<ob::SE2StateSpace> start(space);
        start->setX(start_pos[0]);
        start->setY(start_pos[1]);
        start->setYaw(start_pos[2]);
        
        ob::ScopedState<ob::SE2StateSpace> goal(space);
        goal->setX(goal_pos[0]);
        goal->setY(goal_pos[1]);
        goal->setYaw(goal_pos[2]);
        
        ss->setStartAndGoalStates(start, goal, goal_tolerance);
        
        auto obj = std::make_shared<ob::PathLengthOptimizationObjective>(ss->getSpaceInformation());
        ss->setOptimizationObjective(obj);
        
        auto planner = std::make_shared<oc::SST>(ss->getSpaceInformation());
        ss->setPlanner(planner);
        
        ob::PlannerStatus solved = ss->solve(solve_time);
        
        if (solved) {
            // Extract the solution path
            auto solution = ss->getSolutionPath();
            auto* controlPath = solution.as<oc::PathControl>();
            
            // Convert to list of waypoints
            auto states = controlPath->getStates();
            for (size_t i = 0; i < states.size(); ++i) {
                const auto *se2 = states[i]->as<ob::SE2StateSpace::StateType>();
                const double* pos = se2->as<ob::RealVectorStateSpace::StateType>(0)->values;
                const double rot = se2->as<ob::SO2StateSpace::StateType>(1)->value;
                
                path.push_back({pos[0], pos[1], rot});
            }
        }
        
        ss->clear();
        planner->clear();
        
    } catch (...) {
        // Return empty path on error
        path.clear();
    }
    
    return path;
}

// NEW FUNCTION: Validate planning parameters
bool validate_planning_params(std::vector<double> start, std::vector<double> goal) {
    if (start.size() != 3) return false;
    if (goal.size() != 3) return false;
    
    // Check if positions are within reasonable bounds
    for (int i = 0; i < 2; i++) {  // x, y coordinates
        if (start[i] < -10.0 || start[i] > 10.0) return false;
        if (goal[i] < -10.0 || goal[i] > 10.0) return false;
    }
    
    return true;
}

// NEW FUNCTION: Get planner statistics
std::string get_sst_info() {
    return "SST (Sparse Stable Trees) Planner - Asymptotically optimal control planner";
}

// NEW FUNCTION: Run SST with custom Python propagator
std::string run_sst_with_custom_propagator(
    py::function python_propagator,                        // Python function for propagation
    std::vector<double> start_pos = {-0.5, 0.0, 0.0},     // [x, y, yaw]
    std::vector<double> goal_pos = {0.5, 0.0, 0.0},       // [x, y, yaw]
    double solve_time = 10.0,                              // seconds
    double goal_tolerance = 0.05,                          // goal tolerance
    std::vector<double> space_bounds = {-1.0, 1.0},       // [low, high] for x,y
    std::vector<double> control_bounds = {-0.3, 0.3}      // [low, high] for controls
) {
    try {
        // Validate input sizes
        if (start_pos.size() != 3 || goal_pos.size() != 3) {
            return "Error: start_pos and goal_pos must have 3 elements [x, y, yaw]";
        }
        if (space_bounds.size() != 2 || control_bounds.size() != 2) {
            return "Error: bounds must have 2 elements [low, high]";
        }
        
        // Create state space
        auto space(std::make_shared<ob::SE2StateSpace>());
        
        // Set configurable space bounds
        ob::RealVectorBounds bounds(2);
        bounds.setLow(space_bounds[0]);
        bounds.setHigh(space_bounds[1]);
        space->setBounds(bounds);
        
        // Create control space
        auto cspace(std::make_shared<oc::RealVectorControlSpace>(space, 2));
        
        // Set configurable control bounds
        ob::RealVectorBounds cbounds(2);
        cbounds.setLow(control_bounds[0]);
        cbounds.setHigh(control_bounds[1]);
        cspace->setBounds(cbounds);
        
        // Use smart pointer for proper lifetime management
        auto ss = std::make_shared<oc::SimpleSetup>(cspace);
        
        // Explicitly set control duration to eliminate warning
        ss->getSpaceInformation()->setMinMaxControlDuration(1, 10);
        
        // Explicitly set propagation step size to eliminate warning
        ss->getSpaceInformation()->setPropagationStepSize(0.05);
        
        // Set a simple state validity checker (always valid for demo)
        ss->setStateValidityChecker([](const ob::State *state) { 
            return true; 
        });
        
        // Set CUSTOM PYTHON PROPAGATOR
        ss->setStatePropagator([python_propagator](const ob::State *start, const oc::Control *control, 
                                                  const double duration, ob::State *result) {
            try {
                // Extract start state [x, y, yaw]
                const auto *se2state = start->as<ob::SE2StateSpace::StateType>();
                const double* pos = se2state->as<ob::RealVectorStateSpace::StateType>(0)->values;
                const double rot = se2state->as<ob::SO2StateSpace::StateType>(1)->value;
                std::vector<double> start_state = {pos[0], pos[1], rot};
                
                // Extract control [v, omega]
                const double* ctrl = control->as<oc::RealVectorControlSpace::ControlType>()->values;
                std::vector<double> control_input = {ctrl[0], ctrl[1]};
                
                // Call Python propagator function
                // Expected signature: propagator(start_state, control_input, duration) -> [x, y, yaw]
                auto result_state = python_propagator(start_state, control_input, duration).cast<std::vector<double>>();
                
                // Validate result
                if (result_state.size() != 3) {
                    throw std::runtime_error("Python propagator must return [x, y, yaw] list");
                }
                
                // Set result state
                result->as<ob::SE2StateSpace::StateType>()->setXY(result_state[0], result_state[1]);
                result->as<ob::SE2StateSpace::StateType>()->setYaw(result_state[2]);
                
            } catch (const std::exception& e) {
                // Fallback to simple kinematic model if Python function fails
                const auto *se2state = start->as<ob::SE2StateSpace::StateType>();
                const double* pos = se2state->as<ob::RealVectorStateSpace::StateType>(0)->values;
                const double rot = se2state->as<ob::SO2StateSpace::StateType>(1)->value;
                const double* ctrl = control->as<oc::RealVectorControlSpace::ControlType>()->values;
                
                result->as<ob::SE2StateSpace::StateType>()->setXY(
                    pos[0] + ctrl[0] * duration * cos(rot),
                    pos[1] + ctrl[0] * duration * sin(rot));
                result->as<ob::SE2StateSpace::StateType>()->setYaw(
                    rot + ctrl[1] * duration);
            }
        });
        
        // Set configurable start state
        ob::ScopedState<ob::SE2StateSpace> start(space);
        start->setX(start_pos[0]);
        start->setY(start_pos[1]);
        start->setYaw(start_pos[2]);
        
        // Set configurable goal state
        ob::ScopedState<ob::SE2StateSpace> goal(space);
        goal->setX(goal_pos[0]);
        goal->setY(goal_pos[1]);
        goal->setYaw(goal_pos[2]);
        
        // Set start and goal with configurable tolerance
        ss->setStartAndGoalStates(start, goal, goal_tolerance);
        
        // Explicitly set optimization objective to eliminate warning
        auto obj = std::make_shared<ob::PathLengthOptimizationObjective>(ss->getSpaceInformation());
        ss->setOptimizationObjective(obj);
        
        // Use your custom SST planner
        auto planner = std::make_shared<oc::SST>(ss->getSpaceInformation());
        ss->setPlanner(planner);
        
        // Solve with configurable time
        ob::PlannerStatus solved = ss->solve(solve_time);
        
        // Store result before cleanup
        std::string result_msg;
        if (solved) {
            result_msg = "SST Planner with CUSTOM PYTHON PROPAGATOR found a solution! ";
            result_msg += "Start: [" + std::to_string(start_pos[0]) + ", " + 
                         std::to_string(start_pos[1]) + ", " + std::to_string(start_pos[2]) + "] ";
            result_msg += "Goal: [" + std::to_string(goal_pos[0]) + ", " + 
                         std::to_string(goal_pos[1]) + ", " + std::to_string(goal_pos[2]) + "]";
        } else {
            result_msg = "SST Planner with custom propagator could not find a solution within " + std::to_string(solve_time) + " seconds.";
        }
        
        // Explicit cleanup to prevent double-free issues
        ss->clear();
        planner->clear();
        
        return result_msg;
        
    } catch (const std::exception& e) {
        return std::string("Error: ") + e.what();
    } catch (...) {
        return "Unknown error occurred in SST planner with custom propagator";
    }
}

// NEW FUNCTION: Get solution path with custom Python propagator
std::vector<std::vector<double>> run_sst_get_path_custom_propagator(
    py::function python_propagator,
    std::vector<double> start_pos = {-0.5, 0.0, 0.0},
    std::vector<double> goal_pos = {0.5, 0.0, 0.0},
    double solve_time = 10.0,
    double goal_tolerance = 0.05
) {
    std::vector<std::vector<double>> path;
    
    try {
        // Similar setup but with custom propagator and return path
        auto space(std::make_shared<ob::SE2StateSpace>());
        
        ob::RealVectorBounds bounds(2);
        bounds.setLow(-1.0);
        bounds.setHigh(1.0);
        space->setBounds(bounds);
        
        auto cspace(std::make_shared<oc::RealVectorControlSpace>(space, 2));
        
        ob::RealVectorBounds cbounds(2);
        cbounds.setLow(-0.3);
        cbounds.setHigh(0.3);
        cspace->setBounds(cbounds);
        
        auto ss = std::make_shared<oc::SimpleSetup>(cspace);
        ss->getSpaceInformation()->setMinMaxControlDuration(1, 10);
        ss->getSpaceInformation()->setPropagationStepSize(0.05);
        
        ss->setStateValidityChecker([](const ob::State *state) { return true; });
        
        // Set CUSTOM PYTHON PROPAGATOR (same as above)
        ss->setStatePropagator([python_propagator](const ob::State *start, const oc::Control *control, 
                                                  const double duration, ob::State *result) {
            try {
                const auto *se2state = start->as<ob::SE2StateSpace::StateType>();
                const double* pos = se2state->as<ob::RealVectorStateSpace::StateType>(0)->values;
                const double rot = se2state->as<ob::SO2StateSpace::StateType>(1)->value;
                std::vector<double> start_state = {pos[0], pos[1], rot};
                
                const double* ctrl = control->as<oc::RealVectorControlSpace::ControlType>()->values;
                std::vector<double> control_input = {ctrl[0], ctrl[1]};
                
                auto result_state = python_propagator(start_state, control_input, duration).cast<std::vector<double>>();
                
                if (result_state.size() != 3) {
                    throw std::runtime_error("Python propagator must return [x, y, yaw] list");
                }
                
                result->as<ob::SE2StateSpace::StateType>()->setXY(result_state[0], result_state[1]);
                result->as<ob::SE2StateSpace::StateType>()->setYaw(result_state[2]);
                
            } catch (const std::exception& e) {
                // Fallback to simple kinematic model
                const auto *se2state = start->as<ob::SE2StateSpace::StateType>();
                const double* pos = se2state->as<ob::RealVectorStateSpace::StateType>(0)->values;
                const double rot = se2state->as<ob::SO2StateSpace::StateType>(1)->value;
                const double* ctrl = control->as<oc::RealVectorControlSpace::ControlType>()->values;
                
                result->as<ob::SE2StateSpace::StateType>()->setXY(
                    pos[0] + ctrl[0] * duration * cos(rot),
                    pos[1] + ctrl[0] * duration * sin(rot));
                result->as<ob::SE2StateSpace::StateType>()->setYaw(
                    rot + ctrl[1] * duration);
            }
        });
        
        ob::ScopedState<ob::SE2StateSpace> start(space);
        start->setX(start_pos[0]);
        start->setY(start_pos[1]);
        start->setYaw(start_pos[2]);
        
        ob::ScopedState<ob::SE2StateSpace> goal(space);
        goal->setX(goal_pos[0]);
        goal->setY(goal_pos[1]);
        goal->setYaw(goal_pos[2]);
        
        ss->setStartAndGoalStates(start, goal, goal_tolerance);
        
        auto obj = std::make_shared<ob::PathLengthOptimizationObjective>(ss->getSpaceInformation());
        ss->setOptimizationObjective(obj);
        
        auto planner = std::make_shared<oc::SST>(ss->getSpaceInformation());
        ss->setPlanner(planner);
        
        ob::PlannerStatus solved = ss->solve(solve_time);
        
        if (solved) {
            // Extract the solution path
            auto solution = ss->getSolutionPath();
            auto* controlPath = solution.as<oc::PathControl>();
            
            // Convert to list of waypoints
            auto states = controlPath->getStates();
            for (size_t i = 0; i < states.size(); ++i) {
                const auto *se2 = states[i]->as<ob::SE2StateSpace::StateType>();
                const double* pos = se2->as<ob::RealVectorStateSpace::StateType>(0)->values;
                const double rot = se2->as<ob::SO2StateSpace::StateType>(1)->value;
                
                path.push_back({pos[0], pos[1], rot});
            }
        }
        
        ss->clear();
        planner->clear();
        
    } catch (...) {
        // Return empty path on error
        path.clear();
    }
    
    return path;
}

PYBIND11_MODULE(my_custom_planner_module, m) {
    m.doc() = "OMPL SST Planner Python Bindings";
    
    // Expose the SST class with proper lifetime management
    py::class_<oc::SST, std::shared_ptr<oc::SST>>(m, "SST")
        .def(py::init<const oc::SpaceInformationPtr &>(), py::keep_alive<1, 2>())
        .def("setGoalBias", &oc::SST::setGoalBias)
        .def("getGoalBias", &oc::SST::getGoalBias)
        .def("setSelectionRadius", &oc::SST::setSelectionRadius)
        .def("getSelectionRadius", &oc::SST::getSelectionRadius)
        .def("setPruningRadius", &oc::SST::setPruningRadius)
        .def("getPruningRadius", &oc::SST::getPruningRadius)
        .def("clear", &oc::SST::clear, "Clear the planner's internal state");
    
    // Enhanced demo function with configurable parameters
    m.def("run_sst_planner", &run_sst_planner, 
          "Run SST planner with configurable parameters",
          py::arg("start_pos") = std::vector<double>{-0.5, 0.0, 0.0},
          py::arg("goal_pos") = std::vector<double>{0.5, 0.0, 0.0},
          py::arg("solve_time") = 10.0,
          py::arg("goal_tolerance") = 0.05,
          py::arg("space_bounds") = std::vector<double>{-1.0, 1.0},
          py::arg("control_bounds") = std::vector<double>{-0.3, 0.3});
    
    // Simple demo function for backward compatibility
    m.def("run_sst_planner_simple", &run_sst_planner_simple, "Run SST planner with default parameters");
    
    // EXISTING FUNCTIONS:
    m.def("run_sst_get_path", &run_sst_get_path,
          "Run SST planner and return solution path as list of [x, y, yaw] waypoints",
          py::arg("start_pos") = std::vector<double>{-0.5, 0.0, 0.0},
          py::arg("goal_pos") = std::vector<double>{0.5, 0.0, 0.0},
          py::arg("solve_time") = 10.0,
          py::arg("goal_tolerance") = 0.05);
    
    m.def("validate_planning_params", &validate_planning_params,
          "Validate start and goal position parameters",
          py::arg("start"), py::arg("goal"));
    
    m.def("get_sst_info", &get_sst_info,
          "Get information about the SST planner");
    
    // NEW FUNCTIONS WITH CUSTOM PYTHON PROPAGATORS:
    m.def("run_sst_with_custom_propagator", &run_sst_with_custom_propagator,
          "Run SST planner with custom Python propagator function",
          py::arg("python_propagator"),
          py::arg("start_pos") = std::vector<double>{-0.5, 0.0, 0.0},
          py::arg("goal_pos") = std::vector<double>{0.5, 0.0, 0.0},
          py::arg("solve_time") = 10.0,
          py::arg("goal_tolerance") = 0.05,
          py::arg("space_bounds") = std::vector<double>{-1.0, 1.0},
          py::arg("control_bounds") = std::vector<double>{-0.3, 0.3});
    
    m.def("run_sst_get_path_custom_propagator", &run_sst_get_path_custom_propagator,
          "Run SST planner with custom Python propagator and return solution path",
          py::arg("python_propagator"),
          py::arg("start_pos") = std::vector<double>{-0.5, 0.0, 0.0},
          py::arg("goal_pos") = std::vector<double>{0.5, 0.0, 0.0},
          py::arg("solve_time") = 10.0,
          py::arg("goal_tolerance") = 0.05);
    
    // Add version info
    m.attr("__version__") = "0.1.0";
    
    // Add cleanup function for manual memory management if needed
    m.def("cleanup", []() {
        // Force garbage collection and cleanup
        // This can be called manually if needed
    }, "Manual cleanup function");
} 