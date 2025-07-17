#include "fusion.h"
#include "se2.h"
#include "utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <random>
#include <limits>
#include <ompl-1.6/ompl/control/SpaceInformation.h>
#include <ompl-1.6/ompl/base/spaces/SE2StateSpace.h>
#include <ompl-1.6/ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl-1.6/ompl/control/SimpleSetup.h>
#include <ompl-1.6/ompl/base/goals/GoalState.h>
#include <ompl-1.6/ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl-1.6/ompl/util/RandomNumbers.h>

namespace py = pybind11;
namespace ob = ompl::base;
namespace oc = ompl::control;
using ll = logger::LogLevel;

// Custom ControlSampler class for discrete control sampling
class CustomControlSampler : public oc::ControlSampler
{
public:
    CustomControlSampler(const oc::ControlSpace *space, const std::vector<double>& obj_shape)
        : oc::ControlSampler(space), control_space_(space), obj_shape_(obj_shape)
    {
    }

    void sample(oc::Control *control) override
    {
        auto *rctrl = control->as<oc::RealVectorControlSpace::ControlType>();
        
        const auto *rvc_space = control_space_->as<oc::RealVectorControlSpace>();
        const auto& bounds = rvc_space->getBounds();
        
        // Sample discrete control[0] and convert to continuous
        double discrete_ctrl0 = rng_.uniformReal(bounds.low[0], bounds.high[0]);
        // std::cout << "discrete_ctrl0: " << discrete_ctrl0 << std::endl;
        rctrl->values[0] = static_cast<int>(discrete_ctrl0) * M_PI / 2.0;
        // std::cout << "First control: " << rctrl->values[0] << std::endl;
        
        // Sample control[1] and adjust based on object shape and control[0]
        rctrl->values[1] = rng_.uniformReal(bounds.low[1], bounds.high[1]);
        if (fabs(fmod(rctrl->values[0], M_PI)) < 0.01) {
            rctrl->values[1] *= obj_shape_[1];  // Use height
        } else {
            rctrl->values[1] *= obj_shape_[0];  // Use width
        }
        
        rctrl->values[2] = rng_.uniformReal(bounds.low[2], bounds.high[2]);
    }

    unsigned int sampleStepCount(unsigned int minSteps, unsigned int maxSteps) override
    {
        return 1;
    }

private:
    const oc::ControlSpace *control_space_;
    std::vector<double> obj_shape_;
    ompl::RNG rng_;
};

// Enhanced wrapper function with configurable parameters - returns SimpleSetup object
std::shared_ptr<oc::SimpleSetup> run_planner(
    std::vector<double> start_pos = {0.0, 0.0, 0.0},      // [x, y, yaw]
    std::vector<double> goal_pos = {0.0, 0.0, 0.0},       // [x, y, yaw]
    std::vector<double> obj_shape = {1.0, 1.0, 1.0},      // [width, height, depth] for custom control sampling
    double solve_time = 10.0,                             // seconds
    double goal_tolerance = 0.05,                         // goal tolerance
    std::vector<double> space_bounds = {-1, 1.0, -1.0, 0.0},     // [low_x, high_x, low_y, high_y]
    std::vector<double> control_bounds = {0, 4, -0.4, 0.4, 0, 0.3},  // low and high for 3D controls
    py::object python_propagator = py::none(),            // optional custom Python propagator
    double pruning_radius = 0.1                           // pruning radius for FUSION planner
) {
    try {
        // Validate input sizes
        if (start_pos.size() != 3) {
            throw std::invalid_argument("start_pos must have 3 elements [x, y, yaw]");
        }
        
        // Auto-select goal type based on goal_pos size
        bool use_goal_region = false;
        if (goal_pos.size() != 3) {
            use_goal_region = true;  // Automatically use goal region if goal_pos is not [x, y, yaw]
            logger::log("goal_pos doesn't have 3 elements, automatically using GraspableRegion", ll::INFO);
        } else {
            logger::log("Using simple goal state with tolerance: " + std::to_string(goal_tolerance), ll::INFO);
        }
        
        if (space_bounds.size() != 4) {
            throw std::invalid_argument("space_bounds must have 4 elements [low_x, high_x, low_y, high_y]");
        }
        if (control_bounds.size() != 6) {
            throw std::invalid_argument("control_bounds must have 6 elements [low1, high1, low2, high2, low3, high3]");
        }
        
        // Create state space
        auto space(std::make_shared<ob::SE2StateSpace>());
        
        // Set the bounds for the state space (matching plan_to_edge.py)
        ob::RealVectorBounds bounds(2);
        bounds.setLow(0, space_bounds[0]);
        bounds.setHigh(0, space_bounds[1]);
        bounds.setLow(1, space_bounds[2]);
        bounds.setHigh(1, space_bounds[3]);
        space->setBounds(bounds);
        
        // Create 3D control space
        auto cspace(std::make_shared<oc::RealVectorControlSpace>(space, 3));
        
        // Set different bounds for each control element
        ob::RealVectorBounds cbounds(3);
        cbounds.setLow(0, control_bounds[0]);   // Control element 1 low bound
        cbounds.setHigh(0, control_bounds[1]);  // Control element 1 high bound
        cbounds.setLow(1, control_bounds[2]);   // Control element 2 low bound
        cbounds.setHigh(1, control_bounds[3]);  // Control element 2 high bound
        cbounds.setLow(2, control_bounds[4]);   // Control element 3 low bound
        cbounds.setHigh(2, control_bounds[5]);  // Control element 3 high bound
        cspace->setBounds(cbounds);
        
        // Use smart pointer for proper lifetime management
        auto ss = std::make_shared<oc::SimpleSetup>(cspace);
        
        // Explicitly set control duration to eliminate warning (matching plan_to_edge.py)
        auto si = ss->getSpaceInformation();
        si->setPropagationStepSize(3.0);        // Larger steps like plan_to_edge.py
        si->setMinMaxControlDuration(1, 1);     // Fixed duration like plan_to_edge.py
        
        // Set a simple state validity checker (always valid for demo)
        ss->setStateValidityChecker([](const ob::State *state) { 
            return true; 
        });
        
        // Set propagator (either custom Python or default)
        if (!python_propagator.is_none()) {
            // Set CUSTOM PYTHON PROPAGATOR
            logger::log("Using custom Python propagator", ll::INFO);
            py::function propagator_func = python_propagator.cast<py::function>();
            ss->setStatePropagator([propagator_func, obj_shape, space](const ob::State *start, const oc::Control *control, 
                                                    const double duration, ob::State *result) {
                try {
                    // Extract start state [x, y, yaw]
                    const auto *se2state = start->as<ob::SE2StateSpace::StateType>();
                    const double* pos = se2state->as<ob::RealVectorStateSpace::StateType>(0)->values;
                    const double rot = se2state->as<ob::SO2StateSpace::StateType>(1)->value;
                    SE2Pose start_pose(pos[0], pos[1], rot);
                    
                    // Extract control [v, omega, additional]
                    const double* ctrl = control->as<oc::RealVectorControlSpace::ControlType>()->values;
                    std::vector<double> control_input = {ctrl[0], ctrl[1], ctrl[2]};
                    
                    // Create a wrapper that handles tensor conversion
                    py::object torch = py::module::import("torch");
                    py::object control_tensor = torch.attr("tensor")(control_input).attr("unsqueeze")(0);
                    
                    // Get model device and move tensor to same device
                    try {
                        py::object first_param = py::iter(propagator_func.attr("parameters")()).attr("__next__")();
                        py::object device = first_param.attr("device");
                        control_tensor = control_tensor.attr("to")(device);
                    } catch (...) {
                        // If getting device fails, just use CPU
                    }
                    
                    // Call neural network and extract result
                    py::object output = propagator_func(control_tensor);
                    py::object numpy_output = output.attr("detach")().attr("cpu")().attr("numpy")();
                    py::array_t<double> result_array = numpy_output.cast<py::array_t<double>>();
                    
                    // Extract the first row (batch index 0) as vector
                    std::vector<double> delta;
                    auto buf = result_array.request();
                    double* ptr = static_cast<double*>(buf.ptr);
                    for (int i = 0; i < 3; ++i) {
                        delta.push_back(ptr[i]);
                    }
                    
                    // Convert delta to SE2Pose
                    SE2Pose delta_pose(delta[0], delta[1], delta[2]);

                    // Apply delta to start state
                    SE2Pose final_pose = start_pose * delta_pose;
                    
                    // Set result state
                    result->as<ob::SE2StateSpace::StateType>()->setXY(
                        final_pose.x,
                        final_pose.y);
                    
                    // Manually normalize angle to [-π, π]
                    double normalized_yaw = final_pose.theta;
                    while (normalized_yaw > M_PI) normalized_yaw -= 2.0 * M_PI;
                    while (normalized_yaw < -M_PI) normalized_yaw += 2.0 * M_PI;
                    result->as<ob::SE2StateSpace::StateType>()->setYaw(normalized_yaw);
                    
                    // Force OMPL to enforce bounds (including angle normalization)
                    space->enforceBounds(result);
                    // logger::log("Propagation successful", ll::DEBUG);
                    
                } catch (const std::exception& e) {
                    logger::log("Error in propagator: " + std::string(e.what()), ll::ERROR);
                    logger::log("Falling back to kinematic model", ll::WARNING);
                    // Fallback to simple kinematic model if Python function fails
                    const auto *se2state = start->as<ob::SE2StateSpace::StateType>();
                    const double* pos = se2state->as<ob::RealVectorStateSpace::StateType>(0)->values;
                    const double rot = se2state->as<ob::SO2StateSpace::StateType>(1)->value;
                    const double* ctrl = control->as<oc::RealVectorControlSpace::ControlType>()->values;
                    
                    // Calculate new yaw with normalization
                    double new_yaw = rot + ctrl[1] * duration;
                    while (new_yaw > M_PI) new_yaw -= 2.0 * M_PI;
                    while (new_yaw < -M_PI) new_yaw += 2.0 * M_PI;
                    
                    result->as<ob::SE2StateSpace::StateType>()->setXY(
                        pos[0] + ctrl[0] * duration * cos(rot),
                        pos[1] + ctrl[0] * duration * sin(rot));
                    result->as<ob::SE2StateSpace::StateType>()->setYaw(new_yaw);
                }
            });
        } else {
            // Set default propagator for 3D control (matching plan_to_edge.py control semantics)
            ss->setStatePropagator([space](const ob::State *start, const oc::Control *control, 
                                    const double duration, ob::State *result) {
                const auto *se2state = start->as<ob::SE2StateSpace::StateType>();
                const double* pos = se2state->as<ob::RealVectorStateSpace::StateType>(0)->values;
                const double rot = se2state->as<ob::SO2StateSpace::StateType>(1)->value;
                const double* ctrl = control->as<oc::RealVectorControlSpace::ControlType>()->values;
                
                // Using 3D control like plan_to_edge.py:
                // ctrl[0] = rotation (0-4)
                // ctrl[1] = side offset (-0.4 to 0.4) 
                // ctrl[2] = push distance (0-0.3)
                
                // Apply rotation and translation based on control semantics
                double new_x = pos[0] + ctrl[2] * duration * cos(rot + ctrl[0]);
                double new_y = pos[1] + ctrl[2] * duration * sin(rot + ctrl[0]) + ctrl[1] * duration;
                double new_yaw = rot + ctrl[0] * duration;
                
                // Manually normalize angle to [-π, π]
                while (new_yaw > M_PI) new_yaw -= 2.0 * M_PI;
                while (new_yaw < -M_PI) new_yaw += 2.0 * M_PI;
                
                result->as<ob::SE2StateSpace::StateType>()->setXY(new_x, new_y);
                result->as<ob::SE2StateSpace::StateType>()->setYaw(new_yaw);
                
                // Force OMPL to enforce bounds (including angle normalization)
                space->enforceBounds(result);
            });
        }
        
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
        
        if (use_goal_region) {
            // Create GraspableRegion goal region (matching Python implementation)
            class GraspableRegion : public ob::GoalSampleableRegion {
            public:
                GraspableRegion(const ob::SpaceInformationPtr &si, 
                               const std::vector<double> &goal_pos,
                               const std::vector<double> &obj_shape,
                               double edge,
                               double threshold) 
                    : ob::GoalSampleableRegion(si), goal_point_(goal_pos), 
                      obj_shape_(obj_shape), edge_(edge), threshold_(threshold) {
                    setThreshold(threshold);
                }
                
                double distanceGoal(const ob::State *state) const override {
                    const auto *se2state = state->as<ob::SE2StateSpace::StateType>();
                    const double* pos = se2state->as<ob::RealVectorStateSpace::StateType>(0)->values;
                    const double rot = se2state->as<ob::SO2StateSpace::StateType>(1)->value;
                    
                    double x = pos[0];
                    double y = pos[1];
                    double yaw = rot;
                    
                    // Calculate object corners in world frame (matching Python logic)
                    std::vector<std::vector<double>> corners = {
                        {-obj_shape_[0]/2, -obj_shape_[1]/2, 1},
                        {-obj_shape_[0]/2, +obj_shape_[1]/2, 1},
                        {+obj_shape_[0]/2, -obj_shape_[1]/2, 1},
                        {+obj_shape_[0]/2, +obj_shape_[1]/2, 1}
                    };
                    
                    // Create homogeneous transformation matrix
                    double cos_yaw = cos(yaw);
                    double sin_yaw = sin(yaw);
                    
                    double max_x = -std::numeric_limits<double>::infinity();
                    for (const auto& corner : corners) {
                        // Apply transformation: homogenous_matrix * corner
                        double rotated_x = cos_yaw * corner[0] - sin_yaw * corner[1] + x;
                        max_x = std::max(max_x, rotated_x);
                    }
                    
                    // Check if object edge is past the target edge (goal achieved)
                    if (max_x - edge_ > 0.05 && x < edge_) {
                        return 0.0;  // Goal achieved
                    } else {
                        // Calculate distance to goal point
                        double x_dist = x - goal_point_[0];
                        double y_dist = y - goal_point_[1];
                        double yaw_dist = yaw - goal_point_[2];
                        
                        // Normalize angle difference to [-π, π]
                        yaw_dist = fmod(yaw_dist + M_PI, 2*M_PI) - M_PI;
                        
                        return sqrt(x_dist*x_dist + y_dist*y_dist + yaw_dist*yaw_dist);
                    }
                }
                
                void sampleGoal(ob::State *state) const override {
                    auto *se2state = state->as<ob::SE2StateSpace::StateType>();
                    se2state->setX(goal_point_[0]);
                    se2state->setY(goal_point_[1]);
                    se2state->setYaw(goal_point_[2]);
                }
                
                unsigned int maxSampleCount() const override {
                    return 1;
                }
                
                bool couldSample() const override {
                    return true;
                }
                
            private:
                std::vector<double> goal_point_;
                std::vector<double> obj_shape_;
                double edge_;
                double threshold_;
            };
            
            // Create the goal region with object shape and edge parameters
            // Ensure goal_pos has at least 3 elements for GraspableRegion
            std::vector<double> safe_goal_pos = goal_pos;
            while (safe_goal_pos.size() < 3) {
                safe_goal_pos.push_back(0.0);  // Fill missing elements with 0
            }
            double edge_position = 0.76;  // Default edge position
            auto goal_region = std::make_shared<GraspableRegion>(si, safe_goal_pos, obj_shape, edge_position, goal_tolerance);
            ss->setGoal(goal_region);
            ss->setStartState(start);  // Set start state separately for goal region
            logger::log("Using GraspableRegion goal with edge detection", ll::INFO);
        } else {
            // Use simple goal state with tolerance
            ss->setStartAndGoalStates(start, goal, goal_tolerance);
            // print these with 4 decimal places
            logger::log("Start state: " + std::to_string(start->getX()) + ", " + std::to_string(start->getY()) + ", " + std::to_string(start->getYaw()), ll::INFO);
            logger::log("Goal state: " + std::to_string(goal->getX()) + ", " + std::to_string(goal->getY()) + ", " + std::to_string(goal->getYaw()), ll::INFO);
        }
        
        // Explicitly set optimization objective to eliminate warning
        auto obj = std::make_shared<ob::PathLengthOptimizationObjective>(ss->getSpaceInformation());
        ss->setOptimizationObjective(obj);
        
        // Use your custom FUSION planner
        auto planner = std::make_shared<oc::FUSION>(ss->getSpaceInformation());
        
        // Set pruning radius
        planner->setPruningRadius(pruning_radius);
        
        ss->setPlanner(planner);
        
        // Set up custom control sampler with object shape using the ControlSpace allocator
        auto controlSpace = ss->getSpaceInformation()->getControlSpace();
        controlSpace->setControlSamplerAllocator(
            [obj_shape](const oc::ControlSpace* space) -> oc::ControlSamplerPtr {
                return std::make_shared<CustomControlSampler>(space, obj_shape);
            });
        
        // Solve with configurable time
        ob::PlannerStatus solved = ss->solve(solve_time);

        // Return the SimpleSetup object (don't clear it here)
        return ss;
        
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error: ") + e.what());
    } catch (...) {
        throw std::runtime_error("Unknown error occurred in SST planner");
    }
}


// Get solution path from SimpleSetup object
std::vector<std::vector<double>> get_path(std::shared_ptr<oc::SimpleSetup> ss) {
    std::vector<std::vector<double>> path;
    
    try {
        // Check if there's a solution
        if (ss->haveSolutionPath()) {
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
        
    } catch (...) {
        // Return empty path on error
        logger::log("Error in get_path(), returning empty path", ll::ERROR);
        path.clear();
    }
    
    return path;
}

// Get control inputs from SimpleSetup object's solution path
std::vector<std::vector<double>> get_controls(std::shared_ptr<oc::SimpleSetup> ss) {
    std::vector<std::vector<double>> controls;
    
    try {
        // Check if there's a solution
        if (ss->haveSolutionPath()) {
            // Extract the solution path
            auto solution = ss->getSolutionPath();
            auto* controlPath = solution.as<oc::PathControl>();
            
            // Convert to list of control inputs
            auto path_controls = controlPath->getControls();
            for (size_t i = 0; i < path_controls.size(); ++i) {
                const auto *control = path_controls[i]->as<oc::RealVectorControlSpace::ControlType>();
                const double* ctrl_values = control->values;
                
                // Extract 3D control values [v, omega, additional]
                controls.push_back({ctrl_values[0], ctrl_values[1], ctrl_values[2]});
            }
        }
        
    } catch (...) {
        // Return empty controls on error
        logger::log("Error in get_controls(), returning empty controls", ll::ERROR);
        controls.clear();
    }
    
    return controls;
}

// Get nearest nodes within range R from a given node using the planner's nearestR function
std::vector<std::vector<double>> get_nearest_nodes(
    std::shared_ptr<oc::SimpleSetup> ss, 
    std::vector<double> query_node
) {
    std::vector<std::vector<double>> nearest_nodes;
    
    try {
        // Validate input
        if (query_node.size() != 3) {
            throw std::invalid_argument("query_node must have 3 elements [x, y, yaw]");
        }
        
        // Get the planner from SimpleSetup
        auto planner = ss->getPlanner();
        auto* fusion_planner = dynamic_cast<oc::FUSION*>(planner.get());
        
        if (!fusion_planner) {
            throw std::runtime_error("Planner is not a FUSION planner");
        }
        
        // Create a temporary motion with the query node state
        auto space = ss->getStateSpace();
        auto* se2_space = space->as<ob::SE2StateSpace>();
        
        // Allocate state and set values
        ob::State* query_state = space->allocState();
        auto* se2_state = query_state->as<ob::SE2StateSpace::StateType>();
        se2_state->setXY(query_node[0], query_node[1]);
        se2_state->setYaw(query_node[2]);
        
        // Get selection radius from planner (same as used in the planner's nearestR)
        double selection_radius = fusion_planner->getSelectionRadius();
        selection_radius = 0.1;  // Override with custom value
        
        // Use the new public method to get nearest neighbor states within radius R
        auto state_results = fusion_planner->getNearestStatesR(query_state, selection_radius);
        
        // Convert state results to vector format
        for (const auto* state : state_results) {
            if (state) {
                const ob::SE2StateSpace::StateType *se2 = state->as<ob::SE2StateSpace::StateType>();
                const ob::RealVectorStateSpace::StateType *pos_state = se2->as<ob::RealVectorStateSpace::StateType>(0);
                const ob::SO2StateSpace::StateType *rot_state = se2->as<ob::SO2StateSpace::StateType>(1);
                
                const double* pos = pos_state->values;
                const double rot = rot_state->value;
                
                nearest_nodes.push_back({pos[0], pos[1], rot});
            }
        }
        space->freeState(query_state);
        
    } catch (const std::exception& e) {
        logger::log("Error in get_nearest_nodes(): " + std::string(e.what()), ll::ERROR);
        nearest_nodes.clear();
    } catch (...) {
        logger::log("Unknown error in get_nearest_nodes()", ll::ERROR);
        nearest_nodes.clear();
    }
    
    return nearest_nodes;
}


PYBIND11_MODULE(FUSION, m) {
    m.doc() = "OMPL FUSION Planner Python Bindings";
    
    // Expose the FUSION class with proper lifetime management
    py::class_<oc::FUSION, std::shared_ptr<oc::FUSION>>(m, "FUSION")
        .def(py::init([](oc::SpaceInformation *si) {
            return std::make_shared<oc::FUSION>(oc::SpaceInformationPtr(si));
        }))
        .def("setGoalBias", &oc::FUSION::setGoalBias)
        .def("getGoalBias", &oc::FUSION::getGoalBias)
        .def("setSelectionRadius", &oc::FUSION::setSelectionRadius)
        .def("getSelectionRadius", &oc::FUSION::getSelectionRadius)
        .def("setPruningRadius", &oc::FUSION::setPruningRadius)
        .def("getPruningRadius", &oc::FUSION::getPruningRadius)
        .def("clear", &oc::FUSION::clear, "Clear the planner's internal state");
    
    // Expose the SimpleSetup class with proper lifetime management
    py::class_<oc::SimpleSetup, std::shared_ptr<oc::SimpleSetup>>(m, "SimpleSetup")
        .def("haveSolutionPath", &oc::SimpleSetup::haveSolutionPath)
        .def("clear", &oc::SimpleSetup::clear)
        .def("solve", static_cast<ob::PlannerStatus(oc::SimpleSetup::*)(double)>(&oc::SimpleSetup::solve));
    
    // Enhanced run_planner function that returns SimpleSetup object
    m.def("run_planner", &run_planner, 
          R"pbdoc(
          Run FUSION planner with configurable parameters and return SimpleSetup object.
          
          This function sets up and configures a FUSION motion planner for SE2 state space
          with 3D control inputs. It automatically selects between GraspableRegion (when
          goal_pos doesn't have 3 elements) and simple goal state (when goal_pos has 3 elements).
          Supports both custom Python propagators and default kinematic models.
          
          Parameters
          ----------
          start_pos : list of 3 floats, default [0.0, 0.0, 0.0]
              Initial state [x, y, yaw] in SE2 space
          goal_pos : list of 3 floats, default [0.0, 0.0, 0.0]  
              Target state [x, y, yaw] in SE2 space
          obj_shape : list of 3 floats, default [1.0, 1.0, 1.0]
              Object dimensions [width, height, depth] for control sampling
          solve_time : float, default 10.0
              Maximum planning time in seconds
          goal_tolerance : float, default 0.05
              Distance tolerance for goal achievement
          space_bounds : list of 4 floats, default [-1.0, 1.0, -1.0, 0.0]
              State space bounds [x_min, x_max, y_min, y_max]
          control_bounds : list of 6 floats, default [0, 4, -0.4, 0.4, 0, 0.3]
              Control bounds [rot_min, rot_max, side_min, side_max, push_min, push_max]
          python_propagator : callable or None, default None
              Optional custom Python propagator function for dynamics modeling
          pruning_radius : float, default 0.1
              Pruning radius for the FUSION planner (controls tree density)
              
          Returns
          -------
          SimpleSetup
              Configured OMPL SimpleSetup object ready for planning.
              Use solve_path(ss, time) to run planning, then get_path(ss) for solution.
              
          Examples
          --------
          Basic usage with default parameters:
          >>> ss = FUSION.run_planner()
          >>> solved = FUSION.solve_path(ss, 10.0)
          >>> if solved: path = FUSION.get_path(ss)
          
          Push-to-edge planning (automatic GraspableRegion when goal_pos != 3 elements):
          >>> ss = FUSION.run_planner(
          ...     start_pos=[0.4, -0.6, 0.0],
          ...     goal_pos=[0.7, -0.6],  # Only 2 elements triggers GraspableRegion
          ...     obj_shape=[0.16, 0.21, 0.07]
          ... )
          
          Simple point goal planning (automatic when goal_pos has 3 elements):
          >>> ss = FUSION.run_planner(
          ...     start_pos=[0.0, 0.0, 0.0],
          ...     goal_pos=[1.0, 1.0, 0.0],  # 3 elements triggers simple goal
          ...     goal_tolerance=0.1
          ... )
          
          With custom propagator:
          >>> import torch
          >>> model = torch.load('my_model.pth')  
          >>> ss = FUSION.run_planner(
          ...     start_pos=[0.0, 0.0, 0.0],
          ...     goal_pos=[1.0, 1.0, 0.0],
          ...     python_propagator=model
          ... )
          )pbdoc",
          py::arg("start_pos") = std::vector<double>{0.0, 0.0, 0.0},
          py::arg("goal_pos") = std::vector<double>{0.0, 0.0, 0.0},
          py::arg("obj_shape") = std::vector<double>{1.0, 1.0, 1.0},
          py::arg("solve_time") = 10.0,
          py::arg("goal_tolerance") = 0.05,
          py::arg("space_bounds") = std::vector<double>{-1.0, 1.0, -1.0, 0.0},
          py::arg("control_bounds") = std::vector<double>{0, 4, -0.4, 0.4, 0, 0.3},
          py::arg("python_propagator") = py::none(),
          py::arg("pruning_radius") = 0.1);
    
    // Get path from SimpleSetup object
    m.def("get_path", &get_path,
          "Extract solution path from SimpleSetup object as list of [x, y, yaw] waypoints",
          py::arg("ss"));
    
    // Get controls from SimpleSetup object
    m.def("get_controls", &get_controls,
          "Extract control inputs from SimpleSetup object's solution path as list of [v, omega, additional] controls",
          py::arg("ss"));
    
    // Get nearest nodes within range R from a given node
    m.def("get_nearest_nodes", &get_nearest_nodes,
          "Find all nodes within selection radius R from a given query node using planner's nearestR function",
          py::arg("ss"),
          py::arg("query_node"));
    
    // Add version info
    m.attr("__version__") = "0.1.0";
    
    // Add cleanup function for manual memory management if needed
    m.def("cleanup", []() {
        // Force garbage collection and cleanup
        // This can be called manually if needed
    }, "Manual cleanup function");

    m.def("make_fusion_planner", [](py::object si) {
        // Extract the raw pointer from the SWIG object
        // This is a hack, but works for OMPL's SWIG Python bindings
        auto capsule = si.attr("_get_c_pointer")();
        auto ptr = reinterpret_cast<ompl::control::SpaceInformation*>(capsule.cast<size_t>());
        return std::make_shared<oc::FUSION>(ompl::control::SpaceInformationPtr(ptr));
    });
} 