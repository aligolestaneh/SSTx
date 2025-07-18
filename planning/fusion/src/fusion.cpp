/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2015, Rutgers the State University of New Jersey, New Brunswick
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of Rutgers University nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Authors: Zakary Littlefield */

#include "ompl/control/planners/fusion/fusion.h"
#include "ompl/base/goals/GoalSampleableRegion.h"
#include "ompl/base/goals/GoalRegion.h"
#include "ompl/base/objectives/MinimaxObjective.h"
#include "ompl/base/objectives/MaximizeMinClearanceObjective.h"
#include "ompl/base/objectives/PathLengthOptimizationObjective.h"
#include "ompl/base/objectives/MechanicalWorkOptimizationObjective.h"
#include "ompl/base/spaces/SE2StateSpace.h"
#include "ompl/tools/config/SelfConfig.h"
#include <limits>
#include <set>
#include <functional>
#include <queue>
#include <chrono>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>

ompl::control::Fusion::Fusion(const SpaceInformationPtr &si) : base::Planner(si, "Fusion")
{
    specs_.approximateSolutions = true;
    siC_ = si.get();
    prevSolution_.clear();
    prevSolutionControls_.clear();
    prevSolutionSteps_.clear();

    // Initialize thread-safe tracking variables
    currentBestCost_.store(std::numeric_limits<double>::infinity());
    hasExactSolution_.store(false);
    planningActive_.store(false);

    Planner::declareParam<double>("goal_bias", this, &Fusion::setGoalBias, &Fusion::getGoalBias, "0.:.05:1.");
    Planner::declareParam<double>("selection_radius", this, &Fusion::setSelectionRadius, &Fusion::getSelectionRadius, "0.:.1:"
                                                                                                                "100");
    Planner::declareParam<double>("pruning_radius", this, &Fusion::setPruningRadius, &Fusion::getPruningRadius, "0.:.1:100");
    Planner::declareParam<bool>("terminate_on_first_solution", this, &Fusion::setTerminateOnFirstSolution, &Fusion::getTerminateOnFirstSolution, "0,1");
}

ompl::control::Fusion::~Fusion()
{
    freeMemory();
}

void ompl::control::Fusion::setup()
{
    base::Planner::setup();
    if (!nn_)
        nn_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
    nn_->setDistanceFunction([this](const Motion *a, const Motion *b)
                             {
                                 return distanceFunction(a, b);
                             });
    if (!witnesses_)
        witnesses_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
    witnesses_->setDistanceFunction([this](const Motion *a, const Motion *b)
                                    {
                                        return distanceFunction(a, b);
                                    });

    if (pdef_)
    {
        if (pdef_->hasOptimizationObjective())
        {
            opt_ = pdef_->getOptimizationObjective();
            if (dynamic_cast<base::MaximizeMinClearanceObjective *>(opt_.get()) ||
                dynamic_cast<base::MinimaxObjective *>(opt_.get()))
                OMPL_WARN("%s: Asymptotic near-optimality has only been proven with Lipschitz continuous cost "
                          "functions w.r.t. state and control. This optimization objective will result in undefined "
                          "behavior",
                          getName().c_str());
        }
        else
        {
            OMPL_WARN("%s: No optimization object set. Using path length", getName().c_str());
            opt_ = std::make_shared<base::PathLengthOptimizationObjective>(si_);
            pdef_->setOptimizationObjective(opt_);
        }
    }

    prevSolutionCost_ = opt_->infiniteCost();
}

void ompl::control::Fusion::costTrackingThread(const std::string& filename, 
                                               std::chrono::high_resolution_clock::time_point startTime) const
{
    std::ofstream logFile(filename, std::ios::out);
    logFile << "# Time(s) Cost Type\n";
    
    const double samplingInterval = 0.01; // 10ms sampling interval
    auto lastSampleTime = startTime;
    
    while (planningActive_.load()) {
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(currentTime - startTime).count();
        
        // Sample at regular intervals
        if (std::chrono::duration<double>(currentTime - lastSampleTime).count() >= samplingInterval) {
            double currentCost = currentBestCost_.load();
            bool hasExact = hasExactSolution_.load();
            
            if (currentCost == std::numeric_limits<double>::infinity()) {
                logFile << elapsed << " inf none\n";
            } else {
                std::string solutionType = hasExact ? "exact" : "approximate";
                logFile << elapsed << " " << currentCost << " " << solutionType << "\n";
            }
            
            logFile.flush();
            lastSampleTime = currentTime;
        }
        
        // Sleep for a short time to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // Final sample when planning stops
    auto finalTime = std::chrono::high_resolution_clock::now();
    auto finalElapsed = std::chrono::duration<double>(finalTime - startTime).count();
    double finalCost = currentBestCost_.load();
    bool finalHasExact = hasExactSolution_.load();
    
    if (finalCost == std::numeric_limits<double>::infinity()) {
        logFile << finalElapsed << " inf none\n";
    } else {
        std::string solutionType = finalHasExact ? "exact" : "approximate";
        logFile << finalElapsed << " " << finalCost << " " << solutionType << "\n";
    }
    
    logFile.close();
}

void ompl::control::Fusion::clear()
{
    Planner::clear();
    sampler_.reset();
    controlSampler_.reset();
    freeMemory();
    if (nn_)
        nn_->clear();
    if (witnesses_)
        witnesses_->clear();
    if (opt_)
        prevSolutionCost_ = opt_->infiniteCost();
    
    // Reset tracking variables
    currentBestCost_.store(std::numeric_limits<double>::infinity());
    hasExactSolution_.store(false);
    planningActive_.store(false);
}

void ompl::control::Fusion::freeMemory()
{
    if (nn_)
    {
        std::vector<Motion *> motions;
        nn_->list(motions);
        for (auto &motion : motions)
        {
            if (motion->state_)
                si_->freeState(motion->state_);
            if (motion->control_)
                siC_->freeControl(motion->control_);
            delete motion;
        }
    }
    if (witnesses_)
    {
        std::vector<Motion *> witnesses;
        witnesses_->list(witnesses);
        for (auto &witness : witnesses)
        {
            delete witness;
        }
    }
    for (auto &i : prevSolution_)
    {
        if (i)
            si_->freeState(i);
    }
    prevSolution_.clear();
    for (auto &prevSolutionControl : prevSolutionControls_)
    {
        if (prevSolutionControl)
            siC_->freeControl(prevSolutionControl);
    }
    prevSolutionControls_.clear();
    prevSolutionSteps_.clear();
}

ompl::control::Fusion::Motion *ompl::control::Fusion::selectNode(ompl::control::Fusion::Motion *sample)
{
    std::vector<Motion *> ret;
    Motion *selected = nullptr;
    base::Cost bestCost = opt_->infiniteCost();
    nn_->nearestR(sample, selectionRadius_, ret);
    for (auto &i : ret)
    {
        if (!i->inactive_ && opt_->isCostBetterThan(i->accCost_, bestCost))
        {
            bestCost = i->accCost_;
            selected = i;
        }
    }
    if (selected == nullptr)
    {
        int k = 1;
        while (selected == nullptr)
        {
            nn_->nearestK(sample, k, ret);
            for (unsigned int i = 0; i < ret.size() && selected == nullptr; i++)
                if (!ret[i]->inactive_)
                    selected = ret[i];
            k += 5;
        }
    }
    return selected;
}

ompl::control::Fusion::Witness *ompl::control::Fusion::findClosestWitness(ompl::control::Fusion::Motion *node)
{
    if (witnesses_->size() > 0)
    {
        auto *closest = static_cast<Witness *>(witnesses_->nearest(node));
        if (distanceFunction(closest, node) > pruningRadius_)
        {
            closest = new Witness(siC_);
            closest->linkRep(node);
            si_->copyState(closest->state_, node->state_);
            witnesses_->add(closest);
        }
        return closest;
    }
    else
    {
        auto *closest = new Witness(siC_);
        closest->linkRep(node);
        si_->copyState(closest->state_, node->state_);
        witnesses_->add(closest);
        return closest;
    }
}

ompl::base::PlannerStatus ompl::control::Fusion::solve(const base::PlannerTerminationCondition &ptc)
{
    checkValidity();
    base::Goal *goal = pdef_->getGoal().get();
    auto *goal_s = dynamic_cast<base::GoalSampleableRegion *>(goal);

    while (const base::State *st = pis_.nextStart())
    {
        auto *motion = new Motion(siC_);
        si_->copyState(motion->state_, st);
        siC_->nullControl(motion->control_);
        nn_->add(motion);
        motion->accCost_ = opt_->identityCost();
        findClosestWitness(motion);
    }

    if (nn_->size() == 0)
    {
        OMPL_ERROR("%s: There are no valid initial states!", getName().c_str());
        return base::PlannerStatus::INVALID_START;
    }

    if (!sampler_)
        sampler_ = si_->allocStateSampler();
    if (!controlSampler_)
        controlSampler_ = siC_->allocControlSampler();

    const base::ReportIntermediateSolutionFn intermediateSolutionCallback = pdef_->getIntermediateSolutionCallback();

    OMPL_INFORM("%s: Starting planning with %u states already in datastructure\n", getName().c_str(), nn_->size());

    Motion *solution = nullptr;
    Motion *approxsol = nullptr;
    double approxdif = std::numeric_limits<double>::infinity();
    bool sufficientlyShort = false;

    auto *rmotion = new Motion(siC_);
    base::State *rstate = rmotion->state_;
    Control *rctrl = rmotion->control_;
    base::State *xstate = si_->allocState();

    unsigned iterations = 0;
    
    // Start time-based cost tracking
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize tracking variables
    if (prevSolutionCost_.value() < opt_->infiniteCost().value()) {
        // We have a previous solution
        currentBestCost_.store(prevSolutionCost_.value());
        hasExactSolution_.store(true);  // Assume previous solution was exact
    } else {
        // No previous solution
        currentBestCost_.store(std::numeric_limits<double>::infinity());
        hasExactSolution_.store(false);
    }
    
    // Start the cost tracking thread
    planningActive_.store(true);
    std::thread tracker(&Fusion::costTrackingThread, this, "fusion_solutions.txt", start_time);

    while (ptc == false)
    {
        /* sample random state (with goal biasing) */
        if (goal_s && rng_.uniform01() < goalBias_ && goal_s->canSample())
            goal_s->sampleGoal(rstate);
        else
            sampler_->sampleUniform(rstate);

        /* find closest state in the tree */
        Motion *nmotion = selectNode(rmotion);

        /* sample a random control that attempts to go towards the random state, and also sample a control duration */
        controlSampler_->sample(rctrl);
        unsigned int cd = rng_.uniformInt(siC_->getMinControlDuration(), siC_->getMaxControlDuration());
        unsigned int propCd = siC_->propagateWhileValid(nmotion->state_, rctrl, cd, rstate);

        if (propCd == cd)
        {
            base::Cost incCostMotion = opt_->motionCost(nmotion->state_, rstate);
            base::Cost incCostControl = opt_->controlCost(rctrl, cd);
            base::Cost incCost = opt_->combineCosts(incCostMotion, incCostControl);
            base::Cost cost = opt_->combineCosts(nmotion->accCost_, incCost);
            Witness *closestWitness = findClosestWitness(rmotion);

            if (closestWitness->rep_ == rmotion || opt_->isCostBetterThan(cost, closestWitness->rep_->accCost_))
            {
                Motion *oldRep = closestWitness->rep_;
                /* create a motion */
                auto *motion = new Motion(siC_);
                motion->accCost_ = cost;
                si_->copyState(motion->state_, rmotion->state_);
                siC_->copyControl(motion->control_, rctrl);
                motion->steps_ = cd;
                motion->parent_ = nmotion;
                nmotion->children_.push_back(motion);
                nmotion->numChildren_++;
                closestWitness->linkRep(motion);

                nn_->add(motion);
                double dist = 0.0;
                bool solv = goal->isSatisfied(motion->state_, &dist);
                            
                
                if (solv && opt_->isCostBetterThan(motion->accCost_, prevSolutionCost_))
                {
                    approxdif = dist;
                    solution = motion;

                    for (auto &i : prevSolution_)
                        if (i)
                            si_->freeState(i);
                    prevSolution_.clear();
                    for (auto &prevSolutionControl : prevSolutionControls_)
                        if (prevSolutionControl)
                            siC_->freeControl(prevSolutionControl);
                    prevSolutionControls_.clear();
                    prevSolutionSteps_.clear();

                    Motion *solTrav = solution;
                    while (solTrav->parent_ != nullptr)
                    {
                        prevSolution_.push_back(si_->cloneState(solTrav->state_));
                        prevSolutionControls_.push_back(siC_->cloneControl(solTrav->control_));
                        prevSolutionSteps_.push_back(solTrav->steps_);
                        solTrav = solTrav->parent_;
                    }
                    prevSolution_.push_back(si_->cloneState(solTrav->state_));
                    prevSolutionCost_ = solution->accCost_;

                    // Update thread-safe tracking variables
                    currentBestCost_.store(solution->accCost_.value());
                    hasExactSolution_.store(true);

                    OMPL_INFORM("Found solution with cost %.4f", solution->accCost_.value());
                    bestSolutionCost_ = solution->accCost_.value();

                    if (intermediateSolutionCallback)
                    {
                        // the callback requires a vector with const elements -> create a copy
                        std::vector<const base::State *> prevSolutionConst(prevSolution_.begin(), prevSolution_.end());
                        intermediateSolutionCallback(this, prevSolutionConst, prevSolutionCost_);
                    }
                    sufficientlyShort = opt_->isSatisfied(solution->accCost_);
                    if (sufficientlyShort)
                        break;
                }
                if (solution == nullptr && dist < approxdif)
                {
                    OMPL_INFORM("Found APPROXIMATE solution with cost %.4f and distance %.4f", motion->accCost_.value(), dist);
                    approxdif = dist;
                    approxsol = motion;

                    for (auto &i : prevSolution_)
                        if (i)
                            si_->freeState(i);
                    prevSolution_.clear();
                    for (auto &prevSolutionControl : prevSolutionControls_)
                        if (prevSolutionControl)
                            siC_->freeControl(prevSolutionControl);
                    prevSolutionControls_.clear();
                    prevSolutionSteps_.clear();

                    Motion *solTrav = approxsol;
                    while (solTrav->parent_ != nullptr)
                    {
                        prevSolution_.push_back(si_->cloneState(solTrav->state_));
                        prevSolutionControls_.push_back(siC_->cloneControl(solTrav->control_));
                        prevSolutionSteps_.push_back(solTrav->steps_);
                        solTrav = solTrav->parent_;
                    }
                    prevSolution_.push_back(si_->cloneState(solTrav->state_));

                    // Update thread-safe tracking variables for approximate solution
                    double currentCost = currentBestCost_.load();
                    if (opt_->isCostBetterThan(motion->accCost_, base::Cost(currentCost)) || 
                        currentCost == std::numeric_limits<double>::infinity()) {
                        currentBestCost_.store(motion->accCost_.value());
                        hasExactSolution_.store(false); // This is an approximate solution
                    }
                }

                if (oldRep != rmotion)
                {
                    while (oldRep->inactive_ && oldRep->numChildren_ == 0)
                    {
                        oldRep->inactive_ = true;
                        nn_->remove(oldRep);

                        if (oldRep->state_)
                            si_->freeState(oldRep->state_);
                        if (oldRep->control_)
                            siC_->freeControl(oldRep->control_);

                        oldRep->state_ = nullptr;
                        oldRep->control_ = nullptr;
                        oldRep->parent_->numChildren_--;
                        oldRep->parent_->children_.erase(std::remove(oldRep->parent_->children_.begin(), oldRep->parent_->children_.end(), oldRep), oldRep->parent_->children_.end());
                        Motion *oldRepParent = oldRep->parent_;
                        delete oldRep;
                        oldRep = oldRepParent;
                    }
                }
            }
        }
        iterations++;
    }

    bool solved = false;
    bool approximate = false;
    if (solution == nullptr)
    {
        solution = approxsol;
        approximate = true;
    }

    if (solution != nullptr)
    {
        /* set the solution path */
        auto path(std::make_shared<PathControl>(si_));
        for (int i = prevSolution_.size() - 1; i >= 1; --i)
            path->append(prevSolution_[i], prevSolutionControls_[i - 1],
                         prevSolutionSteps_[i - 1] * siC_->getPropagationStepSize());
        path->append(prevSolution_[0]);
        solved = true;
        pdef_->addSolutionPath(path, approximate, approxdif, getName());
    }

    si_->freeState(xstate);
    if (rmotion->state_)
        si_->freeState(rmotion->state_);
    if (rmotion->control_)
        siC_->freeControl(rmotion->control_);
    delete rmotion;

    // Stop the cost tracking thread
    planningActive_.store(false);
    tracker.join();

    OMPL_INFORM("%s: Created %u states in %u iterations", getName().c_str(), nn_->size(), iterations);

    return {solved, approximate};
}

ompl::base::PlannerStatus ompl::control::Fusion::resolve(const double replanning_time)
{
    checkValidity();
    
    // Ensure the planner has a tree to work with
    if (!nn_ || nn_->size() == 0)
    {
        OMPL_WARN("%s: No tree to resolve.", getName().c_str());
        return base::PlannerStatus::ABORT;
    }
    
    // Get the current solution path and SAVE IT before calling solve()
    ompl::base::PathPtr path = pdef_->getSolutionPath();
    auto pathControl = std::dynamic_pointer_cast<PathControl>(path);

    if (!pathControl || pathControl->getStateCount() < 2)
    {
        OMPL_WARN("%s: No valid solution path with at least 2 states available for resolve.", getName().c_str());
        return base::PlannerStatus::ABORT;
    }
    
    // SAVE the original path states before solve() overwrites them
    std::vector<ompl::base::State*> originalPathStates;
    for (size_t i = 0; i < pathControl->getStateCount(); ++i) {
        ompl::base::State* stateCopy = si_->allocState();
        si_->copyState(stateCopy, pathControl->getState(i));
        originalPathStates.push_back(stateCopy);
    }
    
    // Find the motion in the tree that corresponds to the second state in the solution path.
    // This will become our new root/start motion.
    // Note: We want to replan from the second state, so we need to find the motion
    // that corresponds to the second state in the original path
    ompl::base::State *secondStateInPath = originalPathStates[1];  // Use saved state
    
    Motion *tempMotionForSearch = new Motion(siC_);
    si_->copyState(tempMotionForSearch->state_, secondStateInPath);
    Motion *newStartMotion = nn_->nearest(tempMotionForSearch);
    delete tempMotionForSearch; // Clean up the temporary motion

    if (!newStartMotion)
    {
        OMPL_WARN("%s: Could not find nearest motion to second state.", getName().c_str());
        return base::PlannerStatus::ABORT;
    }
    
    double distance = si_->distance(newStartMotion->state_, secondStateInPath);
    if (distance > 1e-5)
    {
        OMPL_WARN("%s: Could not find the second state of the solution path in the tree (distance=%.8f).", getName().c_str(), distance);
        return base::PlannerStatus::ABORT;
    }

    // Remove everything except the subtree rooted at newStartMotion
    // Collect all motions that should be KEPT (newStartMotion and its descendants)
    std::set<Motion*> motionsToKeep;
    std::queue<Motion*> subtreeQueue;
    subtreeQueue.push(newStartMotion);
    motionsToKeep.insert(newStartMotion);
    
    while (!subtreeQueue.empty()) {
        Motion* current = subtreeQueue.front();
        subtreeQueue.pop();
        
        // Add all children to keep set and queue
        for (Motion* child : current->children_) {
            motionsToKeep.insert(child);
            subtreeQueue.push(child);
        }
    }
    
    // Get all motions currently in the tree
    std::vector<Motion*> allMotions;
    nn_->list(allMotions);
    
    // Remove motions that are NOT in the keep set
    for (Motion* motion : allMotions) {
        if (motionsToKeep.find(motion) == motionsToKeep.end()) {
            // This motion should be removed
            nn_->remove(motion);
            if (motion->state_) si_->freeState(motion->state_);
            if (motion->control_) siC_->freeControl(motion->control_);
            delete motion;
        }
    }
    
    // Make newStartMotion the new root (no parent) but KEEP its children
    newStartMotion->parent_ = nullptr;

    // Rebuild witness set since motions were removed
    if (witnesses_)
    {
        std::vector<Motion*> existing_witnesses;
        witnesses_->list(existing_witnesses);
        for(auto& w : existing_witnesses) {
            delete w;
        }
        witnesses_->clear();

        std::vector<Motion*> remaining_motions;
        nn_->list(remaining_motions);
        for (Motion* m : remaining_motions) {
            findClosestWitness(m);
        }
    }

    // Set the new start state in the problem definition
    pdef_->clearStartStates();
    pdef_->addStartState(newStartMotion->state_);
    
    // Ensure the goal is properly set (it should still be there, but make sure)
    ompl::base::GoalPtr goal = pdef_->getGoal();
    if (!goal) {
        OMPL_ERROR("%s: No goal set in problem definition", getName().c_str());
        return base::PlannerStatus::ABORT;
    }
    
    // Call solve to replan from new start
    pdef_->setGoal(goal);
    base::PlannerTerminationCondition ptc = base::timedPlannerTerminationCondition(replanning_time);
    base::PlannerStatus status = solve(ptc);

    // The solve() call has created a new solution path starting from newStartMotion
    // This new path should be the complete solution (starting from the new start state)
    if (status)
    {
        ompl::base::PathPtr newPath = pdef_->getSolutionPath();
        auto newPathControl = std::dynamic_pointer_cast<PathControl>(newPath);
        
        if (newPathControl && newPathControl->getStateCount() > 0)
        {
            // The new path from solve() already starts from newStartMotion and goes to the goal
            // This is exactly what we want - a complete solution from the new start state
            // No need to modify it further, just ensure it's properly set
            
            OMPL_INFORM("Resolve successful: new solution path created starting from new start state");
        }
    }
    else
    {
        OMPL_WARN("Resolve failed: could not find new solution from replanning");
    }

    // Clean up saved original path states
    for (ompl::base::State* state : originalPathStates) {
        si_->freeState(state);
    }

    return status;
}

ompl::base::PlannerStatus ompl::control::Fusion::simple_resolve(const double replanning_time)
{
    checkValidity();
    
    // 1. Get the current states and control in the solution path
    ompl::base::PathPtr path = pdef_->getSolutionPath();
    auto pathControl = std::dynamic_pointer_cast<PathControl>(path);

    if (!pathControl || pathControl->getStateCount() < 2)
    {
        OMPL_WARN("%s: No valid solution path with at least 2 states available for simple_resolve.", getName().c_str());
        return base::PlannerStatus::ABORT;
    }
    
    // Debug: Print original path info
    OMPL_INFORM("Simple_resolve: Original path has %d states and %d controls", 
                pathControl->getStateCount(), pathControl->getControlCount());
    if (pathControl->getStateCount() > 0)
    {
        ompl::base::State* firstState = pathControl->getState(0);
        double x = firstState->as<ompl::base::SE2StateSpace::StateType>()->getX();
        double y = firstState->as<ompl::base::SE2StateSpace::StateType>()->getY();
        double yaw = firstState->as<ompl::base::SE2StateSpace::StateType>()->getYaw();
        OMPL_INFORM("Simple_resolve: Original path starts at (%.3f, %.3f, %.3f)", x, y, yaw);
    }
    
    // 2. Change the start state to the next state in solution path
    ompl::base::State *nextState = pathControl->getState(1);
    pdef_->clearStartStates();
    pdef_->addStartState(nextState);
    
    // 3. Run the solve function for the given replanning_time
    base::PlannerTerminationCondition ptc = base::timedPlannerTerminationCondition(replanning_time);
    base::PlannerStatus status = solve(ptc);
    
    // 4. If solve succeeded, use its result (which should start from the new start state)
    // If solve failed, create a fallback path by removing the first state/control
    if (status)
    {
        // solve() succeeded and created a new solution path starting from the new start state
        // This is exactly what we want - a replanned solution from the current position
        OMPL_INFORM("Simple_resolve: solve() succeeded, using replanned solution");
        
        // Debug: Check what the new solution path looks like
        ompl::base::PathPtr newPath = pdef_->getSolutionPath();
        auto newPathControl = std::dynamic_pointer_cast<PathControl>(newPath);
        if (newPathControl && newPathControl->getStateCount() > 0)
        {
            ompl::base::State* firstState = newPathControl->getState(0);
            double x = firstState->as<ompl::base::SE2StateSpace::StateType>()->getX();
            double y = firstState->as<ompl::base::SE2StateSpace::StateType>()->getY();
            double yaw = firstState->as<ompl::base::SE2StateSpace::StateType>()->getYaw();
            OMPL_INFORM("Simple_resolve: New solution starts at (%.3f, %.3f, %.3f)", x, y, yaw);
            
            // Check if it matches the expected start state
            double expectedX = nextState->as<ompl::base::SE2StateSpace::StateType>()->getX();
            double expectedY = nextState->as<ompl::base::SE2StateSpace::StateType>()->getY();
            double expectedYaw = nextState->as<ompl::base::SE2StateSpace::StateType>()->getYaw();
            OMPL_INFORM("Simple_resolve: Expected start at (%.3f, %.3f, %.3f)", expectedX, expectedY, expectedYaw);
            
            if (std::abs(x - expectedX) > 0.01 || std::abs(y - expectedY) > 0.01 || std::abs(yaw - expectedYaw) > 0.01)
            {
                OMPL_WARN("Simple_resolve: New solution doesn't start from expected position!");
            }
        }
        
        return status;
    }
    else
    {
        // solve() failed, create a fallback path by removing the first state/control
        OMPL_INFORM("Simple_resolve: solve() failed, using fallback path (removing first state/control)");
        
        auto fallbackPath = std::make_shared<PathControl>(si_);
        
        // Add all states and controls starting from index 1 (skip the first state)
        for (size_t i = 1; i < pathControl->getStateCount(); ++i)
        {
            if (i == 1)
            {
                // First state in fallback path (was second state in original path)
                fallbackPath->append(pathControl->getState(i));
            }
            else
            {
                // Add control and state for remaining segments
                fallbackPath->append(pathControl->getState(i), 
                                   pathControl->getControl(i-1),
                                   pathControl->getControlDuration(i-1));
            }
        }
        
        pdef_->clearSolutionPaths();
        pdef_->addSolutionPath(fallbackPath, true, 0.0, getName());
        return base::PlannerStatus(true, true); // solved, approximate
    }
}

ompl::base::PlannerStatus ompl::control::Fusion::replan(const double replanning_time)
{
    checkValidity();
    
    // 1. Get the second state in the solution path
    ompl::base::PathPtr path = pdef_->getSolutionPath();
    auto pathControl = std::dynamic_pointer_cast<PathControl>(path);

    if (!pathControl || pathControl->getStateCount() < 2)
    {
        OMPL_WARN("%s: No valid solution path with at least 2 states available for replan.", getName().c_str());
        return base::PlannerStatus::ABORT;
    }
    
    ompl::base::State *newState = pathControl->getState(1);
    OMPL_INFORM("Replan: Starting from second state in solution path");
    
    // 2. Find the motion corresponding to this state (newStart)
    Motion *tempMotionForSearch = new Motion(siC_);
    si_->copyState(tempMotionForSearch->state_, newState);
    Motion *newStart = nn_->nearest(tempMotionForSearch);
    delete tempMotionForSearch;
    
    if (!newStart)
    {
        OMPL_WARN("%s: Could not find motion corresponding to new state.", getName().c_str());
        return base::PlannerStatus::ABORT;
    }
    
    double distance = si_->distance(newStart->state_, newState);
    if (distance > 1e-5)
    {
        OMPL_WARN("%s: Could not find exact motion for new state (distance=%.8f).", getName().c_str(), distance);
        return base::PlannerStatus::ABORT;
    }
    
    OMPL_INFORM("Replan: Found motion corresponding to new state");
    
    // 3. Set toKeep to false for all motions initially
    std::vector<Motion*> allMotions;
    nn_->list(allMotions);
    for (Motion* motion : allMotions)
    {
        motion->toKeep_ = false;
    }
    
    // 4. Set toKeep to true for the branch starting from the second motion in solution path
    std::queue<Motion*> branchQueue;
    branchQueue.push(newStart);
    newStart->toKeep_ = true;
    int keepCount = 1; // Start with 1 (the newStart motion)
    
    // Process the branch in breadth-first order
    while (!branchQueue.empty())
    {
        Motion* current = branchQueue.front();
        branchQueue.pop();
        
        // Add all children to the queue and mark them to keep
        for (Motion* child : current->children_)
        {
            child->toKeep_ = true;
            keepCount++;
            branchQueue.push(child);
        }
    }
    
    OMPL_INFORM("Replan: Marked %d motions to keep (branch from second state)", keepCount);
    
    // 6. Remove motions from latest to earliest (reverse order of addition)
    // We'll remove from the end of the list first
    int removedCount = 0;
    for (int i = allMotions.size() - 1; i >= 0; --i)
    {
        Motion* motion = allMotions[i];
        if (!motion->toKeep_)
        {
            nn_->remove(motion);
            if (motion->state_) si_->freeState(motion->state_);
            if (motion->control_) siC_->freeControl(motion->control_);
            delete motion;
            removedCount++;
        }
    }
    
    OMPL_INFORM("Replan: Removed %d motions, kept %d motions", removedCount, keepCount);
    
    // 7. Clear witnesses and rebuild
    if (witnesses_)
    {
        std::vector<Motion*> existing_witnesses;
        witnesses_->list(existing_witnesses);
        for(auto& w : existing_witnesses) {
            delete w;
        }
        witnesses_->clear();
    }
    
    // 8. Make newStart the root (no parent)
    newStart->parent_ = nullptr;
    
    // 9. Recalculate costs for all remaining motions
    // std::queue<Motion*> costQueue;
    // costQueue.push(newStart);
    // newStart->accCost_ = opt_->identityCost(); // Root has identity cost
    
    // while (!costQueue.empty())
    // {
    //     Motion* current = costQueue.front();
    //     costQueue.pop();
        
    //     for (Motion* child : current->children_)
    //     {
    //         // Calculate incremental cost from parent to child
    //         base::Cost incCostMotion = opt_->motionCost(current->state_, child->state_);
    //         base::Cost incCostControl = opt_->controlCost(child->control_, child->steps_);
    //         base::Cost incCost = opt_->combineCosts(incCostMotion, incCostControl);
            
    //         // Set child's accumulated cost
    //         child->accCost_ = opt_->combineCosts(current->accCost_, incCost);
            
    //         costQueue.push(child);
    //     }
    // }
    
    // 10. Rebuild witness set for remaining motions
    std::vector<Motion*> remaining_motions;
    nn_->list(remaining_motions);
    for (Motion* m : remaining_motions) {
        findClosestWitness(m);
    }
    
    OMPL_INFORM("Replan: Rebuilt tree with %d motions, new start cost: %.4f", 
                nn_->size(), newStart->accCost_.value());
    
    // 11. Set the start state in the problem definition to newState
    pdef_->clearStartStates();
    pdef_->addStartState(newState);
    
    // 12. Update the current solution path by removing the first state and control
    // Create a new path without the first state and control
    auto updatedPath = std::make_shared<PathControl>(si_);
    
    // Add all states and controls starting from index 1 (skip the first state)
    for (size_t i = 1; i < pathControl->getStateCount(); ++i)
    {
        if (i == 1)
        {
            // First state in updated path (was second state in original path)
            updatedPath->append(pathControl->getState(i));
        }
        else
        {
            // Add control and state for remaining segments
            updatedPath->append(pathControl->getState(i), 
                               pathControl->getControl(i-1),
                               pathControl->getControlDuration(i-1));
        }
    }
    
    // Clear existing solution paths and add the updated path
    pdef_->clearSolutionPaths();
    pdef_->addSolutionPath(updatedPath, true, 0.0, getName());
    
    OMPL_INFORM("Replan: Updated solution path - removed first state/control, now has %d states and %d controls", 
                updatedPath->getStateCount(), updatedPath->getControlCount());
    
    // 13. Run the solve function to continue planning from the new start state
    base::PlannerTerminationCondition ptc = base::timedPlannerTerminationCondition(replanning_time);
    base::PlannerStatus status = solve(ptc);
    
    OMPL_INFORM("Replan: solve() completed with status: %s", status ? "SUCCESS" : "FAILED");
    
    return status;
}

void ompl::control::Fusion::getPlannerData(base::PlannerData &data) const
{
    Planner::getPlannerData(data);

    std::vector<Motion *> motions;
    std::vector<Motion *> allMotions;
    if (nn_)
        nn_->list(motions);

    for (auto &motion : motions)
    {
        if (motion->numChildren_ == 0)
        {
            allMotions.push_back(motion);
        }
    }
    for (unsigned i = 0; i < allMotions.size(); i++)
    {
        if (allMotions[i]->parent_ != nullptr)
        {
            allMotions.push_back(allMotions[i]->parent_);
        }
    }

    double delta = siC_->getPropagationStepSize();

    if (prevSolution_.size() != 0)
        data.addGoalVertex(base::PlannerDataVertex(prevSolution_[0]));

    for (auto m : allMotions)
    {
        if (m->parent_)
        {
            if (data.hasControls())
                data.addEdge(base::PlannerDataVertex(m->parent_->state_), base::PlannerDataVertex(m->state_),
                             control::PlannerDataEdgeControl(m->control_, m->steps_ * delta));
            else
                data.addEdge(base::PlannerDataVertex(m->parent_->state_), base::PlannerDataVertex(m->state_));
        }
        else
            data.addStartVertex(base::PlannerDataVertex(m->state_));
    }
}
