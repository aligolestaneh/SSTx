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
 
/* Authors: Ali Golestaneh*/
 
#ifndef OMPL_CONTROL_PLANNERS_FUSION_
#define OMPL_CONTROL_PLANNERS_FUSION_
 
#include "ompl/control/planners/PlannerIncludes.h"
#include "ompl/datastructures/NearestNeighbors.h"
 
namespace ompl
{
    namespace control
    {
        class FUSION: public base::Planner
        {
        public: 
            FUSION(const SpaceInformationPtr &si);
 
            ~FUSION() override;
 
            void setup() override;
 
            base::PlannerStatus solve(const base::PlannerTerminationCondition &ptc) override;
 
            void getPlannerData(base::PlannerData &data) const override;
 
            void clear() override;
 
            void setGoalBias(double goalBias)
            {
                goalBias_ = goalBias;
            }
 
            double getGoalBias() const
            {
                return goalBias_;
            }
 
            void setSelectionRadius(double selectionRadius)
            {
                selectionRadius_ = selectionRadius;
            }
 
            double getSelectionRadius() const
            {
                return selectionRadius_;
            }
 
            void setPruningRadius(double pruningRadius)
            {
                pruningRadius_ = pruningRadius;
            }

            double getPruningRadius() const
            {
                return pruningRadius_;
            }
 
            template <template <typename T> class NN>
            void setNearestNeighbors()
            {
                if (nn_ && nn_->size() != 0)
                    OMPL_WARN("Calling setNearestNeighbors will clear all states.");
                clear();
                nn_ = std::make_shared<NN<Motion *>>();
                witnesses_ = std::make_shared<NN<Motion *>>();
                setup();
            }

            std::vector<base::State *> getNearestStatesR(const base::State *state, double radius) const;

            // Getter methods for tree statistics
            std::size_t getNearestNeighborsSize() const
            {
                return nn_ ? nn_->size() : 0;
            }

            std::size_t getWitnessesSize() const
            {
                return witnesses_ ? witnesses_->size() : 0;
            }

            class Motion
            {
            public:
                Motion() = default;
 
                Motion(const SpaceInformation *si)
                  : state_(si->allocState()), control_(si->allocControl())
                {
                }
 
                virtual ~Motion() = default;
 
                virtual base::State *getState() const
                {
                    return state_;
                }
                virtual Motion *getParent() const
                {
                    return parent_;
                }
 
                base::Cost accCost_{0};
 
                base::State *state_{nullptr};
 
                Control *control_{nullptr};
 
                unsigned int steps_{0};
 
                Motion *parent_{nullptr};
 
                unsigned numChildren_{0};
 
                bool inactive_{false};

                bool rewire_{false};
            };
 
            class Witness : public Motion
            {
            public:
                Witness() = default;
 
                Witness(const SpaceInformation *si) : Motion(si)
                {
                }
                base::State *getState() const override
                {
                    return rep_->state_;
                }
                Motion *getParent() const override
                {
                    return rep_->parent_;
                }
 
                void linkRep(Motion *lRep)
                {
                    rep_ = lRep;
                }
 
                Motion *rep_{nullptr};
            };

            // Distance function (made public for testing)
            double distanceFunction(const Motion *a, const Motion *b) const
            {
                // Create temporary states for normalization
                auto temp1 = si_->allocState();
                auto temp2 = si_->allocState();
                
                // Copy states
                si_->copyState(temp1, a->state_);
                si_->copyState(temp2, b->state_);
                
                // Enforce bounds (normalize angles)
                si_->enforceBounds(temp1);
                si_->enforceBounds(temp2);
                
                // Compute distance using normalized states
                double dist = si_->distance(temp1, temp2);
                
                // Clean up
                si_->freeState(temp1);
                si_->freeState(temp2);
                
                return dist;
            }
 
        protected:
            Motion *selectNode(Motion *sample);
 
            Witness *findClosestWitness(Motion *node);
 
            void freeMemory();
 
            base::StateSamplerPtr sampler_;
 
            ControlSamplerPtr controlSampler_;
 
            const SpaceInformation *siC_;
 
            std::shared_ptr<NearestNeighbors<Motion *>> nn_;
 
            std::shared_ptr<NearestNeighbors<Motion *>> witnesses_;
 
            double goalBias_{0.05};
 
            double selectionRadius_{0.2};
 
            double pruningRadius_{0.1};
 
            RNG rng_;
 
            std::vector<base::State *> prevSolution_;
            std::vector<Control *> prevSolutionControls_;
            std::vector<unsigned> prevSolutionSteps_;


 
            base::Cost prevSolutionCost_;
 
            base::OptimizationObjectivePtr opt_;
        };
    }
}
 
#endif