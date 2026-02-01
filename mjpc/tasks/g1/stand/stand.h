// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MJPC_TASKS_G1_STAND_TASK_H_
#define MJPC_TASKS_G1_STAND_TASK_H_

#include <memory>
#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace g1 {

class Stand : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Stand* task) : mjpc::BaseResidualFn(task) {}

    // ------------------ Residuals for G1 stand task -----------------
    //   Number of residuals: 5 + nv_joints + nu
    //     Residual (0): Height: torso height relative to feet
    //     Residual (1): Balance: COM capture point vs average foot position
    //     Residual (2-3): COM xy velocity (should be 0)
    //     Residual (4): Joint velocity (nv - 6 dimensions)
    //     Residual (5): Control (nu dimensions)
    //   Number of parameters: 1
    //     Parameter (0): height_goal
    // ----------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  Stand() : residual_(this) {}

  std::string Name() const override;
  std::string XmlPath() const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace g1
}  // namespace mjpc

#endif  // MJPC_TASKS_G1_STAND_TASK_H_