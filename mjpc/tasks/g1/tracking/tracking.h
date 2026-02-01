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

#ifndef MJPC_TASKS_G1_TRACKING_TASK_H_
#define MJPC_TASKS_G1_TRACKING_TASK_H_

#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
namespace g1 {

// Frame of kinematic reference data
struct ReferenceFrame {
  double root_pos[3];      // Root position (x, y, z)
  double root_quat[4];     // Root orientation quaternion (w, x, y, z)
  std::vector<double> joint_pos;  // Joint positions
};

class Tracking : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Tracking* task, double reference_time = 0)
        : mjpc::BaseResidualFn(task),
          reference_time_(reference_time) {}

    // ------------- Residuals for G1 tracking task -------------
    //   Residual (0): Joint velocity - minimize joint velocities
    //   Residual (1): Control - minimize control effort
    //   Residual (2): Root position - track root position
    //   Residual (3): Root orientation - track root orientation
    //   Residual (4): Joint position - track joint positions
    // -----------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

   private:
    friend class Tracking;
    double reference_time_;
  };

  Tracking() : residual_(this) {}

  // Load reference trajectory from CSV file
  // CSV format: each row is a frame with:
  //   root_pos_x, root_pos_y, root_pos_z,
  //   root_quat_w, root_quat_x, root_quat_y, root_quat_z,
  //   joint_pos_0, joint_pos_1, ..., joint_pos_N
  bool LoadTrajectory(const std::string& csv_path, double fps);

  // Get interpolated reference at given time
  void GetReference(double time, double* root_pos, double* root_quat,
                    double* joint_pos) const;

  // Transition function - updates mocap bodies for visualization
  void TransitionLocked(mjModel* model, mjData* data) override;

  // Reset function - loads trajectory from model custom elements
  void ResetLocked(const mjModel* model) override;

  std::string Name() const override;
  std::string XmlPath() const override;

  // Accessors
  double fps() const { return fps_; }
  int num_frames() const { return static_cast<int>(trajectory_.size()); }
  double duration() const {
    return trajectory_.empty() ? 0.0 : (trajectory_.size() - 1) / fps_;
  }
  bool trajectory_loaded() const { return !trajectory_.empty(); }

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.reference_time_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;

  // Trajectory data
  std::vector<ReferenceFrame> trajectory_;
  double fps_ = 30.0;  // Default FPS
  int num_joints_ = 0;  // Number of joints in reference
};

}  // namespace g1
}  // namespace mjpc

#endif  // MJPC_TASKS_G1_TRACKING_TASK_H_
