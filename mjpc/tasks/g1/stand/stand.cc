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

#include "mjpc/tasks/g1/stand/stand.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"


namespace mjpc::g1 {

std::string Stand::XmlPath() const {
  return GetModelPath("g1/stand/task.xml");
}
std::string Stand::Name() const { return "G1 Stand"; }

// ------------------ Residuals for G1 stand task ------------
//   Number of residuals: 7 + nv_joints + nu
//     Residual (0): Height: torso height relative to feet
//     Residual (1): Balance: COM capture point vs average foot position
//     Residual (2-3): Upright: torso orientation (xy components)
//     Residual (4-5): COM xy velocity (should be 0)
//     Residual (6): Joint velocity (nv - 6 dimensions)
//     Residual (7): Control (nu dimensions)
//   Number of parameters: 1
//     Parameter (0): height_goal
// ----------------------------------------------------------------
void Stand::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                 double* residual) const {
  int counter = 0;

  // ----- Height: torso feet vertical error ----- //

  // feet sensor positions
  double* left_foot_position = SensorByName(model, data, "left_foot_position");
  double* right_foot_position = SensorByName(model, data, "right_foot_position");
  double* torso_position = SensorByName(model, data, "torso_position");
  double torso_feet_error =
      torso_position[2] - 0.5 * (left_foot_position[2] + right_foot_position[2]);
  residual[counter++] = torso_feet_error - parameters_[0];

  // ----- Balance: CoM-feet xy error ----- //

  // capture point
  double* com_position = SensorByName(model, data, "torso_subtreecom");
  double* com_velocity = SensorByName(model, data, "torso_subtreelinvel");
  double kFallTime = 0.2;
  double capture_point[3] = {com_position[0], com_position[1], com_position[2]};
  mju_addToScl3(capture_point, com_velocity, kFallTime);

  // average feet xy position
  double fxy_avg[2] = {0.0};
  mju_addTo(fxy_avg, left_foot_position, 2);
  mju_addTo(fxy_avg, right_foot_position, 2);
  mju_scl(fxy_avg, fxy_avg, 0.5, 2);

  mju_subFrom(fxy_avg, capture_point, 2);
  double com_feet_distance = mju_norm(fxy_avg, 2);
  residual[counter++] = com_feet_distance;

  // ----- Upright: torso orientation ----- //
  double* torso_zaxis = SensorByName(model, data, "torso_zaxis");
  // torso_zaxis is unit vector in world frame; we want it aligned with world up (0,0,1)
  // Error is xy components (z should be 1, but we don't need to enforce)
  residual[counter++] = torso_zaxis[0];  // x component should be 0
  residual[counter++] = torso_zaxis[1];  // y component should be 0

  // ----- COM xy velocity should be 0 ----- //
  mju_copy(&residual[counter], com_velocity, 2);
  counter += 2;

  // ----- joint velocity ----- //
  mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
  counter += model->nv - 6;

  // ----- action ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // sensor dim sanity check
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        counter);
  }
}

}  // namespace mjpc::g1