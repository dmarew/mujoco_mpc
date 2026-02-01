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

#include "mjpc/tasks/g1/tracking/tracking.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace {

// Compute interpolation weights between two frames
std::tuple<int, int, double, double> ComputeInterpolationValues(
    double index, int max_index) {
  int index_0 = static_cast<int>(std::floor(std::clamp(index, 0.0,
                                                        static_cast<double>(max_index))));
  int index_1 = std::min(index_0 + 1, max_index);

  double weight_1 = std::clamp(index, 0.0, static_cast<double>(max_index)) - index_0;
  double weight_0 = 1.0 - weight_1;

  return {index_0, index_1, weight_0, weight_1};
}

// Spherical linear interpolation for quaternions
void SlerpQuat(double* result, const double* q0, const double* q1, double t) {
  // Compute dot product
  double dot = q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3];

  // If dot is negative, negate one quaternion to take shorter path
  double q1_adj[4];
  if (dot < 0.0) {
    dot = -dot;
    for (int i = 0; i < 4; i++) q1_adj[i] = -q1[i];
  } else {
    for (int i = 0; i < 4; i++) q1_adj[i] = q1[i];
  }

  // If quaternions are very close, use linear interpolation
  if (dot > 0.9995) {
    for (int i = 0; i < 4; i++) {
      result[i] = q0[i] + t * (q1_adj[i] - q0[i]);
    }
    // Normalize
    double norm = std::sqrt(result[0]*result[0] + result[1]*result[1] +
                            result[2]*result[2] + result[3]*result[3]);
    for (int i = 0; i < 4; i++) result[i] /= norm;
    return;
  }

  // Spherical interpolation
  double theta_0 = std::acos(dot);
  double theta = theta_0 * t;
  double sin_theta = std::sin(theta);
  double sin_theta_0 = std::sin(theta_0);

  double s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
  double s1 = sin_theta / sin_theta_0;

  for (int i = 0; i < 4; i++) {
    result[i] = s0 * q0[i] + s1 * q1_adj[i];
  }
}

// Parse a line of CSV data
std::vector<double> ParseCsvLine(const std::string& line) {
  std::vector<double> values;
  std::stringstream ss(line);
  std::string token;

  while (std::getline(ss, token, ',')) {
    // Skip empty tokens and whitespace-only tokens
    size_t start = token.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) continue;
    size_t end = token.find_last_not_of(" \t\r\n");
    token = token.substr(start, end - start + 1);

    if (!token.empty()) {
      try {
        values.push_back(std::stod(token));
      } catch (...) {
        // Skip non-numeric values (e.g., header row)
        return {};
      }
    }
  }
  return values;
}

}  // namespace

namespace mjpc::g1 {

std::string Tracking::XmlPath() const {
  return GetModelPath("g1/tracking/task.xml");
}

std::string Tracking::Name() const { return "G1 Track"; }

bool Tracking::LoadTrajectory(const std::string& csv_path, double fps) {
  trajectory_.clear();
  fps_ = fps;

  std::ifstream file(csv_path);
  if (!file.is_open()) {
    mju_warning("G1 Tracking: Failed to open trajectory file: %s",
                csv_path.c_str());
    return false;
  }

  std::string line;
  bool first_line = true;

  while (std::getline(file, line)) {
    // Skip empty lines
    if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;
    }

    std::vector<double> values = ParseCsvLine(line);

    // Skip header or invalid lines
    if (values.size() < 7) {  // At minimum: 3 pos + 4 quat
      if (first_line) {
        first_line = false;
        continue;  // Skip header row
      }
      continue;
    }
    first_line = false;

    ReferenceFrame frame;

    // Root position (first 3 values)
    frame.root_pos[0] = values[0];
    frame.root_pos[1] = values[1];
    frame.root_pos[2] = values[2];

    // Root quaternion (next 4 values) - CSV is xyzw, MuJoCo uses wxyz
    frame.root_quat[0] = values[6];  // w
    frame.root_quat[1] = values[3];  // x
    frame.root_quat[2] = values[4];  // y
    frame.root_quat[3] = values[5];  // z

    // Normalize quaternion
    double qnorm = std::sqrt(frame.root_quat[0]*frame.root_quat[0] +
                             frame.root_quat[1]*frame.root_quat[1] +
                             frame.root_quat[2]*frame.root_quat[2] +
                             frame.root_quat[3]*frame.root_quat[3]);
    if (qnorm > 1e-10) {
      for (int i = 0; i < 4; i++) frame.root_quat[i] /= qnorm;
    }

    // Joint positions (remaining values)
    int num_joints = static_cast<int>(values.size()) - 7;
    frame.joint_pos.resize(num_joints);
    for (int i = 0; i < num_joints; i++) {
      frame.joint_pos[i] = values[7 + i];
    }

    // Set or verify number of joints
    if (trajectory_.empty()) {
      num_joints_ = num_joints;
    } else if (num_joints != num_joints_) {
      mju_warning("G1 Tracking: Inconsistent joint count in CSV");
      trajectory_.clear();
      return false;
    }

    trajectory_.push_back(frame);
  }

  if (trajectory_.empty()) {
    mju_warning("G1 Tracking: No valid frames loaded from trajectory file");
    return false;
  }

  // Log success (mju_warning is used for info messages too)
  char msg[256];
  snprintf(msg, sizeof(msg), "G1 Tracking: Loaded %d frames at %.1f FPS (%.2f sec)",
           static_cast<int>(trajectory_.size()), fps_, duration());
  mju_warning("%s", msg);

  return true;
}

void Tracking::GetReference(double time, double* root_pos, double* root_quat,
                            double* joint_pos) const {
  if (trajectory_.empty()) {
    // No trajectory loaded, return default pose
    if (root_pos) {
      root_pos[0] = 0.0;
      root_pos[1] = 0.0;
      root_pos[2] = 0.75;  // Default standing height
    }
    if (root_quat) {
      root_quat[0] = 1.0;
      root_quat[1] = 0.0;
      root_quat[2] = 0.0;
      root_quat[3] = 0.0;
    }
    if (joint_pos) {
      for (int i = 0; i < num_joints_; i++) {
        joint_pos[i] = 0.0;
      }
    }
    return;
  }

  // Compute frame index
  double frame_index = time * fps_;
  int max_index = static_cast<int>(trajectory_.size()) - 1;

  // Get interpolation values
  int idx0, idx1;
  double w0, w1;
  std::tie(idx0, idx1, w0, w1) = ComputeInterpolationValues(frame_index, max_index);

  const ReferenceFrame& f0 = trajectory_[idx0];
  const ReferenceFrame& f1 = trajectory_[idx1];

  // Interpolate root position
  if (root_pos) {
    for (int i = 0; i < 3; i++) {
      root_pos[i] = w0 * f0.root_pos[i] + w1 * f1.root_pos[i];
    }
  }

  // Interpolate root orientation using SLERP
  if (root_quat) {
    SlerpQuat(root_quat, f0.root_quat, f1.root_quat, w1);
  }

  // Interpolate joint positions
  if (joint_pos) {
    int n = std::min(num_joints_, static_cast<int>(f0.joint_pos.size()));
    for (int i = 0; i < n; i++) {
      joint_pos[i] = w0 * f0.joint_pos[i] + w1 * f1.joint_pos[i];
    }
  }
}

void Tracking::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  const Tracking* task = static_cast<const Tracking*>(task_);
  int counter = 0;

  // Get current reference
  double ref_time = data->time - reference_time_;
  if (ref_time < 0) ref_time = 0;

  // Wrap time if trajectory is loaded and looping
  if (task->trajectory_loaded() && task->duration() > 0) {
    ref_time = std::fmod(ref_time, task->duration());
  }

  double ref_root_pos[3];
  double ref_root_quat[4];
  std::vector<double> ref_joint_pos(model->nq - 7);  // Exclude floating base
  task->GetReference(ref_time, ref_root_pos, ref_root_quat, ref_joint_pos.data());

  // ----- Residual 0: Joint velocity -----
  // Minimize joint velocities (exclude floating base: 6 DoF)
  int nv_joints = model->nv - 6;
  mju_copy(residual + counter, data->qvel + 6, nv_joints);
  counter += nv_joints;

  // ----- Residual 1: Control -----
  // Minimize control effort
  mju_copy(residual + counter, data->ctrl, model->nu);
  counter += model->nu;

  // ----- Residual 2: Root position tracking -----
  // Current root position (first 3 elements of qpos for floating base)
  for (int i = 0; i < 3; i++) {
    residual[counter + i] = data->qpos[i] - ref_root_pos[i];
  }
  counter += 3;

  // ----- Residual 3: Root orientation tracking -----
  // Use mju_subQuat for orientation error
  double quat_error[3];
  mju_subQuat(quat_error, data->qpos + 3, ref_root_quat);
  mju_copy(residual + counter, quat_error, 3);
  counter += 3;

  // ----- Residual 4: Joint position tracking -----
  // Track joint positions (exclude floating base)
  int nq_joints = model->nq - 7;  // Floating base uses 7 (3 pos + 4 quat)
  int n_track = std::min(nq_joints, static_cast<int>(ref_joint_pos.size()));
  for (int i = 0; i < n_track; i++) {
    residual[counter + i] = data->qpos[7 + i] - ref_joint_pos[i];
  }
  // Zero out remaining if reference has fewer joints
  for (int i = n_track; i < nq_joints; i++) {
    residual[counter + i] = 0.0;
  }
  counter += nq_joints;

  CheckSensorDim(model, counter);
}

void Tracking::ResetLocked(const mjModel* model) {
  // Try to load trajectory from model custom text elements
  // Look for "trajectory_path" and "trajectory_fps" custom elements
  char* traj_path = GetCustomTextData(model, "trajectory_path");
  double* fps_data = GetCustomNumericData(model, "trajectory_fps");

  if (traj_path && std::strlen(traj_path) > 0) {
    double fps = fps_data ? fps_data[0] : 30.0;

    // Resolve path relative to task directory if not absolute
    std::string path_str(traj_path);
    if (!path_str.empty() && path_str[0] != '/') {
      // Relative path - resolve relative to task directory
      std::string task_dir = GetModelPath("g1/tracking/");
      path_str = task_dir + path_str;
    }

    LoadTrajectory(path_str, fps);
  }
}

void Tracking::TransitionLocked(mjModel* model, mjData* data) {
  // Reset on simulation reset (time == 0) or mode change
  if (data->time == 0.0) {
    residual_.reference_time_ = 0.0;

    // Set initial pose from trajectory if available
    if (trajectory_loaded()) {
      double ref_root_pos[3];
      double ref_root_quat[4];
      std::vector<double> ref_joint_pos(num_joints_);
      GetReference(0.0, ref_root_pos, ref_root_quat, ref_joint_pos.data());

      // Set root pose
      mju_copy3(data->qpos, ref_root_pos);
      mju_copy(data->qpos + 3, ref_root_quat, 4);

      // Set joint positions
      int nq_joints = model->nq - 7;
      int n_set = std::min(nq_joints, num_joints_);
      for (int i = 0; i < n_set; i++) {
        data->qpos[7 + i] = ref_joint_pos[i];
      }

      // Zero velocities
      mju_zero(data->qvel, model->nv);
    } else {
      // No trajectory loaded - reset to first keyframe from model
      if (model->nkey > 0) {
        mju_copy(data->qpos, model->key_qpos, model->nq);
        mju_zero(data->qvel, model->nv);
      }
    }
  }

  // Update mocap body for visualization (if present)
  int mocap_body_id = mj_name2id(model, mjOBJ_BODY, "ref_pelvis");
  if (mocap_body_id >= 0) {
    int body_mocapid = model->body_mocapid[mocap_body_id];
    if (body_mocapid >= 0 && trajectory_loaded()) {
      double ref_time = data->time - residual_.reference_time_;
      if (ref_time < 0) ref_time = 0;
      if (duration() > 0) {
        ref_time = std::fmod(ref_time, duration());
      }

      double ref_root_pos[3];
      double ref_root_quat[4];
      GetReference(ref_time, ref_root_pos, ref_root_quat, nullptr);

      mju_copy3(data->mocap_pos + 3 * body_mocapid, ref_root_pos);
      mju_copy(data->mocap_quat + 4 * body_mocapid, ref_root_quat, 4);
    }
  }
}

}  // namespace mjpc::g1
