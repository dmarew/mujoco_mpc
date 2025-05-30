// Copyright 2022 DeepMind Technologies Limited
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

#include "mjpc/planners/model_derivatives.h"

#include <algorithm>
#include <random>        
#include <functional> 

#include <mujoco/mujoco.h>
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {

// allocate memory
void ModelDerivatives::Allocate(int dim_state_derivative, int dim_action,
                                int dim_sensor, int T) {
  A.resize(dim_state_derivative * dim_state_derivative * T);
  B.resize(dim_state_derivative * dim_action * T);
  C.resize(dim_sensor * dim_state_derivative * T);
  D.resize(dim_sensor * dim_action * T);
}

// reset memory to zeros
void ModelDerivatives::Reset(int dim_state_derivative, int dim_action,
                             int dim_sensor, int T) {
  std::fill(A.begin(),
            A.begin() + T * dim_state_derivative * dim_state_derivative, 0.0);
  std::fill(B.begin(), B.begin() + T * dim_state_derivative * dim_action, 0.0);
  std::fill(C.begin(), C.begin() + T * dim_sensor * dim_state_derivative, 0.0);
  std::fill(D.begin(), D.begin() + T * dim_sensor * dim_action, 0.0);
}

// compute derivatives at all time steps
void ModelDerivatives::Compute(const mjModel* m,
                               const std::vector<UniqueMjData>& data,
                               const double* x, const double* u,
                               const double* h, int dim_state,
                               int dim_state_derivative, int dim_action,
                               int dim_sensor, int T, double tol, int mode,
                               ThreadPool& pool, int skip) {
  // reset indices
  evaluate_.clear();
  interpolate_.clear();

  // evaluate indices
  int s = skip + 1;
  evaluate_.push_back(0);
  for (int t = s; t < T - s; t += s) {
    evaluate_.push_back(t);
  }
  evaluate_.push_back(T - 2);
  evaluate_.push_back(T - 1);

  // interpolate indices
  for (int t = 0, e = 0; t < T; t++) {
    if (e == evaluate_.size() || evaluate_[e] > t) {
      interpolate_.push_back(t);
    } else {
      e++;
    }
  }

  // evaluate derivatives
  int count_before = pool.GetCount();
  for (int t : evaluate_) {
    pool.Schedule([&m, &data, &A = A, &B = B, &C = C, &D = D, &x, &u, &h,
                   dim_state, dim_state_derivative, dim_action, dim_sensor, tol,
                   mode, t, T]() {
      mjData* d = data[ThreadPool::WorkerId()].get();
      // set state
      SetState(m, d, x + t * dim_state);
      d->time = h[t];

      // set action
      mju_copy(d->ctrl, u + t * dim_action, dim_action);

      // Jacobians
      if (t == T - 1) {
        // Jacobians
        mjd_transitionFD(m, d, tol, mode, nullptr, nullptr,
                         DataAt(C, t * (dim_sensor * dim_state_derivative)),
                         nullptr);
      } else {
        // derivatives
        mjd_transitionFD(
            m, d, tol, mode,
            DataAt(A, t * (dim_state_derivative * dim_state_derivative)),
            DataAt(B, t * (dim_state_derivative * dim_action)),
            DataAt(C, t * (dim_sensor * dim_state_derivative)),
            DataAt(D, t * (dim_sensor * dim_action)));
      }
    });
  }
  pool.WaitCount(count_before + evaluate_.size());
  pool.ResetCount();

  // interpolate derivatives
  count_before = pool.GetCount();
  for (int t : interpolate_) {
    pool.Schedule([&A = A, &B = B, &C = C, &D = D, &evaluate_ = this->evaluate_,
                   dim_state_derivative, dim_action, dim_sensor, t]() {
      // find interval
      int bounds[2];
      FindInterval(bounds, evaluate_, t, evaluate_.size());
      int e0 = evaluate_[bounds[0]];
      int e1 = evaluate_[bounds[1]];

      // normalized input
      double tt = double(t - e0) / double(e1 - e0);
      if (bounds[0] == bounds[1]) {
        tt = 0.0;
      }

      // A
      int nA = dim_state_derivative * dim_state_derivative;
      double* Ai = DataAt(A, t * nA);
      const double* AL = DataAt(A, e0 * nA);
      const double* AU = DataAt(A, e1 * nA);

      mju_scl(Ai, AL, 1.0 - tt, nA);
      mju_addToScl(Ai, AU, tt, nA);

      // B
      int nB = dim_state_derivative * dim_action;
      double* Bi = DataAt(B, t * nB);
      const double* BL = DataAt(B, e0 * nB);
      const double* BU = DataAt(B, e1 * nB);

      mju_scl(Bi, BL, 1.0 - tt, nB);
      mju_addToScl(Bi, BU, tt, nB);

      // C
      int nC = dim_sensor * dim_state_derivative;
      double* Ci = DataAt(C, t * nC);
      const double* CL = DataAt(C, e0 * nC);
      const double* CU = DataAt(C, e1 * nC);

      mju_scl(Ci, CL, 1.0 - tt, nC);
      mju_addToScl(Ci, CU, tt, nC);

      // D
      int nD = dim_sensor * dim_action;
      double* Di = DataAt(D, t * nD);
      const double* DL = DataAt(D, e0 * nD);
      const double* DU = DataAt(D, e1 * nD);

      mju_scl(Di, DL, 1.0 - tt, nD);
      mju_addToScl(Di, DU, tt, nD);
    });
  }

  pool.WaitCount(count_before + interpolate_.size());
  pool.ResetCount();
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
//  Fills A,C once; fills B,D with Gaussian-smoothed average.
//    If terminal=true   ⇒ only A & C are computed (B,D untouched).
// -----------------------------------------------------------------------------
static void ComputeDerivativesSmoothOnce(
    const mjModel* m, mjData* d,
    const double* x, const double* u, double time,
    int dim_state, int dim_sd, int dim_act, int dim_sen,
    double sigma, int nsamples, std::uint64_t seed,
    double tol, int mode, bool terminal,
    double* A, double* B, double* C, double* D)
{
  // ---------- 0)  nominal FD (fills A,C and optionally B,D) ------------------
  SetState(m, d, x);
  d->time  = time;
  mju_copy(d->ctrl, u, dim_act);

  mjd_transitionFD(m, d, tol, mode,
                   A,
                   terminal ? nullptr : B,
                   C,
                   terminal ? nullptr : D);

  if (terminal) return;               // nothing more to do for last knot point

  // ---------- 1)  RNG & scratch ----------------------------------------------
  std::mt19937_64 rng(seed ^
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
  std::normal_distribution<double> N01(0.0, 1.0);

  std::vector<double> eps(dim_act);
  std::vector<double> Btmp(dim_sd * dim_act);
  std::vector<double> Dtmp(dim_sen * dim_act);

  // clear B & D accumulators
  mju_zero(B, dim_sd * dim_act);
  if (dim_sen) mju_zero(D, dim_sen * dim_act);

  const double inv_pairs = 1.0 / (2.0 * nsamples);

  // ---------- 2)  Monte-Carlo average ----------------------------------------
  for (int k = 0; k < nsamples; ++k) {
    for (int j = 0; j < dim_act; ++j) eps[j] = N01(rng);

    for (int sign : {+1, -1}) {
      // perturbed control
      SetState(m, d, x);
      d->time = time;
      for (int j = 0; j < dim_act; ++j)
        d->ctrl[j] = u[j] + sign * sigma * eps[j];

      // FD wrt input only
      mjd_transitionFD(m, d, tol, mode,
                       nullptr, Btmp.data(), nullptr,
                       dim_sen ? Dtmp.data() : nullptr);

      // accumulate
      double* Bout = B;
      for (double v : Btmp) *Bout++ += v * inv_pairs;
      if (dim_sen) {
        double* Dout = D;
        for (double v : Dtmp) *Dout++ += v * inv_pairs;
      }
    }
  }
}
void ModelDerivatives::ComputeSmoothed(const mjModel* m,
                               const std::vector<UniqueMjData>& data,
                               const double* x, const double* u,
                               const double* h,
                               int dim_state, int dim_sd,
                               int dim_act,   int dim_sen,
                               int T,
                               double tol, int mode,
                               ThreadPool& pool, int skip, double sigma, int nsamples)
{
  // 1) resize member buffers --------------------------------------------------
  const std::size_t nA = dim_sd * dim_sd;
  const std::size_t nB = dim_sd * dim_act;
  const std::size_t nC = dim_sen * dim_sd;
  const std::size_t nD = dim_sen * dim_act;

  A.resize(static_cast<std::size_t>(T) * nA);
  B.resize(static_cast<std::size_t>(T) * nB);
  C.resize(static_cast<std::size_t>(T) * nC);
  D.resize(static_cast<std::size_t>(T) * nD);

  // 2) choose indices to evaluate (respect skip) ------------------------------
  evaluate_.clear();
  interpolate_.clear();
  int step = skip + 1;
  for (int t = 0; t < T; t += step) evaluate_.push_back(t);
  if (evaluate_.back() != T - 1) evaluate_.push_back(T - 1);  // make sure last
  for (int t = 0, e = 0; t < T; t++) {
    if (e == evaluate_.size() || evaluate_[e] > t) {
      interpolate_.push_back(t);
    } else {
      e++;
    }
  }
  // 3) schedule the tasks -----------------------------------------------------
  int base = pool.GetCount();

  for (int t : evaluate_) {
    pool.Schedule([&, t] {
      mjData* d = data[ThreadPool::WorkerId()].get();

      double* At = DataAt(A, static_cast<std::size_t>(t) * nA);
      double* Bt = DataAt(B, static_cast<std::size_t>(t) * nB);
      double* Ct = DataAt(C, static_cast<std::size_t>(t) * nC);
      double* Dt = DataAt(D, static_cast<std::size_t>(t) * nD);

      bool terminal = (t == T - 1);

      ComputeDerivativesSmoothOnce(
          m, d,
          x + static_cast<std::size_t>(t) * dim_state,
          u + static_cast<std::size_t>(t) * dim_act,
          h[t],
          dim_state, dim_sd, dim_act, dim_sen,
          /*sigma=*/sigma,
          /*nsamples=*/nsamples, 
          /*seed=*/12345 + 791*t,   // decorrelate frames
          tol, mode,
          terminal,
          At, Bt, Ct, Dt);
    });
  }

  pool.WaitCount(base + static_cast<int>(evaluate_.size()));
  pool.ResetCount();

  // 4) interpolate the derivatives -------------------------------------------
  // interpolate derivatives
  int count_before = pool.GetCount();
  for (int t : interpolate_) {
    pool.Schedule([&A = A, &B = B, &C = C, &D = D, &evaluate_ = this->evaluate_,
                   dim_sd, dim_act, dim_sen, t]() {
      // find interval
      int bounds[2];
      FindInterval(bounds, evaluate_, t, evaluate_.size());
      int e0 = evaluate_[bounds[0]];
      int e1 = evaluate_[bounds[1]];

      // normalized input
      double tt = double(t - e0) / double(e1 - e0);
      if (bounds[0] == bounds[1]) {
        tt = 0.0;
      }

      // A
      int nA = dim_sd * dim_sd;
      double* Ai = DataAt(A, t * nA);
      const double* AL = DataAt(A, e0 * nA);
      const double* AU = DataAt(A, e1 * nA);

      mju_scl(Ai, AL, 1.0 - tt, nA);
      mju_addToScl(Ai, AU, tt, nA);

      // B
      int nB = dim_sd * dim_act;
      double* Bi = DataAt(B, t * nB);
      const double* BL = DataAt(B, e0 * nB);
      const double* BU = DataAt(B, e1 * nB);

      mju_scl(Bi, BL, 1.0 - tt, nB);
      mju_addToScl(Bi, BU, tt, nB);

      // C
      int nC = dim_sen * dim_sd;
      double* Ci = DataAt(C, t * nC);
      const double* CL = DataAt(C, e0 * nC);
      const double* CU = DataAt(C, e1 * nC);

      mju_scl(Ci, CL, 1.0 - tt, nC);
      mju_addToScl(Ci, CU, tt, nC);

      // D
      int nD = dim_sen * dim_act;
      double* Di = DataAt(D, t * nD);
      const double* DL = DataAt(D, e0 * nD);
      const double* DU = DataAt(D, e1 * nD);

      mju_scl(Di, DL, 1.0 - tt, nD);
      mju_addToScl(Di, DU, tt, nD);
    });
  }

  pool.WaitCount(count_before + interpolate_.size());
  pool.ResetCount();
}
}  // namespace mjpc
