#pragma once
#include <mps/flags.h>
#include <mps/algorithms/truncation.h>

namespace mps {

class SimplificationStrategy : public TruncationStrategy {
  double stop_tolerance_{FLAGS.get(MPS_SIMPLIFY_TOLERANCE)};
  index_t max_sweeps_{2};
  int direction_{+1};
  bool normalize_{false};
  bool single_site_{false};
  bool debug_simplification_{FLAGS.get_int(MPS_DEBUG_SIMPLIFY) > 0};

 public:
  SimplificationStrategy &set_simplification_debug_level(bool debug) {
    debug_simplification_ = debug;
    return *this;
  }

  SimplificationStrategy &set_sweeps(index_t sweeps) {
    tensor_assert(sweeps > 0);
    max_sweeps_ = sweeps;
    return *this;
  }

  SimplificationStrategy &set_direction(int direction) {
    tensor_assert(direction != 0);
    direction_ = direction;
    return *this;
  }

  SimplificationStrategy &set_stop_tolerance(double tolerance) {
    stop_tolerance_ = tolerance;
    return *this;
  }

  SimplificationStrategy &set_normalization(bool normalize) {
    normalize_ = normalize;
    return *this;
  }

  SimplificationStrategy &set_single_site_simplification() {
    single_site_ = true;
    return *this;
  }

  SimplificationStrategy &set_two_site_simplification() {
    single_site_ = false;
    return *this;
  }

  index_t sweeps() const { return max_sweeps_; }
  int direction() const { return direction_; }
  bool normalize() const { return normalize_; }
  bool single_site_simplification() const { return single_site_; }
  double stop_relative_tolerance() const { return stop_tolerance_; }
  bool debug_simplification() const { return debug_simplification_; }
};

inline SimplificationStrategy default_simplification_strategy() { return {}; }

/* Open boundary condition algorithms that simplify a state, optimizing over one site */

struct SimplificationOutput {
  double error;
  double norm;
  int sense;
};

SimplificationOutput simplify_obc(RMPS *P, const RMPS &Q,
                                  const SimplificationStrategy &strategy);

SimplificationOutput simplify_obc(CMPS *P, const CMPS &Q,
                                  const SimplificationStrategy &strategy);

SimplificationOutput simplify_obc(RMPS *P, const RTensor &weights,
                                  const vector<RMPS> &Q,
                                  const SimplificationStrategy &strategy);

SimplificationOutput simplify_obc(CMPS *P, const CTensor &weights,
                                  const vector<CMPS> &Q,
                                  const SimplificationStrategy &strategy);

/* Open boundary condition algorithms that simplify a state, optimizing over one site */

inline SimplificationStrategy make_simplification_strategy(int *sense,
                                                           index_t sweeps,
                                                           bool normalize,
                                                           index_t Dmax = 0,
                                                           double tol = -1) {
  SimplificationStrategy strategy;
  if (tol < 0) {
    strategy.use_default_truncation_tolerance();
  } else if (tol >= 1) {
    strategy.do_not_truncate();
  } else if (tol == 0) {
    strategy.truncate_only_zeros();
  } else {
    strategy.set_relative_truncation_tolerance(tol);
  }
  if (sense != nullptr) {
    strategy.set_direction(*sense);
  }
  if (Dmax == 0) {
    strategy.allow_any_dimension();
  } else {
    strategy.set_maximum_dimension(Dmax);
  }
  strategy.set_normalization(normalize);
  strategy.set_sweeps(sweeps);
  return strategy;
}

inline double simplify_obc(RMPS *P, const RMPS &Q, int *sense, index_t sweeps,
                           bool normalize, index_t Dmax = 0, double tol = -1,
                           double *norm = 0) {
  auto [error, old_norm, new_sense] = simplify_obc(
      P, Q, make_simplification_strategy(sense, sweeps, normalize, Dmax, tol));
  if (norm != nullptr) {
    *norm = old_norm;
  }
  if (sense != nullptr) {
    *sense = new_sense;
  }
  return error;
}

inline double simplify_obc(CMPS *P, const CMPS &Q, int *sense, index_t sweeps,
                           bool normalize, index_t Dmax = 0, double tol = -1,
                           double *norm = 0) {
  auto [error, old_norm, new_sense] = simplify_obc(
      P, Q, make_simplification_strategy(sense, sweeps, normalize, Dmax, tol));
  if (norm != nullptr) {
    *norm = old_norm;
  }
  if (sense != nullptr) {
    *sense = new_sense;
  }
  return error;
}

inline double simplify_obc(RMPS *P, const RTensor &weights,
                           const vector<RMPS> &Q, int *sense, index_t sweeps,
                           bool normalize, index_t Dmax = 0, double tol = -1,
                           double *norm = 0) {
  auto [error, old_norm, new_sense] = simplify_obc(
      P, weights, Q,
      make_simplification_strategy(sense, sweeps, normalize, Dmax, tol));
  if (norm != nullptr) {
    *norm = old_norm;
  }
  if (sense != nullptr) {
    *sense = new_sense;
  }
  return error;
}

inline double simplify_obc(CMPS *P, const CTensor &weights,
                           const vector<CMPS> &Q, int *sense, index_t sweeps,
                           bool normalize, index_t Dmax = 0, double tol = -1,
                           double *norm = 0) {
  auto [error, old_norm, new_sense] = simplify_obc(
      P, weights, Q,
      make_simplification_strategy(sense, sweeps, normalize, Dmax, tol));
  if (norm != nullptr) {
    *norm = old_norm;
  }
  if (sense != nullptr) {
    *sense = new_sense;
  }
  return error;
}

}  // namespace mps
