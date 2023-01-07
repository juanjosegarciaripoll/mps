#pragma once
#include <limits>
#include <tensor/tensor.h>
#include <mps/flags.h>

namespace mps {

using namespace tensor;

class TruncationStrategy {
  double tolerance_{FLAGS.get(MPS_TRUNCATION_TOLERANCE)};
  index_t maximum_dimension_{std::numeric_limits<index_t>::max()};
  bool debug_truncation_{FLAGS.get(MPS_DEBUG_TRUNCATION) != 0};

  using truncation_function_t =
      index_t (TruncationStrategy::*)(const RTensor &schmidt_weights) const;
  truncation_function_t dispatch_{
      &TruncationStrategy::worker_truncate_up_to_relative_tolerance};

  index_t worker_truncate_nothing(const RTensor &schmidt_weights) const;
  index_t worker_truncate_only_zeros(const RTensor &schmidt_weights) const;
  index_t worker_truncate_up_to_relative_tolerance(
      const RTensor &schmidt_weights) const;

 public:
  bool debug_truncation() const { return debug_truncation_; }

  index_t where_to_truncate(const RTensor &schmidt_weights) const {
    return ((*this).*dispatch_)(schmidt_weights);
  }

  TruncationStrategy &set_truncation_debug_level(bool debug) {
    debug_truncation_ = debug;
    return *this;
  }

  TruncationStrategy &do_not_truncate() {
    dispatch_ = &TruncationStrategy::worker_truncate_nothing;
    maximum_dimension_ = std::numeric_limits<index_t>::max();
    tolerance_ = 0.0;
    return *this;
  }

  TruncationStrategy &truncate_only_zeros() {
    dispatch_ = &TruncationStrategy::worker_truncate_only_zeros;
    maximum_dimension_ = std::numeric_limits<index_t>::max();
    tolerance_ = 0.0;
    return *this;
  }

  TruncationStrategy &set_relative_truncation_tolerance(double tolerance) {
    tensor_assert(tolerance >= 0 && tolerance <= 1.0);
    tolerance_ = std::max(tolerance, std::numeric_limits<double>::epsilon());
    dispatch_ = &TruncationStrategy::worker_truncate_up_to_relative_tolerance;
    return *this;
  }

  TruncationStrategy &set_maximum_dimension(index_t chi) {
    if (chi == 0) {
      std::cerr << "Deprecated convention of using bond dimension zero to "
                   "represent any dimension.";
      return allow_any_dimension();
    }
    tensor_assert(chi >= 0);
    maximum_dimension_ = chi;
    return *this;
  }

  TruncationStrategy &allow_any_dimension() {
    return set_maximum_dimension(std::numeric_limits<index_t>::max());
  }

  TruncationStrategy &use_default_truncation_tolerance() {
    return set_relative_truncation_tolerance(FLAGS.get(MPS_TRUNCATION_TOLERANCE));
  }

  index_t maximum_dimension() const { return maximum_dimension_; }
  double truncation_relative_tolerance() const { return tolerance_; }
};

inline TruncationStrategy default_truncation_strategy() {
  return TruncationStrategy();
}

}  // namespace mps
