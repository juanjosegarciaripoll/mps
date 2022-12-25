#pragma once
#include <limits>
#include <tensor/tensor.h>
#include <mps/mps.h>

namespace mps {

using namespace tensor;

class TruncationStrategy {
  double tolerance_{FLAGS.get(MPS_TRUNCATION_TOLERANCE)};
  index_t maximum_dimension_{std::numeric_limits<index_t>::max()};
  const bool debug_{FLAGS.get(MPS_DEBUG_TRUNCATION) != 0};

  using truncation_function_t =
      index_t (TruncationStrategy::*)(const RTensor &schmidt_weights) const;
  truncation_function_t dispatch_{
      &TruncationStrategy::worker_truncate_up_to_relative_tolerance};

  index_t worker_truncate_nothing(const RTensor &schmidt_weights) const;
  index_t worker_truncate_only_zeros(const RTensor &schmidt_weights) const;
  index_t worker_truncate_up_to_relative_tolerance(
      const RTensor &schmidt_weights) const;

 public:
  bool debug_truncation() const { return debug_; }
  index_t where_to_truncate(const RTensor &schmidt_weights) const {
    return ((*this).*dispatch_)(schmidt_weights);
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

  TruncationStrategy &set_relative_tolerance(double tolerance) {
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
    return set_relative_tolerance(tolerance_);
  }

  TruncationStrategy &allow_any_dimension() {
    maximum_dimension_ = std::numeric_limits<index_t>::max();
    return set_relative_tolerance(tolerance_);
  }

  TruncationStrategy &use_default_tolerance() {
    return set_relative_tolerance(FLAGS.get(MPS_TRUNCATION_TOLERANCE));
  }
};

inline TruncationStrategy default_truncation_strategy() {
  return TruncationStrategy();
}

}  // namespace mps