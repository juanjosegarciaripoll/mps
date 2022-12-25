// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2012 Juan Jose Garcia Ripoll

    Tensor is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public License as published
    by the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Library General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include <float.h>
#include <memory>
#include <mps/tools.h>
#include <mps/flags.h>
#include <tensor/io.h>
#include <mps/algorithms/truncation.h>

namespace mps {

Indices weights_to_keep(const RTensor &s, double tol, index_t max_dim) {
  const Indices ndx = sort_indices(abs(s), true /* reversed */);
  auto sorted_s = s(ndx);
  auto stop = where_to_truncate(sorted_s, tol, max_dim);
  if (stop == ndx.ssize()) return ndx;
  Indices output(stop);
  std::copy(ndx.begin(), ndx.begin() + stop, output.begin());
  return output;
}

index_t where_to_truncate(const RTensor &s, double tol, index_t max_dim) {
  /* S is a vector of positive numbers arranged in decreasing order.  This
     * routine looks for a point to truncate S such that the norm-2 error made
     * is smaller than the relative tolerance (TOL) or the length of the output
     * is smaller than MAX_DIM.
     */
  const index L = s.ssize();
  bool debug = FLAGS.get(MPS_DEBUG_TRUNCATION);

  if (max_dim == 0 || max_dim > L) {
    max_dim = L;
  }
  if (tol < 0 /* MPS_DEFAULT_TOLERANCE is negative */) {
    tol = FLAGS.get(MPS_TRUNCATION_TOLERANCE);
  }
  if (tol >= 1.0 /* MPS_DO_NOT_TRUNCATE */) {
    if (debug) {
      std::cerr << "Truncation disabled" << '\n';
    }
    return max_dim;
  }
  if (tol == 0 /* MPS_TRUNCATE_ZEROS */) {
    /* If the tolerance is zero, we only drop the trailing zero elements. There
       * is no need to accumulate values. */
    for (index_t i = L; i--;) {
      if (s[i] != 0) {
        if (debug) {
          std::cerr << "Truncated only zeros, new size " << i << " vs " << L
                    << '\n';
        }
        return std::min(i + 1, max_dim);
      }
    }
    if (debug) {
      std::cerr << "Not truncated vector of size " << L << '\n';
    }
    return 0;
  }
  /*
     * cumulated[i] contains the norm of the elements _beyond_ the i-th
     * site. This means that if we keep (i+1) leading elements, the error will
     * be exactly cumulated[i].
     */
  auto cumulated = std::make_unique<double[]>(static_cast<size_t>(L));
  double total = 0;
  for (index_t i = L; i--;) {
    total += square(s[i]);
    cumulated[static_cast<size_t>(i)] = total;
  }
  /* Due to the precision limits in current processors, we automatically
     * relax the tolerance to DBL_EPSILON, which is a floating point number
     * such that added to 1.0 gives 1.0. In other words, a tolerance <=
     * DBL_EPSILON is irrelevant for all purposes.
     */
  double limit = std::max(tol, DBL_EPSILON) * total;
  for (index_t i = 0; i < max_dim; i++) {
    if (cumulated[static_cast<size_t>(i)] <= limit) {
      max_dim = i;
      break;
    }
  }

  if (debug) {
    std::cerr << "Truncated to tolerance " << limit << ", new size " << max_dim
              << " vs " << L << '\n';
  }
  return max_dim;
}

index_t TruncationStrategy::worker_truncate_nothing(
    const RTensor &schmidt_weights) const {
  if (debug_truncation()) {
    std::cerr << "Truncation disabled" << '\n';
  }
  return schmidt_weights.ssize();
}

index_t TruncationStrategy::worker_truncate_only_zeros(
    const RTensor &schmidt_weights) const {
  auto L = schmidt_weights.ssize();
  for (index_t i = 0; i < L; ++i) {
    if (schmidt_weights[i] == 0.0) {
      if (debug_truncation()) {
        std::cerr << "Truncated only zeros, new size " << i << " vs " << L
                  << '\n';
      }
      return std::min<index_t>(i, 1);
    }
  }
  if (debug_truncation()) {
    std::cerr << "Not truncated vector of size " << L << '\n';
  }
  return L;
}

index_t TruncationStrategy::worker_truncate_up_to_relative_tolerance(
    const RTensor &schmidt_weights) const {
  auto L = schmidt_weights.ssize();
  static mps::vector<double> cumulated;
  cumulated.resize(L, 0.0);
  /*
   * cumulated[i] contains the norm of the elements _beyond_ the i-th
   * site. This means that if we keep (i+1) leading elements, the error will
   * be exactly cumulated[i].
   */
  double total = 0;
  for (index_t i = L; i--;) {
    total += square(schmidt_weights[i]);
    cumulated[static_cast<size_t>(i)] = total;
  }
  /* Due to the precision limits in current processors, we automatically
   * relax the tolerance to DBL_EPSILON, which is a floating point number
   * such that added to 1.0 gives 1.0. In other words, a tolerance <=
   * DBL_EPSILON is irrelevant for all purposes.
   */
  double limit = tolerance_ * total;
  auto max_dim = std::min(maximum_dimension_, L);
  for (index_t i = 0; i < max_dim; i++) {
    if (cumulated[static_cast<size_t>(i)] <= limit) {
      max_dim = i;
      break;
    }
  }

  if (debug_truncation()) {
    std::cerr << "Truncated to tolerance " << limit << ", new size " << max_dim
              << " vs " << L << '\n';
  }
  return max_dim;
}

}  // namespace mps
