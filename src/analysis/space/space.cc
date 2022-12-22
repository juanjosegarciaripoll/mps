// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2010 Juan Jose Garcia Ripoll

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

#include <mps/analysis.h>

namespace mps {

Space::Space(domain_t domain)
    : domain_(std::move(domain)), mps_dimensions_(make_mps_dimensions()) {}

index_t Space::dimension_size(index_t axis) const {
  tensor_assert(axis >= 0 && axis < dimensions());
  return domain_[axis].ssize();
}

Indices Space::tensor_dimensions() const {
  Indices output(dimensions(), 0);
  std::transform(domain_.begin(), domain_.end(), output.begin(),
                 [](const interval_t &i) { return i.ssize(); });
  return output;
}

index_t Space::first_qubit(index_t axis) const {
  assert_valid_axis(axis);
  return std::accumulate(
      domain_.begin(), domain_.begin() + axis, 0,
      [](index_t output, const interval_t &i) { return output + i.qubits; });
}

Indices Space::dimension_qubits(index_t axis) const {
  auto start = first_qubit(axis);
  auto this_interval = domain_[axis];
  Indices output(this_interval.qubits);
  for (index_t i = 0; i < this_interval.qubits; ++i) {
    output.at(i) = start + i;
  }
  return output;
}

RSparse Space::extend_matrix(const RSparse &op, index_t axis) const {
  RSparse output = op;
  index_t left_dimension = 1, right_dimension = 1;
  for (index_t n = 0; n < dimensions(); ++n) {
    auto d = dimension_size(n);
    if (n < axis) {
      left_dimension *= d;
    } else if (n > axis) {
      right_dimension *= d;
    }
  }
  return kron2(RSparse::eye(left_dimension),
               kron2(op, RSparse::eye(right_dimension)));
}

index_t Space::total_qubits() const {
  return std::accumulate(
      domain_.begin(), domain_.end(), 0,
      [](index_t x, const interval_t &i) { return x + i.qubits; });
}

index_t Space::total_dimension() const { return index_t(1) << total_qubits(); }

RMPO Space::identity_mpo() const { return RMPO(make_mps_dimensions()); }

RMPO Space::extend_mpo(const RMPO &op, index_t axis) const {
  auto output = identity_mpo();
  auto qubits = dimension_qubits(axis);
  tensor_assert(qubits.ssize() == op.ssize());
  auto it = op.begin();
  for (auto q : dimension_qubits(axis)) {
    output.at(q) = *it;
    ++it;
  }
  return output;
}

}  // namespace mps
