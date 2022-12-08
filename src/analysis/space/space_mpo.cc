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

#include <mps/quantum.h>
#include <mps/analysis.h>
#include <mps/io.h>

namespace mps {

static RMPO interval_position_mpo(const Space::interval_t &interval) {
  Indices dimensions(interval.qubits, 2);
  auto output = initialize_interactions_mpo<RMPO>(dimensions);

  RTensor s = diag(RTensor{0, 1});

  auto L = (interval.end - interval.start);
  for (index_t i = 0; i < interval.qubits; ++i) {
    RTensor d = s * (L / (2 << i));
    if (i == 0) {
      d += Pauli_id * interval.start;
    }
    add_local_term(&output, d, (interval.qubits - i - 1));
  }
  return output;
}

RMPO position_mpo(const Space &space, index_t axis) {
  return space.extend_mpo(interval_position_mpo(space.interval(axis)), axis);
}

static RMPO finite_difference_mpo(double a, double b, double c,
                                  index_t qubits) {
  RTensor A(RTensor::zeros({3, 2, 2, 3}));
  A.at(0, 0, 0, 0) = 1.0;  // Identity
  A.at(0, 1, 1, 0) = 1.0;
  A.at(1, 1, 0, 0) = 1.0;  // Increase
  A.at(1, 0, 1, 1) = 1.0;
  A.at(2, 1, 0, 2) = 1.0;  // Decrease
  A.at(2, 0, 1, 0) = 1.0;

  vector<RTensor> tensors(qubits, A);
  tensors.at(0) = reshape(fold(RTensor{a, b, c}, -1, A, 0), 1, 2, 2, 3);
  tensors.at(qubits - 1) = reshape(sum(A, -1), 3, 2, 2, 1);

  return RMPO(tensors);
}

static RMPO interval_first_derivative_mpo(const Space::interval_t &interval) {
  auto dx = interval.step();
  return finite_difference_mpo(0, (-1) / dx, (+1) / dx, interval.qubits);
}

static RMPO interval_second_derivative_mpo(const Space::interval_t &interval) {
  auto dx2 = square(interval.step());
  return finite_difference_mpo(-2 / dx2, 1 / dx2, 1 / dx2, interval.qubits);
}

RMPO first_derivative_mpo(const Space &space, index_t axis) {
  return space.extend_mpo(interval_first_derivative_mpo(space.interval(axis)),
                          axis);
}

RMPO second_derivative_mpo(const Space &space, index_t axis) {
  return space.extend_mpo(interval_second_derivative_mpo(space.interval(axis)),
                          axis);
}

}  // namespace mps
