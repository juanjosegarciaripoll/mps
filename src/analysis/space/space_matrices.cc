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

static RSparse interval_position_matrix(const Space::interval_t &interval) {
  RTensor coordinates = linspace(interval.start, interval.end - interval.step(),
                                 interval.ssize());
  return RSparse::diag(coordinates, 0);
}

RSparse position_matrix(const Space &space, index_t axis) {
  return space.extend_matrix(interval_position_matrix(space.interval(axis)),
                             axis);
}

static RSparse finite_difference_matrix(double a, double b, double c,
                                        index_t qubits) {
  std::vector<SparseTriplet<double>> triplets;
  index_t L = 1 << qubits;
  triplets.reserve(static_cast<size_t>(L * 3));
  for (index_t i = 0; i < L; ++i) {
    triplets.emplace_back(i, (i + 1) % L, c);
    triplets.emplace_back(i, (i + L - 1) % L, b);
    triplets.emplace_back(i, i, a);
  }
  return RSparse(triplets, L, L);
}

static RSparse interval_first_derivative_matrix(
    const Space::interval_t &interval) {
  auto dx = interval.step();
  return finite_difference_matrix(0, (-1.0) / dx, (+1.0) / dx, interval.qubits);
}

static RSparse interval_second_derivative_matrix(
    const Space::interval_t &interval) {
  auto dx = interval.step();
  return finite_difference_matrix(-2.0 / dx, 1.0 / dx, 1.0 / dx,
                                  interval.qubits);
}

RSparse first_derivative_matrix(const Space &space, index_t axis) {
  return space.extend_matrix(
      interval_first_derivative_matrix(space.interval(axis)), axis);
}

RSparse second_derivative_matrix(const Space &space, index_t axis) {
  return space.extend_matrix(
      interval_second_derivative_matrix(space.interval(axis)), axis);
}

}  // namespace mps
