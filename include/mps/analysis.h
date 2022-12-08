#pragma once
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

#ifndef MPS_ANALYSIS_H
#define MPS_ANALYSIS_H

#include <tensor/sparse.h>
#include <mps/vector.h>
#include <mps/mps.h>
#include <mps/mpo.h>

namespace mps {

/** Class defining an affine space for numerical analysis.*/
class Space {
 public:
  struct interval_t {
    double start{0}, end{1};
    index_t qubits{1};

    index_t ssize() const { return 1 << qubits; }
    double step() const { return (end - start) / static_cast<double>(ssize()); }
  };

  using domain_t = vector<interval_t>;

  Space(domain_t domain);
  Space() = delete;
  Space(const Space &) = default;
  Space(Space &&) = default;
  ~Space() = default;
  Space &operator=(const Space &) = default;
  Space &operator=(Space &&) = default;

  /** Number of coordinates in this Space. */
  index_t dimensions() const { return domain_.ssize(); }

  /** Properties of given interval. */
  const interval_t &interval(index_t axis) const {
    assert_valid_axis(axis);
    return domain_[axis];
  }

  /** Start of given dimension. */
  double dimension_start(index_t axis) const { return interval(axis).start; }

  /** End of given dimension. */
  double dimension_end(index_t axis) const { return interval(axis).end; }

  /** Lattice step of given dimension. */
  double dimension_step(index_t axis) const { return interval(axis).step(); }

  /** Dimension of the selected coordinate. */
  index_t dimension_size(index_t axis) const;

  /** Qubit positions associated to this dimension. */
  Indices dimension_qubits(index_t axis) const;

  /** Total number of qubits to represent this space. */
  index_t total_qubits() const;

  /** Dimensions of an MPS that represents a function in this space. */
  const Indices &mps_dimensions() const { return mps_dimensions_; }

  RSparse extend_matrix(const RSparse &op, index_t axis) const;
  RMPO extend_mpo(const RMPO &op, index_t axis) const;
  RMPO identity_mpo() const;

 private:
  domain_t domain_;
  Indices mps_dimensions_;

  Indices make_mps_dimensions() const { return Indices(total_qubits(), 2); }

  void assert_valid_axis(index_t axis) const {
    tensor_assert((axis >= 0) && (axis < dimensions()));
  }

  index_t first_qubit(index_t axis) const;
};

/** Finite difference derivative operator as sparse matrix. */
RSparse first_derivative_matrix(const Space &space, index_t axis = 0);

/** Finite difference second order derivative operator as sparse matrix. */
RSparse second_derivative_matrix(const Space &space, index_t axis);

/** Coordinate operator as sparse matrix. */
RSparse position_matrix(const Space &space, index_t axis = 0);

/** Finite difference derivative operator as MPO. */
RMPO first_derivative_mpo(const Space &space, index_t axis = 0);

/** Finite difference second order derivative operator as MPO. */
RMPO second_derivative_mpo(const Space &space, index_t axis = 0);

/** Coordinate operator as MPO. */
RMPO position_mpo(const Space &space, index_t axis = 0);

}  // namespace mps

#endif  // MPS_ANALYSIS_H
