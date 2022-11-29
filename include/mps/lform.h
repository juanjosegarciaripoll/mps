// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
#pragma once
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

#ifndef MPS_LFORM_H
#define MPS_LFORM_H

#include <mps/vector.h>
#include <mps/mps.h>

namespace mps {

/** Internal representation of a MPS as a linear form.  This object
      works with a scalar product $$\sum_i c_i\langle\phi_i|\psi\rangle$$, where
      the bra and ked, $\phi_i$ and $\psi$, are two MPS. This scalar
      product, from the point of view of the k-th site, it may be seen
      as a linear form $w^{T} v$, where $w$ and $v$ are the k-th
      elements of MPS associated to $\sum_ic_i\psi$ and $\phi$[k].
  */
template <class Tensor>
class LinearForm {
 public:
  using mps_t = MPS<Tensor>;
  using tensor_t = Tensor;
  using number_t = tensor_scalar_t<Tensor>;

  LinearForm(const mps_t &bra, const mps_t &ket, index start = 0);
  LinearForm(const tensor_t &weights, const vector<mps_t> &bra,
             const mps_t &ket, index start = 0);

  /** Update the linear form, with a new value of the state it is applied on. */
  void propagate_right(const tensor_t &ketP);
  /** Update the linear form, with a new value of the state it is applied on. */
  void propagate_left(const tensor_t &ketP);
  /** Implement propagate_right (sense>0) or left (sense<0). */
  void propagate(const tensor_t &ketP, int sense);

  /** The site at which the quadratic form is defined. */
  index here() const { return current_site_; }
  /** Number of sites in the lattice. */
  index size() const { return bra_[0].ssize(); }
  /** Number of vectors that create the linear form. */
  index number_of_bras() const { return bra_.ssize(); }

  /** Vector representation of the linear form with respect to site here().*/
  const tensor_t single_site_vector() const;
  /** Vector representation of the linear form with respect to
	sites here() and here()+1.*/
  const tensor_t two_site_vector(int sense) const;

  /** Norm-2 of the linear form. */
  double norm2() const;

 private:
  using matrix_array_t = vector<tensor_t>;
  using matrix_database_t = vector<matrix_array_t>;

  const tensor_t weight_;
  const vector<mps_t> bra_;
  index size_, current_site_;
  matrix_database_t matrix_;

  tensor_t &left_matrix(index i, index site) { return matrix_[i][site]; }
  tensor_t &right_matrix(index i, index site) { return matrix_[i][site + 1]; }
  const tensor_t &left_matrix(index i, index site) const {
    return matrix_[i][site];
  }
  const tensor_t &right_matrix(index i, index site) const {
    return matrix_[i][site + 1];
  }

  void initialize_matrices(index start, const mps_t &ket);
  matrix_database_t make_matrix_array();
};

extern template class LinearForm<RTensor>;
using RLForm = LinearForm<RTensor>;

extern template class LinearForm<CTensor>;
using CLForm = LinearForm<CTensor>;

}  // namespace mps

#endif  // MPS_LFORM_H
