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

#ifndef MPO_MPO_H
#define MPO_MPO_H

#include <vector>
#include <mps/mps.h>
#include <mps/rmpo.h>
#include <mps/cmpo.h>

namespace mps {

  using namespace tensor;

  RMPO local_Hamiltonian_mpo(const std::vector<RTensor> &Hloc);

  void add_local_term(RMPO *mpdo, const RTensor &Hloc, index k);

  void add_interaction(RMPO *mpdo, const RTensor &Hi, index i, const RTensor &Hj);

  void add_interaction(RMPO *mpdo, const std::vector<RTensor> &Hi, index i, const RTensor *sign = NULL);

  void add_product_term(RMPO *mpdo, const std::vector<RTensor> &Hi);

  void add_hopping_matrix(RMPO *mpdo, const RTensor &J, const RTensor &ad, const RTensor &a);

  void add_jordan_wigner_matrix(RMPO *mpdo, const RTensor &J, const RTensor &ad, const RTensor &a, const RTensor &sign);

  CMPO local_Hamiltonian_mpo(const std::vector<CTensor> &Hloc);

  void add_local_term(CMPO *mpdo, const CTensor &Hloc, index i);

  void add_interaction(CMPO *mpdo, const CTensor &Hi, index i, const CTensor &Hj);

  void add_interaction(CMPO *mpdo, const std::vector<CTensor> &Hi, index i, const CTensor *sign = NULL);

  void add_product_term(CMPO *mpdo, const std::vector<CTensor> &Hi);

  void add_hopping_matrix(CMPO *mpdo, const CTensor &J, const CTensor &ad, const CTensor &a);

  void add_jordan_wigner_matrix(CMPO *mpdo, const CTensor &J, const CTensor &ad, const CTensor &a, const CTensor &sign);

  /** Apply a Matrix Product Operator onto a state. */
  const RMPS apply(const RMPO &mpdo, const RMPS &state);

  /** Apply a Matrix Product Operator onto a state. */
  const CMPS apply(const CMPO &mpdo, const CMPS &state);

  /** Apply a Matrix Product Operator onto a state, obtaining a canonical form. */
  const RMPS apply_canonical(const RMPO &mpdo, const RMPS &state, int sense = +1, bool truncate = true);

  /** Apply a Matrix Product Operator onto a state, obtaining a canonical form. */
  const CMPS apply_canonical(const CMPO &mpdo, const CMPS &state, int sense = -1, bool truncate = true);

  /** Expectation value of an MPO between two MPS. */
  double expected(const RMPS &bra, const RMPO &op, const RMPS &ket);

  /** Expectation value of an MPO between two MPS. */
  double expected(const RMPS &bra, const RMPO &op);

  /** Expectation value of an MPO between two MPS. */
  cdouble expected(const CMPS &bra, const CMPO &op, const CMPS &ket);

  /** Expectation value of an MPO between two MPS. */
  cdouble expected(const CMPS &bra, const CMPO &op);

  /** Adjoint of a Matrix Product Operator. */
  const CMPO adjoint(const CMPO &mpo);

  /** Adjoint (Transpose) of a Matrix Product Operator. */
  const RMPO adjoint(const RMPO &mpo);

  /** Combine two Matrix Product Operators, multiplying them A*B. */
  const CMPO mmult(const CMPO &A, const CMPO &B);

  /** Combine two Matrix Product Operators, multiplying them A*B. */
  const RMPO mmult(const RMPO &A, const RMPO &B);

  /** Return the matrix that represents the MPO acting on the full Hilbert
      space. Use with care with only small tensors, as this may exhaust the
      memory of your computer. */
  const RTensor mpo_to_matrix(const RMPO &A);

  /** Return the matrix that represents the MPO acting on the full Hilbert
      space. Use with care with only small tensors, as this may exhaust the
      memory of your computer. */
  const CTensor mpo_to_matrix(const CMPO &A);

  const RMPO simplify(const RMPO &old_mpo, int sense=+1, bool debug=false);

  const CMPO simplify(const CMPO &old_mpo, int sense=+1, bool debug=false);

  /** Compute a fidelity between two operators. Fidelity is defined as $\mathrm{tr}(Ua^\dagger Ub)/\sqrt{\mathrm{tr}(Ua^\dagger Ua)\mathrm{tr}(Ub^\dagger Ub)}\$ */
  double fidelity(const RMPO &Ua, const RMPO &Ub);

  /** Compute a fidelity between two operators. Fidelity is defined as $\mathrm{tr}(Ua^\dagger Ub)/\sqrt{\mathrm{tr}(Ua^\dagger Ua)\mathrm{tr}(Ub^\dagger Ub)}\$ */
  double fidelity(const CMPO &Ua, const CMPO &Ub);

  const RMPS mpo_to_mps(const RMPO &A);

  const CMPS mpo_to_mps(const CMPO &A);
}

#endif /* !MPO_MPO_H */
