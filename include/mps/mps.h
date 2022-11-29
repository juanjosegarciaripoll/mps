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

#ifndef MPS_MPS_H
#define MPS_MPS_H

#include <mps/vector.h>
#include <mps/mps/types.h>
#include <mps/algorithms/expectation.h>

namespace mps {

/**Physical dimensions of the state. */
template <typename Tensor>
inline Indices dimensions(const MPS<Tensor> &psi) {
  return psi.dimensions();
}

/**Create a product state. */
template <typename Tensor>
inline MPS<Tensor> product_state(index length, const Tensor &local_state) {
  return MPS<Tensor>::product_state(length, local_state);
}

/**Create a product state. */
template <typename Tensor>
inline MPS<Tensor> product_state(const vector<Tensor> &local_states) {
  MPS<Tensor> output(local_states.size());
  std::transform(std::begin(local_states), std::end(local_states),
                 std::begin(output),
                 [](const Tensor &t) { return reshape(t, 1, t.size(), 1); });
  return output;
}

/**Create a GHZ state.*/
const RMPS ghz_state(index length, bool periodic = false);

/**Create a cluster state.*/
const RMPS cluster_state(index length);

/** Apply a local operator on the given site. */
const RMPS apply_local_operator(const RMPS &psi, const RTensor &op, index site);

/** Apply a local operator on the given site. */
const CMPS apply_local_operator(const CMPS &psi, const CTensor &op, index site);

/**Convert an MPS to a complex vector, contracting all tensors.*/
template <typename Tensor>
inline Tensor mps_to_vector(const MPS<Tensor> &mps) {
  return mps.to_vector();
}

/**Norm of a RMPS.*/
extern template double norm2(const RMPS &psi);

/**Norm of a CMPS.*/
extern template double norm2(const CMPS &psi);

/**Scalar product between MPS.*/
extern template double scprod(const RMPS &psi1, const RMPS &psi2,
                              int direction);

/**Scalar product between MPS.*/
extern template cdouble scprod(const CMPS &psi1, const CMPS &psi2,
                               int direction);

/**Compute a single-site expected value.*/
extern template double expected(const RMPS &a, const RTensor &Op1, index k,
                                int direction);

/**Compute a single-site expected value.*/
extern template double expected(const RMPS &a, const RTensor &Op1, index k,
                                int direction);

/**Compute a single-site expected value.*/
extern template cdouble expected(const CMPS &a, const CTensor &Op1, index k,
                                 int direction);

/**Compute all expected values of a single operator over the chain.*/
extern template RTensor expected_vector(const RMPS &a, const RTensor &Op1);

/**Compute all expected values, with a different operator over each site of the chain.*/
extern template RTensor expected_vector(const RMPS &a,
                                        const vector<RTensor> &Op1);

/**Compute all expected values of a single operator over the chain.*/
extern template CTensor expected_vector(const CMPS &a, const CTensor &Op1);

/**Compute all expected values, with a different operator over each site of the chain.*/
extern template CTensor expected_vector(const CMPS &a,
                                        const vector<CTensor> &Op1);

/**Compute a two-site correlation.*/
extern template double expected(const RMPS &a, const RTensor &op1, index k1,
                                const RTensor &op2, index k2, int direction);

/**Compute a two-site correlation.*/
extern template cdouble expected(const RMPS &a, const CTensor &op1, index k1,
                                 const CTensor &op2, index k2, int direction);

/**Compute a two-site correlation.*/
extern template cdouble expected(const CMPS &a, const CTensor &op1, index k1,
                                 const CTensor &op2, index k2, int direction);

extern template RTensor all_correlations_fast(const RMPS &a,
                                              const vector<RTensor> &op1,
                                              const vector<RTensor> &op2,
                                              const RMPS &b, bool symmetric,
                                              const RTensor *jordan_wigner_op);

extern template CTensor all_correlations_fast(const CMPS &a,
                                              const vector<CTensor> &op1,
                                              const vector<CTensor> &op2,
                                              const CMPS &b, bool symmetric,
                                              const CTensor *jordan_wigner_op);

/**Compute all two-site correlations.*/
extern template RTensor expected(const RMPS &a, const RTensor &op1,
                                 const RTensor &op2);

/**Compute all two-site correlations.*/
extern template CTensor expected(const CMPS &a, const CTensor &op1,
                                 const CTensor &op2);

/**Compute all two-site correlations.*/
extern template RTensor expected(const RMPS &a, const vector<RTensor> &op1,
                                 const vector<RTensor> &op2);

/**Compute all two-site correlations.*/
extern template CTensor expected(const CMPS &a, const vector<CTensor> &op1,
                                 const vector<CTensor> &op2);

/**Store a tensor in a matrix product state in the canonical form.*/
void set_canonical(RMPS &psi, index site, const RTensor &A, int sense,
                   bool truncate = true);

/**Store a tensor in a matrix product state in the canonical form.*/
void set_canonical(CMPS &psi, index site, const CTensor &A, int sense,
                   bool truncate = true);

/**Rewrite a RMPS in canonical form. A value of 'sense' -1 or +1 determines
     the right-to-left or left-to-right direction in this form. */
RMPS canonical_form(const RMPS &psi, int sense = -1);

/**Rewrite a CMPS in canonical form. A value of 'sense' -1 or +1 determines
     the right-to-left or left-to-right direction in this form.*/
CMPS canonical_form(const CMPS &psi, int sense = -1);

/**Rewrite a RMPS in canonical form, normalizing. A value of 'sense' -1 or +1 determines
     the right-to-left or left-to-right direction in this form.*/
RMPS normal_form(const RMPS &psi, int sense = -1);

/**Rewrite a CMPS in canonical form, normalizing. A value of 'sense' -1 or +1 determines
     the right-to-left or left-to-right direction in this form.*/
CMPS normal_form(const CMPS &psi, int sense = -1);

/**Rewrite a RMPS in canonical form on both sides of 'site'. */
RMPS canonical_form_at(const RMPS &psi, index site);

/**Rewrite a CMPS in canonical form on both sides of 'site'. */
CMPS canonical_form_at(const CMPS &psi, index site);

/**Rewrite a RMPS in canonical form, normalizing. */
RMPS normal_form_at(const RMPS &psi, index site);

/**Rewrite a CMPS in canonical form, normalizing. */
CMPS normal_form_at(const CMPS &psi, index site);

/** Update an MPS with a tensor that spans two sites, (site,site+1). Dmax is
   * the maximum bond dimension that is used. Actually, tol and Dmax are the
   * arguments to where_to_truncate. */
void set_canonical_2_sites(RMPS &P, const RTensor &Pij, index site, int sense,
                           index Dmax = 0, double tol = -1,
                           bool canonicalize_both = true);

/** Update an MPS with a tensor that spans two sites, (site,site+1). Dmax is
   * the maximum bond dimension that is used. Actually, tol and Dmax are the
   * arguments to where_to_truncate. */
void set_canonical_2_sites(CMPS &P, const CTensor &Pij, index site, int sense,
                           index Dmax = 0, double tol = -1,
                           bool canonicalize_both = true);

/* Return a single-site density matrix out of an MPS. */
const RTensor density_matrix(const RMPS &psi, index site);

/* Return a single-site density matrix out of an MPS. */
const CTensor density_matrix(const CMPS &psi, index site);

}  // namespace mps

#endif /* !TENSOR_MPS_H */
