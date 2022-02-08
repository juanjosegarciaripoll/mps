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

#ifndef MPS_MPS_H
#define MPS_MPS_H

#include <algorithm>
#include <mps/tools.h>
#include <mps/mp_base.h>

namespace mps {

using namespace tensor;

/**Generic basi for matrix product state.*/
template <typename Tensor>
class MPS : public MP<Tensor> {
 public:
  MPS() = default;
  MPS(const MPS &) = default;
  MPS(MPS &&) = default;
  MPS &operator=(const MPS &) = default;
  MPS &operator=(MPS &&) = default;

  template <typename otherT>
  explicit MPS(const MPS<otherT> &mps) : parent(mps.size()) {
    std::transform(std::begin(mps), std::end(mps), this->begin(),
                   [](const otherT &t) { return Tensor(t); });
  }

  MPS(index size, index physical_dimension = 0, index bond_dimension = 1,
      bool periodic = false)
      : parent(size) {
    if (physical_dimension) {
      auto d = tensor::Indices::empty(size);
      std::fill(d.begin(), d.end(), physical_dimension);
      presize(d, bond_dimension, periodic);
    }
  }

  MPS(const tensor::Indices &physical_dimensions, index bond_dimension = 1,
      bool periodic = false)
      : parent(physical_dimensions.size()) {
    presize(physical_dimensions, bond_dimension, periodic);
  }

  explicit MPS(const std::vector<Tensor> &data) : parent(data){};

  /**Can the RMP be used for a periodic boundary condition problem?*/
  bool is_periodic() const {
    if (this->size()) {
      index d0 = (*this)[0].dimension(0);
      index dl = (*this)[-1].dimension(2);
      if (d0 == dl && d0 > 1) return true;
    }
    return false;
  }

  /**Create a random MPS. */
  static MPS<Tensor> random(index length, index physical_dimension,
                            index bond_dimension, bool periodic = false) {
    MPS<Tensor> output(length, physical_dimension, bond_dimension, periodic);
    randomize(output);
    return output;
  }

  /**Create a random MPS. */
  static MPS random(const tensor::Indices &physical_dimensions,
                    index bond_dimension, bool periodic = false) {
    MPS<Tensor> output(physical_dimensions.size(), bond_dimension, periodic);
    randomize(output);
    return output;
  }

 private:
  typedef MP<Tensor> parent;

  inline void presize(const tensor::Indices &physical_dimensions,
                      tensor::index bond_dimension, bool periodic) {
    assert(bond_dimension > 0);
    assert(this->size() == physical_dimensions.size());
    tensor::index l = physical_dimensions.size();
    tensor::Indices dimensions = {bond_dimension, tensor::index(0),
                                  bond_dimension};
    for (tensor::index i = 0; i < l; i++) {
      assert(physical_dimensions[i] > 0);
      dimensions.at(1) = physical_dimensions[i];
      dimensions.at(0) = (periodic || (i > 0)) ? bond_dimension : 1;
      dimensions.at(2) = (periodic || (i < (l - 1))) ? bond_dimension : 1;
      this->at(i) = Tensor::zeros(dimensions);
    }
  }

  static void randomize(MPS<Tensor> &mps) {
    for (auto &t : mps) {
      t.randomize();
    }
  }
};

extern template class MPS<RTensor>;
extern template class MPS<CTensor>;
#ifdef DOXYGEN_ONLY
/**Real matrix product structure.*/
struct RMPS : public MPS<RTensor> {};
/**Complex matrix product structure.*/
struct CMPS : public MPS<CTensor> {};
#else
typedef MPS<RTensor> RMPS;
typedef MPS<CTensor> CMPS;
#endif

/*!\defgroup TheMPS Matrix product states

   A Matrix Product State (MPS) represents a 1D quantum state made of N sites,
   particles or spins. Each site has a tensor associated to it, so that the
   whole state may be written as a contraction of these tensors
   \f[
   |\psi\rangle = \sum_{i,\alpha,\beta} A^{i_1}_{\alpha_1\alpha_2}
   A^{i_2}_{\alpha_2\alpha_3}\cdot A^{i_N}_{\alpha_N,\alpha_1}
   |i_1,i_2,\ldots,i_N\rangle
   \f]
   Each of the tensors in this product may be different and have different sizes.
   The indices \f$i_k\f$ denote the physical state of the k-th site. The indices
   \f$\alpha_k\f$ have no direct physical interpretation. However, the larger the
   these indices can be, the more accurately we can approximate arbitrary states
   using the previous representation. Finally, in problems with open boundary
   conditions, the index \f$\alpha_1\f$ need only have size 1.

   From the point of view of the programmer, a MPS is just a collection of
   tensors. The user is allowed to put and retrieve the tensor associated to
   the k-th site using get(k) or set(k,A), where A is the new tensor. Each tensor
   typically has three indices and is organized as A(a,i,b), where \c a and \c b
   are the \f$\alpha\f$ and \f$\beta\f$ from the previous formula, and \c i
   is the phyisical degree of freedom.

   Other operations, such as orthogonalize() or orthonormalize() are related
   to the \ref Algorithms algorithms for evolution and computation of ground
   states.

   Finally, since the MPS are actually vectors, one can compute the norm(),
   a scalar product with scprod(), expected values with correlation(), or
   obtain a vector that represents the same state with to_basis().
*/

/**Physical dimensions of the state. */
const Indices dimensions(const RMPS &psi);

/**Physical dimensions of the state. */
const Indices dimensions(const CMPS &psi);

/**Create a product state. */
const RMPS product_state(index length, const tensor::RTensor &local_state);

/**Create a product state. */
const CMPS product_state(index length, const tensor::CTensor &local_state);

/**Create a GHZ state.*/
const RMPS ghz_state(index length, bool periodic = false);

/**Create a cluster state.*/
const RMPS cluster_state(index length);

/** Apply a local operator on the given site. */
const RMPS apply_local_operator(const RMPS &psi, const RTensor &op, index site);

/** Apply a local operator on the given site. */
const CMPS apply_local_operator(const CMPS &psi, const CTensor &op, index site);

/**Convert a RMPS to a complex vector, contracting all tensors.*/
const RTensor mps_to_vector(const RMPS &mps);

/**Convert a CMPS to a complex vector, contracting all tensors.*/
const CTensor mps_to_vector(const CMPS &mps);

/**Norm of a RMPS.*/
double norm2(const RMPS &psi);

/**Norm of a CMPS.*/
double norm2(const CMPS &psi);

/**Scalar product between MPS.*/
double scprod(const RMPS &psi1, const RMPS &psi2, int sense = +1);

/**Scalar product between MPS.*/
cdouble scprod(const CMPS &psi1, const CMPS &psi2, int sense = +1);

/**Compute a single-site expected value.*/
double expected(const RMPS &a, const RTensor &Op1, index k);

/**Compute all expected values of a single operator over the chain.*/
RTensor expected_vector(const RMPS &a, const RTensor &Op1);

/**Compute all expected values, with a different operator over each site of the chain.*/
RTensor expected_vector(const RMPS &a, const std::vector<RTensor> &Op1);

/**Compute a single-site expected value.*/
cdouble expected(const RMPS &a, const CTensor &Op1, index k);

/**Compute all expected values of a single operator over the chain.*/
CTensor expected_vector(const CMPS &a, const CTensor &Op1);

/**Compute all expected values, with a different operator over each site of the chain.*/
CTensor expected_vector(const CMPS &a, const std::vector<CTensor> &Op1);

/**Compute a single-site expected value.*/
cdouble expected(const CMPS &a, const CTensor &Op1, index k);

/**Compute a two-site correlation.*/
double expected(const RMPS &a, const RTensor &op1, index k1, const RTensor &op2,
                index k2);

/**Compute a two-site correlation.*/
cdouble expected(const RMPS &a, const CTensor &op1, index k1,
                 const CTensor &op2, index k2);

/**Compute a two-site correlation.*/
cdouble expected(const CMPS &a, const CTensor &op1, index k1,
                 const CTensor &op2, index k2);

/**Compute all two-site correlations.*/
RTensor expected(const RMPS &a, const RTensor &op1, const RTensor &op2);

/**Compute all two-site correlations.*/
CTensor expected(const CMPS &a, const CTensor &op1, const CTensor &op2);

/**Compute all two-site correlations.*/
RTensor expected(const RMPS &a, const std::vector<RTensor> &op1,
                 const std::vector<RTensor> &op2);

/**Compute all two-site correlations.*/
CTensor expected(const CMPS &a, const std::vector<CTensor> &op1,
                 const std::vector<CTensor> &op2);

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
