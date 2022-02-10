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

#ifndef MPS_MPS_TYPES_H
#define MPS_MPS_TYPES_H

#include <numeric>
#include <algorithm>
#include <mps/tools.h>
#include <mps/mp_base.h>

namespace mps {

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

  /**Return the physical dimensions of the state. */
  Indices dimensions() const {
    Indices d(this->size());
    std::transform(this->begin(), this->end(), std::begin(d),
                   [](const Tensor &t) { return t.dimension(1); });
    return d;
  }

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
    MPS<Tensor> output(physical_dimensions, bond_dimension, periodic);
    randomize(output);
    return output;
  }

  /**Create a product state with all sites sharing the same vector.*/
  static MPS<Tensor> product_state(index length, const Tensor &local_state) {
    if ((local_state.rank() != 1) || (local_state.size() == 0)) {
      throw std::invalid_argument(
          "Not a valid quantum state in MPS::product_state().");
    }
    MPS<Tensor> output(length);
    std::fill(output.begin(), output.end(),
              reshape(local_state, 1, local_state.size(), 1));
    return output;
  }

  /**Create a one-dimensional vector for this MPS's wavefunction.*/
  Tensor to_vector() const {
    if (this->size() == 0) {
      return Tensor::empty(0);
    } else {
      auto output = std::accumulate(
          this->begin() + 1, this->end(), (*this)[0],
          [](const Tensor &previous, const Tensor &tn) {
            /* previous(i, D, j) is a tensor that results from contracting (n-1)
             * sites. D is the total physical dimension of those sites. tn(j,
             * dn, k) is a tensor for the n-th site. The result is a tensor with
             * dimensions  (i, D*dn, k). */
            auto product = fold(previous, -1, tn, 0);
            return reshape(product, product.dimension(0),
                           product.dimension(1) * product.dimension(2),
                           product.dimension(3));
          });
      return trace(output, 0, -1);
    }
  }

 private:
  typedef MP<Tensor> parent;

  inline void presize(const tensor::Indices &physical_dimensions,
                      index bond_dimension, bool periodic) {
    assert(bond_dimension > 0);
    assert(this->size() == physical_dimensions.size());
    index l = physical_dimensions.size();
    tensor::Indices dimensions = {bond_dimension, index(0), bond_dimension};
    for (index i = 0; i < l; i++) {
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

template <typename T>
struct mp_tensor_t_inner<MPS<T>> {
  typedef T type;
};

}  // namespace mps

#endif  // MPS_MPS_TYPES_H