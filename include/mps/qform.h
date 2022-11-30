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

#ifndef MPS_QFORM_H
#define MPS_QFORM_H

#include <mps/vector.h>
#include <mps/mpo.h>
#include <mps/hamiltonian.h>
#include <mps/algorithms/mpo_environments.h>

namespace mps {

/** Internal representation of a MPO as a quadratic form.
      This object works with a scalar product $$\langle\psi|O|\phi\rangle$$,
      where $O$ is a MPO, and the bra and ked, $\psi$ and $\phi$, are two
      MPS. This scalar product, from the point of view of the k-th site, it
      may be seen as a quadratic form $w^{T} \bar{O} v$, where $w$ and $v$
      are the elements of the tensors $\psi$[k] and $\phi$[k] of the MPS
      on that given site.
  */
template <class Tensor>
class QuadraticForm {
 public:
  typedef MPO<Tensor> mpo_t;
  typedef MPS<Tensor> mps_t;
  typedef Tensor tensor_t;

  /** Initialize with the given MPO and bra and ket states. This function
	assumes that we are inspecting site 'start', which may be at the
	beginning or the end of the chain.*/
  QuadraticForm(const mpo_t &mpo, const mps_t &bra, const mps_t &ket,
                index_t start = 0)
      : current_site_{0}, mpo_(mpo), envs_(mpo.ssize() + 1, env_t(DIR_LEFT)) {
    initialize_environments(bra, ket, start);
  }
  /** Update the form changing the tensors of the bra and ket states. The
	function updates the $\psi$ and $\phi$ states in $\langle\psi|O|\phi\rangle$
	changing the tensor of those states at the site here(), and moving to
	the next site, here()+/-1 depending on the direction.*/
  void propagate(const tensor_t &braP, const tensor_t &ketP, Dir sense) {
    if (sense == DIR_RIGHT) {
      propagate_right(braP, ketP);
    } else if (sense == DIR_LEFT) {
      propagate_left(braP, ketP);
    } else {
      tensor_terminate(std::invalid_argument(
          "QuadraticForm(), received an invalid direction"));
    }
  }
  /** The site at which the quadratic form is defined. */
  index_t here() const { return current_site_; }
  /** Number of sites in the lattice. */
  index_t ssize() const { return mpo_.ssize(); }
  /** Number of sites in the lattice. */
  size_t size() const { return mpo_.size(); }
  /** Last site in the lattice. */
  index_t last_site() const { return ssize() - 1; }

  /** Matrix representation of the quadratic form with respect to site here().*/
  tensor_t single_site_matrix() const {
    index_t n = here();
    return compose(left_environment(n), mpo_[n], right_environment(n));
  }

  /** Matrix representation of the quadratic form with respect to
     * sites here() and here()+1.*/
  tensor_t two_site_matrix(mps::Dir sense) const {
    index_t i, j;
    get_index_pair(sense, i, j);
    return compose(left_environment(i), mpo_[i], mpo_[j], right_environment(j));
  }

  /** Apply the two_site_matrix() onto a tensor representing one site. */
  linalg::LinearMap<Tensor> single_site_map() const {
    index_t n = here();
    return single_site_linear_map(left_environment(n), mpo_[n],
                                  right_environment(n));
  }

  /** Apply the two_site_matrix() onto a tensor representing two sites. */
  linalg::LinearMap<Tensor> two_site_map(mps::Dir direction) const {
    index_t i, j;
    get_index_pair(direction, i, j);
    return two_site_linear_map(left_environment(i), mpo_[i], mpo_[j],
                               right_environment(j));
  }

  Tensor apply_single_site_matrix(const Tensor &P) const {
    return single_site_map()(P);
  }

  Tensor apply_two_site_matrix(const Tensor &P, mps::Dir direction) const {
    return two_site_map(direction)(P);
  }

  Tensor take_two_site_matrix_diag(mps::Dir direction) const {
    Tensor output;
    index_t i, j;
    get_index_pair(direction, i, j);
    const auto &Lenv = left_environment(i);
    const auto &Renv = right_environment(j);
    for (const auto &op1 : mpo_[i]) {
      for (const auto &op2 : mpo_[j]) {
        if (op1.right_index == op2.left_index) {
          // L(a1,b1,a2,b2)
          const auto &L = Lenv[op1.left_index].tensor();
          // R(a3,b3,a1,b1)
          const auto &R = Renv[op2.right_index].tensor();
          if (!L.is_empty() && !R.is_empty()) {
            index_t a2 = L.dimension(2);
            index_t b2 = L.dimension(3);
            index_t a3 = R.dimension(0);
            index_t b3 = R.dimension(1);
            // We implement this
            // Q12(a2,i,j,a3) = L(a1,a1,a2,a2) O1(i,i) O2(j,j) R(a3,a3,a1,a1)
            // where a1 = 1, because of periodic boundary conditions
            Tensor Q12 = kron2(
                kron2(take_diag(reshape(L, a2, b2)), take_diag(op1.matrix)),
                kron2(take_diag(op2.matrix), take_diag(reshape(R, a3, b3))));
            output = maybe_add(output, Q12);
          }
        }
      }
    }
    return output;
  }

 private:
  typedef MPOEnvironment<Tensor> env_t;
  typedef vector<MPOEnvironment<Tensor>> env_list_t;
  typedef SparseMPO<Tensor> sparse_mpo_t;

  index_t current_site_;
  sparse_mpo_t mpo_;
  env_list_t envs_;

  env_t &left_environment(index_t site) { return envs_[site]; }
  env_t &right_environment(index_t site) { return envs_[site + 1]; }

  const env_t &left_environment(index_t site) const { return envs_[site]; }
  const env_t &right_environment(index_t site) const { return envs_[site + 1]; }

  void get_index_pair(Dir sense, index_t &i, index_t &j) const {
    index_t n = here();
    if (sense == DIR_RIGHT) {
      tensor_assert2(
          n < last_site(),
          std::out_of_range(
              "Out of range index in QuadraticForm::two_site_matrix"));
      i = n;
      j = n + 1;
    } else if (sense == DIR_LEFT) {
      tensor_assert2(
          n > 0, std::out_of_range(
                     "Out of range index in QuadraticForm::two_site_matrix"));
      i = n - 1;
      j = n;
    } else {
      tensor_terminate(
          std::invalid_argument("Not a valid direction (Dir type)"));
    }
  }

  void initialize_environments(const mps_t &bra, const mps_t &ket,
                               index_t start) {
    tensor_assert2(bra.size() == size() && ket.size() == size(),
                   std::invalid_argument("Wrong sizes of MPS in QForm"));
    // Prepare the right matrices from site start to size()-1 to 0
    current_site_ = last_site();
    right_environment(current_site_) = env_t(DIR_LEFT);
    while (here() > start) {
      propagate_left(bra[here()], ket[here()]);
    }
    current_site_ = 0;
    left_environment(current_site_) = env_t(DIR_RIGHT);
    while (here() != start) {
      propagate_right(bra[here()], ket[here()]);
    }
  }

  void propagate_left(const tensor_t &bra, const tensor_t &ket) {
    if (here() > 0) {
      right_environment(here() - 1) =
          right_environment(here()).propagate(bra, ket, mpo_[here()]);
      --current_site_;
    }
  }

  void propagate_right(const tensor_t &bra, const tensor_t &ket) {
    if (here() < last_site()) {
      left_environment(here() + 1) =
          left_environment(here()).propagate(bra, ket, mpo_[here()]);
      ++current_site_;
    }
  }
};

extern template class QuadraticForm<RMPO>;
typedef QuadraticForm<RMPO> RQForm;

extern template class QuadraticForm<CMPO>;
typedef QuadraticForm<CMPO> CQForm;

}  // namespace mps

#endif /* !MPS_DMRG_H */
