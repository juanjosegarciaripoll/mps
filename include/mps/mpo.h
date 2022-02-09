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
#include <mps/hamiltonian.h>

namespace mps {

using namespace tensor;

template <typename dest, typename orig>
inline dest safe_tensor_coercion(orig t) {
  return dest(t);
}

template <typename dest>
inline dest safe_tensor_coercion(dest) {
  return t;
}

template <>
inline RTensor safe_tensor_coercion<RTensor, CTensor>(CTensor data) {
  if (std::any_of(std::begin(data), std::end(data),
                  [](const cdouble &z) { return z.imag() != 0; })) {
    throw std::domain_error("Cannot convert complex tensor to real.");
  }
  return real(data);
}

extern template RTensor safe_tensor_coercion<RTensor, CTensor>(CTensor);

/*!\addtogroup TheMPS*/
/* @{ */

/**Matrix Product Operator structure.*/
template <typename Tensor>
class MPO : public MP<Tensor> {
 public:
  typedef MPS<Tensor> MPS;
  MPO() = default;
  MPO(const MPO &) = default;
  MPO(MPO &&) = default;
  MPO &operator=(const MPO &) = default;
  MPO &operator=(MPO &&) = default;

  MPO(index length, index physical_dimension) : parent(length) {
    tensor::Indices dims(length);
    std::fill(dims.begin(), dims.end(), physical_dimension);
    clear(dims);
  }

  MPO(const tensor::Indices &physical_dimensions)
      : parent(physical_dimensions.size()) {
    clear(physical_dimensions);
  }

  MPO(const Hamiltonian &H, double t = 0.0) : parent(H.size()) {
    clear(H.dimensions());
    add_Hamiltonian(*this, H, t);
  }

 private:
  typedef MP<Tensor> parent;

  void clear(const tensor::Indices &physical_dimensions) {
    if (physical_dimensions.size() < 2) {
      std::cerr << "Cannot create MPO with size 0 or 1.\n";
      abort();
    }
    // TODO: Simplify. We only need sizes (1,d,d,1) for the add_local/add_interaction to succeed.
    Tensor P;
    for (index i = 0; i < this->ssize(); i++) {
      index d = physical_dimensions[i];
      Tensor Id = reshape(Tensor::eye(d, d), 1, d, d, 1);
      if (i == 0) {
        /* first */
        P = Tensor::zeros(1, d, d, 2);
        P.at(range(0), _, _, range(0)) = Id;
      } else if (i + 1 < this->ssize()) {
        /* last */
        P = Tensor::zeros(2, d, d, 2);
        P.at(range(1), _, _, range(1)) = Id;
        P.at(range(0), _, _, range(0)) = Id;
      } else {
        /* otherwise */
        P = Tensor::zeros(2, d, d, 1);
        P.at(range(1), _, _, range(0)) = Id;
      }
      this->at(i) = P;
    }
  }
};

extern template class MPO<RTensor>;
extern template class MPO<CTensor>;
#ifdef DOXYGEN_ONLY
/**Real matrix product structure.*/
struct RMPO : public MPS<RTensor> {};
/**Complex matrix product structure.*/
struct CMPO : public MPS<CTensor> {};
#else
typedef MPO<RTensor> RMPO;
typedef MPO<CTensor> CMPO;
#endif

/* @} */

RMPO local_Hamiltonian_mpo(const std::vector<RTensor> &Hloc);

void add_local_term(RMPO *mpdo, const RTensor &Hloc, index k);

void add_interaction(RMPO *mpdo, const RTensor &Hi, index i, const RTensor &Hj);

void add_interaction(RMPO *mpdo, const std::vector<RTensor> &Hi, index i,
                     const RTensor *sign = nullptr);

void add_product_term(RMPO *mpdo, const std::vector<RTensor> &Hi);

void add_hopping_matrix(RMPO *mpdo, const RTensor &J, const RTensor &ad,
                        const RTensor &a);

void add_jordan_wigner_matrix(RMPO *mpdo, const RTensor &J, const RTensor &ad,
                              const RTensor &a, const RTensor &sign);

CMPO local_Hamiltonian_mpo(const std::vector<CTensor> &Hloc);

void add_local_term(CMPO *mpdo, const CTensor &Hloc, index i);

void add_interaction(CMPO *mpdo, const CTensor &Hi, index i, const CTensor &Hj);

void add_interaction(CMPO *mpdo, const std::vector<CTensor> &Hi, index i,
                     const CTensor *sign = nullptr);

void add_product_term(CMPO *mpdo, const std::vector<CTensor> &Hi);

void add_hopping_matrix(CMPO *mpdo, const CTensor &J, const CTensor &ad,
                        const CTensor &a);

void add_jordan_wigner_matrix(CMPO *mpdo, const CTensor &J, const CTensor &ad,
                              const CTensor &a, const CTensor &sign);

template <typename Tensor>
MPO<Tensor> &add_Hamiltonian(MPO<Tensor> &mpo, const Hamiltonian &H, double t) {
  for (index i = 0; i < mpo.ssize(); i++) {
    auto Hi = safe_tensor_coercion<Tensor, CTensor>(H.local_term(i, t));
    add_local_term(&mpo, Hi, i);
  }
  for (index i = 0; i < mpo.ssize(); i++) {
    for (index j = 0; j < H.interaction_depth(i, t); j++) {
      auto Hi =
          safe_tensor_coercion<Tensor, CTensor>(H.interaction_left(i, j, t));
      if (!Hi.is_empty()) {
        auto Hj =
            safe_tensor_coercion<Tensor, CTensor>(H.interaction_right(i, j, t));
        add_interaction(&mpo, Hi, i, Hj);
      }
    }
  }
  return mpo;
}

/** Apply a Matrix Product Operator onto a state. */
const RMPS apply(const RMPO &mpdo, const RMPS &state);

/** Apply a Matrix Product Operator onto a state. */
const CMPS apply(const CMPO &mpdo, const CMPS &state);

/** Apply a Matrix Product Operator onto a state, obtaining a canonical form. */
const RMPS apply_canonical(const RMPO &mpdo, const RMPS &state, int sense = +1,
                           bool truncate = true);

/** Apply a Matrix Product Operator onto a state, obtaining a canonical form. */
const CMPS apply_canonical(const CMPO &mpdo, const CMPS &state, int sense = -1,
                           bool truncate = true);

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

const RMPO simplify(const RMPO &old_mpo, int sense = +1, bool debug = false);

const CMPO simplify(const CMPO &old_mpo, int sense = +1, bool debug = false);

/** Compute a fidelity between two operators. Fidelity is defined as $\mathrm{tr}(Ua^\dagger Ub)/\sqrt{\mathrm{tr}(Ua^\dagger Ua)\mathrm{tr}(Ub^\dagger Ub)}\$ */
double fidelity(const RMPO &Ua, const RMPO &Ub);

/** Compute a fidelity between two operators. Fidelity is defined as $\mathrm{tr}(Ua^\dagger Ub)/\sqrt{\mathrm{tr}(Ua^\dagger Ua)\mathrm{tr}(Ub^\dagger Ub)}\$ */
double fidelity(const CMPO &Ua, const CMPO &Ub);

const RMPS mpo_to_mps(const RMPO &A);

const CMPS mpo_to_mps(const CMPO &A);
}  // namespace mps

#endif /* !MPO_MPO_H */
