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

#ifndef MPS_MPO_INTERACTIONS_H
#define MPS_MPO_INTERACTIONS_H

#include <stdexcept>
#include <mps/vector.h>
#include <mps/mpo/types.h>
#include <mps/hamiltonian.h>

namespace mps {

/* This is how we encode MPO:

     - An operator is a collection of tensors A(a,i,j,b), where
       for fixed "a" and "b", the matrix A(a,i,j,b) is an operator
       acting on the state.

     - "a" = 0 means we are free to choose what to do.
     - "a" = 1 means we can only apply identities.
     - A(0,i,j,b>1) is the first operator from a nearest-neighbor
       interaction, which is paired by A(b,i,j,1)

   */

template <class MPO>
inline MPO initialize_interactions_mpo(const tensor::Indices &dimensions) {
  using tensor_t = typename MPO::elt_t;
  vector<tensor_t> tensors;
  tensors.reserve(dimensions.ssize());
  for (auto d : dimensions) {
    auto Id = tensor_t::eye(d, d);
    if (tensors.ssize() == 0) {
      /* first */
      auto P = tensor_t::zeros(1, d, d, 2);
      P.at(range(0), _, _, range(0)) = Id;
      tensors.emplace_back(P);
    } else if (tensors.ssize() + 1 < dimensions.ssize()) {
      /* last */
      auto P = tensor_t::zeros(2, d, d, 2);
      P.at(range(1), _, _, range(1)) = Id;
      P.at(range(0), _, _, range(0)) = Id;
      tensors.emplace_back(P);
    } else {
      /* otherwise */
      auto P = tensor_t::zeros(2, d, d, 1);
      P.at(range(1), _, _, range(0)) = Id;
      tensors.emplace_back(P);
    }
  }
  return MPO(tensors);
}

template <class Tensor>
inline void add_local_term(MPO<Tensor> *mpo, const Tensor &Hloc, index_t i) {
  if (i < 0 || i >= mpo->ssize()) {
    std::cerr << "In add_local_term(), the index_t " << i
              << " is outside the lattice.\n";
    abort();
  }
  if (Hloc.columns() != Hloc.rows()) {
    std::cerr << "In add_local_term(MPO, Tensor, index), the Tensor is not a "
                 "square matrix.\n";
    abort();
  }
  Tensor Pi = (*mpo)[i];
  index_t d = Pi.dimension(1);
  if (d != Hloc.rows()) {
    std::cerr << "In add_local_term(MPO, Tensor, index), the dimensions of the "
                 "tensor do not match those of the MPO.\n";
    abort();
  }
  bool unprepared_mpo = (Pi.dimension(3) == 1 && Pi.dimension(0) == 1);
  if (unprepared_mpo) {
    std::cerr << "In add_local_term(MPO, Tensor, index), the MPO has not been "
                 "prepared\n";
    abort();
  }
  if (i == 0) {
    Pi.at(range(0), _, _, range(1)) =
        Tensor(Pi(range(0), _, _, range(1))) + reshape(Hloc, 1, d, d, 1);
  } else if (i == mpo->last_index()) {
    Pi.at(range(0), _, _, range(0)) =
        Tensor(Pi(range(0), _, _, range(0))) + reshape(Hloc, 1, d, d, 1);
  } else {
    Pi.at(range(0), _, _, range(1)) =
        Tensor(Pi(range(0), _, _, range(1))) + reshape(Hloc, 1, d, d, 1);
  }
  mpo->at(i) = Pi;
}

template <class Tensor>
inline void add_interaction(MPO<Tensor> *mpo, const Tensor &Hi, index_t i,
                            const Tensor &Hj) {
  if (i < 0) {
    std::cerr << "In add_interaction(), the index " << i
              << " is outside the lattice.\n";
    abort();
  }
  if ((i + 1) >= mpo->ssize()) {
    std::cerr << "In add_interaction(), the index " << i
              << "+1 is outside the lattice.\n";
    abort();
  }
  if (Hi.rows() != Hi.columns()) {
    std::cerr << "In add_interaction(MPO, ...), second argument is not a "
                 "square matrix\n";
    abort();
  }
  if (Hj.rows() != Hj.columns()) {
    std::cerr << "In add_interaction(MPO, ...), fourth argument is not a "
                 "square matrix\n";
    abort();
  }
  index_t di = Hi.rows(), dj = Hj.rows();
  Tensor Pi = mpo->at(i);
  Tensor Pj = mpo->at(i + 1);
  index_t b = Pi.dimension(3) + 1;
  Pi = change_dimension(Pi, 3, b);
  Pj = change_dimension(Pj, 0, b);

  if (Pi.dimension(1) != di) {
    std::cerr << "In add_interaction(MPO, ...), the second argument has wrong "
                 "dimensions.\n";
    abort();
  }
  if (Pj.dimension(1) != dj) {
    std::cerr << "In add_interaction(MPO, ...), the second argument has wrong "
                 "dimensions.\n";
    abort();
  }

  Pi.at(range(0), _, _, range(b - 1)) = Hi;
  if (i + 2 == mpo->ssize()) {
    Pj.at(range(b - 1), _, _, range(0)) = Hj;
  } else {
    Pj.at(range(b - 1), _, _, range(1)) = Hj;
  }

  mpo->at(i) = Pi;
  mpo->at(i + 1) = Pj;
}

template <class Tensor>
MPO<Tensor> local_Hamiltonian_mpo(const vector<Tensor> &Hloc) {
  MPO<Tensor> output(Hloc.ssize(), 1);
  index_t i = 0, last = static_cast<index_t>(Hloc.size()) - 1;
  for (auto &H : Hloc) {
    index_t d = H.rows();
    Tensor aux = Tensor::zeros(2, d, d, 2);
    aux.at(range(0), _, _, range(1)) = H;

    Tensor id = Tensor::eye(d);
    aux.at(range(1), _, _, range(1)) = id;
    aux.at(range(0), _, _, range(0)) = id;

    if (i == 0) aux = aux(range(0), _, _, _);
    if (i == last) aux = aux(_, _, _, range(1));
    output.at(i) = aux;
    ++i;
  }
  return output;
}

template <class Tensor>
inline void add_product_term(MPO<Tensor> *mpo, const vector<Tensor> &H) {
  //
  // This function adds a term \prod_j H[j] to a Hamiltonian.
  //
  if (!std::all_of(mpo->begin(), mpo->end(), [](const Tensor &t) -> bool {
        return (t.rank() == 2) && (t.columns() == t.rows());
      })) {
    throw std::invalid_argument(
        "in add_interaction(), operator is not a square matrix.");
  }
  auto is_identity = [](const Tensor &t) -> bool {
    return all_of(t == Tensor::eye(t.rows()));
  };
  index_t closing = 0, opening = 0;
  index_t start = 0, end = mpo->ssize();
  while (start < end && is_identity(H[start])) {
    ++start;
  }
  while (end > 0 && is_identity(H[end - 1])) {
    --end;
    closing = 1;
  }
  for (index_t j = start; j < end; ++j) {
    const Tensor &Hj = H[j];
    Tensor Pj = (*mpo)[j];
    index_t dl = Pj.dimension(0);
    index_t dr = Pj.dimension(3);
    if (j > start) {
      Pj = change_dimension(Pj, 0, dl + 1);
    } else {
      dl = opening;
    }
    if (j + 1 < end) {
      Pj = change_dimension(Pj, 3, dr + 1);
    } else {
      dr = closing;
    }
    if (dl == opening && dr == closing) {
      Pj.at(range(dl), _, _, range(dr)) = Hj + Pj(range(dl), _, _, range(dr));
    } else {
      Pj.at(range(dl), _, _, range(dr)) = Hj;
    }
    mpo->at(j) = Pj;
  }
}
template <class Tensor>
void add_interaction(MPO<Tensor> *mpo, const vector<Tensor> &H, index_t i,
                     const Tensor *sign = nullptr) {
  //
  // This function add terms \sum_{j,j\neq i} H[i]*H[j] to a Hamiltonian.
  // The origin of interactions is thus marked by "i"
  //
  index_t start = 0, end = mpo->ssize();
  if (i < 0 || i >= end) {
    throw std::invalid_argument(
        "In add_interaction(), lattice site out of boundaries.");
  }
  if (norm2(H[i]) == 0) {
    return;
  }
  index_t last_closing = 0;
  while (start < end && norm2(H[start]) == 0) {
    ++start;
  }
  while (end && (sum(abs(H[end - 1])) == 0)) {
    --end;
    last_closing = 1;
  }
  for (index_t j = start; j < end; ++j) {
    const Tensor &Hj = H[j];
    Tensor Pj = mpo->at(j);
    index_t dl = Pj.dimension(0);
    index_t dr = Pj.dimension(3);
    index_t opening = 0, closing = 1;
    if (j + 1 < end) {
      Pj = change_dimension(Pj, 3, dr + 1);
      if (j <= i) {
        Pj.at(range(opening), _, _, range(dr)) = Hj;
      }
    } else {
      closing = last_closing;
    }
    if (j > start) {
      Pj = change_dimension(Pj, 0, dl + 1);
      if (i <= j) {
        Pj.at(range(dl), _, _, range(closing)) = Hj;
      }
      if (i != j && j + 1 < end) {
        Pj.at(range(dl), _, _, range(dr)) =
            sign ? *sign : Tensor::eye(Hj.rows());
      }
    }
    mpo->at(j) = Pj;
  }
}

template <class Tensor>
inline void add_hopping_matrix(MPO<Tensor> *mpo, const Tensor &J,
                               const Tensor &ad, const Tensor &a,
                               const Tensor &sign = Tensor()) {
  //
  // This function add terms
  // \sum_{j,j\neq i} J(i,j) a[i]*sign[i+1]*...*sign[j-1]*ad[j] to a Hamiltonian.
  //
  if ((J.rank() != 2) || (J.rows() != J.columns()) ||
      (J.rows() != ssize(*mpo))) {
    throw std::invalid_argument("Invalid matrix J in add_hopping_matrix().");
  }

  index_t L = mpo->ssize();
  for (index_t j = 0; j < L; ++j) {
    vector<Tensor> ops(L);
    for (index_t i = 0; i < L; ++i) {
      if (i == j)
        ops.at(j) = ad;
      else
        ops.at(i) = a * J(j, i);
    }
    add_interaction(mpo, ops, j, &sign);
  }
}

template <class Tensor>
inline void add_jordan_wigner_matrix(MPO<Tensor> *mpo, const Tensor &J,
                                     const Tensor &ad, const Tensor &a,
                                     const Tensor &sign) {
  add_hopping_matrix(mpo, J, ad, a, sign);
}

template <class Tensor>
inline void add_Hamiltonian(MPO<Tensor> *mpo, const Hamiltonian &H, double t) {
  for (index_t i = 0; i < mpo->ssize(); i++) {
    auto Hi = tensor_cast(*mpo, H.local_term(i, t));
    add_local_term(mpo, Hi, i);
  }
  for (index_t i = 0; i < mpo->ssize(); i++) {
    for (index_t j = 0; j < H.interaction_depth(i, t); j++) {
      auto Hi = tensor_cast(*mpo, H.interaction_left(i, j, t));
      if (!Hi.is_empty()) {
        auto Hj = tensor_cast(*mpo, H.interaction_right(i, j, t));
        add_interaction(mpo, Hi, i, Hj);
      }
    }
  }
}

template <class MPO>
inline MPO Hamiltonian_to(const Hamiltonian &H, double t = 0.0) {
  auto output = initialize_interactions_mpo<MPO>(H.dimensions());
  add_Hamiltonian(&output, H, t);
  return output;
}

}  // namespace mps

#endif  // MPS_MPO_INTERACTIONS_H
