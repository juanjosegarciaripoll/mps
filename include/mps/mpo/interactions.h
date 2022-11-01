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

#ifndef MPS_MPO_INTERACTIONS_H
#define MPS_MPO_INTERACTIONS_H

#include <stdexcept>
#include <vector>
#include <mps/mpo/types.h>

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

template <class Tensor>
inline void add_local_term(MPO<Tensor> *mpo, const Tensor &Hloc, index i) {
  if (i < 0 || i >= mpo->ssize()) {
    std::cerr << "In add_local_term(), the index " << i
              << " is outside the lattice.\n";
    abort();
  }
  if (Hloc.columns() != Hloc.rows()) {
    std::cerr << "In add_local_term(MPO, Tensor, index), the Tensor is not a "
                 "square matrix.\n";
    abort();
  }
  Tensor Pi = (*mpo)[i];
  index d = Pi.dimension(1);
  if (d != Hloc.rows()) {
    std::cerr << "In add_local_term(MPO, Tensor, index), the dimensions of the "
                 "tensor do not match those of the MPO.\n";
    abort();
  }
  if (Pi.dimension(0) == 1) {
    /* First */
    Pi.at(range(0), _, _, range(1)) =
        Tensor(Pi(range(0), _, _, range(1))) + reshape(Hloc, 1, d, d, 1);
  } else if (Pi.dimension(3) == 1) {
    /* Last */
    Pi.at(range(0), _, _, range(0)) =
        Tensor(Pi(range(0), _, _, range(0))) + reshape(Hloc, 1, d, d, 1);
  } else {
    /* Middle */
    Pi.at(range(0), _, _, range(1)) =
        Tensor(Pi(range(0), _, _, range(1))) + reshape(Hloc, 1, d, d, 1);
  }
  mpo->at(i) = Pi;
}

template <class Tensor>
inline void add_interaction(MPO<Tensor> *mpo, const Tensor &Hi, index i,
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
  index di = Hi.rows(), dj = Hj.rows();
  Tensor Pi = mpo->at(i);
  Tensor Pj = mpo->at(i + 1);
  index b = Pi.dimension(3) + 1;
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
MPO<Tensor> local_Hamiltonian_mpo(const std::vector<Tensor> &Hloc) {
  MPO<Tensor> output(Hloc.size(), 1);
  index i = 0, last = static_cast<index>(Hloc.size()) - 1;
  for (auto &H : Hloc) {
    index d = H.rows();
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
inline void add_product_term(MPO<Tensor> *mpo, const std::vector<Tensor> &H) {
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
  index closing = 0, opening = 0;
  index start = 0, end = mpo->size();
  while (start < end && is_identity(H[start])) {
    ++start;
  }
  while (end > 0 && is_identity(H[end - 1])) {
    --end;
    closing = 1;
  }
  for (index j = start; j < end; ++j) {
    const Tensor &Hj = H[j];
    Tensor Pj = (*mpo)[j];
    index dl = Pj.dimension(0);
    index dr = Pj.dimension(3);
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
void add_interaction(MPO<Tensor> *mpo, const std::vector<Tensor> &H, index i,
                     const Tensor *sign = nullptr) {
  //
  // This function add terms \sum_{j,j\neq i} H[i]*H[j] to a Hamiltonian.
  // The origin of interactions is thus marked by "i"
  //
  index start = 0, end = mpo->size();
  if (i < 0 || i >= end) {
    throw std::invalid_argument(
        "In add_interaction(), lattice site out of boundaries.");
  }
  if (norm2(H[i]) == 0) {
    return;
  }
  index last_closing = 0;
  while (start < end && norm2(H[start]) == 0) {
    ++start;
  }
  while (end && (sum(abs(H[end - 1])) == 0)) {
    --end;
    last_closing = 1;
  }
  for (index j = start; j < end; ++j) {
    const Tensor &Hj = H[j];
    Tensor Pj = mpo->at(j);
    index dl = Pj.dimension(0);
    index dr = Pj.dimension(3);
    index opening = 0, closing = 1;
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

  index L = mpo->size();
  for (index j = 0; j < L; ++j) {
    std::vector<Tensor> ops(L);
    for (index i = 0; i < L; ++i) {
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
  for (index i = 0; i < mpo->ssize(); i++) {
    auto Hi = tensor_cast(*mpo, H.local_term(i, t));
    add_local_term(mpo, Hi, i);
  }
  for (index i = 0; i < mpo->ssize(); i++) {
    for (index j = 0; j < H.interaction_depth(i, t); j++) {
      auto Hi = tensor_cast(*mpo, H.interaction_left(i, j, t));
      if (!Hi.is_empty()) {
        auto Hj = tensor_cast(*mpo, H.interaction_right(i, j, t));
        add_interaction(mpo, Hi, i, Hj);
      }
    }
  }
}

}  // namespace mps

#endif  // MPS_MPO_INTERACTIONS_H