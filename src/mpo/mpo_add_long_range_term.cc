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

#include <mps/mpo.h>
#include <tensor/io.h>

namespace mps {

/* This is how we encode MPO:

     - An operator is a collection of tensors A(a,i,j,b), where
       for fixed "a" and "b", the matrix A(a,i,j,b) is an operator
       acting on the state.

     - "a" = 0 means we are free to choose what to do.
     - "a" = 1 means we can only apply identities.
     - A(0,i,j,b>1) is the first operator from a two-body
       interaction, which is paired by some A(b,i,j,1)
    
   */

template <class MPO, class Tensor>
static void do_add_interaction(MPO &mpo, const std::vector<Tensor> &H, index i,
                               const Tensor *sign) {
  //
  // This function add terms \sum_{j,j\neq i} H[i]*H[j] to a Hamiltonian.
  // The origin of interactions is thus marked by "i"
  //
  index start = 0, end = mpo.size();
  if (i < 0 || i >= end) {
    std::cerr << "In add_interaction(), the index " << i
              << " is outside the lattice.\n";
    abort();
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
    Tensor Pj = mpo[j];
    index dl = Pj.dimension(0);
    index dr = Pj.dimension(3);
    index opening = 0, closing = 1;
#if 1
    if (j + 1 < end) {
      Pj = change_dimension(Pj, 3, dr + 1);
      if (j <= i) {
        Pj.at(range(opening), range(), range(), range(dr)) = Hj;
      }
    } else {
      closing = last_closing;
    }
    if (j > start) {
      Pj = change_dimension(Pj, 0, dl + 1);
      if (i <= j) {
        Pj.at(range(dl), range(), range(), range(closing)) = Hj;
      }
      if (i != j && j + 1 < end) {
        Pj.at(range(dl), range(), range(), range(dr)) =
            sign ? *sign : Tensor::eye(Hj.rows());
      }
    }
#else
    if (j > start) {
      Pj = change_dimension(Pj, 0, dl + 1);
    }
    if (j + 1 < end) {
      Pj = change_dimension(Pj, 3, dr + 1);
    } else {
      closing = last_closing;
    }
    if (j == i) {
      // We are on the origin of the interaction
      // We close all terms that came from the left
      if (j > start) {
        Pj.at(range(dl), range(), range(), range(closing)) = Hj;
      }
      // We open all terms that did not have anything on the left
      if (j + 1 < end) {
        Pj.at(range(opening), range(), range(), range(dr)) = Hj;
      }
    } else if (j < i) {
      // We open new terms
      Pj.at(range(opening), range(), range(), range(dr)) = Hj;
      if (j > start) {
        Pj.at(range(dl), range(), range(), range(dr)) =
            sign ? *sign : Tensor::eye(Hj.rows());
      }
    } else {
      // We close terms opened by the qubit
      Pj.at(range(dl), range(), range(), range(closing)) = Hj;
      if (j + 1 < end) {
        Pj.at(range(dl), range(), range(), range(dr)) =
            sign ? *sign : Tensor::eye(Hj.rows());
      }
    }
#endif
    mpo.at(j) = Pj;
  }
}

}  // namespace mps
