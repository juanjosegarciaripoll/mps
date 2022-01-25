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

template <class Tensor>
static bool is_identity(const Tensor &t) {
  return (t.rank() == 2) && (t.columns() == t.rows()) &&
         all_equal(t, Tensor::eye(t.rows()));
}

template <class MPO, class Tensor>
static void do_add_interaction(MPO &mpo, const std::vector<Tensor> &H) {
  //
  // This function adds a term \prod_j H[j] to a Hamiltonian.
  //
  index closing = 0, opening = 0;
  int start = 0, end = mpo.size();
  while (start < end && is_identity(H[start])) {
    ++start;
  }
  while (end > 0 && is_identity(H[end - 1])) {
    --end;
    closing = 1;
  }
  for (int j = start; j < end; ++j) {
    const Tensor &Hj = H[j];
    Tensor Pj = mpo[j];
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
      Pj.at(range(dl), range(), range(), range(dr)) =
          Hj + Pj(range(dl), range(), range(), range(dr));
    } else {
      Pj.at(range(dl), range(), range(), range(dr)) = Hj;
    }
    mpo.at(j) = Pj;
  }
}

}  // namespace mps
