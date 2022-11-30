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

namespace mps {

using namespace tensor;

template <class MPO, class Tensor>
static const Tensor to_matrix(const MPO &A) {
  Tensor B = Tensor::ones(1, 1, 1);
  index_t D = 1;

  for (int i = 0; i < A.ssize(); ++i) {
    Tensor Ai = A[i];
    index_t d = Ai.dimension(1);
    index_t b = Ai.dimension(3);
    /* B(D,D',a)*A(a,d,d',b) -> B(D,D',d,d',b) */
    B = fold(B, 2, Ai, 0);
    /* B(D,D',a)*A(a,d,d',b) -> B(D,d,D',d',,b) */
    B = reshape(permute(B, 1, 2), D * d, D * d, b);
    D = D * d;
  }
  return reshape(B, D, D);
}

}  // namespace mps
