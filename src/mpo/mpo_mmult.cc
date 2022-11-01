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

template <class MPO>
static const MPO do_mmult(const MPO &A, const MPO &B) {
  typedef typename MPO::elt_t Tensor;
  tensor_assert(A.size() == B.size());

  index L = A.size();
  MPO C = A;

  for (index n = 0; n < L; n++) {
    const Tensor &tA = A[n]; /* tA(a1,i,j,a2) */
    const Tensor &tB = B[n]; /* tB(c1,j,k,c2) */

    /* tC(a1,i,c1,k,c2,a2) = tB(c1,j,k,c2) tA(a1,i,j,a2) */
    Tensor tC = foldin(tB, 1, tA, 2);
    index a1, i, a2, c1, k, c2;
    tC.get_dimensions(&a1, &i, &c1, &k, &c2, &a2);
    tC = reshape(permute(permute(tC, 1, 2), 4, 5), a1 * c1, i, k, a2 * c2);

    C.at(n) = tC;
  }
  return C;
}

}  // namespace mps
