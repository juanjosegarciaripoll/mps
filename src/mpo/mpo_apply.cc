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

template <class MPS, class MPO>
static const MPS do_apply(const MPO &mpdo, const MPS &psi) {
  typedef typename MPS::elt_t Tensor;
  tensor_assert(mpdo.size() == psi.size());

  index a1, c1, j, c2, a2;
  index L = mpdo.size();
  MPS chi(psi.size());

  for (index i = 0; i < L; i++) {
    const Tensor &A = psi[i];  /* A(a1,i,a2) */
    const Tensor &O = mpdo[i]; /* O(c1,j,i,c2) */

    /* B(a1,c1,j,c2,a2) = O(c1,j,i,c2) A(a1,i,a2) */
    Tensor B = foldin(O, 2, A, 1);
    B.get_dimensions(&a1, &c1, &j, &c2, &a2);

    chi.at(i) = reshape(permute(B, 0, 1), c1 * a1, j, a2 * c2);
  }
  return chi;
}

}  // namespace mps
