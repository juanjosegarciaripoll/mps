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
static void do_apply_rightwards(MPS &chi, const MPO &mpdo, const MPS &psi,
                                bool truncate) {
  typedef typename MPS::elt_t Tensor;
  index b = 1, a1, c1, di, dj, c2, a2;
  Tensor L = Tensor::ones(1, 1, 1); /* L(b,c1,a1) */

  for (index ndx = 0; ndx < psi.ssize(); ++ndx) {
    const Tensor &A = psi[ndx]; /* A(a1,i,a2) */
    A.get_dimensions(&a1, &di, &a2);

    const Tensor &O = mpdo[ndx]; /* O(c1,j,i,c2) */
    O.get_dimensions(&c1, &dj, &di, &c2);

    /* C(b,c1,di,a2) = L(b,c1,a1) * A(a1,i,a2) */
    Tensor C = fold(L, -1, A, 0);

    /* C(b,dj,c2,a2) = O(c1,j,i,c2) C(b,c1,i,a2) */
    C = foldin(reshape(permute(O, 1, 2), c1 * di, dj, c2), 0,
               reshape(C, b, c1 * di, a2), 1);
    C = reshape(C, b, dj, c2 * a2);

    if (ndx + 1 == psi.ssize()) {
      chi.at(ndx) = C;
    } else {
      L = split(&chi.at(ndx), C, +1, truncate);
      L = reshape(L, b = L.dimension(0), c2, a2);
    }
  }
}

template <class MPS, class MPO>
static void do_apply_leftwards(MPS &chi, const MPO &mpdo, const MPS &psi,
                               bool truncate) {
  typedef typename MPS::elt_t Tensor;
  index b = 1, a1, c1, di, dj, c2, a2;
  Tensor L = Tensor::ones(1, 1, 1); /* L(a2,c2,b) */

  for (index ndx = psi.ssize(); ndx--;) {
    const Tensor &A = psi[ndx]; /* A(a1,i,a2) */
    A.get_dimensions(&a1, &di, &a2);

    const Tensor &O = mpdo[ndx]; /* O(c1,j,i,c2) */
    O.get_dimensions(&c1, &dj, &di, &c2);

    /* C(a1,i,c2,b) = A(a1,i,a2) * L(a2,c2,b)*/
    Tensor C = fold(A, 2, L, 0);

    /* C(a1,c1,j,b) = O(c1,j,i,c2) C(a1,i,c2,b) */
    C = foldin(reshape(O, c1, dj, di * c2), 2, reshape(C, a1, di * c2, b), 1);
    C = reshape(C, a1 * c1, dj, b);

    if (ndx == 0) {
      chi.at(ndx) = C;
    } else {
      L = split(&chi.at(ndx), C, -1, truncate);
      L = reshape(L, a1, c1, b = L.dimension(1));
    }
  }
}

template <class MPS, class MPO>
static const MPS do_apply(const MPO &mpdo, const MPS &psi, int sense,
                          bool truncate) {
  tensor_assert(mpdo.size() == psi.size());

  MPS chi(psi.ssize());
  if (sense > 0) {
    do_apply_rightwards(chi, mpdo, psi, truncate);
  } else {
    do_apply_leftwards(chi, mpdo, psi, truncate);
  }
  return chi;
}

}  // namespace mps
