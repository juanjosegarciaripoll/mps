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

#include <mps/algorithms.h>

namespace mps {

template <typename mps, typename t>
static inline t do_density_matrix(const mps &psi, index_t site) {
  t ML, MR;
  tensor_assert(site < psi.ssize());
  for (index_t i = 0; i < site; i++) ML = prop_matrix(ML, +1, psi[i], psi[i]);
  for (index_t i = psi.ssize() - 1; i > site; i--)
    MR = prop_matrix(MR, -1, psi[i], psi[i]);
  /* Dimensions:
     *	ML(a1,b1,a2,b2)
     *	MR(a3,b3,a1,b1)
     *  A(a2,d,a3)
     *  A*(b2,d,b3)
     */
  index_t a1, b1, a2, b2, a3, b3;
  ML.get_dimensions(&a1, &b1, &a2, &b2);
  MR.get_dimensions(&a3, &b3, &a1, &b1);
  t A = psi[site];
  index_t d = A.dimension(1);

  /* E(a3,b3,a2,b2) <- MR(a3,b3,[a1b1]) ML([a1b1],a2,b2) */
  t E =
      foldin(reshape(MR, a3, b3, a1 * b1), -1, reshape(ML, a1 * b1, a2, b2), 0);
  /* E(b2,b3,a2,a3) <- E(a3,b3,a2,b2) */
  E = permute(E, 0, 3);
  /* A(d,[a2a3]) <- A(a2,d,a3) */
  A = reshape(permute(A, 0, 1), d, a2 * a3);
  /* E(d,[a2a3]) <- A*(d,[b2b3]) E([b2b3],[a2a3]) */
  E = foldc(A, -1, reshape(E, b2 * b3, a2 * a3), 0);
  /* E(d,d') <- E(d,[a2a3]) A(d',[a2a3]) */
  E = fold(E, -1, A, -1);
  return transpose(E);
}

}  // namespace mps
