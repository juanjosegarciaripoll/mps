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

#include <algorithm>
#include <tensor/linalg.h>
#include <mps/mps.h>
#include <mps/tools.h>

namespace mps {

template <class Tensor>
static const Tensor do_split(Tensor *pU, Tensor psi, int sense, bool truncate) {
  Tensor &U = *pU, V;
  Indices d = psi.dimensions();
  if (sense > 0) {
    index r = psi.rank();
    index b = d[r - 1];
    index ai = psi.size() / b;
    RTensor s =
        mps::FLAGS.get(MPS_USE_BLOCK_SVD)
            ? linalg::block_svd(reshape(psi, ai, b), &U, &V, SVD_ECONOMIC)
            : linalg::svd(reshape(psi, ai, b), &U, &V, SVD_ECONOMIC);
    index l = s.size();
    index new_l = where_to_truncate(
        s, truncate ? MPS_DEFAULT_TOLERANCE : MPS_TRUNCATE_ZEROS,
        std::min<index>(ai, b));
    if (new_l != l) {
      U = change_dimension(U, 1, new_l);
      V = change_dimension(V, 0, new_l);
      s = change_dimension(s, 0, new_l);
      l = new_l;
    }
    d.at(r - 1) = l;
    scale_inplace(V, 0, s);
  } else {
    index a = d[0];
    index ib = psi.size() / a;
    RTensor s =
        mps::FLAGS.get(MPS_USE_BLOCK_SVD)
            ? linalg::block_svd(reshape(psi, a, ib), &V, &U, SVD_ECONOMIC)
            : linalg::svd(reshape(psi, a, ib), &V, &U, SVD_ECONOMIC);
    index l = s.size();
    index new_l = where_to_truncate(
        s, truncate ? MPS_DEFAULT_TOLERANCE : MPS_TRUNCATE_ZEROS,
        std::min<index>(a, ib));
    if (new_l != l) {
      U = change_dimension(U, 0, new_l);
      V = change_dimension(V, 1, new_l);
      s = change_dimension(s, 0, new_l);
      l = new_l;
    }
    d.at(0) = l;
    scale_inplace(V, -1, s);
  }
  U = reshape(U, d);
  return V;
}

}  // namespace mps
