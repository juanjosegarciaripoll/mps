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

namespace mps {

template <class MPS, class Tensor>
static void set_canonical_inner(MPS &psi, index_t ndx, const Tensor &t, int sense,
                                bool truncate) {
  if (sense > 0) {
    if (ndx + 1 == psi.ssize()) {
      psi.at(ndx) = t;
    } else {
      Tensor V = split(&psi.at(ndx), t, +1, truncate);
      psi.at(ndx + 1) = fold(V, -1, psi[ndx + 1], 0);
    }
  } else {
    if (ndx == 0) {
      psi.at(ndx) = t;
    } else {
      Tensor V = split(&psi.at(ndx), t, -1, truncate);
      psi.at(ndx - 1) = fold(psi[ndx - 1], -1, V, 0);
    }
  }
}

template <class MPS>
static MPS either_form_inner(MPS psi, index_t site, bool normalize) {
  index_t i;
  for (i = psi.last(); i > site; i--) set_canonical(psi, i, psi[i], -1);
  for (i = 0; i < site; i++) set_canonical(psi, i, psi[i], +1);
  if (normalize) psi.at(i) /= norm2(psi[i]);
  return psi;
}

}  // namespace mps
