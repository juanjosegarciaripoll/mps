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

#include <tensor/io.h>
#include <tensor/tools.h>
#include <mps/lattice.h>

namespace mps {

template <class Tensor>
const Tensor apply_lattice(const Tensor &psi, const Lattice &L, const Tensor &J,
                           const Tensor &U, Lattice::particle_kind_t kind) {
  Tensor output = Tensor::zeros(psi.dimensions());
  {
    RTensor values;
    Indices ndx;
    for (int i = 0; i < J.rows(); i++) {
      for (int j = 0; j < J.rows(); j++) {
        if (abs(J(i, j))) {
          L.hopping_inner(&values, &ndx, i, j, kind);
          output += J(i, j) * values * psi(range(sort_indices(ndx)));
        }
        if (j >= i && abs(U(i, j))) {
          values = L.interaction_inner(i, j);
          output += (U(i, j) + U(j, i)) * values * psi;
        }
      }
    }
  }
  return output;
}

}  // namespace mps
