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
#include <mps/mpo.h>

namespace mps {

template <class MPO, class MPS, class Tensor>
MPO do_simplify(const MPO &old_mpo, int sense, bool debug) {
  MPS psi = canonical_form(mpo_to_mps(old_mpo), sense);
  MPO mpo = old_mpo;

  for (int i = 0; i < mpo.ssize(); i++) {
    const Tensor &m1 = mpo[i];
    const Tensor &m2 = psi[i];
    mpo.at(i) = reshape(m2, m2.dimension(0), m1.dimension(1), m1.dimension(2),
                        m2.dimension(2));
    if (debug) {
      std::cerr << old_mpo[i].dimensions() << " -> " << mpo[i].dimensions()
                << '\n';
    }
  }
  return mpo;
}

}  // namespace mps
