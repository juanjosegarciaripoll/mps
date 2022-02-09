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
#include <mps/io.h>

namespace mps {

template <class MPO>
static std::ostream &do_dump_mpo(std::ostream &s, const MPO &mpo,
                                 const char *name) {
  for (index n = 0; n < mpo.ssize(); n++) {
    const typename MPO::elt_t P = mpo[n];
    index r = P.dimension(1);
    index c = P.dimension(2);
    for (index i = 0; i < P.dimension(0); i++) {
      for (index j = 0; j < P.dimension(3); j++) {
        s << name << '[' << n << "](" << i << ",:,:," << j << ")=\n"
          << matrix_form(reshape(P(range(i), _, _, range(j)).copy(), r, c))
          << std::endl;
      }
    }
  }
  return s;
}

}  // namespace mps
