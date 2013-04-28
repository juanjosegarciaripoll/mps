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

  RMPO::RMPO() :
    parent()
  {
  }

  RMPO::RMPO(index length, index physical_dimension) :
    parent(length)
  {
    tensor::Indices dims(length);
    std::fill(dims.begin(), dims.end(), physical_dimension);
    *this = RMPO(dims);
  }

  RMPO::RMPO(const tensor::Indices &physical_dimensions) :
    parent(physical_dimensions.size())
  {
    for (index i = 0; i < size(); i++) {
      index d = physical_dimensions[i];
      if (i == 0) { /* first */
        at(i) = elt_t::zeros(igen << 1 << d << d << 2);
      } else if (i+1 < size()) { /* last */
        elt_t P = elt_t::zeros(igen << 2 << d << d <<2);
        P.at(range(1),range(),range(),range(1)) =
          reshape(elt_t::eye(d,d), 1,d,d,1);
        at(i) = P;
      } else { /* otherwise */
        elt_t P = elt_t::zeros(igen << 2 << d << d << 1);
        P.at(range(1),range(),range(),range(0)) =
          reshape(elt_t::eye(d,d), 1,d,d,1);
        at(i) = P;
      }
    }
  }

} // namespace mps
