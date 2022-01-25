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

CMPO::CMPO() : parent() {}

CMPO::CMPO(index length, index physical_dimension) : parent(length) {
  if (length < 2) {
    std::cerr << "Cannot create MPO with size 0 or 1.\n";
    abort();
  }
  tensor::Indices dims(length);
  std::fill(dims.begin(), dims.end(), physical_dimension);
  clear(dims);
}

CMPO::CMPO(const tensor::Indices &physical_dimensions)
    : parent(physical_dimensions.size()) {
  clear(physical_dimensions);
}

void CMPO::clear(const tensor::Indices &physical_dimensions) {
  if (physical_dimensions.size() < 2) {
    std::cerr << "Cannot create MPO with size 0 or 1.\n";
    abort();
  }
  elt_t P;
  for (index i = 0; i < size(); i++) {
    index d = physical_dimensions[i];
    elt_t Id = reshape(elt_t::eye(d, d), 1, d, d, 1);
    if (i == 0) {
      /* first */
      P = elt_t::zeros(1, d, d, 2);
      P.at(range(0), range(), range(), range(0)) = Id;
    } else if (i + 1 < size()) {
      /* last */
      P = elt_t::zeros(2, d, d, 2);
      P.at(range(1), range(), range(), range(1)) = Id;
      P.at(range(0), range(), range(), range(0)) = Id;
    } else {
      /* otherwise */
      P = elt_t::zeros(2, d, d, 1);
      P.at(range(1), range(), range(), range(0)) = Id;
    }
    at(i) = P;
  }
}

}  // namespace mps
