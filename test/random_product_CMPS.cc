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

#include <cmath>
#include <tensor/tensor.h>
#include "loops.h"

namespace tensor_test {

  using namespace mps;

  CMPS random_product_CMPS(int n, int dimension)
  {
    CMPS output(n);
    for (int i = 0; i < n; i++) {
      CTensor v = CTensor::random(1, dimension, 1);
      output.at(i) = v / norm2(v);
    }
    return output;
  }

} // namespace tensor_test
