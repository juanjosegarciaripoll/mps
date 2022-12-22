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
#include <mps/mps.h>

namespace mps {

using namespace tensor;

RMPS cluster_state(index_t length) {
  if (length == 1) {
    return ghz_state(length);
  } else {
    RMPS output(length, 2, 2);
    // We choose to normalize the cluster state by modifying the first tensor to
    // contain the normalization factor. This has the advantage that the tensor
    // contraction is more precise. This factor is 1.0/sqrt(2 ** length)
    double normalization = pow(2.0, -static_cast<double>(length) / 2);

    RTensor &P0 = output.at(0);
    P0.fill_with_zeros();
    P0.at(0, 0, 0) = P0.at(0, 1, 1) = normalization;

    RTensor &PL = output.at(length - 1);
    PL.at(0, 0, 0) = 1;
    PL.at(0, 1, 0) = 1;
    PL.at(1, 0, 0) = 1;
    PL.at(1, 1, 0) = -1;

    for (index_t i = 1; i < (length - 1); i++) {
      RTensor &P = output.at(i);
      P.fill_with_zeros();
      P.at(0, 0, 0) = 1;
      P.at(1, 0, 0) = 1;
      P.at(0, 1, 1) = 1;
      P.at(1, 1, 1) = -(1);
    }
    return output;
  }
}

}  // namespace mps
