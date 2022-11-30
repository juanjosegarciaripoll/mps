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
#include <tensor/linalg.h>
#include <mps/imath.h>
#include <mps/quantum.h>
#include <mps/hamiltonian.h>

namespace mps {

using mps::imath::isqrt;

void split_interaction(const CTensor &H12, vector<CTensor> *v1,
                       vector<CTensor> *v2) {
  tensor_assert(H12.rank() == 2);
  /*
     * Notice the funny reordering of indices in O1 and O2, which is due to the
     * following statement and which simplifies the application of O1 and O2 on a
     * given vector.
     */
  index_t d1 = isqrt(H12.rows());
  index_t d2 = d1;

  CTensor O1, O2;
  CTensor U =
      reshape(permute(reshape(H12, d1, d2, d1, d2), 1, 2), d1 * d1, d2 * d2);
  RTensor s = mps::limited_svd(U, &O1, &O2, 1e-13);

  index_t n_op = s.ssize();
  v1->resize(n_op);
  v2->resize(n_op);
  for (index_t i = 0; i < n_op; i++) {
    double sqrts = sqrt(s[i]);
    v1->at(i) = sqrts * tensor::reshape(CTensor{O1(_, range(i))}, d1, d1);
    v2->at(i) =
        tensor::conj(sqrts * tensor::reshape(CTensor{O2(range(i), _)}, d2, d2));
  }
}

}  // namespace mps
