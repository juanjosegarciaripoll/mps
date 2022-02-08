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

#include <mps/mps.h>
#include <mps/mps_algorithms.h>

namespace mps {

using namespace tensor;

/* SINGLE-SITE CORRELATION FUNCTION */

template <class MPS, class Tensor>
tensor_scalar_t<Tensor> single_site_expected(const MPS &a, const Tensor &Op1,
                                             index k1) {
  Tensor M;
  const Tensor *op;
  k1 = a.normal_index(k1);
  for (index k = 0; k < a.ssize(); k++) {
    Tensor Pk = a[k];
    if (k == k1) {
      op = &Op1;
    } else {
      op = nullptr;
    }
    M = prop_matrix(M, +1, Pk, Pk, op);
  }
  return prop_matrix_close(M)[0];
}

/* STATE NORM */

template <class MPS>
static double state_norm(const MPS &a) {
  typename MPS::elt_t M;
  for (index k = 0; k < a.ssize(); k++) {
    M = prop_matrix(M, +1, a[k], a[k], nullptr);
  }
  return sqrt(tensor::abs(real(prop_matrix_close(M)[0])));
}

/* STATE NORM */

template <class MPS>
static typename MPS::number_t scalar_product(const MPS &a, const MPS &b,
                                             int sense) {
  typename MPS::elt_t M;
  assert(a.size() == b.size());
  if (sense >= 0) {
    for (index k = 0; k < a.ssize(); k++) {
      M = prop_matrix(M, +1, a[k], b[k], nullptr);
    }
  } else {
    for (index k = a.ssize(); k--;) {
      M = prop_matrix(M, -1, a[k], b[k], nullptr);
    }
  }
  return prop_matrix_close(M)[0];
}

}  // namespace mps
