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

/* TWO-SITE CORRELATION FUNCTION */

template <class MPS, class Tensor>
Tensor all_expected_vector_fast(const MPS &a, const std::vector<Tensor> &op,
                                const MPS &b) {
  index L = a.ssize();
  if (b.ssize() != L) {
    std::cerr << "In expected_vector(), two MPS of different size were passed";
    abort();
  }
  if (ssize(op) != L) {
    std::cerr << "In expected_vector(), there are less operators than the "
                 "state size.";
    abort();
  }
  std::vector<Tensor> auxLeft(L);
  Tensor left;
  for (index i = 1; i < L; i++) {
    auxLeft[i] = left = prop_matrix(left, +1, a[i - 1], b[i - 1], 0);
  }
  Tensor right;
  auto output = Tensor::empty(L);
  for (index i = L; i--;) {
    left = prop_matrix(auxLeft[i], +1, a[i], b[i], &op[i]);
    output.at(i) = prop_matrix_close(left, right)[0];
    right = prop_matrix(right, -1, a[i], b[i], 0);
  }
  return output;
}

}  // namespace mps
