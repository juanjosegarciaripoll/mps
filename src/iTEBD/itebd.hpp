// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2012 Juan Jose Garcia Ripoll

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
#include <mps/tools.h>
#include <mps/quantum.h>
#include <mps/itebd.h>

namespace mps {

template <class Tensor>
static const Tensor normalize(const Tensor &v) {
  return v / norm2(v);
}

template <class Tensor>
iTEBD<Tensor>::iTEBD(index_t dimension)
    : A_(normalize<Tensor>(Tensor::random(1, dimension, 1))),
      B_(normalize<Tensor>(Tensor::random(1, dimension, 1))),
      lA_{1},
      lB_{1},
      AlA_(A_),
      BlB_(B_),
      canonical_(true) {
  tensor_assert(dimension > 0);
}

template <class Tensor>
iTEBD<Tensor>::iTEBD(const Tensor &A)
    : A_(reshape(normalize<Tensor>(A), 1, A.dimension(0), 1)),
      B_(A_),
      lA_{1.0},
      lB_{1.0},
      AlA_(A_),
      BlB_(B_),
      canonical_(true) {
  tensor_assert(A.rank() == 1);
}

template <class Tensor>
iTEBD<Tensor>::iTEBD(const Tensor &A, const Tensor &B)
    : A_(reshape(normalize<Tensor>(A), 1, A.dimension(0), 1)),
      B_(reshape(normalize<Tensor>(B), 1, B.dimension(0), 1)),
      lA_{1.0},
      lB_{1.0},
      AlA_(A_),
      BlB_(B_),
      canonical_(true) {
  tensor_assert(A.rank() == 1);
  tensor_assert(B.rank() == 1);
}

template <class Tensor>
iTEBD<Tensor>::iTEBD(const Tensor &A, const Tensor &lA, const Tensor &B,
                     const Tensor &lB, bool canonical)
    : A_(A),
      B_(B),
      lA_(lA),
      lB_(lB),
      AlA_(scale(A, -1, lA)),
      BlB_(scale(B, -1, lB)),
      canonical_(canonical) {
  tensor_assert(A_.rank() == 3);
  tensor_assert(A_.dimension(0) == lB.dimension(0));
  tensor_assert(A_.dimension(2) == lA.dimension(0));
  tensor_assert(B_.dimension(0) == lA.dimension(0));
  tensor_assert(B_.dimension(2) == lB.dimension(0));
}

template <class Tensor>
double iTEBD<Tensor>::entropy(int) const {
  if (!is_canonical()) {
    return canonical_form().entropy();
  } else {
    /* In canonical form, the 'lA' is the vector of Schmidt coefficients
       * for the bipartition in between sites A and B */
    return mps::entropy(tensor::abs(lA_ * lA_));
  }
}

template <class Tensor>
const Tensor iTEBD<Tensor>::schmidt(int site) const {
  return tensor::square(tensor::abs(left_vector(site)));
}

}  // namespace mps
