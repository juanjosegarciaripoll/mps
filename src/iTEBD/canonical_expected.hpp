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

#include <mps/tools.h>
#include <mps/itebd.h>

namespace mps {

template <class Tensor>
static inline const tensor_scalar_t<Tensor> do_expected12(
    const iTEBD<Tensor> &psi, const Tensor &Op12, int site) {
  if (!psi.is_canonical()) {
    return do_expected12(psi.canonical_form(), Op12, site);
  } else {
    index_t a, i, b, j;
    const Tensor &AlA = psi.combined_matrix(0);
    const Tensor &BlB = psi.combined_matrix(1);
    AlA.get_dimensions(&a, &i, &b);
    BlB.get_dimensions(&b, &j, &a);
    Tensor v1 = psi.left_boundary(0);
    Tensor v2 = v1;
    if (site & 1) {
      v1 = propagate_right(v1, AlA);
      v2 = v1;
      const Tensor BlBAlA = reshape(fold(BlB, -1, AlA, 0), b, j * i, b);
      v1 = propagate_right(v1, BlBAlA, Op12);
      v2 = propagate_right(v2, BlBAlA);
      v1 = propagate_right(v1, BlB);
      v2 = propagate_right(v2, BlB);
    } else {
      const Tensor AlABlB = reshape(fold(AlA, -1, BlB, 0), a, i * j, a);
      v1 = propagate_right(v1, AlABlB, Op12);
      v2 = propagate_right(v2, AlABlB);
    }
    return trace(v1) / trace(v2);
  }
}

template <class Tensor>
static inline const tensor_scalar_t<Tensor> do_string_order(
    const iTEBD<Tensor> &psi, const Tensor &Opi, int i, const Tensor &Opmiddle,
    const Tensor &Opj, int j) {
  if (i == j) {
    return expected(psi, mmult(Opi, Opj), i);
  } else if (i > j) {
    return do_string_order(psi, Opj, j, Opmiddle, Opi, i);
  } else if (!psi.is_canonical()) {
    return do_string_order(psi.canonical_form(), Opi, i, Opmiddle, Opj, j);
  } else {
    j = j - i;
    i = i & 1;
    j = j + i;
    Tensor v1 = psi.left_boundary(0);
    Tensor v2 = v1;
    const Tensor none;
    const Tensor *op;
    for (int site = 0; (site <= j) || !(site & 1); ++site) {
      if (site == i)
        op = &Opi;
      else if (site == j)
        op = &Opj;
      else if (site > i && site < j)
        op = &Opmiddle;
      else
        op = &none;
      v1 = propagate_right(v1, psi.combined_matrix(site), *op);
      v2 = propagate_right(v2, psi.combined_matrix(site));
    }
    return trace(v1) / trace(v2);
  }
}

template <class Tensor>
static inline const Tensor do_string_order_many(const iTEBD<Tensor> &psi,
                                                const Tensor &Opi,
                                                const Tensor &Opmiddle,
                                                const Tensor &Opj, int N) {
  if (!psi.is_canonical()) {
    return do_string_order_many(psi.canonical_form(), Opi, Opmiddle, Opj, N);
  } else {
    Tensor v1 = psi.left_boundary(0);
    Tensor v2 = v1;
    Tensor output = Tensor::empty(N);
    Tensor nextv2;
    for (int site = 0; (site < N); ++site) {
      const Tensor &aux = psi.combined_matrix(site);
      Tensor v = propagate_right(v1, aux, site ? Opj : mmult(Opi, Opj));
      if (nextv2.size()) {
        v2 = nextv2;
      } else {
        v2 = propagate_right(v2, aux);
      }
      if (!(site & 1)) {
        const Tensor &tmp = psi.combined_matrix(site + 1);
        nextv2 = propagate_right(v2, tmp);
        v = propagate_right(v, tmp);
        output.at(site) = trace(v) / trace(nextv2);
      } else {
        nextv2 = Tensor();
        output.at(site) = trace(v) / trace(v2);
      }
      if (site) {
        v1 = propagate_right(v1, aux, Opmiddle);
      } else {
        v1 = propagate_right(v1, aux, Opi);
      }
    }
    return output;
  }
}

}  // namespace mps
