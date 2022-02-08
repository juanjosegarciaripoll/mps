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

#ifndef ITEBD_SLOW_EXPECTED_HPP
#define ITEBD_SLOW_EXPECTED_HPP
#include <mps/itebd.h>
#include <mps/tools.h>

namespace mps {

template <class t>
static inline const t infinite_power(t R) {
  for (size_t i = 1; i <= 10; i++) {
    R = mmult(R, R);
    R /= norm2(R);
  }
  return R;
}

template <class t>
static inline typename t::elt_t slow_string_order(const t &Op1, int i,
                                                  const t &Opmid, const t &Op2,
                                                  int j, const t &A,
                                                  const t &lA, const t &B,
                                                  const t &lB) {
  if (i > j) return slow_string_order(Op2, j, Opmid, Op1, i, A, lA, B, lB);

  t AlA = scale(A, -1, lA);
  t BlB = scale(B, -1, lB);

  t EA = build_E_matrix(AlA);
  t EB = build_E_matrix(BlB);

  t R0 = infinite_power<t>(mmult(EA, EB));
  t R1 = R0;

  t Op;
  // We have to cover an even number of A and B sites
  // so that we can trace with a power of (EA * EB)
  int first = (i & 1) ? i - 1 : i;
  int last = (j & 1) ? j : j + 1;
  for (int k = first; k <= last; k++) {
    const t &AorB = (k & 1) ? BlB : AlA;
    const t &E = (k & 1) ? EB : EA;
    if (k < i || k > j) {
      Op = t();
    } else if (k == i) {
      if (i == j)
        Op = mmult(Op1, Op2);
      else
        Op = Op1;
    } else if (k == j) {
      Op = Op2;
    } else {
      Op = Opmid;
    }
    if (Op.is_empty())
      R1 = mmult(R1, E);
    else
      R1 = mmult(R1, build_E_matrix(foldin(Op, -1, AorB, 1), AorB));
    R0 = mmult(R0, E);
    double n = norm2(R0);
    R0 = R0 / n;
    R1 = R1 / n;
  }
  typename t::elt_t norm = trace(R0);
  typename t::elt_t value = trace(R1);
  return value / norm;
}

template <class t>
static inline const t ensure_3_indices(const t &A) {
  index l = A.size();
  index a = A.dimension(0);
  index b = A.dimension(A.rank() - 1);
  return reshape(A, a, l / (a * b), b);
}

template <class t>
static inline typename t::elt_t slow_expected(const t &Op1, t A, const t &lA) {
  A = scale(A, -1, lA);
  t R0 = build_E_matrix(A);
  t R = infinite_power<t>(R0);
  t R2 = build_E_matrix(foldin(Op1, -1, A, 1), A);
  typename t::elt_t N = trace(mmult(R0, R));
  typename t::elt_t E = trace(mmult(R2, R));
  return (E / N);
}

template <class t>
static inline typename t::elt_t slow_expected12(const t &Op12, const t &A,
                                                const t &lA, const t &B,
                                                const t &lB) {
  return slow_expected<t>(
      Op12, ensure_3_indices<t>(fold(scale(A, -1, lA), -1, B, 0)), lB);
}

template <class t>
static inline typename t::elt_t slow_expected12(const iTEBD<t> &psi,
                                                const t &Op12, int site) {
  if (site & 1)
    return slow_expected12(Op12, psi.matrix(1), psi.right_vector(1),
                           psi.matrix(0), psi.right_vector(0));
  else
    return slow_expected12(Op12, psi.matrix(0), psi.right_vector(0),
                           psi.matrix(1), psi.right_vector(1));
}

}  // namespace mps

#endif  // ITEBD_SLOW_EXPECTED_HPP
