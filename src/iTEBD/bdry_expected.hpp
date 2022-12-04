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

#ifndef ITEBD_BDRY_EXPECTED_H
#define ITEBD_BDRY_EXPECTED_H
#include <tensor/linalg.h>
#include <mps/itebd.h>
#include <mps/tools.h>
#include <mps/algorithms.h>

namespace mps {

template <class Tensor>
static Tensor left_boundary(const Tensor &A) {
  Tensor v = prop_matrix(Tensor(), +1, A, A);
  v /= norm2(v);
  linalg::eig_power(
      [&A](const Tensor &left_vector) {
        return prop_matrix(left_vector, +1, A, A);
      },
      v.size(), &v);
  return v;
}

template <class Tensor>
static Tensor left_boundary(const Tensor &A, const Tensor &B) {
  Tensor v = prop_matrix(prop_matrix(Tensor(), +1, A, A), +1, B, B);
  v /= norm2(v);
  linalg::eig_power(
      [&A, &B](const Tensor &left_vector) {
        return prop_matrix(prop_matrix(left_vector, +1, A, A), +1, B, B);
      },
      v.size(), &v);
  return v;
}

template <class t>
static inline typename t::elt_t bdry_string_order(const t &Op1, int i,
                                                  const t &Opmid, const t &Op2,
                                                  int j, const t &A,
                                                  const t &lA, const t &B,
                                                  const t &lB) {
  if (i > j) return bdry_string_order(Op2, j, Opmid, Op1, i, A, lA, B, lB);

  std::cerr << "bdry_string_order\n";
  t AlA = scale(A, -1, lA);
  t BlB = scale(B, -1, lB);
  t R0 = left_boundary(AlA, BlB);
  t R1 = R0;

  t Op;
  // We have to cover an even number of A and B sites
  // so that we can trace with a power of (EA * EB)
  int first = (i & 1) ? i - 1 : i;
  int last = (j & 1) ? j : j + 1;
  for (int k = first; k <= last; k++) {
    const t &AorB = (k & 1) ? BlB : AlA;
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
    R1 = prop_matrix(R1, +1, AorB, AorB, &Op);
    R0 = prop_matrix(R0, +1, AorB, AorB);
    double n = norm2(R0);
    R0 /= n;
    R1 /= n;
  }
  std::cerr << "* " << prop_matrix_close(R1)[0] << " "
            << prop_matrix_close(R0)[0] << " "
            << prop_matrix_close(R1)[0] / prop_matrix_close(R0)[0] << '\n';
  return prop_matrix_close(R1)[0] / prop_matrix_close(R0)[0];
}

template <class t>
static inline typename t::elt_t bdry_expected(const t &Op1, const t &A,
                                              const t &lA) {
  t AlA = scale(A, -1, lA);
  t R0 = left_boundary(AlA);
  t R1 = prop_matrix(R0, +1, AlA, AlA, &Op1);
  R0 = prop_matrix(R0, +1, AlA, AlA);
  return prop_matrix_close(R1)[0] / prop_matrix_close(R0)[0];
}

template <class t>
static inline typename t::elt_t bdry_expected12(const t &Op12, const t &A,
                                                const t &lA, const t &B,
                                                const t &lB) {
  return bdry_expected<t>(
      Op12, ensure_3_indices<t>(fold(scale(A, -1, lA), -1, B, 0)), lB);
}

template <class t>
static inline typename t::elt_t bdry_expected12(const iTEBD<t> &psi,
                                                const t &Op12, int site) {
  if (site & 1)
    return bdry_expected12(Op12, psi.matrix(1), psi.right_vector(1),
                           psi.matrix(0), psi.right_vector(0));
  else
    return bdry_expected12(Op12, psi.matrix(0), psi.right_vector(0),
                           psi.matrix(1), psi.right_vector(1));
}

}  // namespace mps

#endif  // ITEBD_BDRY_EXPECTED_HPP
