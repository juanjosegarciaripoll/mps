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
#include <mps/qform.h>
#include <mps/mps_algorithms.h>
#include <tensor/io.h>

namespace mps {

template <class MPO>
QuadraticForm<MPO>::QuadraticForm(const MPO &mpo, const mps_t &bra,
                                  const mps_t &ket, int start)
    : size_(mpo.size()),
      matrix_(make_matrix_database(mpo)),
      pairs_(make_pairs(mpo)) {
  // Boundary conditions not supported
  assert(bra[0].dimension(0) == 1 && ket[0].dimension(0) == 1);
  // Prepare the right matrices from site start to size()-1 to 0
  current_site_ = last_site();
  while (here() > start) {
    propagate_left(bra[here()], ket[here()]);
  }
  current_site_ = 0;
  while (here() != start) {
    propagate_right(bra[here()], ket[here()]);
  }
  // dump_matrices();
}

template <class MPO>
typename QuadraticForm<MPO>::matrix_database_t
QuadraticForm<MPO>::make_matrix_database(const MPO &mpo) {
  // We only support open boundary condition problems
  assert(mpo[0].dimension(0) == 1);
  index d, L = mpo.size();
  matrix_database_t output(L + 1);
  typename MPO::elt_t tensor;
  for (index i = 1; i < L; i++) {
    output.at(i) = matrix_array_t(mpo[i].dimension(0), elt_t());
  }
  output.at(0) = output.at(L) = matrix_array_t(1, elt_t::ones(1, 1, 1, 1));
  return output;
}

template <class MPO>
void QuadraticForm<MPO>::dump_matrices() {
  // We only support open boundary condition problems
  std::cout << "All matrices around " << here() << std::endl;
  for (index i = 0; i < matrix_.size(); i++) {
    for (index j = 0; j < matrix_[i].size(); j++) {
      std::cout << " matrix(" << i << "," << j << ")=" << matrix_[i][j]
                << std::endl;
    }
  }
}

template <class MPO>
typename QuadraticForm<MPO>::pair_tree_t QuadraticForm<MPO>::make_pairs(
    const MPO &mpo) {
  pair_tree_t output(mpo.size());
  for (index m = 0; m < mpo.size(); m++) {
    const elt_t &tensor = mpo[m];
    for (index i = 0; i < tensor.dimension(0); i++) {
      for (index j = 0; j < tensor.dimension(3); j++) {
        Pair p(i, j, tensor);
        if (!p.is_empty()) {
          output.at(m).push_back(p);
        }
      }
    }
  }
  return output;
}

template <class tensor>
static void maybe_add(tensor *a, const tensor &b) {
  if (a->is_empty())
    *a = b;
  else
    *a += b;
}

template <class MPO>
void QuadraticForm<MPO>::propagate(const elt_t &braP, const elt_t &ketP,
                                   int sense) {
  if (sense > 0)
    propagate_right(braP, ketP);
  else
    propagate_left(braP, ketP);
}

template <class MPO>
void QuadraticForm<MPO>::propagate_left(const elt_t &braP, const elt_t &ketP) {
  if (here() == 0) return;
  const matrix_array_t &mr = right_matrices(here());
  matrix_array_t &new_mr = right_matrices(here() - 1);
  // std::cout << "Original right matrices\n";
  // for (int i = 0; i < mr.size(); i++) {
  //   std::cout << " mr[" << i << "]=" << mr[i] << std::endl;
  // }
  std::fill(new_mr.begin(), new_mr.end(), elt_t());
  for (pair_iterator_t it = pairs_[here()].begin(), end = pairs_[here()].end();
       it != end; it++) {
    if (!mr[it->right_ndx].is_empty()) {
      maybe_add<elt_t>(&new_mr.at(it->left_ndx),
                       prop_matrix(mr[it->right_ndx], -1, braP, ketP, &it->op));
    }
  }
  // std::cout << "New matrix right locations\n";
  // for (int i = 0; i < new_mr.size(); i++) {
  //   std::cout << " mr'[" << i << "]=" << new_mr[i] << std::endl;
  // }
  --current_site_;
}

template <class MPO>
void QuadraticForm<MPO>::propagate_right(const elt_t &braP, const elt_t &ketP) {
  if (here() == last_site()) return;
  const matrix_array_t &ml = left_matrices(here());
  matrix_array_t &new_ml = left_matrices(here() + 1);
  // std::cout << "Original left matrices\n";
  // for (int i = 0; i < ml.size(); i++) {
  //   std::cout << " ml[" << i << "]=" << ml[i] << std::endl;
  // }
  std::fill(new_ml.begin(), new_ml.end(), elt_t());
  for (pair_iterator_t it = pairs_[here()].begin(), end = pairs_[here()].end();
       it != end; it++)
    if (!ml[it->left_ndx].is_empty()) {
      maybe_add<elt_t>(&new_ml.at(it->right_ndx),
                       prop_matrix(ml[it->left_ndx], +1, braP, ketP, &it->op));
    }
  // std::cout << "New matrix left locations\n";
  // for (int i = 0; i < new_ml.size(); i++) {
  //   std::cout << " ml'[" << i << "]=" << new_ml[i] << std::endl;
  // }
  ++current_site_;
}

template <class elt_t>
static elt_t compose(const elt_t &L, const elt_t &op, const elt_t &R) {
  // std::cout << "Compose\n"
  //           << " L=" << L << std::endl
  //           << " R=" << R << std::endl
  //           << " op=" << op << std::endl;
  index a1, a2, b1, b2, a3, b3;
  // L(a1,b1,a2,b2) op(i,j) R(a3,b3,a1,b1) -> H([a2,i,a3],[b2,j,b3])
  L.get_dimensions(&a1, &b1, &a2, &b2);
  R.get_dimensions(&a3, &b3, &a1, &b1);
  assert(a1 == 1 && b1 == 1);
  // Remember that kron(A(i,j),B(k,l)) -> C([k,i],[l,j])
  return kron(kron(reshape(R, a3, b3), op), reshape(L, a2, b2));
}

template <class elt_t>
static elt_t compose(const elt_t &L, const elt_t &op1, const elt_t &op2,
                     const elt_t &R) {
  // std::cout << "Compose\n"
  //           << " L=" << L << std::endl
  //           << " R=" << R << std::endl
  //           << " op1=" << op1 << std::endl
  //           << " op2=" << op2 << std::endl;
  index a1, a2, b1, b2, a3, b3;
  // L(a1,b1,a2,b2) op(i,j) R(a3,b3,a1,b1) -> H([a2,i,a3],[b2,j,b3])
  L.get_dimensions(&a1, &b1, &a2, &b2);
  R.get_dimensions(&a3, &b3, &a1, &b1);
  assert(a1 == 1 && b1 == 1);
  // Remember that kron(A(i,j),B(k,l)) -> C([k,i],[l,j])
  return kron(kron(kron(reshape(R, a3, b3), op2), op1), reshape(L, a2, b2));
}

template <class MPO>
typename QuadraticForm<MPO>::elt_t QuadraticForm<MPO>::single_site_matrix()
    const {
  elt_t output;
  for (pair_iterator_t it = pairs_[here()].begin(), end = pairs_[here()].end();
       it != end; it++) {
    const elt_t &vl = left_matrix(here(), it->left_ndx);
    const elt_t &vr = right_matrix(here(), it->right_ndx);
    if (!vl.is_empty() && !vr.is_empty())
      maybe_add<elt_t>(&output, compose(vl, it->op, vr));
  }
  return output;
}

template <class MPO>
typename QuadraticForm<MPO>::elt_t QuadraticForm<MPO>::two_site_matrix(
    int sense) const {
  elt_t output;
  index i, j;
  if (sense > 0) {
    i = here();
    j = i + 1;
    assert(j < size());
  } else {
    j = here();
    assert(j > 0);
    i = j - 1;
  }
  for (pair_iterator_t it1 = pairs_[i].begin(), end1 = pairs_[i].end();
       it1 != end1; it1++) {
    for (pair_iterator_t it2 = pairs_[j].begin(), end2 = pairs_[j].end();
         it2 != end2; it2++)
      if (it1->right_ndx == it2->left_ndx) {
        const elt_t &vl = left_matrix(i, it1->left_ndx);
        const elt_t &vr = right_matrix(j, it2->right_ndx);
        if (!vl.is_empty() && !vr.is_empty())
          maybe_add(&output, compose(vl, it1->op, it2->op, vr));
      }
  }
  return output;
}

template <class MPO>
typename QuadraticForm<MPO>::elt_t QuadraticForm<MPO>::apply_one_site_matrix(
    const elt_t &P) const {
  elt_t output;
  for (pair_iterator_t it = pairs_[here()].begin(), end = pairs_[here()].end();
       it != end; it++) {
    // L(a1,b1,a2,b2)
    const elt_t &L = left_matrix(here(), it->left_ndx);
    // R(a3,b3,a1,b1)
    const elt_t &R = right_matrix(here(), it->right_ndx);
    if (!L.is_empty() && !R.is_empty()) {
      index a2 = L.dimension(2);
      index b2 = L.dimension(3);
      index a3 = R.dimension(0);
      index b3 = R.dimension(1);
      // We implement this
      // Q(a2,i,a3) = L(a1,b1,a2,b2) O1(i,k) P(b2,k,b3) R(a3,b3,a1,b1)
      // where a1=b1 = 1, because of periodic boundary conditions
      elt_t Q = fold(fold(reshape(L, a2, b2), 1, foldin(it->op, -1, P, 1), 0),
                     2, reshape(R, a3, b3), 1);
      maybe_add(&output, Q);
    }
  }
  return output;
}

template <class MPO>
typename QuadraticForm<MPO>::elt_t
QuadraticForm<MPO>::take_single_site_matrix_diag() const {
  elt_t output;
  for (pair_iterator_t it = pairs_[here()].begin(), end = pairs_[here()].end();
       it != end; it++) {
    // L(a1,b1,a2,b2)
    const elt_t &L = left_matrix(here(), it->left_ndx);
    // R(a3,b3,a1,b1)
    const elt_t &R = right_matrix(here(), it->right_ndx);
    if (!L.is_empty() && !R.is_empty()) {
      index a2 = L.dimension(2);
      index b2 = L.dimension(3);
      index a3 = R.dimension(0);
      index b3 = R.dimension(1);
      // We implement this
      // Q(a2,i,a3) = L(a1,b1,a2,a2) O1(i,i) R(a3,a3,a1,b1)
      // where a1=b1 = 1, because of periodic boundary conditions
      elt_t Q =
          kron2_sum(kron2_sum(take_diag(reshape(L, a2, b2)), take_diag(it->op)),
                    take_diag(reshape(R, a3, b3)));
      maybe_add(&output, Q);
    }
  }
  return output;
}

template <class MPO>
typename QuadraticForm<MPO>::elt_t QuadraticForm<MPO>::apply_two_site_matrix(
    const elt_t &P12, int sense) const {
  elt_t output;
  index i, j;
  if (sense > 0) {
    i = here();
    j = i + 1;
    assert(j < size());
  } else {
    j = here();
    assert(j > 0);
    i = j - 1;
  }
  for (pair_iterator_t it1 = pairs_[i].begin(), end1 = pairs_[i].end();
       it1 != end1; it1++) {
    for (pair_iterator_t it2 = pairs_[j].begin(), end2 = pairs_[j].end();
         it2 != end2; it2++)
      if (it1->right_ndx == it2->left_ndx) {
        // L(a1,b1,a2,b2)
        const elt_t &L = left_matrix(i, it1->left_ndx);
        // R(a3,b3,a1,b1)
        const elt_t &R = right_matrix(j, it2->right_ndx);
        if (!L.is_empty() && !R.is_empty()) {
          index a2 = L.dimension(2);
          index b2 = L.dimension(3);
          index a3 = R.dimension(0);
          index b3 = R.dimension(1);
          // We implement this
          // Q12(a2,i,a3) = L(a1,b1,a2,b2) O1(i,k) O2(j,l)
          //                     P12(b2,k,l,b3) R(a3,b3,a1,b1)
          // where a1=b1 = 1, because of periodic boundary conditions
          elt_t Q12 =
              fold(fold(reshape(L, a2, b2), 1,
                        foldin(it1->op, -1, foldin(it2->op, -1, P12, 2), 1), 0),
                   3, reshape(R, a3, b3), 1);
          maybe_add(&output, Q12);
        }
      }
  }
  return output;
}

template <class MPO>
typename QuadraticForm<MPO>::elt_t
QuadraticForm<MPO>::take_two_site_matrix_diag(int sense) const {
  elt_t output;
  index i, j;
  if (sense > 0) {
    i = here();
    j = i + 1;
    assert(j < size());
  } else {
    j = here();
    assert(j > 0);
    i = j - 1;
  }
  for (pair_iterator_t it1 = pairs_[i].begin(), end1 = pairs_[i].end();
       it1 != end1; it1++) {
    for (pair_iterator_t it2 = pairs_[j].begin(), end2 = pairs_[j].end();
         it2 != end2; it2++) {
      if (it1->right_ndx == it2->left_ndx) {
        // L(a1,b1,a2,b2)
        const elt_t &L = left_matrix(i, it1->left_ndx);
        // R(a3,b3,a1,b1)
        const elt_t &R = right_matrix(j, it2->right_ndx);
        if (!L.is_empty() && !R.is_empty()) {
          index a2 = L.dimension(2);
          index b2 = L.dimension(3);
          index a3 = R.dimension(0);
          index b3 = R.dimension(1);
          // We implement this
          // Q12(a2,i,j,a3) = L(a1,a1,a2,a2) O1(i,i) O2(j,j) R(a3,a3,a1,a1)
          // where a1 = 1, because of periodic boundary conditions
          elt_t Q12 =
              kron2(kron2(take_diag(reshape(L, a2, b2)), take_diag(it1->op)),
                    kron2(take_diag(it2->op), take_diag(reshape(R, a3, b3))));
          // std::cout << "L= " << matrix_form(take_diag(reshape(L, a2,b2)))
          //           << std::endl
          //           << "o1=" << matrix_form(take_diag(it1->op))
          //           << std::endl
          //           << "o2=" << matrix_form(take_diag(it2->op))
          //           << std::endl
          //           << "R= " << matrix_form(take_diag(reshape(R, a3,b3)))
          //           << std::endl;
          // std::cout << "Q= " << matrix_form(Q12) << std::endl;
          maybe_add(&output, Q12);
        }
      }
    }
  }
  return output;
}

}  // namespace mps
