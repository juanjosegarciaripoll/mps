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

namespace mps {

  template<class MPO>
  QuadraticForm<MPO>::QuadraticForm(const MPO &mpo, const MPS &bra, const MPS &ket, int start) :
    bond_dimensions_(mpo_inner_dimensions(mpo)),
    left_matrix_(make_matrix_database(bond_dimensions_, true)),
    right_matrix_(make_matrix_database(bond_dimensions_, false)),
    pairs_(make_pairs(mpo))
  {
    if (start == 0) {
      current_site_ = size() - 1;
      while (here() != 0)
	propagate_left(bra[here()], ket[here()]);
    } else {
      current_site_ = 0;
      while (here() != size()-1)
	propagate_right(bra[here()], ket[here()]);
    }
  }

  template<class MPO>
  Indices QuadraticForm<MPO>::mpo_inner_dimensions(const MPO &mpo)
  {
    index L = mpo.size();
    Indices output(L+1);
    output.at(0) = 1;
    output.at(L) = 1;
    for (index i = 0; i < L; i++)
      output.at(i+1) = mpo[i].dimension(3);
    return output;
  }

  template<class MPO>
  typename QuadraticForm<MPO>::matrix_database_t
  QuadraticForm<MPO>::make_matrix_database(const Indices &dimensions, bool left)
  {
    index delta = left? 0 : 1;
    index L = dimensions.size() - 1;
    matrix_database_t output(L);
    for (index i = 0; i < L; i++) {
      output.at(i) = matrix_array_t(dimensions[i+delta], elt_t());
    }
    return output;
  }

  template<class MPO>
  typename QuadraticForm<MPO>::pair_tree_t
  QuadraticForm<MPO>::make_pairs(const MPO &mpo)
  {
    pair_tree_t output(mpo.size());
    for (index m = 0; m < mpo.size(); m++) {
      const elt_t &tensor = mpo[m];
      for (index i = 0; i < tensor.dimension(0); i++) {
	for (index j = 0; j < tensor.dimension(3); j++) {
	  Pair p(i, j, tensor);
	  if (!p.is_empty())
	    output.at(m).push_front(p);
	}
      }
    }
    return output;
  }

  template<class tensor>
  static void maybe_add(tensor *a, const tensor &b)
  {
    *a = (a->is_empty())? b : (*a + b);
  }

  template<class MPO>
  void QuadraticForm<MPO>::propagate_left(const elt_t &braP, const elt_t &ketP)
  {
    if (here() == 0) {
      std::cerr << "Cannot propagate_left() beyond site " << here();
      abort();
    }
    const matrix_array_t &vr = left_matrix_[here()];
    matrix_array_t &new_vr = left_matrix_[here()-1];
    std::fill(new_vr.begin(), new_vr.end(), elt_t());
    for (pair_iterator_t it = pairs_[here()].begin(), end = pairs_[here()].end();
	 it != end;
	 it++)
      {
	maybe_add<elt_t>(&new_vr.at(it->right_ndx),
			 prop_matrix(vr[it->left_ndx], +1, braP, ketP, &it->op));
      }
    --current_site_;
  }

  template<class MPO>
  void QuadraticForm<MPO>::propagate_right(const elt_t &braP, const elt_t &ketP)
  {
    if (here()+1 >= left_matrix_.size()) {
      std::cerr << "Cannot propagate_left() beyond site " << here();
      abort();
    }
    const matrix_array_t &vl = left_matrix_[here()];
    matrix_array_t &new_vl = left_matrix_[here()+1];
    std::fill(new_vl.begin(), new_vl.end(), elt_t());
    for (pair_iterator_t it = pairs_[here()].begin(), end = pairs_[here()].end();
	 it != end;
	 it++)
      {
	maybe_add<elt_t>(&new_vl.at(it->right_ndx),
			 prop_matrix(vl[it->left_ndx], +1, braP, ketP, &it->op));
      }
    ++current_site_;
  }

  template<class elt_t>
  static elt_t compose(const elt_t &L, const elt_t &op, const elt_t &R)
  {
    // L(a1,a2,b1,b2) op(i,j) R(a3,a1,b3,b1) -> H([a2,i,a3],[b2,j,b3])
    index a1,a2,b1,b2;
    L.get_dimensions(&a1, &a2, &b1, &b2);
    index a3,b3;
    R.get_dimensions(&a3, &a1, &b3, &b1);
    if (a1 != 1 || b1 != 1) {
      std::cerr << "Due to laziness of their programmers, mps does not implement QForm for PBC";
      abort();
    }
    // Remember that kron(A(i,j),B(k,l)) -> C([k,i],[l,j])
    return kron(kron(reshape(R, a3,b3), op), reshape(L, a2,b2));
  }

  template<class elt_t>
  static elt_t compose(const elt_t &L, const elt_t &op1, const elt_t &op2, const elt_t &R)
  {
    // L(a1,a2,b1,b2) op(i,j) R(a3,a1,b3,b1) -> H([a2,i,a3],[b2,j,b3])
    index a1,a2,b1,b2;
    L.get_dimensions(&a1, &a2, &b1, &b2);
    index a3,b3;
    R.get_dimensions(&a3, &a1, &b3, &b1);
    if (a1 != 1 || b1 != 1) {
      std::cerr << "Due to laziness of their programmers, mps does not implement QForm for PBC";
      abort();
    }
    return kron(kron(kron(reshape(R, a3,b3), op2), op1), reshape(L, a2,b2));
  }

  template<class MPO>
  const typename QuadraticForm<MPO>::elt_t
  QuadraticForm<MPO>::single_site_matrix() const
  {
    elt_t output;
    for (pair_iterator_t it = pairs_[here()].begin(), end = pairs_[here()].end();
	 it != end;
	 it++)
      {
	const elt_t &vl = left_matrix_[here()][it->left_ndx];
	const elt_t &vr = right_matrix_[here()][it->right_ndx];
	if (!vl.is_empty() && !vr.is_empty()) {
	  maybe_add<elt_t>(&output, compose(vl, it->op, vr));
	}
      }
    return output;
  }


  template<class MPO>
  const typename QuadraticForm<MPO>::elt_t
  QuadraticForm<MPO>::two_site_matrix() const
  {
    elt_t output;
    if (here() + 1 >= size()) {
      std::cerr << "Cannot extract two-site matrix from site " << here();
      abort();
    }
    for (pair_iterator_t it1 = pairs_[here()].begin(), end1 = pairs_[here()].end();
	 it1 != end1;
	 it1++)
      {
	for (pair_iterator_t it2 = pairs_[here()+1].begin(), end2 = pairs_[here()+1].end();
	     it2 != end2;
	     it2++)
	  if (it1->right_ndx == it2->left_ndx) {
	    const elt_t &vl = left_matrix_[here()][it1->left_ndx];
	    const elt_t &vr = right_matrix_[here()+1][it2->right_ndx];
	    if (!vl.is_empty() && !vr.is_empty()) {
	      maybe_add(&output, compose(vl, it1->op, it2->op, vr));
	    }
	  }
      }
    return output;
  }

}
