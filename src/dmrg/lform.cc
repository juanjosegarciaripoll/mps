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
#include <mps/lform.h>
#include <mps/mps_algorithms.h>

namespace mps {

  template<class MPS>
  LinearForm<MPS>::LinearForm(const MPS &bra, const MPS &ket, int start) :
    bra_(bra), ket_(ket),
    bond_dimensions_(mps_inner_dimensions(bra)),
    left_matrix_(make_matrix_array(bond_dimensions_, true)),
    right_matrix_(make_matrix_array(bond_dimensions_, false))
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

  template<class MPS>
  Indices LinearForm<MPS>::mps_inner_dimensions(const MPS &mps)
  {
    index L = mps.size();
    Indices output(L+1);
    output.at(0) = 1;
    output.at(L) = 1;
    for (index i = 0; i < L; i++)
      output.at(i+1) = mps[i].dimension(2);
    return output;
  }

  template<class MPS>
  typename LinearForm<MPS>::matrix_array_t
  LinearForm<MPS>::make_matrix_array(const Indices &dimensions, bool left)
  {
    return matrix_array_t(dimensions.size()-1, elt_t());
  }

  template<class MPS>
  void LinearForm<MPS>::propagate_left(const elt_t &braP, const elt_t &ketP)
  {
    if (here() == 0) {
      std::cerr << "Cannot propagate_left() beyond site " << here();
      abort();
    }
    bra_.at(here()) = braP;
    ket_.at(here()) = ketP;
    right_matrix_[here()-1] = prop_matrix(right_matrix_[here()], -1, braP, ketP);
    --current_site_;
  }

  template<class MPS>
  void LinearForm<MPS>::propagate_right(const elt_t &braP, const elt_t &ketP)
  {
    if (here()+1 >= left_matrix_.size()) {
      std::cerr << "Cannot propagate_left() beyond site " << here();
      abort();
    }
    bra_.at(here()) = braP;
    ket_.at(here()) = ketP;
    left_matrix_[here()+1] = prop_matrix(left_matrix_[here()], +1, braP, ketP);
    ++current_site_;
  }

  template<class elt_t>
  static elt_t compose(const elt_t &L, const elt_t &P, const elt_t &R)
  {
    index a1,a2,b1,b2,a3,b3,i;
    L.get_dimensions(&a1, &a2, &b1, &b2);
    R.get_dimensions(&a3, &a1, &b3, &b1);
    P.get_dimensions(&a2, &i, &a3);
    if (a1 != 1 || b1 != 1) {
      std::cerr << "Due to laziness of their programmers, mps does not implement LForm for PBC";
      abort();
    }
    // Reshape L -> L(a2,b2), R -> R(a3,b3)
    // and return L(a2,b2) P(a2,i,a3) R(a3,b3)
    return fold(fold(reshape(L, a2,b2), 1, P, 0), -1,
		reshape(R, a3, b3), 0);
  }

  template<class MPS>
  const typename LinearForm<MPS>::elt_t
  LinearForm<MPS>::single_site_vector() const
  {
    return compose(left_matrix_[here()], bra_[here()], right_matrix_[here()]);
  }

  template<class elt_t>
  static elt_t compose(const elt_t &L, const elt_t &P1, const elt_t &P2, const elt_t &R)
  {
    index a1,a2,b1,b2,a3,b3,a4,b4,i,j;
    L.get_dimensions(&a1, &a2, &b1, &b2);
    R.get_dimensions(&a4, &a1, &b4, &b1);
    P1.get_dimensions(&a2, &i, &a3);
    P2.get_dimensions(&a3, &j, &a4);
    if (a1 != 1 || b1 != 1) {
      std::cerr << "Due to laziness of their programmers, mps does not implement LForm for PBC";
      abort();
    }
    // Reshape L -> L(a2,b2), R -> R(a4,b4)
    // and return L(a2,b2) P(a2,i,i,a4) R(a4,b4)
    elt_t P = fold(P1, -1, P2, 0);
    return fold(fold(reshape(L, a2,b2), 1, P, 0), -1,
		reshape(R, a4, b4), 0);
  }

  template<class MPS>
  const typename LinearForm<MPS>::elt_t
  LinearForm<MPS>::two_site_vector() const
  {
    elt_t output;
    if (here() + 1 >= size()) {
      std::cerr << "Cannot extract two-site matrix from site " << here();
      abort();
    }
    return compose(left_matrix_[here()], bra_[here()], bra_[here()+1],
		   right_matrix_[here()+1]);
  }

}
