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

#include <mps/mpo.h>
#include <tensor/io.h>

namespace mps {

  template<class MPO, class Tensor>
  static void do_add_hopping_matrix(MPO &mpo, const Tensor &J,
				    const Tensor &ad, const Tensor &a,
				    const Tensor &sign)
  {
    //
    // This function add terms
    // \sum_{j,j\neq i} J(i,j) a[i]*sign[i+1]*...*sign[j-1]*ad[j] to a Hamiltonian.
    //
    assert(J.rank() == 2);
    assert(J.rows() == J.columns());
    assert(J.rows() == mpo.size());

    int L = mpo.size();
    for (int j = 0; j < L; ++j) {
      std::vector<Tensor> ops(L);
      for (int i = 0; i < L; ++i) {
	if (i == j)
          ops.at(j) = ad;
        else
	  ops.at(i) = a * J(j, i);
      }
      add_interaction(&mpo, ops, j, &sign);
    }
  }

} // namespace mps
