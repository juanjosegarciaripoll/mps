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

  /* This is how we encode MPO:

     - An operator is a collection of tensors A(a,i,j,b), where
       for fixed "a" and "b", the matrix A(a,i,j,b) is an operator
       acting on the state.

     - "a" = 0 means we are free to choose what to do.
     - "a" = 1 means we can only apply identities.
     - A(0,i,j,b>1) is the first operator from a nearest-neighbor
       interaction, which is paired by A(b,i,j,1)
    
   */

  template<class MPO, class Tensor>
  static void do_add_local_term(MPO &mpo, const Tensor &Hloc, index i)
  {
    if (i < 0 || i >= mpo.size()) {
      std::cerr << "In add_local_term(), the index " << i << " is outside the lattice.\n";
      abort();
    }
    if (Hloc.columns() != Hloc.rows()) {
      std::cerr << "In add_local_term(MPO, Tensor, index), the Tensor is not a square matrix.\n";
      abort();
    }
    Tensor Pi = mpo[i];
    index d = Pi.dimension(1);
    if (d != Hloc.rows()) {
      std::cerr << "In add_local_term(MPO, Tensor, index), the dimensions of the tensor do not match those of the MPO.\n";
      abort();
    }
    if (Pi.dimension(0) == 1) {
      /* First */
      Pi.at(range(0),range(),range(),range(1)) =
        Tensor(Pi(range(0),range(),range(),range(1))) +
        reshape(Hloc, 1, d, d, 1);
    } else if (Pi.dimension(3) == 1) {
      /* Last */
      Pi.at(range(0),range(),range(),range(0)) =
        Tensor(Pi(range(0),range(),range(),range(0))) +
        reshape(Hloc, 1, d, d, 1);
    } else {
      /* Middle */
      Pi.at(range(0),range(),range(),range(1)) =
        Tensor(Pi(range(0),range(),range(),range(1))) +
        reshape(Hloc, 1, d, d, 1);
    }
    mpo.at(i) = Pi;
  }

  template<class MPO, class Tensor>
  static void do_add_interaction(MPO &mpo, const Tensor &Hi, index i,
				 const Tensor &Hj)
  {
    if (i < 0 || (i+1) >= mpo.size()) {
      std::cerr << "In add_local_term(), the index " << i << " is outside the lattice.\n";
      abort();
    }
    if (Hi.rows() != Hi.columns()) {
      std::cerr << "In add_interaction(MPO, ...), second argument is not a square matrix\n";
      abort();
    }
    if (Hj.rows() != Hj.columns()) {
      std::cerr << "In add_interaction(MPO, ...), fourth argument is not a square matrix\n";
      abort();
    }
    index di = Hi.rows(), dj = Hj.rows();
    Tensor Pi = mpo[i];
    Tensor Pj = mpo[i+1];
    index b = Pi.dimension(3) + 1;
    Pi = change_dimension(Pi, 3, b);
    Pj = change_dimension(Pj, 0, b);

    if (Pi.dimension(1) != di) {
      std::cerr << "In add_interaction(MPO, ...), the second argument has wrong dimensions.\n";
      abort();
    }
    if (Pj.dimension(1) != dj) {
      std::cerr << "In add_interaction(MPO, ...), the second argument has wrong dimensions.\n";
      abort();
    }

    Pi.at(range(0), range(),range(), range(b-1)) = reshape(Hi, 1, di, di, 1);
    if (i+2 == mpo.size()) {
      Pj.at(range(b-1), range(), range(), range(0)) = reshape(Hj, 1, dj, dj, 1);
    } else {
      Pj.at(range(b-1), range(), range(), range(1)) = reshape(Hj, 1, dj, dj, 1);
    }

    mpo.at(i) = Pi;
    mpo.at(i+1) = Pj;
  }

} // namespace mps
