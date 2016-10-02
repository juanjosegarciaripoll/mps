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
  static MPO do_local_Hamiltonian_mpo(const std::vector<Tensor> &Hloc)
  {
    MPO output(Hloc.size(), 1);
    
    for (index i = 0; i < Hloc.size(); ++i) {
      index d = Hloc[i].rows();
      Tensor aux = Tensor::zeros(2, d, d, 2);
      aux.at(range(0), range(), range(), range(1)) = Hloc[i];
      std::cout << aux.dimensions() << std::endl;

      Tensor id = Tensor::eye(d);
      aux.at(range(1), range(), range(), range(1)) = id;
      aux.at(range(0), range(), range(), range(0)) = id;

      if (i == 0)
	aux = aux(range(0),range(),range(),range());
      if (i+1 == Hloc.size())
	aux = aux(range(),range(),range(),range(1));
      output.at(i) = aux;
    }
    return output;
  }

} // namespace mps
