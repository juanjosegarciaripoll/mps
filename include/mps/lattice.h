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

#ifndef MPS_LATTICE_H
#define MPS_LATTICE_H

#include <tensor/sparse.h>

namespace mps {

  using namespace tensor;

  class Lattice {

    const tensor::index number_of_sites;
    const int number_of_particles;
    const Indices configurations;

    static Indices filtered_states(int sites, int number_of_particles);

  public:

    enum {
      BOSONIC = 0,
      FERMIONIC = 1
    };
    
    Lattice(int sites, int N);

    const RSparse hopping(int site1, int site2, bool fermionic = false);
    const RSparse number(int site1);
    const RSparse interaction(int site1, int site2);

    const RSparse Hamiltonian(const RTensor &J, const RTensor &interactions,
			      double mu, bool fermionic = false);
    const CSparse Hamiltonian(const CTensor &J, const RTensor &interactions,
			      double mu, bool fermionic = false);

  };
  
}

#endif /* !MPS_LATTICE_H */
