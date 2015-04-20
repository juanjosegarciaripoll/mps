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

#include <mps/lattice.h>

namespace mps {

  Lattice::Lattice(int sites, int N) :
    number_of_sites(sites),
    number_of_particles(N),
    configurations(filtered_states(number_of_sites, number_of_particles))
  {
  }

  const RSparse
  Lattice::hopping(int to_site, int from_site, bool fermionic)
  {
    if (to_site == from_site)
      return number(from_site);

    tensor::index L = configurations.size();
    RTensor values(L);

    tensor::index from_mask = (tensor::index)1 << from_site;
    tensor::index to_mask = (tensor::index)1 << to_site;
    tensor::index mask = from_mask | to_mask;
    RTensor::iterator v = values.begin();
    Indices ndx = configurations;
    for (Indices::iterator it = ndx.begin(), end = ndx.end(); it != end; ++it, ++v)
      {
	tensor::index other = (*it ^ mask);
	*v = (other & mask == to_mask);
      }
    Indices n = iota(0, L-1);
    return RSparse(n,sort_indices(ndx),values,L,L);
  }

  const RSparse
  Lattice::number(int site)
  {
    return interaction(site, site);
  }

  const RSparse
  Lattice::interaction(int site1, int site2)
  {
    tensor::index L = configurations.size();
    RTensor values(L);

    tensor::index mask1 = (tensor::index)1 << site1;
    tensor::index mask2 = (tensor::index)1 << site2;
    tensor::index target = mask1 | mask2;
    RTensor::iterator v = values.begin();
    for (Indices::const_iterator it = configurations.begin(), end = configurations.end();
	 it != end;
	 ++it, ++v)
      {
	*v = (*it & target == target);
      }
    Indices n = iota(0, L-1);
    return RSparse(n,n,values,L,L);
  }

  template<class Sparse>
  void maybe_add(Sparse *H, const Sparse &Op)
  {
    if (H->rows())
      *H = *H + Op;
    else
      *H = Op;
  }

  const RSparse
  Lattice::Hamiltonian(const RTensor &J, const RTensor &U, double mu, bool fermionic)
  {
    RSparse H;
    for (int i = 0; i < J.rows(); i++) {
      for (int j = 0; j < J.columns(); j++) {
	double Jij = J(i,j) - (i == j) * mu;
	if (Jij) {
	  maybe_add<RSparse>(&H, Jij * hopping(i, j, fermionic));
	}
	double Uij = U(i,j);
	if (Uij) {
	  maybe_add<RSparse>(&H, Uij * interaction(i, j));
	}
      }
    }
    return H;
  }
  
  const CSparse
  Lattice::Hamiltonian(const CTensor &J, const RTensor &U, double mu,
		       bool fermionic)
  {
    CSparse H;
    for (int i = 0; i < J.rows(); i++) {
      for (int j = 0; j < J.columns(); j++) {
	cdouble Jij = J(i,j) - (i == j) * mu;
	if (real(Jij) || imag(Jij)) {
	  maybe_add<CSparse>(&H, Jij * hopping(i, j, fermionic));
	}
	double Uij = U(i,j);
	if (Uij) {
	  maybe_add<CSparse>(&H, Uij * interaction(i, j));
	}
      }
    }
    return H;
  }
  
}
