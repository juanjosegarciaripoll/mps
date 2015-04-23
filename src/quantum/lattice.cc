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

#include <tensor/io.h>
#include <mps/lattice.h>

namespace mps {

  static const int byte[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3,
    3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
    3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2,
    2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
    3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
    5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3,
    2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
    4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4,
    4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
    5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5,
    5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
  };

  static int count(tensor::index w)
  {
    if (sizeof(w) == 4) {
      int i = byte[w & 0xff];
      w >>= 8;
      i += byte[w & 0xff];
      w >>= 8;
      i += byte[w & 0xff];
      w >>= 8;
      i += byte[w & 0xff];
      return i;
    } else if (sizeof(w) == 8) {
      int i = byte[w & 0xff];
      w >>= 8;
      i += byte[w & 0xff];
      w >>= 8;
      i += byte[w & 0xff];
      w >>= 8;
      i += byte[w & 0xff];
      w >>= 8;
      i += byte[w & 0xff];
      w >>= 8;
      i += byte[w & 0xff];
      w >>= 8;
      i += byte[w & 0xff];
      w >>= 8;
      i += byte[w & 0xff];
      return i;
    } else {
      int count = 0;
      for (int i = sizeof(w); i; --i) {
        count += byte[w & 0xff];
        w >>= 8;
      }
      return count;
    }
  }

  const Indices
  Lattice::filtered_states(int sites, int filling)
  {
    tensor::index n = 0;
    for (tensor::index c = 0, l = (tensor::index)1 << sites; c < l; c++) {
      if (count(c) == filling)
	n++;
    }
    Indices output(n);
    n = 0;
    for (tensor::index c = 0, l = (tensor::index)1 << sites; c < l; c++) {
      if (count(c) == filling)
	output.at(n++) = c;
    }
    return output;
  }

  Lattice::Lattice(int sites, int N) :
    number_of_sites(sites),
    number_of_particles(N),
    configurations(filtered_states(number_of_sites, number_of_particles))
  {
  }

  const RSparse
  Lattice::hopping_operator(int to_site, int from_site, particle_kind_t kind) const
  {
    if (to_site == from_site)
      return number_operator(from_site);

    tensor::index L = configurations.size();
    RTensor values(L);

    tensor::index from_mask = (tensor::index)1 << from_site;
    tensor::index to_mask = (tensor::index)1 << to_site;
    tensor::index mask11 = from_mask | to_mask;
    tensor::index mask01 = from_mask;
    tensor::index mask10 = to_mask;

    RTensor::iterator v = values.begin();
    Indices ndx = configurations;
    if (kind == HARD_CORE_BOSONS) {
      for (Indices::iterator it = ndx.begin(), end = ndx.end();
	   it != end; ++it, ++v)
	{
	  tensor::index other = *it;
	  tensor::index aux = other & mask11;
	  if (aux == mask01) {
	    *v = 1.0;
	    other ^= mask11;
	  } else if (aux == mask10) {
	    *v = 0.0;
	    other ^= mask11;
	  } else {
	    *v = 0.0;
	  }
	  *it = other;
	}
    } else {
      tensor::index sign_mask, sign_value;
      if (from_site < to_site) {
	sign_mask = (to_mask-1) & (~(from_mask-1));
	sign_value = 0;
      } else {
	sign_mask = (from_mask-1) & (~(to_mask-1));
	sign_value = 1;
      }
      for (Indices::iterator it = ndx.begin(), end = ndx.end();
	   it != end; ++it, ++v)
	{
	  tensor::index other = *it;
	  tensor::index aux = other & mask11;
	  if (aux == mask01) {
	    if ((count(other & sign_mask) & 1) == sign_value)
	      *v = -1.0;
	    else
	      *v = 1.0;
	    other ^= mask11;
	  } else if (aux == mask10) {
	    *v = 0.0;
	    other ^= mask11;
	  } else {
	    *v = 0.0;
	  }
	  *it = other;
	}
    }
    Indices cols = iota(0, L-1);
    Indices rows = sort_indices(ndx);
    return RSparse(rows,cols,values,L,L);
  }

  const RSparse
  Lattice::number_operator(int site) const
  {
    return interaction_operator(site, site);
  }

  const RSparse
  Lattice::interaction_operator(int site1, int site2) const
  {
    tensor::index L = configurations.size();
    RTensor values(L);

    tensor::index mask1 = (tensor::index)1 << site1;
    tensor::index mask2 = (tensor::index)1 << site2;
    tensor::index target = mask1 | mask2;
    RTensor::iterator v = values.begin();
    for (Indices::const_iterator it = configurations.begin(),
	   end = configurations.end();
	 it != end;
	 ++it, ++v)
      {
	*v = (*it & target) == target;
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
  Lattice::Hamiltonian(const RTensor &J, const RTensor &U, double mu,
		       particle_kind_t kind) const
  {
    RSparse H;
    for (int i = 0; i < J.rows(); i++) {
      for (int j = 0; j < J.columns(); j++) {
	double Jij = J(i,j) - (i == j) * mu;
	if (Jij) {
	  maybe_add<RSparse>(&H, Jij * hopping_operator(i, j, kind));
	}
	double Uij = U(i,j);
	if (Uij) {
	  maybe_add<RSparse>(&H, Uij * interaction_operator(i, j));
	}
      }
    }
    return H;
  }
  
  const CSparse
  Lattice::Hamiltonian(const CTensor &J, const CTensor &U, double mu,
		       particle_kind_t kind) const
  {
    CSparse H;
    for (int i = 0; i < J.rows(); i++) {
      for (int j = 0; j < J.columns(); j++) {
	cdouble Jij = J(i,j) - (i == j) * mu;
	if (real(Jij) || imag(Jij)) {
	  maybe_add<CSparse>(&H, Jij * hopping_operator(i, j, kind));
	}
	cdouble Uij = U(i,j);
	if (abs(Uij)) {
	  maybe_add<CSparse>(&H, Uij * interaction_operator(i, j));
	}
      }
    }
    return H;
  }

  int
  Lattice::size() const
  {
    return number_of_sites;
  }
  
  int
  Lattice::particles() const
  {
    return number_of_particles;
  }
  
  tensor::index
  Lattice::dimension() const
  {
    return configurations.size();
  }
  
}
