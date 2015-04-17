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

  int count(index w)
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
    } else {
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
    }
  }

  const Indices
  Lattice::filtered_states(int sites, int filling)
  {
    index n = 0;
    for (index c = 0, l = (index)1 << sites; c < l; c++) {
      if (count(c) == filling)
	n++;
    }
    Indices output(n);
    n = 0;
    for (index c = 0, l = (index)1 << sites; c < l; c++) {
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
  Lattice::hopping(int to_site, int from_site, bool fermionic)
  {
    if (to_site == from_site)
      return number(from_site);

    index L = configurations.size();
    RTensor values(L);

    index from_mask = (index)1 << from_site;
    index to_mask = (index)1 << to_site;
    index mask = from_mask | to_mask;
    index sign_mask;

    if (from_site < to_site)
      sign_mask = (to_mask-1) & (~(from_mask-1));
    else
      sign_mask = (from_mask-1) & (~(to_mask-1));
    
    RTensor::iterator v = values.begin();
    Indices ndx = configurations;
    if (fermionic) {
      for (Indices::iterator it = ndx.begin(), end = ndx.end();
	   it != end; ++it, ++v)
	{
	  index other = (*it ^ mask);
	  *v = (other & mask == to_mask);
	}
    } else {
      for (Indices::iterator it = ndx.begin(), end = ndx.end();
	   it != end; ++it, ++v)
	{
	  index other = (*it ^ mask);
	  bool sign = count(*it & sign_mask) & 1;
	  *v = (other & mask == to_mask);
	  if (sign & 1)
	    *v = -*v;
	}
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
    index L = configurations.size();
    RTensor values(L);

    index mask1 = (index)1 << site1;
    index mask2 = (index)1 << site2;
    index target = mask1 | mask2;
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
