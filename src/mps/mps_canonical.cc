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
#include <tensor/linalg.h>
#include <mps/mps.h>

namespace mps {

  template<class MPS, class Tensor>
  static void set_canonical_inner(MPS &psi, index ndx, const Tensor &t,
				  int sense, bool truncate)
  {
    index b1, i1, b2;
    if (sense == 0) {
      std::cerr << "In MPS::set_canonical(), " << sense
		<< " is not a valid direction.";
      abort();
    } else if (sense > 0) {
      if (ndx+1 == psi.size()) {
	psi.at(ndx) = t;
      } else {
	Tensor U, V;
	t.get_dimensions(&b1, &i1, &b2);
	RTensor s = linalg::svd(reshape(t, b1*i1, b2), &U, &V, SVD_ECONOMIC);
	index l = s.size();
	index new_l = where_to_truncate(s, truncate?
                                        MPS_DEFAULT_TOLERANCE :
                                        MPS_TRUNCATE_ZEROS,
                                        std::min<index>(b1*i1,b2));
	if (new_l != l) {
	  U = change_dimension(U, 1, new_l);
	  V = change_dimension(V, 0, new_l);
	  s = change_dimension(s, 0, new_l);
	  l = new_l;
	}
	psi.at(ndx) = reshape(U, b1,i1,l);
	scale_inplace(V, 0, s);
	psi.at(ndx+1) = fold(V, -1, psi[ndx+1], 0);
      }
    } else {
      if (ndx == 0) {
	psi.at(ndx) = t;
      } else {
	Tensor U, V;
	t.get_dimensions(&b1, &i1, &b2);
	RTensor s = linalg::svd(reshape(t, b1, i1*b2), &V, &U, SVD_ECONOMIC);
	index l = s.size();
	index new_l = where_to_truncate(s, truncate?
                                        MPS_DEFAULT_TOLERANCE :
                                        MPS_TRUNCATE_ZEROS,
                                        std::min<index>(b1,i1*b2));
	if (new_l != l) {
	  U = change_dimension(U, 0, new_l);
	  V = change_dimension(V, 1, new_l);
	  s = change_dimension(s, 0, new_l);
	  l = new_l;
	}
	psi.at(ndx) = reshape(U, l,i1,b2);
	scale_inplace(V, -1, s);
	psi.at(ndx-1) = fold(psi[ndx-1], -1, V, 0);
      }
    }
  }

  template<class MPS>
  static const MPS either_form_inner(MPS psi, int sense, bool normalize)
  {
    index L = psi.size();
    if (sense < 0) {
      for (index i = L; i; ) {
	--i;
	set_canonical(psi, i, psi[i], sense);
      }
      if (normalize) psi.at(0) /= norm2(psi[0]);
    } else {
      for (index i = 0; i < L; i++) {
	set_canonical(psi, i, psi[i], sense);
      }
      if (normalize) psi.at(L-1) /= norm2(psi[L-1]);
    }
    return psi;
  }

} // namespace mps
