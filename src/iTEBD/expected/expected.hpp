// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2012 Juan Jose Garcia Ripoll

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

#include <mps/tools.h>
#include <mps/itebd.h>

namespace mps {

  template<class Tensor>
  static inline const typename Tensor::elt_t
  do_expected12(const iTEBD<Tensor> &psi, const Tensor &Op12, int site)
  {
    if (!psi.is_canonical()) {
      return do_expected12(psi.canonical_form(), Op12, site);
    } else {
      tensor::index a, i, b, j, c;
      const Tensor &AlA = psi.combined_matrix(site);
      const Tensor &BlB = psi.combined_matrix(site+1);
      AlA.get_dimensions(&a, &i, &b);
      BlB.get_dimensions(&b, &j, &c);
      Tensor AlABlB = reshape(fold(AlA, -1, BlB, 0), a, i*j, c);
      Tensor v = psi.left_boundary(site);
      typename Tensor::elt_t value = trace(propagate_right(v, AlABlB, Op12));
      typename Tensor::elt_t norm = trace(propagate_right(v, AlABlB));
      return value / real(norm);
    }
  }

  template<class Tensor>
  static inline const typename Tensor::elt_t
  do_string_order(const iTEBD<Tensor> &psi,
		  const Tensor &Opi, int i, const Tensor &Opmiddle,
		  const Tensor &Opj, int j)
  {
    if (i > j) {
      return do_string_order(psi, Opj, j, Opmiddle, Opi, i);
    } else if (!psi.is_canonical()) {
      return do_string_order(psi.canonical_form(), Opj, j, Opmiddle, Opi, i);
    } else {
      j = j - i;
      i = i & 1;
      j = j + i;
      Tensor v1 = psi.left_boundary(0);
      Tensor v2 = v1;
      const Tensor none;
      const Tensor *op;
      for (int site = 0; (site <= j) || !(site & 1); ++site) {
        if (site == i)
          op = &Opi;
        else if (site == j)
          op = &Opj;
        else if (site > i && site < j)
          op = &Opmiddle;
        else
          op = &none;
        v1 = propagate_right(v1, psi.combined_matrix(site), *op);
        v2 = propagate_right(v2, psi.combined_matrix(site));
      }
      return trace(v1) / trace(v2);
    }
  }

}
