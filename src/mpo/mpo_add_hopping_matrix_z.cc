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

#include "mpo_add_hopping_matrix.cc"

namespace mps {

  void add_hopping_matrix(CMPO *mpo, const CTensor &J,
			  const CTensor &ad, const CTensor &a)
  {
    do_add_hopping_matrix(*mpo, J, ad, a, CTensor::eye(a.rows()));
  }

  void add_jordan_wigner_matrix(CMPO *mpo, const CTensor &J,
				const CTensor &ad, const CTensor &a,
				const CTensor &sign)
  {
    do_add_hopping_matrix(*mpo, J, ad, a, sign);
  }

} // namespace mps
