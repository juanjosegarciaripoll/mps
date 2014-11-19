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

#include <mps/flags.h>
#include "slow_expected.hpp"
#include "bdry_expected.hpp"
#include "canonical_expected.hpp"

namespace mps {

  double
  string_order(const iTEBD<RTensor> &psi,
	       const RTensor &Opi, int i, const RTensor &Opmiddle,
	       const RTensor &Opj, int j)
  {
    if (FLAGS.get(MPS_ITEBD_EXPECTED_METHOD) == MPS_ITEBD_CANONICAL_EXPECTED) {
      return do_string_order(psi, Opi, i, Opmiddle, Opj, j);
    } else if (FLAGS.get(MPS_ITEBD_EXPECTED_METHOD) == MPS_ITEBD_SLOW_EXPECTED){
      return slow_string_order(Opi, i, Opmiddle, Opj, j,
                               psi.matrix(0), psi.right_vector(0),
                               psi.matrix(1), psi.right_vector(1));
    } else {
      return bdry_string_order(Opi, i, Opmiddle, Opj, j,
                               psi.matrix(0), psi.right_vector(0),
                               psi.matrix(1), psi.right_vector(1));
    }
  }

}
