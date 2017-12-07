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

namespace mps {

  double fidelity(const RMPO &Ua, const RMPO &Ub)
  {
    RMPS psi1 = canonical_form(mpo_to_mps(Ua), -1);
    RMPS psi2 = canonical_form(mpo_to_mps(Ub), -1);
    double n1 = norm2(psi1);
    double n2 = norm2(psi2);
    return abs(scprod(psi1,psi2)/(n1*n2));    
  }

} // namespace mps
