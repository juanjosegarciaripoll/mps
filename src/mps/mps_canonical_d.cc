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
#include <mps/mps.h>
#include "mps_canonical.cc"

namespace mps {

void set_canonical(RMPS &psi, index site, const RTensor &t, int sense,
                   bool truncate) {
  set_canonical_inner(psi, site, t, sense, truncate);
}

const RMPS canonical_form(const RMPS &psi, int sense) {
  return canonical_form_at(psi, (sense < 0) ? 0 : (psi.size() - 1));
}

const RMPS normal_form(const RMPS &psi, int sense) {
  return normal_form_at(psi, (sense < 0) ? 0 : (psi.size() - 1));
}

const RMPS canonical_form_at(const RMPS &psi, index site) {
  return either_form_inner(psi, site, false);
}

const RMPS normal_form_at(const RMPS &psi, index site) {
  return either_form_inner(psi, site, true);
}

}  // namespace mps
