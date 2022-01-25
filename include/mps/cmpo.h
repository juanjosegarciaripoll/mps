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

#ifndef MPS_CMPO_H
#define MPS_CMPO_H

#include <mps/hamiltonian.h>

/*!\addtogroup TheMPS*/
/* @{ */

namespace mps {

/**Real matrix product structure.*/
class CMPO : public MP<tensor::CTensor> {
 public:
  typedef CMPS MPS;

  CMPO(index size, index physical_dimension);
  CMPO(const tensor::Indices &physical_dimension);
  CMPO(const Hamiltonian &H, double t = 0.0);
  CMPO();

 private:
  typedef MP<elt_t> parent;

  void clear(const tensor::Indices &dimensions);
};

}  // namespace mps

/* @} */

#endif /* !MPS_CMPO_H */
