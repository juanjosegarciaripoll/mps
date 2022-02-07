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

#ifndef MPS_CMPS_H
#define MPS_CMPS_H

#include <mps/rmps.h>

/*!\addtogroup TheMPS*/
/* @{ */

namespace mps {

/**Real matrix product structure.*/
class CMPS : public MP<tensor::CTensor> {
 public:
  CMPS(const RMPS &real_mps);
  CMPS(index size, index physical_dimension = 0, index bond_dimension = 1,
       bool periodic = false);
  CMPS(const tensor::Indices &physical_dimension, index bond_dimension = 1,
       bool periodic = false);
  CMPS(const std::vector<elt_t> &data) : parent(data){};

  index normal_index(index i) const;

  /**Can the CMP be used for a periodic boundary condition problem?*/
  bool is_periodic() const;

  /**Create a random MPS. */
  static CMPS random(index length, index physical_dimension,
                     index bond_dimension, bool periodic = false);

  /**Create a random MPS. */
  static CMPS random(const tensor::Indices &physical_dimensions,
                     index bond_dimension, bool periodic = false);

 private:
  typedef MP<elt_t> parent;
};

}  // namespace mps

/* @} */

#endif /* !MPS_CMPS_H */
