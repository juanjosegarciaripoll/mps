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

#include <mps/quantum.h>
#include <mps/analysis.h>
#include <mps/io.h>

namespace mps {

static RMPO
interval_position_mpo(const Space::interval_t &interval)
{
  Indices dimensions(interval.qubits, 2);
  auto output = initialize_interactions_mpo<RMPO>(dimensions);

  RTensor s = 0.5 * (Pauli_id - Pauli_z);

  auto L = (interval.end - interval.start);
  for (index_t i = 0; i < interval.qubits; ++i) {
	RTensor d = s * (L / (2 << i));
	if (i == 0) {
	  d += Pauli_id * interval.start;
	}
	add_local_term(&output, d, (interval.qubits - i - 1));
  }
  return output;
}

RMPO
position_mpo(const Space &space, index_t axis)
{
  return space.extend_mpo(interval_position_mpo(space.interval(axis)), axis);
}


}  // namespace mps
