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
#include <mps/io.h>

namespace mps {

template class MPO<RTensor>;

template void add_local_term(RMPO *mpdo, const RTensor &Hloc, index k);

template void add_interaction(RMPO *mpdo, const RTensor &Hi, index i,
                              const RTensor &Hj);

template void add_product_term(RMPO *mpdo, const vector<RTensor> &Hi);

template void add_interaction(RMPO *mpdo, const vector<RTensor> &Hi, index i,
                              const RTensor *sign = nullptr);

template void add_hopping_matrix(RMPO *mpdo, const RTensor &J,
                                 const RTensor &ad, const RTensor &a,
                                 const RTensor &sign);

template void add_jordan_wigner_matrix(RMPO *mpdo, const RTensor &J,
                                       const RTensor &ad, const RTensor &a,
                                       const RTensor &sign);

template RMPO local_Hamiltonian_mpo(const vector<RTensor> &Hloc);

template std::ostream &impl::text_dump(std::ostream &s, const RMPO &mpo,
                                       const char *name);

}  // namespace mps
