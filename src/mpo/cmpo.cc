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

template class MPO<CTensor>;

template void add_local_term(CMPO *mpdo, const CTensor &Hloc, index k);

template void add_interaction(CMPO *mpdo, const CTensor &Hi, index i,
                              const CTensor &Hj);

template void add_product_term(CMPO *mpdo, const std::vector<CTensor> &Hi);

template void add_interaction(CMPO *mpdo, const std::vector<CTensor> &Hi,
                              index i, const CTensor *sign = nullptr);

template void add_hopping_matrix(CMPO *mpdo, const CTensor &J,
                                 const CTensor &ad, const CTensor &a,
                                 const CTensor &sign);

template void add_jordan_wigner_matrix(CMPO *mpdo, const CTensor &J,
                                       const CTensor &ad, const CTensor &a,
                                       const CTensor &sign);

template CMPO local_Hamiltonian_mpo(const std::vector<CTensor> &Hloc);

}