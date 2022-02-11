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

#include <mps/algorithms/linalg.h>

namespace mps {

template double norm2(const RMPS &psi);

template double scprod(const RMPS &psi1, const RMPS &psi2, int direction);

template double expected(const RMPS &a, const RTensor &Op1, index k,
                         int direction);

template cdouble expected(const RMPS &a, const CTensor &Op1, index k,
                          int direction);

template double expected(const RMPS &a, const RTensor &Op1, index k1,
                         const RTensor &Op2, index k2, int direction);

template cdouble expected(const RMPS &a, const CTensor &Op1, index k1,
                          const CTensor &Op2, index k2, int direction);

template RTensor expected_vector(const RMPS &a, const RTensor &Op1);

template RTensor expected_vector(const RMPS &a,
                                 const std::vector<RTensor> &Op1);

template RTensor expected(const RMPS &a, const RTensor &op1,
                          const RTensor &op2);

template RTensor expected(const RMPS &a, const std::vector<RTensor> &op1,
                          const std::vector<RTensor> &op2);

template RTensor all_correlations_fast(const RMPS &a,
                                       const std::vector<RTensor> &op1,
                                       const std::vector<RTensor> &op2,
                                       const RMPS &b, bool symmetric = false,
                                       const RTensor *jordan_wigner_op = 0);

}  // namespace mps
