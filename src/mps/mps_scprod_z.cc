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

#include <mps/algorithms/expectation.h>

namespace mps {

template double norm2(const CMPS &psi);

template cdouble scprod(const CMPS &psi1, const CMPS &psi2, int direction);

template cdouble expected(const CMPS &a, const CTensor &Op1, index_t k,
                          int direction);

template cdouble expected(const CMPS &a, const CTensor &Op1, index_t k1,
                          const CTensor &Op2, index_t k2, int direction);

template CTensor expected_vector(const CMPS &a, const CTensor &Op1);

template CTensor expected_vector(const CMPS &a, const vector<CTensor> &Op1);

template CTensor expected(const CMPS &a, const CTensor &op1,
                          const CTensor &op2);

template CTensor expected(const CMPS &a, const vector<CTensor> &op1,
                          const vector<CTensor> &op2);

template CTensor all_correlations_fast(const CMPS &a,
                                       const vector<CTensor> &op1,
                                       const vector<CTensor> &op2,
                                       const CMPS &b, bool symmetric,
                                       const CTensor *jordan_wigner_op);

}  // namespace mps
