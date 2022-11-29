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

#ifndef MPS_TOOLS_H
#define MPS_TOOLS_H

#include <tensor/tensor.h>
#include <mps/flags.h>

namespace mps {

using namespace tensor;

enum { MPS_DEBUG_NONE = 0, MPS_DEBUG_SIMPLIFY_MANY = 2 };
extern int debug_flags;

index_t where_to_truncate(const RTensor &s, double tol, index_t max_dim);

Indices weights_to_keep(const RTensor &s, double tol, index max_dim);

const RTensor limited_svd(RTensor A, RTensor *U, RTensor *V, double tolerance,
                          index max_dim = 0);

const RTensor split(RTensor *A, const RTensor &origA, int sense, bool truncate);

const RTensor build_E_matrix(const RTensor &A, index *a = 0, index *b = 0);

const RTensor build_E_matrix(const RTensor &A, const RTensor &B, index *a = 0,
                             index *b = 0);

const RTensor propagate_right(const RTensor &v, const RTensor &A,
                              const RTensor &op);

const RTensor limited_svd(CTensor A, CTensor *U, CTensor *V, double tolerance,
                          index max_dim = 0);

const CTensor split(CTensor *A, const CTensor &origA, int sense, bool truncate);

const CTensor build_E_matrix(const CTensor &A, index *a = 0, index *b = 0);

const CTensor build_E_matrix(const CTensor &A, const CTensor &B, index *a = 0,
                             index *b = 0);

const CTensor propagate_right(const CTensor &v, const CTensor &A,
                              const CTensor &op);

template <class Tensor>
inline const Tensor propagate_right(const Tensor &v, const Tensor &A) {
  return propagate_right(v, A, Tensor());
}

}  // namespace mps

#endif  // MPS_TOOLS_H
