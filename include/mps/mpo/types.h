// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
#pragma once
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

#ifndef MPS_MPO_TYPES_H
#define MPS_MPO_TYPES_H

#include <mps/mp_base.h>
#include <mps/hamiltonian.h>

namespace mps {

/*!\addtogroup TheMPS*/
/* @{ */

/**Matrix Product Operator structure.*/
template <typename Tensor>
class MPO : public MP<Tensor> {
 public:
  using parent_t = MP<Tensor>;
  using tensor_array_t = typename parent_t::tensor_array_t;
  using MPS = MPS<Tensor>;

  MPO() = default;
  MPO(const MPO &) = default;
  MPO(MPO &&) = default;
  MPO &operator=(const MPO &) = default;
  MPO &operator=(MPO &&) = default;

  MPO(index_t length, index_t physical_dimension)
      : parent_t(empty_mpo_tensors(Indices(length, physical_dimension))) {}

  MPO(const tensor::Indices &physical_dimensions)
      : parent_t(empty_mpo_tensors(physical_dimensions)) {}

  MPO(tensor_array_t tensors) : parent_t(std::move(tensors)) {}

 private:
  static tensor_array_t empty_mpo_tensors(
      const tensor::Indices &physical_dimensions) {
    tensor_array_t output;
    output.reserve(physical_dimensions.ssize());
    for (auto d : physical_dimensions) {
      output.emplace_back(reshape(Tensor::eye(d,d), 1,d,d,1));
    }
    return output;
  }
};

template <typename T>
struct mp_tensor_t_inner<MPO<T>> {
  typedef T type;
};

extern template class MPO<RTensor>;
extern template class MPO<CTensor>;
#ifdef DOXYGEN_ONLY
/**Real matrix product structure.*/
struct RMPO : public MPS<RTensor> {};
/**Complex matrix product structure.*/
struct CMPO : public MPS<CTensor> {};
#else
typedef MPO<RTensor> RMPO;
typedef MPO<CTensor> CMPO;
#endif

/* @} */

}  // namespace mps

#endif  // MPS_MPO_TYPES_H
