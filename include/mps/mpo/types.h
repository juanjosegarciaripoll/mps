#pragma once
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

#ifndef MPS_MPO_TYPES_H
#define MPS_MPO_TYPES_H

#include <mps/mps.h>

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

  MPO(const std::initializer_list<Tensor> &l) : parent_t(tensor_array_t(l)) {}

 private:
  static tensor_array_t empty_mpo_tensors(
      const tensor::Indices &physical_dimensions) {
    tensor_array_t output;
    output.reserve(physical_dimensions.ssize());
    for (auto d : physical_dimensions) {
      output.emplace_back(reshape(Tensor::eye(d, d), 1, d, d, 1));
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
using RMPO = MPO<RTensor>;
using CMPO = MPO<CTensor>;
#endif

/**Matrix Product Operator list.*/

template <typename MPO>
class MPOList : public vector<MPO> {
  using parent_t = vector<MPO>;

 public:
  using elt_t = MPO;
  using mpo_t = MPO;
  using tensor_t = typename MPO::tensor_t;
  using mpo_array_t = vector<mpo_t>;

  MPOList() = default;
  MPOList(const MPOList &) = default;
  MPOList(MPOList &&) = default;
  MPOList &operator=(const MPOList &) = default;
  MPOList &operator=(MPOList &&) = default;
  ~MPOList() = default;

  MPOList(const mpo_array_t &mpos) : parent_t(mpos) {}
  MPOList(mpo_array_t &&mpos) : parent_t(std::move(mpos)) {}
};

extern template class MPOList<RMPO>;
extern template class MPOList<CMPO>;
#ifdef DOXYGEN_ONLY
/**Real matrix product structure.*/
struct RMPOList : public MPOList<RMPO> {};
/**Complex matrix product structure.*/
struct CMPOList : public MPOList<CMPO> {};
#else
using RMPOList = MPOList<RMPO>;
using CMPOList = MPOList<CMPO>;
#endif

/* @} */

}  // namespace mps

#endif  // MPS_MPO_TYPES_H
