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

#include <mps/hamiltonian.h>

namespace mps {

/*!\addtogroup TheMPS*/
/* @{ */

/**Matrix Product Operator structure.*/
template <typename Tensor>
class MPO : public MP<Tensor> {
 public:
  typedef MPS<Tensor> MPS;
  MPO() = default;
  MPO(const MPO &) = default;
  MPO(MPO &&) = default;
  MPO &operator=(const MPO &) = default;
  MPO &operator=(MPO &&) = default;

  MPO(index_t length, index_t physical_dimension) : parent(length) {
    tensor::Indices dims(length);
    std::fill(dims.begin(), dims.end(), physical_dimension);
    clear(dims);
  }

  MPO(const tensor::Indices &physical_dimensions)
      : parent(physical_dimensions.ssize()) {
    clear(physical_dimensions);
  }

  MPO(const Hamiltonian &H, double t = 0.0) : parent(H.size()) {
    clear(H.dimensions());
    add_Hamiltonian(this, H, t);
  }

 private:
  typedef MP<Tensor> parent;

  void clear(const tensor::Indices &physical_dimensions) {
    if (physical_dimensions.size() < 2) {
      std::cerr << "Cannot create MPO with size 0 or 1.\n";
      abort();
    }
    // TODO: Simplify. We only need sizes (1,d,d,1) for the add_local/add_interaction to succeed.
    Tensor P;
    for (index_t i = 0; i < this->ssize(); i++) {
      index_t d = physical_dimensions[i];
      Tensor Id = Tensor::eye(d, d);
      if (i == 0) {
        /* first */
        P = Tensor::zeros(1, d, d, 2);
        P.at(range(0), _, _, range(0)) = Id;
      } else if (i + 1 < this->ssize()) {
        /* last */
        P = Tensor::zeros(2, d, d, 2);
        P.at(range(1), _, _, range(1)) = Id;
        P.at(range(0), _, _, range(0)) = Id;
      } else {
        /* otherwise */
        P = Tensor::zeros(2, d, d, 1);
        P.at(range(1), _, _, range(0)) = Id;
      }
      this->at(i) = P;
    }
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
