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

#ifndef MPS_MP_BASE_H
#define MPS_MP_BASE_H

#include <mps/vector.h>
#include <tensor/traits.h>
#include <tensor/tensor.h>
#include <mps/except.h>
#include <mps/vector.h>

namespace mps {

using namespace tensor;

class Sweeper {
 public:
  Sweeper(index_t L, index_t sense);
  index_t operator*() const { return k_; };
  bool operator--();
  bool is_last() const { return k_ == kN_; };
  int sense() const { return tensor::narrow_cast<int>(dk_); };
  index_t site() const { return k_; };
  void flip();

 private:
  index_t k_, k0_, kN_, dk_;
};

template <class Tensor>
class MP {

 public:
  using elt_t = Tensor;
  using number_t = tensor_scalar_t<Tensor>;
  using tensor_array_t = vector<Tensor>;
  using iterator = typename tensor_array_t::iterator;
  using const_iterator = typename tensor_array_t::const_iterator;

  MP() = default;
  MP(const MP &) = default;
  MP(MP &&) = default;
  MP &operator=(const MP &) = default;
  MP &operator=(MP &&) = default;
  explicit MP(index_t size) : data_(size) {}
  explicit MP(const vector<Tensor> &other) : data_(other) {}
  explicit MP(vector<Tensor> &&other) : data_(std::move(other)) {}

  size_t size() const { return data_.size(); }
  index_t ssize() const { return data_.ssize(); }
  index_t last() const { return ssize() - 1; }
  index_t last_index() const { return ssize() - 1; }
  void resize(index_t new_size) { data_.resize(new_size); }

  const Tensor &operator[](index_t n) const { return data_[normal_index(n)]; }
  Tensor &at(index_t n) { return data_.at(normal_index(n)); }

  iterator begin() { return data_.begin(); }
  const_iterator begin() const { return data_.begin(); }
  const_iterator end() const { return data_.end(); }
  iterator end() { return data_.end(); }
  const vector<Tensor> to_vector() const { return data_; }

  Sweeper sweeper(index_t sense) const { return Sweeper(ssize(), sense); }

  index_t normal_index(index_t mps_index) const {
    index_t mps_size = ssize();
    if (mps_index < 0) {
      mps_index += mps_size;
      tensor_assert2(mps_index >= 0 && mps_index < mps_size,
                     mps_out_of_range());
      return mps_index;
    } else {
      tensor_assert2(mps_index < mps_size, mps_out_of_range());
      return mps_index;
    }
  }

 private:
  tensor_array_t data_{};
};

template <typename T>
struct mp_tensor_t_inner {};

template <typename T>
struct mp_tensor_t_inner<MP<T>> {
  using type = T;
};

template <typename T>
using mp_tensor_t = typename mp_tensor_t_inner<T>::type;

template <typename Tensor>
inline index_t largest_bond_dimension(const MP<Tensor> &mp) {
  index_t output = 0;
  for (auto &t : mp) {
    output = std::max(output, t.dimension(0));
  }
  return output;
}

template <typename dest, typename orig>
inline dest tensor_cast(const MP<dest> & /*mp*/, orig t) {
  return dest(t);
}

template <>
inline RTensor tensor_cast(const MP<RTensor> & /*mp*/, CTensor data) {
  tensor_assert2(std::all_of(std::begin(data), std::end(data),
                             [](const cdouble &z) { return z.imag() == 0; }),
                 std::domain_error("Cannot convert complex tensor to real."));
  return real(data);
}

extern template RTensor tensor_cast(const MP<RTensor> &mp, CTensor t);

}  // namespace mps

#endif  //!MPS_MP_BASE_H
