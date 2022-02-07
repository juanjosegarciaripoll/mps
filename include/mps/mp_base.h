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

#include <vector>
#include <tensor/tensor.h>

namespace mps {

using tensor::index;

class Sweeper {
 public:
  Sweeper(index L, index sense);
  index operator*() const { return k_; };
  bool operator--();
  bool is_last() const { return k_ == kN_; };
  index sense() const { return dk_; };
  index site() const { return k_; };
  void flip();

 private:
  index k_, k0_, kN_, dk_;
};

template <class Tensor>
class MP {
  typedef typename std::vector<Tensor> data_type;

 public:
  typedef Tensor elt_t;
  typedef typename Tensor::elt_t number_t;
  typedef typename data_type::iterator iterator;
  typedef typename data_type::const_iterator const_iterator;

  MP() = default;
  explicit MP(size_t size) : data_(size) {}
  MP(const std::vector<Tensor> &other) : data_(other) {}

  size_t ssize() const { return data_.size(); }
  index ssize() const { return static_cast<index>(size()); }
  index last() const { return size() - 1; }
  void resize(index new_size) { data_.resize(new_size); }

  const Tensor &operator[](index n) const {
    assert((n >= 0) && (n < size()));
    return data_[n];
  }
  Tensor &at(index n) {
    assert((n >= 0) && (n < size()));
    return data_.at(n);
  }

  iterator begin() { return data_.begin(); }
  const_iterator begin() const { return data_.begin(); }
  const_iterator end() const { return data_.end(); }
  iterator end() { return data_.end(); }
  const std::vector<Tensor> to_vector() const { return data_; }

  Sweeper sweeper(index sense) const { return Sweeper(size(), sense); }

 private:
  data_type data_;
};

index largest_bond_dimension(const MP<tensor::RTensor> &mp);
index largest_bond_dimension(const MP<tensor::CTensor> &mp);

}  // namespace mps

#endif  //!MPS_MP_BASE_H
