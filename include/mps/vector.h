#pragma once
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

#ifndef MPS_VECTOR_H
#define MPS_VECTOR_H

#include <vector>
#include <tensor/numbers.h>

namespace mps {

using index_t = tensor::index_t;
using index = const char *;

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

template <class elt>
class vector : public std::vector<elt> {
  using super_t = std::vector<elt>;

 public:
  vector() = default;
  vector(const std::initializer_list<elt> &l) : super_t(l) {}
  vector(const vector<elt> &v) : super_t(v) {}
  vector(vector<elt> &&v) : super_t(std::move(v)){};
  vector(const std::vector<elt> &v) : super_t(v) {}
  vector(std::vector<elt> &&v) : super_t(std::move(v)){};
  vector &operator=(const vector<elt> &v) = default;
  vector &operator=(vector<elt> &&v) = default;
  ~vector() = default;

  explicit vector(index_t size) : super_t(static_cast<size_t>(size)) {}
  explicit vector(index_t size, elt value)
      : super_t(static_cast<size_t>(size), value) {}

  elt &operator[](index_t pos) {
    return super_t::operator[](static_cast<size_t>(pos));
  }
  elt &at(index_t pos) { return super_t::at(static_cast<size_t>(pos)); }
  const elt &operator[](index_t pos) const {
    return super_t::operator[](static_cast<size_t>(pos));
  }
  const elt &at(index_t pos) const {
    return super_t::at(static_cast<size_t>(pos));
  }

  index_t ssize() const { return static_cast<index_t>(super_t::size()); }
  void resize(index_t new_size) {
    return super_t::resize(static_cast<size_t>(new_size));
  }
  void resize(index_t new_size, const elt &value) {
    return super_t::resize(static_cast<size_t>(new_size), value);
  }
  void reserve(index_t new_size) {
    return super_t::reserve(static_cast<size_t>(new_size));
  }
};

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

}  // namespace mps

#endif  // MPS_VECTOR_H
