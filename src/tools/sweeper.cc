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

#include <algorithm>
#include <mps/mp_base.h>

namespace mps {

Sweeper::Sweeper(index L, index sense) {
  tensor_assert(L > 0);
  if (sense > 0) {
    k_ = k0_ = 0;
    kN_ = L - 1;
    dk_ = +1;
  } else {
    k_ = k0_ = L - 1;
    kN_ = 0;
    dk_ = -1;
  }
}

bool Sweeper::operator--() {
  if (k_ == kN_) {
    return false;
  } else {
    k_ += dk_;
    return true;
  }
}

void Sweeper::flip() {
  std::swap(k0_, kN_);
  dk_ = -dk_;
  k_ = k0_;
}

}  // namespace mps
