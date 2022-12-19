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

//----------------------------------------------------------------------
// TRANSLATIONALLY INVARIANT HAMILTONIAN
//

#include <mps/imath.h>
#include <mps/hamiltonian.h>

namespace mps {

using mps::imath::isqrt;

/**Create a translationally invariant Hamiltonian. 'N' is the number of lattice
   * sites, 'H12' is the nearest neighbor interaction between every two sites,
   * 'H1' is the local term and 'periodic' determines whether there is also
   * interaction between sites 1 and N. */
TIHamiltonian::TIHamiltonian(index_t N, const CTensor &H12, const CTensor &H1,
                             bool periodic)
    : size_(N), H12_(H12), H1_(H1), periodic_(periodic) {
  if (H1.is_empty()) {
    if (H12.is_empty()) {
      std::cerr << "In TIHamiltonian(). You have to provide at least a local "
                   "term or an interaction.\n";
      abort();
    } else {
      index_t d = isqrt(H12.rows());
      H1_ = RTensor::zeros(d, d);
    }
  }
  if (H12.is_empty()) {
    index_t d = H1_.rows();
    H12_ = RTensor::zeros(d * d, d * d);
  } else if (norm2(H12)) {
    split_interaction(H12_, &H12_left_, &H12_right_);
  }
}

std::unique_ptr<const Hamiltonian> TIHamiltonian::duplicate() const {
  return std::unique_ptr<const Hamiltonian>(new TIHamiltonian(*this));
}

index_t TIHamiltonian::size() const { return size_; }

bool TIHamiltonian::is_constant() const { return 1; }

bool TIHamiltonian::is_periodic() const { return periodic_; }

const CTensor TIHamiltonian::interaction(index_t k, double /*t*/) const {
  if (k + 1 < size_) {
    return H12_;
  } else {
    return CTensor();
  }
}

const CTensor TIHamiltonian::interaction_left(index_t k, index_t ndx,
                                              double /*t*/) const {
  if (k + 1 < size_) {
    return H12_left_[ndx];
  } else {
    return CTensor();
  }
}

const CTensor TIHamiltonian::interaction_right(index_t k, index_t ndx,
                                               double /*t*/) const {
  if (k + 1 < size_) {
    return H12_right_[ndx];
  } else {
    return CTensor();
  }
}

index_t TIHamiltonian::interaction_depth(index_t /*k*/, double /*t*/) const {
  return H12_left_.ssize();
}

const CTensor TIHamiltonian::local_term(index_t /*k*/, double /*t*/) const {
  return H1_;
}

}  // namespace mps
