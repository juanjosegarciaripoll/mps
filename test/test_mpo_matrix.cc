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

#include "loops.h"
#include <gtest/gtest.h>
#include <mps/mpo.h>
#include <mps/io.h>
#include <mps/quantum.h>

namespace tensor_test {

using namespace mps;
using namespace tensor;

////////////////////////////////////////////////////////////
// ZERO MPO
//

template <class MPO>
void test_zero_mpo(int size) {
  typedef typename MPO::elt_t Tensor;

  Tensor first = Tensor::zeros(1, 2, 2, 2);
  first.at(0, 0, 0, 0) = 1.0;
  first.at(0, 1, 1, 0) = 1.0;

  Tensor last = Tensor::zeros(2, 2, 2, 1);
  last.at(1, 0, 0, 0) = 1.0;
  last.at(1, 1, 1, 0) = 1.0;

  Tensor middle = Tensor::zeros(2, 2, 2, 2);
  middle.at(0, 0, 0, 0) = 1.0;
  middle.at(0, 1, 1, 0) = 1.0;
  middle.at(1, 0, 0, 1) = 1.0;
  middle.at(1, 1, 1, 1) = 1.0;

  MPO mpo(size, 2);
  for (int i = 0; i < size; i++) {
    Tensor target;
    if (i == 0)
      target = first;
    else if (i + 1 == size)
      target = last;
    else
      target = middle;
    EXPECT_TRUE(all_equal(mpo[i], target));
  }
  EXPECT_CEQ(mpo_to_matrix(mpo), Tensor::zeros(1 << size, 1 << size));
}

////////////////////////////////////////////////////////////
// EXPLICIT CONSTRUCTION OF MPOS AND RESULTING MATRICES
//

//
// RMPO
//

TEST(RMPO, Zero) { test_over_integers(2, 5, test_zero_mpo<RMPO>); }

TEST(RMPO, LocalMatrix2x2) {
  RTensor x = real(mps::Pauli_x);
  RTensor z = real(mps::Pauli_z);
  RTensor i2 = real(mps::Pauli_id);
  RMPO mpo(2, 2);
  add_local_term(&mpo, z, 0);
  add_local_term(&mpo, x, 1);
  EXPECT_CEQ(mpo_to_matrix(mpo), kron2(z, i2) + kron2(i2, x));
}

TEST(RMPO, LocalMatrix3x2) {
  RTensor x = real(mps::Pauli_x);
  RTensor z = real(mps::Pauli_z);
  RTensor i2 = real(mps::Pauli_id);
  RMPO mpo(3, 2);
  add_local_term(&mpo, z, 0);
  add_local_term(&mpo, x, 1);
  add_local_term(&mpo, x + z, 2);
  EXPECT_CEQ(mpo_to_matrix(mpo), kron2(z, kron2(i2, i2)) +
                                     kron2(i2, kron2(x, i2)) +
                                     kron2(i2, kron2(i2, x + z)));
}

TEST(RMPO, IntMatrix2x2) {
  RTensor x = real(mps::Pauli_x);
  RTensor z = real(mps::Pauli_z);
  RMPO mpo(2, 2);
  add_interaction(&mpo, z, 0, x);
  add_interaction(&mpo, x, 0, z + x);
  EXPECT_CEQ(mpo_to_matrix(mpo), kron2(z, x) + kron2(x, z + x));
}

TEST(RMPO, IntMatrix3x2) {
  RTensor x = real(mps::Pauli_x);
  RTensor z = real(mps::Pauli_z);
  RTensor i2 = real(mps::Pauli_id);
  {
    RMPO mpo(3, 2);
    add_interaction(&mpo, z, 0, x);
    EXPECT_CEQ(mpo_to_matrix(mpo), kron2(z, kron2(x, i2)));
  }
  {
    RMPO mpo(3, 2);
    add_interaction(&mpo, z, 0, x);
    add_interaction(&mpo, x, 1, z);
    EXPECT_CEQ(mpo_to_matrix(mpo),
               kron2(z, kron2(x, i2)) + kron2(i2, kron2(x, z)));
  }
}

//
// CMPO
//

TEST(CMPO, Zero) { test_over_integers(2, 5, test_zero_mpo<CMPO>); }

TEST(CMPO, LocalMatrix2x2) {
  CTensor x = mps::Pauli_x;
  CTensor z = mps::Pauli_z;
  CTensor i2 = mps::Pauli_id;
  CMPO mpo(2, 2);
  add_local_term(&mpo, z, 0);
  add_local_term(&mpo, x, 1);
  EXPECT_CEQ(mpo_to_matrix(mpo), kron2(z, i2) + kron2(i2, x));
}

TEST(CMPO, LocalMatrix3x2) {
  CTensor x = mps::Pauli_x;
  CTensor z = mps::Pauli_z;
  CTensor i2 = mps::Pauli_id;
  CMPO mpo(3, 2);
  add_local_term(&mpo, z, 0);
  add_local_term(&mpo, x, 1);
  add_local_term(&mpo, x + z, 2);
  EXPECT_CEQ(mpo_to_matrix(mpo), kron2(z, kron2(i2, i2)) +
                                     kron2(i2, kron2(x, i2)) +
                                     kron2(i2, kron2(i2, x + z)));
}

TEST(CMPO, IntMatrix2x2) {
  CTensor x = mps::Pauli_x;
  CTensor z = mps::Pauli_z;
  CMPO mpo(2, 2);
  add_interaction(&mpo, z, 0, x);
  add_interaction(&mpo, x, 0, z + x);
  EXPECT_CEQ(mpo_to_matrix(mpo), kron2(z, x) + kron2(x, z + x));
}

TEST(CMPO, IntMatrix3x2) {
  CTensor x = mps::Pauli_x;
  CTensor z = mps::Pauli_z;
  CTensor i2 = mps::Pauli_id;
  {
    CMPO mpo(3, 2);
    add_interaction(&mpo, z, 0, x);
    EXPECT_CEQ(mpo_to_matrix(mpo), kron2(z, kron2(x, i2)));
  }
  {
    CMPO mpo(3, 2);
    add_interaction(&mpo, z, 0, x);
    add_interaction(&mpo, x, 1, z);
    EXPECT_CEQ(mpo_to_matrix(mpo),
               kron2(z, kron2(x, i2)) + kron2(i2, kron2(x, z)));
  }
}

}  // namespace tensor_test
