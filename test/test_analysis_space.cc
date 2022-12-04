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
#include <mps/quantum.h>
#include <mps/analysis.h>

namespace tensor_test {

using namespace mps;
using namespace tensor;

TEST(Space, OneDimension) {
  double start = 0.0, end = 1.0;
  index_t qubits = 2;
  Space s({{start, end, qubits}});

  EXPECT_EQ(s.dimensions(), 1);
  EXPECT_EQ(s.total_qubits(), 2);

  EXPECT_EQ(s.dimension_size(0), 4);
  EXPECT_EQ(s.dimension_start(0), start);
  EXPECT_EQ(s.dimension_end(0), end);

  EXPECT_ALL_EQUAL(s.dimension_qubits(0), Indices({0, 1}));

  ASSERT_ERROR_DETECTED(s.dimension_start(-1));
  ASSERT_ERROR_DETECTED(s.dimension_start(2));
}

TEST(Space, TwoDimensions) {
  double start1 = 0.0, end1 = 1.0;
  index_t qubits1 = 2;
  double start2 = 0.0, end2 = 1.0;
  index_t qubits2 = 3;
  Space s({{start1, end1, qubits1}, {start2, end2, qubits2}});

  EXPECT_EQ(s.dimensions(), 2);
  EXPECT_EQ(s.total_qubits(), qubits1 + qubits2);

  EXPECT_EQ(s.dimension_start(0), start1);
  EXPECT_EQ(s.dimension_end(0), end1);
  EXPECT_EQ(s.dimension_size(0), 4);
  EXPECT_ALL_EQUAL(s.dimension_qubits(0), Indices({0, 1}));

  EXPECT_EQ(s.dimension_start(1), start2);
  EXPECT_EQ(s.dimension_end(1), end2);
  EXPECT_EQ(s.dimension_size(1), 8);
  EXPECT_ALL_EQUAL(s.dimension_qubits(1), Indices({2, 3, 4}));

  ASSERT_ERROR_DETECTED(s.dimension_start(2));
}

TEST(Space, ExtendMatrix) {
  double start = 0.0, end = 0.0;
  index_t qubits = 1;
  Space s({{start, end, qubits}, {start, end, qubits}});

  auto sigma_z = RSparse(Pauli_z);
  auto id = RSparse::eye(2);
  EXPECT_ALL_EQUAL(kron(sigma_z, id), s.extend_matrix(sigma_z, 0));
  EXPECT_ALL_EQUAL(kron(id, sigma_z), s.extend_matrix(sigma_z, 1));
}

TEST(Space, ExtendMPO) {
  double start = 0.0, end = 0.0;
  index_t qubits = 1;
  Space s({{start, end, qubits}, {start, end, qubits}});

  RTensor sigma_x_tensor = reshape(Pauli_x, 1, 2, 2, 1);
  RTensor identity_tensor = reshape(RTensor::eye(2), 1, 2, 2, 1);
  EXPECT_ALL_EQUAL(RMPO({sigma_x_tensor, identity_tensor}),
                   s.extend_mpo(RMPO({sigma_x_tensor}), 0));
  EXPECT_ALL_EQUAL(RMPO({identity_tensor, sigma_x_tensor}),
                   s.extend_mpo(RMPO({sigma_x_tensor}), 1));
}

}  // namespace tensor_test
