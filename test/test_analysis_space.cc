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

  EXPECT_ALL_EQUAL(s.tensor_dimensions(), Indices({4, 8}));

  ASSERT_ERROR_DETECTED(s.dimension_start(2));
}

TEST(Space, TensorDimensions) {
  {
    Space space({{0.0, 1.0, 2}});
    EXPECT_ALL_EQUAL(space.tensor_dimensions(), Indices({4}));
  }
  {
    Space space({{0.0, 1.0, 2}, {0.0, 1.0, 3}});
    EXPECT_ALL_EQUAL(space.tensor_dimensions(), Indices({4, 8}));
  }
}

TEST(Space, ExtendMatrix) {
  double start = 0.0, end = 0.0;
  index_t qubits = 1;
  Space s({{start, end, qubits}, {start, end, qubits}});

  auto sigma_z = RSparse(Pauli_z);
  auto id = RSparse::eye(2);
  EXPECT_ALL_EQUAL(kron2(sigma_z, id), s.extend_matrix(sigma_z, 0));
  EXPECT_ALL_EQUAL(kron2(id, sigma_z), s.extend_matrix(sigma_z, 1));
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

TEST(Space, OneDimensionalPositionMPO) {
  double start = 1.0, end = 2.0;
  index_t qubits = 2;
  Space s({{start, end, qubits}});

  auto mpo = position_mpo(s, 0);
  ASSERT_EQ(mpo.ssize(), s.total_qubits());

  double dx = (end - start) / (1 << qubits);
  EXPECT_ALL_EQUAL(
      mpo_to_matrix(mpo),
      diag(RTensor({start, start + dx, start + 2 * dx, start + 3 * dx})));
}

TEST(Space, TwoDimensionalPositionMPO) {
  double start = 1.0, end = 2.0;
  index_t qubits = 2;
  Space space({{start, end, qubits}, {2 * start, 2 * end, qubits}});

  ASSERT_EQ(position_mpo(space, 0).ssize(), space.total_qubits());
  ASSERT_EQ(position_mpo(space, 1).ssize(), space.total_qubits());

  double dx = (end - start) / (1 << qubits);
  auto X = diag(RTensor({start, start + dx, start + 2 * dx, start + 3 * dx}));
  auto Id = RTensor::eye(4);

  EXPECT_ALL_EQUAL(mpo_to_matrix(position_mpo(space, 0)), kron2(X, Id));
  EXPECT_ALL_EQUAL(mpo_to_matrix(position_mpo(space, 1)), kron2(Id, 2.0 * X));
}

TEST(Space, OneDimensionalDerivativeMPO) {
  double start = 1.0, end = 2.0;
  index_t qubits = 2;
  Space space({{start, end, qubits}});

  auto mpo = first_derivative_mpo(space, 0);
  ASSERT_EQ(mpo.ssize(), space.total_qubits());

  double dx = (end - start) / (1 << qubits);
  RTensor first_order_finite_differences{{0.0, 1 / dx, 0.0, -1 / dx},
                                         {-1 / dx, 0.0, 1 / dx, 0.0},
                                         {0.0, -1 / dx, 0.0, 1 / dx},
                                         {1 / dx, 0.0, -1 / dx, 0.0}};
  EXPECT_ALL_EQUAL(mpo_to_matrix(first_derivative_mpo(space, 0)),
                   first_order_finite_differences);
}

TEST(Space, OneDimensionalSecondDerivativeMPO) {
  double start = 1.0, end = 2.0;
  index_t qubits = 2;
  Space space({{start, end, qubits}});

  auto mpo = second_derivative_mpo(space, 0);
  ASSERT_EQ(mpo.ssize(), space.total_qubits());

  double dx2 = square((end - start) / (1 << qubits));
  RTensor second_finite_differences{{-2 / dx2, 1 / dx2, 0.0, 1 / dx2},
                                    {1 / dx2, -2 / dx2, 1 / dx2, 0.0},
                                    {0.0, 1 / dx2, -2 / dx2, 1 / dx2},
                                    {1 / dx2, 0.0, 1 / dx2, -2 / dx2}};
  EXPECT_ALL_EQUAL(mpo_to_matrix(second_derivative_mpo(space, 0)),
                   second_finite_differences);
}

TEST(Space, OneDimensionalPositionMatrix) {
  double start = 1.0, end = 2.0;
  index_t qubits = 2;
  Space space({{start, end, qubits}});

  double dx = (end - start) / (1 << qubits);
  EXPECT_ALL_EQUAL(
      full(position_matrix(space)),
      diag(RTensor({start, start + dx, start + 2 * dx, start + 3 * dx})));
}

TEST(Space, TwoDimensionalPositionMatrix) {
  double start = 1.0, end = 2.0;
  index_t qubits = 2;
  Space space({{start, end, qubits}, {2 * start, 2 * end, qubits}});

  double dx = (end - start) / (1 << qubits);
  auto X = diag(RTensor({start, start + dx, start + 2 * dx, start + 3 * dx}));
  auto Id = RTensor::eye(4);

  EXPECT_ALL_EQUAL(full(position_matrix(space, 0)), kron2(X, Id));
  EXPECT_ALL_EQUAL(full(position_matrix(space, 1)), kron2(Id, 2.0 * X));
}

TEST(Space, OneDimensionalDerivativeMatrix) {
  double start = 1.0, end = 2.0;
  index_t qubits = 2;
  Space space({{start, end, qubits}});

  double dx = (end - start) / (1 << qubits);
  RTensor first_order_finite_differences{{0.0, 1 / dx, 0.0, -1 / dx},
                                         {-1 / dx, 0.0, 1 / dx, 0.0},
                                         {0.0, -1 / dx, 0.0, 1 / dx},
                                         {1 / dx, 0.0, -1 / dx, 0.0}};
  EXPECT_ALL_EQUAL(full(first_derivative_matrix(space, 0)),
                   first_order_finite_differences);
}

TEST(Space, OneDimensionalSecondDerivativeMatrix) {
  double start = 1.0, end = 2.0;
  index_t qubits = 2;
  Space space({{start, end, qubits}});

  double dx2 = square((end - start) / (1 << qubits));
  RTensor second_finite_differences{{-2 / dx2, 1 / dx2, 0.0, 1 / dx2},
                                    {1 / dx2, -2 / dx2, 1 / dx2, 0.0},
                                    {0.0, 1 / dx2, -2 / dx2, 1 / dx2},
                                    {1 / dx2, 0.0, 1 / dx2, -2 / dx2}};
  EXPECT_ALL_EQUAL(full(second_derivative_matrix(space, 0)),
                   second_finite_differences);
}

}  // namespace tensor_test
