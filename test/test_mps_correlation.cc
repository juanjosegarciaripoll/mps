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
#include "test_states.h"
#include <gtest/gtest.h>
#include <mps/except.h>
#include <mps/mps.h>
#include <mps/quantum.h>

namespace tensor_test {

using namespace tensor;
using namespace mps;

template <class MPS>
void test_correlation_basic() {
  typename MPS::elt_t e0 = RTensor({2}, {1.0, 0.0});
  typename MPS::elt_t e1 = RTensor({2}, {0.0, 1.0});
  {
    // A product state with two vectors
    MPS psi = product_state(3, e0);
    psi.at(1) = reshape(e1, 1, 2, 1);
    psi.at(2) = reshape((e1 + e0) / sqrt(2), 1, 2, 1);

    EXPECT_ERROR_DETECTED(expected(psi, mps::Pauli_id, 3, mps::Pauli_id, 0),
                          mps_out_of_range);
    EXPECT_ERROR_DETECTED(expected(psi, mps::Pauli_id, -4, mps::Pauli_id, 0),
                          mps_out_of_range);
    EXPECT_ERROR_DETECTED(expected(psi, mps::Pauli_id, 0, mps::Pauli_id, 3),
                          mps_out_of_range);
    EXPECT_ERROR_DETECTED(expected(psi, mps::Pauli_id, 0, mps::Pauli_id, -4),
                          mps_out_of_range);

    EXPECT_CEQ(expected(psi, mps::Pauli_z, 0, mps::Pauli_z, 1), -1.0);
    EXPECT_CEQ(expected(psi, mps::Pauli_z, 0, mps::Pauli_z, 2), 0.0);
    EXPECT_CEQ(expected(psi, mps::Pauli_z, 0, mps::Pauli_x, 2), 1.0);
    EXPECT_CEQ(expected(psi, mps::Pauli_z, 1, mps::Pauli_x, 2), -1.0);
    EXPECT_CEQ(expected(psi, mps::Pauli_z, 1, mps::Pauli_z, 2), 0.0);

    EXPECT_CEQ(expected(psi, mps::Pauli_z, -3, mps::Pauli_z, 1), -1.0);
    EXPECT_CEQ(expected(psi, mps::Pauli_z, -3, mps::Pauli_z, 2), 0.0);
    EXPECT_CEQ(expected(psi, mps::Pauli_z, -3, mps::Pauli_x, 2), 1.0);
    EXPECT_CEQ(expected(psi, mps::Pauli_z, -2, mps::Pauli_x, 2), -1.0);

    EXPECT_CEQ(expected(psi, mps::Pauli_z, 0, mps::Pauli_z, -2), -1.0);
    EXPECT_CEQ(expected(psi, mps::Pauli_z, 0, mps::Pauli_z, -1), 0.0);
    EXPECT_CEQ(expected(psi, mps::Pauli_z, 0, mps::Pauli_x, -1), 1.0);
    EXPECT_CEQ(expected(psi, mps::Pauli_z, 1, mps::Pauli_x, -1), -1.0);

    // When we repeat the site, the operators are multiplied
    EXPECT_CEQ(expected(psi, mps::Pauli_x, 0, mps::Pauli_x, 0), 1.0);
    EXPECT_CEQ(expected(psi, mps::Pauli_z, 1, mps::Pauli_z, 1), 1.0);
    EXPECT_CEQ(expected(psi, mps::Pauli_z, 2, mps::Pauli_z, 2), 1.0);
  }
}

template <class Tensor>
Tensor product_state_correlations(const vector<Tensor> &states,
                                  const Tensor &op1, const Tensor &op2) {
  index L = ssize(states);
  auto output = Tensor::empty(L, L);
  for (index i = 0; i < L; ++i) {
    auto factor = scprod(states[i], mmult(op1, states[i]));
    for (index j = 0; j < L; ++j) {
      if (j != i) {
        output.at(i, j) = factor * scprod(states[j], mmult(op2, states[j]));
      } else {
        output.at(i, i) = scprod(states[i], mmult(op1, mmult(op2, states[i])));
      }
    }
  }
  return output;
}

template <class MPS>
void test_correlation_order(int size) {
  /*
     * We create a random product state and verify that the
     * expectation value over the k-th site is the same as
     * that of the single-site operator on the associated state.
     */
  auto states = random_product_state<mp_tensor_t<MPS>>(size);
  mp_tensor_t<MPS> op1 = mps::Pauli_z;
  mp_tensor_t<MPS> op2 = op1 + 0.1 * op1.eye(2);
  auto exact_correlations = product_state_correlations(states, op1, op2);
  MPS psi = product_state(states);
  auto correlations = exact_correlations.zeros(size, size);
  for (index i = 0; i < size; i++)
    for (index j = 0; j < size; j++)
      correlations.at(i, j) = expected(psi, op1, i, op2, j);
  ASSERT_CEQ(correlations, exact_correlations);
}

template <class MPS>
void test_fast_correlations(int size) {
  /*
     * We create a random product state and verify that the
     * expectation value over the k-th site is the same as
     * that of the single-site operator on the associated state.
     */
  auto states = random_product_state<mp_tensor_t<MPS>>(size);
  mp_tensor_t<MPS> op1 = mps::Pauli_z;
  mp_tensor_t<MPS> op2 = op1 + 0.1 * op1.eye(2);
  auto exact_correlations = product_state_correlations(states, op1, op2);
  MPS psi = product_state(states);
  auto correlations = expected(psi, op1, op2);
  ASSERT_CEQ(correlations, exact_correlations);
}

////////////////////////////////////////////////////////////
// EXPECTATION VALUES OVER RMPS
//

TEST(RMPSCorrelation, Basic) { test_correlation_basic<RMPS>(); }

TEST(RMPSCorrelation, Order) {
  test_over_integers(1, 10, test_correlation_order<RMPS>);
}

TEST(RMPSCorrelation, GHZ) {
  for (index i = 1; i < 4; i++) {
    RMPS ghz = ghz_state(i);
    for (index j = 0; j < i; j++) {
      for (index k = 0; k < i; k++) {
        EXPECT_DOUBLE_EQ(expected(ghz, mps::Pauli_z, j, mps::Pauli_z, k), 1.0);
        EXPECT_DOUBLE_EQ(expected(ghz, mps::Pauli_x, j, mps::Pauli_z, k), 0.0);
      }
    }
  }
}

TEST(RMPSCorrelation, FastCorrelations) {
  test_over_integers(1, 10, test_fast_correlations<RMPS>);
}

////////////////////////////////////////////////////////////
// EXPECTATION VALUES OVER CMPS
//

TEST(CMPSCorrelation, Basic) { test_correlation_basic<CMPS>(); }

TEST(CMPSCorrelation, Order) {
  test_over_integers(1, 10, test_correlation_order<CMPS>);
}

TEST(CMPSCorrelation, FastCorrelations) {
  test_over_integers(1, 10, test_fast_correlations<CMPS>);
}

}  // namespace tensor_test
