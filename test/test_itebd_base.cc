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
#include <mps/itebd.h>

namespace tensor_test {

using namespace mps;
using namespace tensor;

template <class t>
void test_random_iTEBD() {
  ASSERT_ERROR_DETECTED(iTEBD<t>(0));
  for (int d = 1; d <= 3; d++) {
    iTEBD<t> psi(d);
    t A = psi.matrix(0);
    t B = psi.matrix(1);
    t lA = psi.right_vector(0);
    t lB = psi.right_vector(1);
    EXPECT_ALL_EQUAL(A.dimensions(), Dimensions({1, d, 1}));
    EXPECT_ALL_EQUAL(B.dimensions(), Dimensions({1, d, 1}));
    EXPECT_ALL_EQUAL(lA, t({1}, {1}));
    EXPECT_ALL_EQUAL(lB, lA);
  }
}

template <class t>
void test_product_iTEBD() {
  /* Die if not a vector */
  ASSERT_ERROR_DETECTED(iTEBD<t>(t::empty(1, 1)));
  for (int d = 1; d < 3; d++) {
    auto A = t::random(d);
    A = A / norm2(A);
    /* This creates a product state |A>|A>|A>... */
    iTEBD<t> psi(A);
    t lA = psi.right_vector(0);
    t lB = psi.right_vector(1);
    EXPECT_TRUE(approx_eq(psi.matrix(0), reshape(A, 1, d, 1)));
    EXPECT_TRUE(approx_eq(psi.matrix(1), reshape(A, 1, d, 1)));
    EXPECT_ALL_EQUAL(lA, t({1}, {1}));
    EXPECT_ALL_EQUAL(lB, lA);
    EXPECT_ALL_EQUAL(psi.matrix(0), psi.matrix(1));
  }
}

template <class t>
void test_product_alternated_iTEBD() {
  /* Die if not a vector */
  ASSERT_ERROR_DETECTED(iTEBD<t>(t::empty(1, 1), t::empty(2)));
  ASSERT_ERROR_DETECTED(iTEBD<t>(t::empty(2), t::empty(1, 1)));
  for (int d = 1; d < 3; d++) {
    auto A = t::random(d);
    A = A / norm2(A);
    auto B = t::random(d);
    B = B / norm2(B);
    /* This creates a product state |A>|B>|A>... */
    iTEBD<t> psi(A, B);
    t lA = psi.right_vector(0);
    t lB = psi.right_vector(1);
    EXPECT_TRUE(approx_eq(psi.matrix(0), reshape(A, 1, d, 1)));
    EXPECT_TRUE(approx_eq(psi.matrix(1), reshape(B, 1, d, 1)));
    EXPECT_ALL_EQUAL(lA, t({1}, {1}));
    EXPECT_ALL_EQUAL(lB, lA);
    EXPECT_ALL_EQUAL(psi.matrix(0), psi.matrix(2));
    EXPECT_ALL_EQUAL(psi.matrix(0), psi.matrix(-2));
    EXPECT_ALL_EQUAL(psi.matrix(1), psi.matrix(3));
    EXPECT_ALL_EQUAL(psi.matrix(1), psi.matrix(-1));
  }
}

template <class t>
void test_small_canonical_iTEBD() {
  t A = RTensor({2, 3, 2},
                {-0.95912, -0.29373, -0.32075, 0.82915, -0.053633, 0.29376,
                 -0.29375, -0.070285, 0.82919, 0.42038, 0.29377, -1.2573});
  t lA = RTensor({2}, {0.91919, 0.39382});
  t B = RTensor({2, 3, 2},
                {-0.05363, 0.29376, 0.32074, -0.82919, -0.95913, -0.29374,
                 0.29375, -1.2573, -0.82915, -0.42036, -0.29373, -0.070279});
  t lB = RTensor({2}, {0.91923, 0.39373});
  t H12 = RTensor(
      {9, 9}, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,  0, -1,
               0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,  0, 0,
               1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0,
               0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1});

  iTEBD<t> psi(A, lA, B, lB);
  iTEBD<t> psic = psi.canonical_form();
  EXPECT_CEQ3(expected12(psi, H12), expected12(psic, H12), 2e-6);
}

////////////////////////////////////////////////////////////
/// ITEBD WITH REAL TENSORS
///

TEST(RiTEBDTest, RandomProductState) { test_random_iTEBD<RTensor>(); }

TEST(RiTEBDTest, ProductState) { test_product_iTEBD<RTensor>(); }

TEST(RiTEBDTest, AlternatedProductState) {
  test_product_alternated_iTEBD<RTensor>();
}

////////////////////////////////////////////////////////////
/// ITEBD WITH COMPLEX TENSORS
///

TEST(CiTEBDTest, RandomProductState) { test_random_iTEBD<CTensor>(); }

TEST(CiTEBDTest, ProductState) { test_product_iTEBD<CTensor>(); }

TEST(CiTEBDTest, AlternatedProductState) {
  test_product_alternated_iTEBD<CTensor>();
}

}  // namespace tensor_test
