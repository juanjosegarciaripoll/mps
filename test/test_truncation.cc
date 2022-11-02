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
#include <tensor/tensor.h>
#include <mps/tools.h>

namespace tensor_test {

using mps::where_to_truncate;

/*
 * Algorithms for truncating singular values in tensor decompositions
 */

TEST(SingularValueTruncation, LargeToleranceDisablesTruncation) {
  EXPECT_EQ(where_to_truncate({1e-3}, 1 /*tol*/, 10), 1);
  EXPECT_EQ(where_to_truncate({1e-3, 0.0}, 1 /*tol*/, 10), 2);
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 0.0}, 1 /*tol*/, 10), 3);
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 1e-17, 0.0}, 1 /*tol*/, 10), 4);
}

TEST(SingularValueTruncation, ZeroToleranceEliminatesZeros) {
  EXPECT_EQ(where_to_truncate({0.0}, 0 /*tol*/, 10), 0);
  EXPECT_EQ(where_to_truncate({1e-3}, 0 /*tol*/, 10), 1);
  EXPECT_EQ(where_to_truncate({1e-3, 0.0}, 0 /*tol*/, 10), 1);
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 0.0}, 0 /*tol*/, 10), 2);
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 1e-17, 0.0}, 0 /*tol*/, 10), 3);
}

TEST(SingularValueTruncation, EliminatesValuesBelowTolerance) {
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 1e-6, 0.0}, 0.99999 /*tol*/, 10), 1);
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 1e-6, 0.0}, 1e-2 /*tol*/, 10), 1);
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 1e-6, 0.0}, 1e-10 /*tol*/, 10), 2);
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 1e-6, 0.0}, 1e-16 /*tol*/, 10), 3);
}

/*
 * Algorithms for selecting eigenvalues in other contexts
 */

using mps::weights_to_keep;

TEST(WeightsTruncation, LargeToleranceDisablesTruncation) {
  EXPECT_ALL_EQUAL(weights_to_keep({1e-3}, 1 /*tol*/, 10), (Indices{0}));
  EXPECT_ALL_EQUAL(weights_to_keep({1e-3, 0.0}, 1 /*tol*/, 10),
                   (Indices{0, 1}));
  EXPECT_ALL_EQUAL(weights_to_keep({0.0, 1e-3}, 1 /*tol*/, 10),
                   (Indices{1, 0}));
  EXPECT_ALL_EQUAL(weights_to_keep({1.0, 1e-3, 0.0}, 1 /*tol*/, 10),
                   (Indices{0, 1, 2}));
  EXPECT_ALL_EQUAL(weights_to_keep({1.0, 0.0, 1e-3, 1e-17}, 1 /*tol*/, 10),
                   (Indices{0, 2, 3, 1}));
}

TEST(WeightsTruncation, ZeroToleranceEliminatesZeros) {
  EXPECT_ALL_EQUAL(weights_to_keep({0.0}, 0 /*tol*/, 10), (Indices{}));
  EXPECT_ALL_EQUAL(weights_to_keep({1e-3}, 0 /*tol*/, 10), (Indices{0}));
  EXPECT_ALL_EQUAL(weights_to_keep({1e-3, 0.0}, 0 /*tol*/, 10), (Indices{0}));
  EXPECT_ALL_EQUAL(weights_to_keep({1.0, 1e-3, 0.0}, 0 /*tol*/, 10),
                   (Indices{0, 1}));
  EXPECT_ALL_EQUAL(weights_to_keep({1.0, 0.0, 1e-3}, 0 /*tol*/, 10),
                   (Indices{0, 2}));
  EXPECT_ALL_EQUAL(weights_to_keep({1.0, 1e-3, 1e-17, 0.0}, 0 /*tol*/, 10),
                   (Indices{0, 1, 2}));
  EXPECT_ALL_EQUAL(weights_to_keep({1e-17, 1.0, 1e-3, 0.0}, 0 /*tol*/, 10),
                   (Indices{1, 2, 0}));
}

TEST(WeightsTruncation, EliminatesValuesBelowTolerance) {
  EXPECT_ALL_EQUAL(weights_to_keep({1.0, 1e-3, 1e-6, 0.0}, 0.99999 /*tol*/, 10),
                   (Indices{0}));
  EXPECT_ALL_EQUAL(weights_to_keep({1.0, 1e-3, 1e-6, 0.0}, 1e-2 /*tol*/, 10),
                   (Indices{0}));
  EXPECT_ALL_EQUAL(weights_to_keep({0.0, 1e-3, 1.0, 1e-6}, 1e-2 /*tol*/, 10),
                   (Indices{2}));
  EXPECT_ALL_EQUAL(weights_to_keep({1.0, 1e-3, 1e-6, 0.0}, 1e-10 /*tol*/, 10),
                   (Indices{0, 1}));
  EXPECT_ALL_EQUAL(weights_to_keep({1e-6, 0.0, 1.0, 1e-3}, 1e-10 /*tol*/, 10),
                   (Indices{2, 3}));
  EXPECT_ALL_EQUAL(weights_to_keep({1.0, 1e-3, 1e-6, 0.0}, 1e-16 /*tol*/, 10),
                   (Indices{0, 1, 2}));
  EXPECT_ALL_EQUAL(weights_to_keep({1e-6, 0.0, 1.0, 1e-3}, 1e-16 /*tol*/, 10),
                   (Indices{2, 3, 0}));
}

}  // namespace tensor_test