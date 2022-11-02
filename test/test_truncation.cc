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

TEST(Truncation, LargeToleranceDisablesTruncation) {
  EXPECT_EQ(where_to_truncate({1e-3}, 1 /*tol*/, 10), 1);
  EXPECT_EQ(where_to_truncate({1e-3, 0.0}, 1 /*tol*/, 10), 2);
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 0.0}, 1 /*tol*/, 10), 3);
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 1e-17, 0.0}, 1 /*tol*/, 10), 4);
}

TEST(Truncation, ZeroToleranceEliminatesZeros) {
  EXPECT_EQ(where_to_truncate({0.0}, 0 /*tol*/, 10), 0);
  EXPECT_EQ(where_to_truncate({1e-3}, 0 /*tol*/, 10), 1);
  EXPECT_EQ(where_to_truncate({1e-3, 0.0}, 0 /*tol*/, 10), 1);
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 0.0}, 0 /*tol*/, 10), 2);
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 1e-17, 0.0}, 0 /*tol*/, 10), 3);
}

TEST(Truncation, EliminatesValuesBelowTolerance) {
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 1e-6, 0.0}, 0.99999 /*tol*/, 10), 1);
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 1e-6, 0.0}, 1e-2 /*tol*/, 10), 1);
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 1e-6, 0.0}, 1e-10 /*tol*/, 10), 2);
  EXPECT_EQ(where_to_truncate({1.0, 1e-3, 1e-6, 0.0}, 1e-16 /*tol*/, 10), 3);
}

}  // namespace tensor_test