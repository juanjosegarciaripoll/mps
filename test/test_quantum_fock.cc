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
#include <tensor/io.h>
#include <mps/quantum.h>
#include <gtest/gtest.h>

using namespace mps;

TEST(Fock, Number) {
  EXPECT_ALL_EQUAL(number_operator(0), RTensor::zeros(1, 1));
  EXPECT_ALL_EQUAL(number_operator(1), RTensor({{0.0, 0.0}, {0.0, 1.0}}));
  EXPECT_ALL_EQUAL(
      number_operator(2),
      RTensor({{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 2.0}}));
}

TEST(Fock, Destruction) {
  EXPECT_ALL_EQUAL(destruction_operator(0), RTensor::zeros(1, 1));
  EXPECT_ALL_EQUAL(destruction_operator(1), RTensor({{0.0, 1.0}, {0.0, 0.0}}));
  EXPECT_ALL_EQUAL(
      destruction_operator(2),
      RTensor({{0.0, 1.0, 0.0}, {0.0, 0.0, sqrt(2.0)}, {0.0, 0.0, 0.0}}));
}

TEST(Fock, Creation) {
  EXPECT_ALL_EQUAL(creation_operator(0), RTensor::zeros(1, 1));
  EXPECT_ALL_EQUAL(creation_operator(1), RTensor({{0.0, 0.0}, {1.0, 0.0}}));
  EXPECT_ALL_EQUAL(
      creation_operator(2),
      RTensor({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, sqrt(2.0), 0.0}}));
}
