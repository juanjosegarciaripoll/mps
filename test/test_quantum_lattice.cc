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

#include <gtest/gtest.h>
#include "loops.h"
#include <mps/lattice.h>

namespace tensor_test {

using namespace tensor;
using namespace mps;

//////////////////////////////////////////////////////////////////////
// HARD-CORE BOSONS LATTICE
//
// Lattice sites are numbered as bits in a binary number. We construct
// by hand the operators that represent the states and their updates
// due to hopping and interaction.

//
// RATHER TRIVIAL LATTICES
//

TEST(AnyLatticeTest, EmptyLattice) {
  for (int size = 1; size < 16; size++) {
    Lattice L(size, 0);

    RTensor zero = RTensor::zeros(1, 1);
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        EXPECT_ALL_EQUAL(
            zero, L.hopping_operator(j, i, Lattice::HARD_CORE_BOSONS));
        EXPECT_TRUE(
            all_equal(zero, L.hopping_operator(j, i, Lattice::FERMIONS)));
        EXPECT_ALL_EQUAL(zero, L.interaction_operator(j, i));
      }
    }

    for (int i = 0; i < size; i++) {
      EXPECT_ALL_EQUAL(zero, L.number_operator(i));
    }
  }
}

TEST(AnyLatticeTest, FullLattice) {
  for (int size = 1; size < 16; size++) {
    Lattice L(size, size);

    RTensor zero = RTensor::zeros(1, 1);
    RTensor one = RTensor::ones(1, 1);
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        if (i == j) {
          EXPECT_ALL_EQUAL(
              one, L.hopping_operator(j, i, Lattice::HARD_CORE_BOSONS));
          EXPECT_TRUE(
              all_equal(one, L.hopping_operator(j, i, Lattice::FERMIONS)));
        } else {
          EXPECT_ALL_EQUAL(
              zero, L.hopping_operator(j, i, Lattice::HARD_CORE_BOSONS));
          EXPECT_TRUE(
              all_equal(zero, L.hopping_operator(j, i, Lattice::FERMIONS)));
        }
        EXPECT_ALL_EQUAL(one, L.interaction_operator(j, i));
      }
    }

    for (int i = 0; i < size; i++) {
      EXPECT_ALL_EQUAL(one, L.number_operator(i));
    }
  }
}

TEST(AnyLatticeTest, OneParticleLattice) {
  for (int size = 1; size < 16; size++) {
    Lattice L(size, 1);

    RTensor zero = RTensor::zeros(size, size);
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        RTensor hop = zero;
        hop.at(j, i) = 1.0;
        EXPECT_ALL_EQUAL(
            hop, L.hopping_operator(j, i, Lattice::HARD_CORE_BOSONS));
        EXPECT_TRUE(
            all_equal(hop, L.hopping_operator(j, i, Lattice::FERMIONS)));
        RTensor inter = zero;
        inter.at(j, i) = (i == j);
        EXPECT_ALL_EQUAL(inter, L.interaction_operator(j, i));
      }
    }

    for (int i = 0; i < size; i++) {
      RTensor n = zero;
      n.at(i, i) = 1.0;
      EXPECT_ALL_EQUAL(n, L.number_operator(i));
    }
  }
}

TEST(AnyLatticeTest, AdjointHopping) {
  for (int size = 1; size < 16; ++size) {
    for (int N = 1; N <= size; ++N) {
      Lattice L(size, N);
      for (int i = 0; i < size; i++) {
        for (int j = i; j < size; j++) {
          EXPECT_ALL_EQUAL(L.interaction_operator(i, j),
                                adjoint(L.interaction_operator(j, i)));
        }
      }
    }
  }
}

TEST(AnyLatticeTest, OnSiteInteraction) {
  for (int size = 1; size < 16; ++size) {
    for (int N = 1; N <= size; ++N) {
      Lattice L(size, N);
      for (int i = 0; i < size; i++) {
        EXPECT_TRUE(
            all_equal(L.interaction_operator(i, i), L.number_operator(i)));
      }
    }
  }
}

TEST(AnyLatticeTest, PermutationInvariantInteraction) {
  for (int size = 1; size < 16; size++) {
    for (int N = 1; N <= size; ++N) {
      Lattice L(size, N);
      for (int i = 0; i < size; i++) {
        for (int j = i; j < size; j++) {
          EXPECT_ALL_EQUAL(L.interaction_operator(i, j),
                                L.interaction_operator(j, i));
        }
      }
    }
  }
}

//
// TWO-SITE-LATTICE
//

TEST(HCBLatticeTest, OneInTwo) {
  // One particle in two sites.
  // Configurations: 01, 10
  {
    Lattice L(2, 1);

    RTensor hop0to1 = RTensor::zeros(2, 2);
    hop0to1.at(1, 0) = 1.0;
    EXPECT_ALL_EQUAL(hop0to1,
                          L.hopping_operator(1, 0, Lattice::HARD_CORE_BOSONS));
    EXPECT_TRUE(
        all_equal(hop0to1, L.hopping_operator(1, 0, Lattice::FERMIONS)));

    RTensor hop1to0 = adjoint(hop0to1);
    EXPECT_ALL_EQUAL(hop1to0,
                          L.hopping_operator(0, 1, Lattice::HARD_CORE_BOSONS));
    EXPECT_TRUE(
        all_equal(hop1to0, L.hopping_operator(0, 1, Lattice::FERMIONS)));

    RTensor number0 = RTensor::zeros(2, 2);
    number0.at(0, 0) = 1.0;
    EXPECT_ALL_EQUAL(number0, L.number_operator(0));

    RTensor number1 = RTensor::zeros(2, 2);
    number1.at(1, 1) = 1.0;
    EXPECT_ALL_EQUAL(number1, L.number_operator(1));

    RTensor int01 = RTensor::zeros(2, 2);
    EXPECT_ALL_EQUAL(int01, L.interaction_operator(0, 1));
  }
}

//
// THREE-SITE-LATTICE
//

TEST(HCBLatticeTest, OneInThree) {
  // One particle in two sites.
  // Configurations: 001, 010, 100
  Lattice L(3, 1);
  {
    RTensor hop0to1 = RTensor::zeros(3, 3);
    hop0to1.at(1, 0) = 1.0;
    EXPECT_ALL_EQUAL(hop0to1,
                          L.hopping_operator(1, 0, Lattice::HARD_CORE_BOSONS));
    RTensor hop1to0 = adjoint(hop0to1);
    EXPECT_ALL_EQUAL(hop1to0,
                          L.hopping_operator(0, 1, Lattice::HARD_CORE_BOSONS));
  }
  {
    RTensor hop0to2 = RTensor::zeros(3, 3);
    hop0to2.at(2, 0) = 1.0;
    EXPECT_ALL_EQUAL(hop0to2,
                          L.hopping_operator(2, 0, Lattice::HARD_CORE_BOSONS));
    RTensor hop2to0 = adjoint(hop0to2);
    EXPECT_ALL_EQUAL(hop2to0,
                          L.hopping_operator(0, 2, Lattice::HARD_CORE_BOSONS));
  }
  {
    RTensor hop1to2 = RTensor::zeros(3, 3);
    hop1to2.at(2, 1) = 1.0;
    EXPECT_ALL_EQUAL(hop1to2,
                          L.hopping_operator(2, 1, Lattice::HARD_CORE_BOSONS));
    RTensor hop2to1 = adjoint(hop1to2);
    EXPECT_ALL_EQUAL(hop2to1,
                          L.hopping_operator(1, 2, Lattice::HARD_CORE_BOSONS));
  }
  {
    RTensor number0 = RTensor::zeros(3, 3);
    number0.at(0, 0) = 1.0;
    EXPECT_ALL_EQUAL(number0, L.number_operator(0));

    RTensor number1 = RTensor::zeros(3, 3);
    number1.at(1, 1) = 1.0;
    EXPECT_ALL_EQUAL(number1, L.number_operator(1));

    RTensor number2 = RTensor::zeros(3, 3);
    number2.at(2, 2) = 1.0;
    EXPECT_ALL_EQUAL(number2, L.number_operator(2));

    RTensor int01 = RTensor::zeros(3, 3);
    EXPECT_ALL_EQUAL(int01, L.interaction_operator(0, 1));
    RTensor int02 = RTensor::zeros(3, 3);
    EXPECT_ALL_EQUAL(int02, L.interaction_operator(0, 2));
    RTensor int12 = RTensor::zeros(3, 3);
    EXPECT_ALL_EQUAL(int12, L.interaction_operator(1, 2));
  }
}

TEST(FermionLatticeTest, OneInThree) {
  // One particle in two sites.
  // Configurations: 001, 010, 100
  // The only differences from HCB should reside in the 0 to 2 hopping
  Lattice L(3, 1);
  {
    RTensor hop0to1 = RTensor::zeros(3, 3);
    hop0to1.at(1, 0) = 1.0;
    EXPECT_ALL_EQUAL(hop0to1,
                          L.hopping_operator(1, 0, Lattice::HARD_CORE_BOSONS));
    RTensor hop1to0 = adjoint(hop0to1);
    EXPECT_ALL_EQUAL(hop1to0,
                          L.hopping_operator(0, 1, Lattice::HARD_CORE_BOSONS));
  }
  {
    RTensor hop0to2 = RTensor::zeros(3, 3);
    hop0to2.at(2, 0) = 1.0;
    EXPECT_ALL_EQUAL(hop0to2,
                          L.hopping_operator(2, 0, Lattice::HARD_CORE_BOSONS));
    RTensor hop2to0 = adjoint(hop0to2);
    EXPECT_ALL_EQUAL(hop2to0,
                          L.hopping_operator(0, 2, Lattice::HARD_CORE_BOSONS));
  }
  {
    RTensor hop1to2 = RTensor::zeros(3, 3);
    hop1to2.at(2, 1) = 1.0;
    EXPECT_ALL_EQUAL(hop1to2,
                          L.hopping_operator(2, 1, Lattice::HARD_CORE_BOSONS));
    RTensor hop2to1 = adjoint(hop1to2);
    EXPECT_ALL_EQUAL(hop2to1,
                          L.hopping_operator(1, 2, Lattice::HARD_CORE_BOSONS));
  }
}

TEST(HCBLatticeTest, TwoInThree) {
  // Two particles in three sites.
  // Configurations: 011, 101, 110
  Lattice L(3, 2);
  {
    RTensor hop0to1 = RTensor::zeros(3, 3);
    hop0to1.at(2, 1) = 1.0;
    EXPECT_ALL_EQUAL(hop0to1,
                          L.hopping_operator(1, 0, Lattice::HARD_CORE_BOSONS));
    RTensor hop1to0 = adjoint(hop0to1);
    EXPECT_ALL_EQUAL(hop1to0,
                          L.hopping_operator(0, 1, Lattice::HARD_CORE_BOSONS));
  }
  {
    RTensor hop0to2 = RTensor::zeros(3, 3);
    hop0to2.at(2, 0) = 1.0;
    EXPECT_ALL_EQUAL(hop0to2,
                          L.hopping_operator(2, 0, Lattice::HARD_CORE_BOSONS));
    RTensor hop2to0 = adjoint(hop0to2);
    EXPECT_ALL_EQUAL(hop2to0,
                          L.hopping_operator(0, 2, Lattice::HARD_CORE_BOSONS));
  }
  {
    RTensor hop1to2 = RTensor::zeros(3, 3);
    hop1to2.at(1, 0) = 1.0;
    EXPECT_ALL_EQUAL(hop1to2,
                          L.hopping_operator(2, 1, Lattice::HARD_CORE_BOSONS));
    RTensor hop2to1 = adjoint(hop1to2);
    EXPECT_ALL_EQUAL(hop2to1,
                          L.hopping_operator(1, 2, Lattice::HARD_CORE_BOSONS));
  }
  {
    RTensor number0 = RTensor::zeros(3, 3);
    number0.at(0, 0) = 1.0;
    number0.at(1, 1) = 1.0;
    EXPECT_ALL_EQUAL(number0, L.number_operator(0));

    RTensor number1 = RTensor::zeros(3, 3);
    number1.at(0, 0) = 1.0;
    number1.at(2, 2) = 1.0;
    EXPECT_ALL_EQUAL(number1, L.number_operator(1));

    RTensor number2 = RTensor::zeros(3, 3);
    number2.at(1, 1) = 1.0;
    number2.at(2, 2) = 1.0;
    EXPECT_ALL_EQUAL(number2, L.number_operator(2));

    RTensor int01 = RTensor::zeros(3, 3);
    int01.at(0, 0) = 1.0;
    EXPECT_ALL_EQUAL(int01, L.interaction_operator(0, 1));
    RTensor int02 = RTensor::zeros(3, 3);
    int02.at(1, 1) = 1.0;
    RTensor int12 = RTensor::zeros(3, 3);
    int12.at(2, 2) = 1.0;
    EXPECT_ALL_EQUAL(int12, L.interaction_operator(1, 2));
  }
}

TEST(FermionLatticeTest, TwoInThree) {
  // Two particles in three sites.
  // Configurations: 011, 101, 110
  Lattice L(3, 2);
  {
    RTensor hop0to1 = RTensor::zeros(3, 3);
    hop0to1.at(2, 1) = 1.0;
    EXPECT_TRUE(
        all_equal(hop0to1, L.hopping_operator(1, 0, Lattice::FERMIONS)));
    RTensor hop1to0 = adjoint(hop0to1);
    EXPECT_TRUE(
        all_equal(hop1to0, L.hopping_operator(0, 1, Lattice::FERMIONS)));
  }
  {
    RTensor hop0to2 = RTensor::zeros(3, 3);
    hop0to2.at(2, 0) = -1.0;
    EXPECT_TRUE(
        all_equal(hop0to2, L.hopping_operator(2, 0, Lattice::FERMIONS)));
    RTensor hop2to0 = adjoint(hop0to2);
    EXPECT_TRUE(
        all_equal(hop2to0, L.hopping_operator(0, 2, Lattice::FERMIONS)));
  }
  {
    RTensor hop1to2 = RTensor::zeros(3, 3);
    hop1to2.at(1, 0) = 1.0;
    EXPECT_TRUE(
        all_equal(hop1to2, L.hopping_operator(2, 1, Lattice::FERMIONS)));
    RTensor hop2to1 = adjoint(hop1to2);
    EXPECT_TRUE(
        all_equal(hop2to1, L.hopping_operator(1, 2, Lattice::FERMIONS)));
  }
}

TEST(FermionLatticeTest, TwoInFour) {
  // Two particles in three sites.
  // Configurations: 0011, 0101, 0110, 1001, 1010, 1100
  // Indices:           0,    1,    2,    3,    4,    5
  Lattice L(4, 2);
  {
    RTensor hop0to1 = RTensor::zeros(6, 6);
    hop0to1.at(2, 1) = 1.0;
    hop0to1.at(4, 3) = 1.0;
    EXPECT_TRUE(
        all_equal(hop0to1, L.hopping_operator(1, 0, Lattice::FERMIONS)));
    RTensor hop1to0 = adjoint(hop0to1);
    EXPECT_TRUE(
        all_equal(hop1to0, L.hopping_operator(0, 1, Lattice::FERMIONS)));
  }
  {
    RTensor hop0to2 = RTensor::zeros(6, 6);
    hop0to2.at(2, 0) = -1.0;
    hop0to2.at(5, 3) = 1.0;
    EXPECT_TRUE(
        all_equal(hop0to2, L.hopping_operator(2, 0, Lattice::FERMIONS)));
    RTensor hop2to0 = adjoint(hop0to2);
    EXPECT_TRUE(
        all_equal(hop2to0, L.hopping_operator(0, 2, Lattice::FERMIONS)));
  }
  {
    RTensor hop1to2 = RTensor::zeros(6, 6);
    hop1to2.at(1, 0) = 1.0;
    hop1to2.at(5, 4) = 1.0;
    EXPECT_TRUE(
        all_equal(hop1to2, L.hopping_operator(2, 1, Lattice::FERMIONS)));
    RTensor hop2to1 = adjoint(hop1to2);
    EXPECT_TRUE(
        all_equal(hop2to1, L.hopping_operator(1, 2, Lattice::FERMIONS)));
  }
  {
    RTensor hop1to3 = RTensor::zeros(6, 6);
    hop1to3.at(3, 0) = 1.0;
    hop1to3.at(5, 2) = -1.0;
    EXPECT_TRUE(
        all_equal(hop1to3, L.hopping_operator(3, 1, Lattice::FERMIONS)));
    RTensor hop3to1 = adjoint(hop1to3);
    EXPECT_TRUE(
        all_equal(hop3to1, L.hopping_operator(1, 3, Lattice::FERMIONS)));
  }
}

}  // namespace tensor_test
