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
#include <tensor/gen.h>
#include <mps/lattice.h>

namespace tensor_test {

using namespace tensor;
using namespace mps;

//////////////////////////////////////////////////////////////////////
// HARD-CORE BOSONS LATTICE PARTITION
//
// Lattice sites are numbered as bits in a binary number. We split
// the lattice into two halves and count the states that are in each
// partition.

void test_lattice_partition(int sites_left, int sites_right, int N,
                            const Indices &expected_left,
                            const Indices &expected_right,
                            const Indices &expected_indices) {
  Lattice L(sites_left + sites_right, N);
  Indices left_states, right_states, matrix_indices;

  L.bipartition(sites_left, &left_states, &right_states, &matrix_indices);
  EXPECT_TRUE(all_equal(left_states, expected_left));
  EXPECT_TRUE(all_equal(right_states, expected_right));
  EXPECT_TRUE(all_equal(matrix_indices, expected_indices));

  if (!all_equal(left_states, expected_left) ||
      !all_equal(right_states, expected_right) ||
      !all_equal(matrix_indices, expected_indices))
    std::cerr << left_states << '\n'
              << right_states << '\n'
              << matrix_indices << '\n';
}

//
// TWO SITES
//

TEST(LatticePartition, Lattice0in1plus1) {
  // Left states
  //     0
  Indices left = igen << 0;
  // Right states
  //     0
  Indices right = igen << 0;
  // Matrix indices
  //     00	->	0,0	->	0
  Indices indices = igen << 0;
  test_lattice_partition(1, 1, 0, left, right, indices);
}

TEST(LatticePartition, Lattice1in1plus1) {
  // Left states
  //     0
  //     1
  Indices left = igen << 0 << 1;
  // Right states
  //     0
  //     1
  Indices right = igen << 0 << 1;
  // Matrix indices
  //     01	->	0,1	->	1
  //     10	->	1,0	->	2
  Indices indices = igen << 1 << 2;
  test_lattice_partition(1, 1, 1, left, right, indices);
}

TEST(LatticePartition, Lattice2in1plus1) {
  // Left states
  //     1
  Indices left = igen << 1;
  // Right states
  //     1
  Indices right = igen << 1;
  // Matrix indices
  //     11	->	1,1	->      0
  Indices indices = igen << 0;
  test_lattice_partition(1, 1, 2, left, right, indices);
}

////////////////////////////////////////////////////////////
// THREE SITES, 1 + 2 BIPARTITION
//

TEST(LatticePartition, Lattice0in1plus2) {
  // Left states
  //     0
  Indices left = igen << 0;
  // Right states
  //     0
  Indices right = igen << 0;
  // Matrix indices
  //     00	->	0,0	->	0
  Indices indices = igen << 0;
  test_lattice_partition(1, 2, 0, left, right, indices);
}

TEST(LatticePartition, Lattice1in1plus2) {
  // Left states
  //     0
  //     1
  Indices left = igen << 0 << 1;
  // Right states
  //     00
  //     01
  //     10
  Indices right = igen << 0 << 1 << 2;
  // Matrix indices
  //     001	->	0,01	->	1
  //     010	->      0,10	->	2
  //     100  ->      1,00    ->      3
  Indices indices = igen << 1 << 2 << 3;
  test_lattice_partition(1, 2, 1, left, right, indices);
}

TEST(LatticePartition, Lattice2in1plus2) {
  // Left states
  //     0
  //     1
  Indices left = igen << 0 << 1;
  // Right states
  //     01   ->  0
  //     10   ->  1
  //     11   ->  2
  Indices right = igen << 1 << 2 << 3;
  // Matrix indices
  //     011	->  0,11  ->  (0,2)  ->  2
  //     101	->  1,01  ->  (1,0)  ->  3
  //     110  ->  1,10  ->  (1,1)  ->  4
  Indices indices = igen << 2 << 3 << 4;
  test_lattice_partition(1, 2, 2, left, right, indices);
}

TEST(LatticePartition, Lattice3in1plus2) {
  // Left states
  //     1
  Indices left = igen << 1;
  // Right states
  //     11
  Indices right = igen << 3;
  // Matrix indices
  //     111 ->       1,11	->      0
  Indices indices = igen << 0;
  test_lattice_partition(1, 2, 3, left, right, indices);
}

////////////////////////////////////////////////////////////
// FOUR SITES, 2 + 2 BIPARTITION
//

TEST(LatticePartition, Lattice0in2plus2) {
  // Left states
  //     00
  Indices left = igen << 0;
  Indices right = left;
  // Matrix indices
  //     00	->	00,00	->	0
  Indices indices = igen << 0;
  test_lattice_partition(2, 2, 0, left, right, indices);
}

TEST(LatticePartition, Lattice1in2plus2) {
  // Left states
  //     00   -> 0 (index)
  //     01   -> 1
  //     10   -> 2
  Indices left = igen << 0 << 1 << 2;
  Indices right = left;
  // Matrix indices
  //     0001	-> 00,01 -> (0,1) -> 1
  //     0010	-> 00,10 -> (0,2) -> 2
  //     0100 -> 01,00 -> (1,0) -> 3
  //     1000 -> 10,00 -> (2,0) -> 6
  Indices indices = igen << 1 << 2 << 3 << 6;
  test_lattice_partition(2, 2, 1, left, right, indices);
}

TEST(LatticePartition, Lattice2in2plus2) {
  // Left states
  //     00   -> 0 (index)
  //     01   -> 1
  //     10   -> 2
  //     11   -> 3
  Indices left = igen << 0 << 1 << 2 << 3;
  Indices right = left;
  // Matrix indices
  //     0011	-> 00,11 -> (0,3) -> 3
  //     0101	-> 01,01 -> (1,1) -> 5
  //     0110 -> 01,10 -> (1,2) -> 6
  //     1001 -> 10,01 -> (2,1) -> 9
  //     1010 -> 10,10 -> (2,2) -> 10
  //     1100 -> 11,00 -> (3,0) -> 12
  Indices indices = igen << 3 << 5 << 6 << 9 << 10 << 12;
  test_lattice_partition(2, 2, 2, left, right, indices);
}

TEST(LatticePartition, Lattice3in2plus2) {
  // Left states
  //     01   -> 0 (index)
  //     10   -> 1
  //     11   -> 2
  Indices left = igen << 1 << 2 << 3;
  Indices right = left;
  // Matrix indices
  //     0111	-> 01,11 -> (0,2) -> 2
  //     1011	-> 10,11 -> (1,2) -> 5
  //     1101 -> 11,01 -> (2,0) -> 6
  //     1110 -> 11,10 -> (2,1) -> 7
  Indices indices = igen << 2 << 5 << 6 << 7;
  test_lattice_partition(2, 2, 3, left, right, indices);
}

TEST(LatticePartition, Lattice4in2plus2) {
  // Left states
  //     11
  Indices left = igen << 3;
  // Right states
  //     11
  Indices right = igen << 3;
  // Matrix indices
  //     1111 ->       11,11	->      0
  Indices indices = igen << 0;
  test_lattice_partition(2, 2, 4, left, right, indices);
}

////////////////////////////////////////////////////////////
// FOUR SITES, 1 + 3 BIPARTITION
//

TEST(LatticePartition, Lattice0in1plus3) {
  // Left states
  //     0
  Indices left = igen << 0;
  // Right states
  //     0
  Indices right = igen << 0;
  // Matrix indices
  //     00	->	0,0	->	0
  Indices indices = igen << 0;
  test_lattice_partition(1, 3, 0, left, right, indices);
}

TEST(LatticePartition, Lattice1in1plus3) {
  // Left states
  //     0
  //     1
  Indices left = igen << 0 << 1;
  // Right states
  //     000 -> 0 (index)
  //     001 -> 1
  //     010 -> 2
  //     100 -> 3
  Indices right = igen << 0 << 1 << 2 << 4;
  // Matrix indices
  //     0001	-> 0,001 -> (0,1) -> 1
  //     0010	-> 0,010 -> (0,2) -> 2
  //     0100 -> 0,100 -> (0,3) -> 3
  //     1000 -> 1,000 -> (1,0) -> 4
  Indices indices = igen << 1 << 2 << 3 << 4;
  test_lattice_partition(1, 3, 1, left, right, indices);
}

TEST(LatticePartition, Lattice2in1plus3) {
  // Left states
  //     0
  //     1
  Indices left = igen << 0 << 1;
  // Right states
  //     001   ->  0 (index)
  //     010   ->  1
  //     011   ->  2
  //     100   ->  3
  //     101   ->  4
  //     110   ->  5
  Indices right = igen << 1 << 2 << 3 << 4 << 5 << 6;
  // Matrix indices
  //     0011	-> 0,011 -> (0,2)  ->  2
  //     0101	-> 0,101 -> (0,4)  ->  4
  //     0110 -> 0,110 -> (0,5)  ->  5
  //     1001 -> 1,001 -> (1,0)  ->  6
  //     1010 -> 1,010 -> (1,1)  ->  7
  //     1100 -> 1,100 -> (1,3)  ->  9
  Indices indices = igen << 2 << 4 << 5 << 6 << 7 << 9;
  test_lattice_partition(1, 3, 2, left, right, indices);
}

TEST(LatticePartition, Lattice4in1plus3) {
  // Left states
  //     1
  Indices left = igen << 1;
  // Right states
  //     111
  Indices right = igen << 7;
  // Matrix indices
  //     1111 ->  1,111 -> 0
  Indices indices = igen << 0;
  test_lattice_partition(1, 3, 4, left, right, indices);
}

////////////////////////////////////////////////////////////
// GENERAL PROPERTIES
//

TEST(LatticePartition, EmptyLattice) {
  for (int sites = 2; sites <= 6; sites++) {
    Lattice L(sites, 0);
    Indices expected_left = igen << 0;
    Indices expected_right = expected_left;
    Indices expected_ndx = igen << 0;
    for (int left_sites = 1; left_sites < sites; left_sites++) {
      Indices il, ir, ndx;
      L.bipartition(left_sites, &il, &ir, &ndx);
      EXPECT_TRUE(all_equal(il, expected_left));
      EXPECT_TRUE(all_equal(ir, expected_right));
      EXPECT_TRUE(all_equal(ndx, expected_ndx));
    }
  }
}

TEST(LatticePartition, FullLattice) {
  for (int sites = 2; sites <= 6; sites++) {
    Lattice L(sites, sites);
    for (int left_sites = 1; left_sites < sites; left_sites++) {
      int right_sites = sites - left_sites;
      Indices expected_left = igen << (1 << left_sites) - 1;
      Indices expected_right = igen << (1 << right_sites) - 1;
      Indices expected_ndx = igen << 0;
      Indices il, ir, ndx;
      L.bipartition(left_sites, &il, &ir, &ndx);
      EXPECT_TRUE(all_equal(il, expected_left));
      EXPECT_TRUE(all_equal(ir, expected_right));
      EXPECT_TRUE(all_equal(ndx, expected_ndx));
    }
  }
}

TEST(LatticePartition, AllowedStates) {
  for (int sites = 2; sites <= 6; sites++) {
    Lattice L(sites, sites);
    for (int left_sites = 1; left_sites < sites; left_sites++) {
      int right_sites = sites - left_sites;
      Indices expected_left = igen << (1 << left_sites) - 1;
      Indices expected_right = igen << (1 << right_sites) - 1;
      Indices expected_ndx = igen << 0;
      Indices il, ir, ndx;
      L.bipartition(left_sites, &il, &ir, &ndx);
      EXPECT_TRUE(all_equal(il, expected_left));
      EXPECT_TRUE(all_equal(ir, expected_right));
      EXPECT_TRUE(all_equal(ndx, expected_ndx));
    }
  }
}

TEST(LatticePartition, Symmetry) {
  for (int sites = 2; sites <= 6; sites++) {
    for (int nparticles = 0; nparticles <= sites; nparticles++) {
      Lattice L(sites, nparticles);
      for (int left_sites = 1; left_sites < sites; left_sites++) {
        Indices il, ir, ndx;
        L.bipartition(left_sites, &il, &ir, &ndx);
        Indices ilb, irb, ndxb;
        L.bipartition(sites - left_sites, &ilb, &irb, &ndxb);
        EXPECT_TRUE(all_equal(il, irb));
        EXPECT_TRUE(all_equal(ir, ilb));
      }
    }
  }
}

}  // namespace tensor_test
