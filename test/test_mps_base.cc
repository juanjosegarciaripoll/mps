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
#include <mps/mps.h>

namespace tensor_test {

using namespace tensor;
using namespace mps;

//////////////////////////////////////////////////////////////////////
// CONSTRUCTING SMALL MPS
//

template <class MPS>
void test_mps_constructor() {
  {
    MPS psi(/* size */ 1, /* physical dimension */ 2, /* bond dimension */ 3,
            /* periodic */ false);
    EXPECT_EQ(psi.size(), 1);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({1, 2, 1}));
  }
  {
    MPS psi(/* size */ 1, /* physical dimension */ 2, /* bond dimension */ 3,
            /* periodic */ true);
    EXPECT_EQ(psi.size(), 1);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({3, 2, 3}));
  }
  {
    MPS psi(/* size */ 2, /* physical dimension */ 2, /* bond dimension */ 3,
            /* periodic */ false);
    EXPECT_EQ(psi.size(), 2);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({1, 2, 3}));
    EXPECT_EQ(psi[1].dimensions(), Dimensions({3, 2, 1}));
  }
  {
    MPS psi(/* size */ 3, /* physical dimension */ 2, /* bond dimension */ 3,
            /* periodic */ false);
    EXPECT_EQ(psi.size(), 3);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({1, 2, 3}));
    EXPECT_EQ(psi[1].dimensions(), Dimensions({3, 2, 3}));
    EXPECT_EQ(psi[2].dimensions(), Dimensions({3, 2, 1}));
  }
  {
    MPS psi(/* size */ 3, /* physical dimension */ 2, /* bond dimension */ 3,
            /* periodic */ true);
    EXPECT_EQ(psi.size(), 3);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({3, 2, 3}));
    EXPECT_EQ(psi[1].dimensions(), Dimensions({3, 2, 3}));
    EXPECT_EQ(psi[2].dimensions(), Dimensions({3, 2, 3}));
  }
}

template <class MPS>
void test_mps_dimensions_constructor() {
  {
    MPS psi(Indices{2}, /* bond dimension */ 3,
            /* periodic */ false);
    EXPECT_EQ(psi.size(), 1);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({1, 2, 1}));
  }
  {
    MPS psi(Indices{2}, /* bond dimension */ 3,
            /* periodic */ true);
    EXPECT_EQ(psi.size(), 1);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({3, 2, 3}));
  }
  {
    MPS psi(Indices{2, 5}, /* bond dimension */ 3,
            /* periodic */ false);
    EXPECT_EQ(psi.size(), 2);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({1, 2, 3}));
    EXPECT_EQ(psi[1].dimensions(), Dimensions({3, 5, 1}));
  }
  {
    MPS psi(Indices{2, 5, 7}, /* bond dimension */ 3,
            /* periodic */ false);
    EXPECT_EQ(psi.size(), 3);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({1, 2, 3}));
    EXPECT_EQ(psi[1].dimensions(), Dimensions({3, 5, 3}));
    EXPECT_EQ(psi[2].dimensions(), Dimensions({3, 7, 1}));
  }
  {
    MPS psi(Indices{2, 5, 7}, /* bond dimension */ 3,
            /* periodic */ true);
    EXPECT_EQ(psi.size(), 3);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({3, 2, 3}));
    EXPECT_EQ(psi[1].dimensions(), Dimensions({3, 5, 3}));
    EXPECT_EQ(psi[2].dimensions(), Dimensions({3, 7, 3}));
  }
}

template <class MPS>
void test_mps_copy_constructor() {
  /*
   * Test that operator=(const MPS &) copies the content.
   */
  MPS state(/* size */ 2);
  auto psi = mp_tensor_t<MPS>::random(1, 2, 1);
  state.at(0) = state.at(1) = psi;
  EXPECT_EQ(state.size(), 2);
  MPS new_state(state);
  EXPECT_EQ(new_state.size(), 2);
  EXPECT_TRUE(all_equal(new_state[0], psi));
  EXPECT_TRUE(all_equal(new_state[1], psi));
  EXPECT_EQ(state.size(), 2);
  EXPECT_TRUE(all_equal(state[0], psi));
  EXPECT_TRUE(all_equal(state[1], psi));
}

template <class MPS>
void test_mps_operator_eq() {
  /*
   * Test that operator=(const MPS &) copies the content.
   */
  MPS state(/* size */ 2);
  auto psi = mp_tensor_t<MPS>::random(1, 2, 1);
  state.at(0) = state.at(1) = psi;
  EXPECT_EQ(state.size(), 2);
  MPS new_state;
  EXPECT_EQ(new_state.size(), 0);
  new_state = state;
  EXPECT_EQ(new_state.size(), 2);
  EXPECT_TRUE(all_equal(new_state[0], psi));
  EXPECT_TRUE(all_equal(new_state[1], psi));
  EXPECT_EQ(state.size(), 2);
  EXPECT_TRUE(all_equal(state[0], psi));
  EXPECT_TRUE(all_equal(state[1], psi));
}

template <class MPS>
void test_mps_move_constructor() {
  /*
   * Test that operator=(MPS &&) steals the content from the origin.
   */
  MPS state(/* size */ 2);
  auto psi = mp_tensor_t<MPS>::random(1, 2, 1);
  state.at(0) = state.at(1) = psi;
  EXPECT_EQ(state.size(), 2);
  MPS new_state(std::move(state));
  EXPECT_EQ(new_state.size(), 2);
  EXPECT_TRUE(all_equal(new_state[0], psi));
  EXPECT_TRUE(all_equal(new_state[1], psi));
  EXPECT_EQ(state.size(), 0);
}

template <class MPS>
void test_mps_operator_eq_move() {
  /*
   * Test that operator=(MPS &&) steals the content from the origin.
   */
  MPS state(/* size */ 2);
  auto psi = mp_tensor_t<MPS>::random(1, 2, 1);
  state.at(0) = state.at(1) = psi;
  EXPECT_EQ(state.size(), 2);
  MPS new_state;
  EXPECT_EQ(new_state.size(), 0);
  new_state = std::move(state);
  EXPECT_EQ(new_state.size(), 2);
  EXPECT_TRUE(all_equal(new_state[0], psi));
  EXPECT_TRUE(all_equal(new_state[1], psi));
  EXPECT_EQ(state.size(), 0);
}

template <class MPS>
void test_mps_random() {
  {
    MPS psi = MPS::random(/* size */ 1, /* physical dimension */ 2,
                          /* bond dimension */ 3,
                          /* periodic */ false);
    EXPECT_EQ(psi.size(), 1);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({1, 2, 1}));
  }
  {
    MPS psi = MPS::random(/* size */ 1, /* physical dimension */ 2,
                          /* bond dimension */ 3,
                          /* periodic */ true);
    EXPECT_EQ(psi.size(), 1);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({3, 2, 3}));
  }
  {
    MPS psi = MPS::random(/* size */ 2, /* physical dimension */ 2,
                          /* bond dimension */ 3,
                          /* periodic */ false);
    EXPECT_EQ(psi.size(), 2);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({1, 2, 3}));
    EXPECT_EQ(psi[1].dimensions(), Dimensions({3, 2, 1}));
  }
  {
    MPS psi = MPS::random(/* size */ 3, /* physical dimension */ 2,
                          /* bond dimension */ 3,
                          /* periodic */ false);
    EXPECT_EQ(psi.size(), 3);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({1, 2, 3}));
    EXPECT_EQ(psi[1].dimensions(), Dimensions({3, 2, 3}));
    EXPECT_EQ(psi[2].dimensions(), Dimensions({3, 2, 1}));
  }
  {
    MPS psi = MPS::random(/* size */ 3, /* physical dimension */ 2,
                          /* bond dimension */ 3,
                          /* periodic */ true);
    EXPECT_EQ(psi.size(), 3);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({3, 2, 3}));
    EXPECT_EQ(psi[1].dimensions(), Dimensions({3, 2, 3}));
    EXPECT_EQ(psi[2].dimensions(), Dimensions({3, 2, 3}));
  }
}

template <class MPS>
void test_mps_random_with_dimensions() {
  {
    MPS psi = MPS::random(Indices{2},
                          /* bond dimension */ 3,
                          /* periodic */ false);
    EXPECT_EQ(psi.size(), 1);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({1, 2, 1}));
  }
  {
    MPS psi = MPS::random(Indices{2},
                          /* bond dimension */ 3,
                          /* periodic */ true);
    EXPECT_EQ(psi.size(), 1);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({3, 2, 3}));
  }
  {
    MPS psi = MPS::random(Indices{2, 5},
                          /* bond dimension */ 3,
                          /* periodic */ false);
    EXPECT_EQ(psi.size(), 2);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({1, 2, 3}));
    EXPECT_EQ(psi[1].dimensions(), Dimensions({3, 5, 1}));
  }
  {
    MPS psi = MPS::random(Indices{2, 5, 7},
                          /* bond dimension */ 3,
                          /* periodic */ false);
    EXPECT_EQ(psi.size(), 3);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({1, 2, 3}));
    EXPECT_EQ(psi[1].dimensions(), Dimensions({3, 5, 3}));
    EXPECT_EQ(psi[2].dimensions(), Dimensions({3, 7, 1}));
  }
  {
    MPS psi = MPS::random(Indices{2, 5, 7},
                          /* bond dimension */ 3,
                          /* periodic */ true);
    EXPECT_EQ(psi.size(), 3);
    EXPECT_EQ(psi[0].dimensions(), Dimensions({3, 2, 3}));
    EXPECT_EQ(psi[1].dimensions(), Dimensions({3, 5, 3}));
    EXPECT_EQ(psi[2].dimensions(), Dimensions({3, 7, 3}));
  }
}

template <class MPS>
void test_mps_product_state(int size) {
  const auto psi = MPS::elt_t::random(3);
  MPS state = product_state(size, psi);
  EXPECT_EQ(state.size(), size);
  const auto tensor_psi = reshape(psi, 1, psi.size(), 1);
  EXPECT_TRUE(std::all_of(
      begin(state), end(state),
      [&](const mp_tensor_t<MPS> &t) { return all_equal(t, tensor_psi); }));
}

void test_ghz_state(int size) {
  RMPS ghz = ghz_state(size);
  RTensor psi = mps_to_vector(ghz);
  double v = 1 / sqrt((double)2.0);
  EXPECT_EQ(ghz.size(), size);
  EXPECT_EQ(psi.size(), 2 << (size - 1));
  for (index i = 0; i < psi.size(); i++) {
    double psi_i = ((i == 0) || (i == psi.size() - 1)) ? v : 0.0;
    EXPECT_DOUBLE_EQ(psi[i], psi_i);
  }
  EXPECT_DOUBLE_EQ(norm2(psi), 1.0);
}

const RMPS apply_cluster_state_stabilizer(RMPS state, int site) {
  state = apply_local_operator(state, mps::Pauli_x, site);
  if (site >= 0) state = apply_local_operator(state, mps::Pauli_z, site - 1);
  if (site < state.last())
    state = apply_local_operator(state, mps::Pauli_z, site + 1);
  return state;
}

void test_cluster_state(int size) {
  RMPS cluster = cluster_state(size);
  RTensor psi = mps_to_vector(cluster);
  EXPECT_EQ(cluster.size(), size);
  EXPECT_EQ(psi.size(), 2 << (size - 1));
  EXPECT_DOUBLE_EQ(norm2(psi), 1.0);
  for (index i = 1; i < cluster.size(); i++) {
    RTensor psi2 = mps_to_vector(apply_cluster_state_stabilizer(cluster, i));
    if (!simeq(psi, psi2)) {
      RMPS aux = apply_cluster_state_stabilizer(cluster, i);
      std::cout << "site=" << i << std::endl;
      std::cout << "psi2=" << psi2 << std::endl;
      std::cout << "psi1=" << psi << std::endl;
    }
    EXPECT_CEQ(psi, psi2);
  }
}

//////////////////////////////////////////////////////////////////////
// REAL SPECIALIZATIONS
//

TEST(RMPS, Constructor) {
  test_mps_constructor<RMPS>();
  test_mps_dimensions_constructor<RMPS>();
}

TEST(RMPS, Random) {
  test_mps_random<RMPS>();
  test_mps_random_with_dimensions<RMPS>();
}

TEST(RMPS, CopySemantics) {
  test_mps_copy_constructor<RMPS>();
  test_mps_operator_eq<RMPS>();
}

TEST(RMPS, MoveSemantics) {
  test_mps_move_constructor<RMPS>();
  test_mps_operator_eq_move<RMPS>();
}

TEST(RMPS, ProductState) {
  test_over_integers(1, 4, test_mps_product_state<CMPS>);
}

TEST(RMPS, GHZState) { test_over_integers(1, 10, test_ghz_state); }

TEST(RMPS, ClusterState) { test_over_integers(3, 10, test_cluster_state); }

//////////////////////////////////////////////////////////////////////
// COMPLEX SPECIALIZATIONS
//

TEST(CMPS, Constructor) {
  test_mps_constructor<CMPS>();
  test_mps_dimensions_constructor<CMPS>();
}

TEST(CMPS, Random) {
  test_mps_random<CMPS>();
  test_mps_random_with_dimensions<RMPS>();
}

TEST(CMPS, CopySemantics) {
  test_mps_copy_constructor<CMPS>();
  test_mps_operator_eq<CMPS>();
}

TEST(CMPS, MoveSemantics) {
  test_mps_move_constructor<CMPS>();
  test_mps_operator_eq_move<CMPS>();
}

TEST(CMPS, ProductState) {
  test_over_integers(1, 10, test_mps_product_state<CMPS>);
}

TEST(CMPS, ComplexUpgrade) {
  RMPS rpsi =
      RMPS::random(/*size*/ 2, /*physical_dimension*/ 3, /*bond dimension*/ 1);
  CMPS zpsi(rpsi);
  EXPECT_EQ(zpsi.size(), 2);
  EXPECT_EQ(rpsi.size(), 2);
  EXPECT_TRUE(std::equal(std::begin(zpsi), std::end(zpsi), std::begin(rpsi),
                         [](const CTensor &z, const RTensor &r) {
                           return all_equal(z, to_complex(r));
                         }));
}

}  // namespace tensor_test
