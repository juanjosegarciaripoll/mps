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
  EXPECT_ALL_EQUAL(new_state[0], psi);
  EXPECT_ALL_EQUAL(new_state[1], psi);
  EXPECT_EQ(state.size(), 2);
  EXPECT_ALL_EQUAL(state[0], psi);
  EXPECT_ALL_EQUAL(state[1], psi);
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
  EXPECT_ALL_EQUAL(new_state[0], psi);
  EXPECT_ALL_EQUAL(new_state[1], psi);
  EXPECT_EQ(state.size(), 2);
  EXPECT_ALL_EQUAL(state[0], psi);
  EXPECT_ALL_EQUAL(state[1], psi);
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
  EXPECT_ALL_EQUAL(new_state[0], psi);
  EXPECT_ALL_EQUAL(new_state[1], psi);
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
  EXPECT_ALL_EQUAL(new_state[0], psi);
  EXPECT_ALL_EQUAL(new_state[1], psi);
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
void test_mps_dimensions() {
  {
    auto d = Indices{2};
    auto psi = MPS(d, 3);
    EXPECT_ALL_EQUAL(psi.dimensions(), d);
    EXPECT_ALL_EQUAL(dimensions(psi), d);
  }
  {
    auto d = Indices{2, 5};
    auto psi = MPS(d, 3);
    EXPECT_ALL_EQUAL(psi.dimensions(), d);
    EXPECT_ALL_EQUAL(dimensions(psi), d);
  }
  {
    auto d = Indices{2, 5, 7};
    auto psi = MPS(d, 3);
    EXPECT_ALL_EQUAL(psi.dimensions(), d);
    EXPECT_ALL_EQUAL(dimensions(psi), d);
  }
}

template <class MPS>
void test_mps_const_access() {
  {
    const MPS psi;
    EXPECT_EQ(psi.size(), 0);
    EXPECT_ERROR_DETECTED(psi[0], mps_out_of_range);
    EXPECT_ERROR_DETECTED(psi[1], mps_out_of_range);
  }
  {
    auto state = MPS::elt_t::random(3);
    const MPS psi = MPS::product_state(1, state);
    state = reshape(state, 1, 3, 1);
    EXPECT_ALL_EQUAL(state, psi[0]);
    EXPECT_ERROR_DETECTED(psi[1], mps_out_of_range);
    EXPECT_ALL_EQUAL(state, psi[-1]);
    EXPECT_ERROR_DETECTED(psi[-2], mps_out_of_range);
  }
  {
    auto state = MPS::elt_t::random(3);
    const MPS psi = MPS::product_state(2, state);
    state = reshape(state, 1, 3, 1);
    EXPECT_ALL_EQUAL(state, psi[0]);
    EXPECT_ALL_EQUAL(state, psi[1]);
    EXPECT_ERROR_DETECTED(psi[2], mps_out_of_range);
    EXPECT_ALL_EQUAL(state, psi[-1]);
    EXPECT_ALL_EQUAL(state, psi[-2]);
    EXPECT_ERROR_DETECTED(psi[-3], mps_out_of_range);
  }
}

template <class MPS>
void test_mps_access() {
  {
    MPS psi;
    EXPECT_EQ(psi.size(), 0);
    EXPECT_ERROR_DETECTED(psi.at(0), mps_out_of_range);
    EXPECT_ERROR_DETECTED(psi.at(1), mps_out_of_range);
    EXPECT_ERROR_DETECTED(psi.at(-1), mps_out_of_range);
  }
  {
    auto state = MPS::elt_t::random(3);
    MPS psi = MPS::product_state(1, state);
    state = reshape(state, 1, 3, 1);
    EXPECT_ALL_EQUAL(state, psi.at(0));
    EXPECT_ERROR_DETECTED(psi.at(1), mps_out_of_range);
    EXPECT_ALL_EQUAL(state, psi.at(-1));
    EXPECT_ERROR_DETECTED(psi.at(-2), mps_out_of_range);
  }
  {
    auto state = MPS::elt_t::random(3);
    MPS psi = MPS::product_state(2, state);
    state = reshape(state, 1, 3, 1);
    EXPECT_ALL_EQUAL(state, psi.at(0));
    EXPECT_ALL_EQUAL(state, psi.at(1));
    EXPECT_ERROR_DETECTED(psi.at(2), mps_out_of_range);
    EXPECT_ALL_EQUAL(state, psi.at(-1));
    EXPECT_ALL_EQUAL(state, psi.at(-2));
    EXPECT_ERROR_DETECTED(psi.at(-3), mps_out_of_range);
  }
  {
    auto state0 = MPS::elt_t::random(1, 3, 1);
    auto state1 = MPS::elt_t::random(1, 3, 1);
    auto state2 = MPS::elt_t::random(1, 3, 1);
    MPS psi(3);
    psi.at(0) = state0;
    psi.at(1) = state1;
    psi.at(2) = state2;
    EXPECT_ALL_EQUAL(state0, psi[0]);
    EXPECT_ALL_EQUAL(state1, psi[1]);
    EXPECT_ALL_EQUAL(state2, psi[2]);
    EXPECT_ALL_EQUAL(state0, psi[-3]);
    EXPECT_ALL_EQUAL(state1, psi[-2]);
    EXPECT_ALL_EQUAL(state2, psi[-1]);
  }
}

template <class MPS>
void test_mps_product_state(int size) {
  const auto psi = MPS::elt_t::random(3);
  // Throw if we do not provide a vector as local state
  EXPECT_ERROR_DETECTED(product_state(size, reshape(psi, 3, 1)),
                        std::invalid_argument);
  // Throw if we the vector has size zero
  EXPECT_ERROR_DETECTED(product_state(size, mp_tensor_t<MPS>::empty(0)),
                        std::invalid_argument);
  EXPECT_ERROR_DETECTED(product_state(size, mp_tensor_t<MPS>()),
                        std::invalid_argument);
  {
    MPS state = product_state(size, psi);
    EXPECT_EQ(state.size(), size);
    const auto tensor_psi = reshape(psi, 1, psi.size(), 1);
    EXPECT_TRUE(std::all_of(
        begin(state), end(state),
        [&](const mp_tensor_t<MPS> &t) { return all_equal(t, tensor_psi); }));
  }
  {
    mp_tensor_t<MPS> psi0 = {0, 1}, psi1 = {1, 0, 2};
    vector<mp_tensor_t<MPS>> v = {psi0, psi1};
    MPS state = product_state(v);
    EXPECT_EQ(state.size(), 2);
    EXPECT_ALL_EQUAL(state[0], reshape(psi0, 1, 2, 1));
    EXPECT_ALL_EQUAL(state[1], reshape(psi1, 1, 3, 1));
  }
}

template <class MPS>
void test_mps_to_vector() {
  {
    MPS psi;
    auto v = psi.to_vector();
    EXPECT_ALL_EQUAL(v.dimensions(), Dimensions{0});
  }
  {
    auto psi0 = MPS::elt_t::random(3);
    MPS psi = MPS::product_state(2, psi0);
    auto v = psi.to_vector();
    auto videal = kron(psi0, psi0);
    EXPECT_CEQ3(v, videal, STRICT_EPSILON);
  }
  {
    MPS psi = MPS::random(Dimensions{3, 2}, /*bond dim*/ 4);
    auto v = psi.to_vector();
    auto psi0 = reshape(psi[0], 3, 4);
    auto psi1 = reshape(psi[1], 4, 2);
    auto videal = reshape(mmult(psi0, psi1), 3 * 2);
    EXPECT_CEQ3(v, videal, STRICT_EPSILON);
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

TEST(RMPS, Dimensions) { test_mps_dimensions<RMPS>(); }

TEST(RMPS, AccesOperators) {
  test_mps_const_access<RMPS>();
  test_mps_access<RMPS>();
}

TEST(RMPS, ToVector) { test_mps_to_vector<RMPS>(); }

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

TEST(CMPS, Dimensions) { test_mps_dimensions<CMPS>(); }

TEST(CMPS, AccesOperators) {
  test_mps_const_access<CMPS>();
  test_mps_access<CMPS>();
}

TEST(CMPS, ToVector) { test_mps_to_vector<CMPS>(); }

}  // namespace tensor_test
