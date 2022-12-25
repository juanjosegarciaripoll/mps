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
#include <mps/mpo.h>
#include <mps/io.h>
#include <mps/quantum.h>

namespace tensor_test {

using namespace mps;
using namespace tensor;

template <class MPO>
void test_empty_mpo_list() {
  MPOList<MPO> list;
  EXPECT_EQ(list.size(), 0);
}

template <class MPO>
void test_small_mpo_list() {
  using MPS = typename MPO::MPS;

  auto psi = MPS(cluster_state(2));
  auto Pauli_z = tensor_cast(psi, mps::Pauli_z);
  auto Pauli_x = tensor_cast(psi, mps::Pauli_x);

  auto mpo1 = initialize_interactions_mpo<MPO>({2, 2});
  add_interaction(&mpo1, Pauli_z, 0, Pauli_x);

  auto mpo2 = initialize_interactions_mpo<MPO>({2, 2});
  add_local_term(&mpo2, Pauli_x, 0);
  add_local_term(&mpo2, Pauli_z, 1);

  MPOList<MPO> list({mpo1, mpo2});
  EXPECT_EQ(list[0], mpo1);
  EXPECT_EQ(list[1], mpo2);

  EXPECT_CEQ(mps::apply(list, psi), mps::apply(mpo2, mps::apply(mpo1, psi)));
}

////////////////////////////////////////////////////////////

TEST(RMPOList, Empty) { test_empty_mpo_list<RMPO>(); }

TEST(RMPOList, SmallMPOList) { test_small_mpo_list<RMPO>(); }

////////////////////////////////////////////////////////////

TEST(CMPOList, Empty) { test_empty_mpo_list<CMPO>(); }

TEST(CMPOList, SmallMPOList) { test_small_mpo_list<CMPO>(); }

}  // namespace tensor_test
