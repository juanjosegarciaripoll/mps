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
#include <gtest/gtest-death-test.h>
#include <mps/mpo.h>
#include <mps/io.h>
#include <mps/quantum.h>

namespace tensor_test {

  using namespace mps;
  using namespace tensor;

  template<class MPO>
  void test_zero_mpo(int size)
  {
    typedef typename MPO::elt_t Tensor;

    Tensor first = Tensor::zeros(1,2,2,2);
    first.at(0,0,0,0) = 1.0;
    first.at(0,1,1,0) = 1.0;

    Tensor last = Tensor::zeros(2,2,2,1);
    last.at(1,0,0,0) = 1.0;
    last.at(1,1,1,0) = 1.0;

    Tensor middle = Tensor::zeros(2,2,2,2);
    middle.at(0,0,0,0) = 1.0;
    middle.at(0,1,1,0) = 1.0;
    middle.at(1,0,0,1) = 1.0;
    middle.at(1,1,1,1) = 1.0;

    MPO mpo(size, 2);
    for (int i = 0; i < size; i++) {
      Tensor target;
      if (i == 0)
	target = first;
      else if (i+1 == size)
	target = last;
      else
	target = middle;
      EXPECT_TRUE(all_equal(mpo[i], target));
    }

    typename MPO::MPS psi = cluster_state(size);
    EXPECT_CEQ(norm2(apply(mpo, psi)), 0.0);
  }

  /*
   * Test MPO using only local terms in small problems.
   */
  template<class MPO>
  void
  test_small_local_mpo()
  {
    typedef typename MPO::elt_t Tensor;
    typedef typename MPO::MPS MPS;

    MPS psi = cluster_state(2);
    {
      MPO mpo(2,2);
      add_local_term(mpo, mps::Pauli_z, 0);
      Tensor Hpsi1 = mps_to_vector(apply(mpo, psi));

      Tensor H = kron2(mps::Pauli_z, mps::Pauli_id);
      Tensor Hpsi2 = mmult(H, mps_to_vector(psi));

      EXPECT_CEQ(norm2(Hpsi1 - Hpsi2), 0.0);
    }
    {
      MPO mpo(2,2);
      add_local_term(mpo, mps::Pauli_z, 1);
      Tensor Hpsi1 = mps_to_vector(apply(mpo, psi));

      Tensor H = kron2(mps::Pauli_id, mps::Pauli_z);
      Tensor Hpsi2 = mmult(H, mps_to_vector(psi));

      EXPECT_CEQ(norm2(Hpsi1 - Hpsi2), 0.0);
    }
    {
      MPO mpo(2,2);
      add_local_term(mpo, mps::Pauli_z, 0);
      add_local_term(mpo, mps::Pauli_z, 1);
      Tensor Hpsi1 = mps_to_vector(apply(mpo, psi));

      Tensor H = kron2(mps::Pauli_id, mps::Pauli_z) +
        kron2(mps::Pauli_z, mps::Pauli_id);
      Tensor Hpsi2 = mmult(H, mps_to_vector(psi));

      EXPECT_CEQ(norm2(Hpsi1 - Hpsi2), 0.0);
    }

    psi = cluster_state(3);
    {
      MPO mpo(3,2);
      add_local_term(mpo, mps::Pauli_z, 0);
      Tensor Hpsi1 = mps_to_vector(apply(mpo, psi));

      Tensor H = kron2(mps::Pauli_z, kron2(mps::Pauli_id, mps::Pauli_id));
      Tensor Hpsi2 = mmult(H, mps_to_vector(psi));

      EXPECT_CEQ(norm2(Hpsi1 - Hpsi2), 0.0);
    }
    {
      MPO mpo(3,2);
      add_local_term(mpo, mps::Pauli_z, 1);
      Tensor Hpsi1 = mps_to_vector(apply(mpo, psi));

      Tensor H = kron2(mps::Pauli_id, kron2(mps::Pauli_z, mps::Pauli_id));
      Tensor Hpsi2 = mmult(H, mps_to_vector(psi));

      EXPECT_CEQ(norm2(Hpsi1 - Hpsi2), 0.0);
    }
    {
      MPO mpo(3,2);
      add_local_term(mpo, mps::Pauli_z, 2);
      Tensor Hpsi1 = mps_to_vector(apply(mpo, psi));

      Tensor H = kron2(mps::Pauli_id, kron2(mps::Pauli_id, mps::Pauli_z));
      Tensor Hpsi2 = mmult(H, mps_to_vector(psi));

      EXPECT_CEQ(norm2(Hpsi1 - Hpsi2), 0.0);
    }
    {
      MPO mpo(3,2);
      add_local_term(mpo, mps::Pauli_z, 0);
      add_local_term(mpo, mps::Pauli_z, 2);
      Tensor Hpsi1 = mps_to_vector(apply(mpo, psi));

      Tensor H = kron2(mps::Pauli_z, kron2(mps::Pauli_id, mps::Pauli_id)) +
        kron2(mps::Pauli_id, kron2(mps::Pauli_id, mps::Pauli_z));
      Tensor Hpsi2 = mmult(H, mps_to_vector(psi));

      EXPECT_CEQ(norm2(Hpsi1 - Hpsi2), 0.0);
    }
  }

  template<class MPO>
  void
  test_small_nn_mpo()
  {
    typedef typename MPO::elt_t Tensor;
    typedef typename MPO::MPS MPS;

    MPS psi = cluster_state(2);
    {
      MPO mpo(2,2);
      add_interaction(mpo, mps::Pauli_z, 0, mps::Pauli_z);
      Tensor Hpsi1 = mps_to_vector(apply(mpo, psi));

      Tensor H = kron2(mps::Pauli_z, mps::Pauli_z);
      Tensor Hpsi2 = mmult(H, mps_to_vector(psi));

      EXPECT_CEQ(norm2(Hpsi1 - Hpsi2), 0.0);
    }
    psi = cluster_state(3);
    {
      MPO mpo(3,2);
      add_interaction(mpo, mps::Pauli_z, 0, mps::Pauli_z);
      Tensor Hpsi1 = mps_to_vector(apply(mpo, psi));

      Tensor H = kron2(mps::Pauli_z, kron2(mps::Pauli_z, mps::Pauli_id));
      Tensor Hpsi2 = mmult(H, mps_to_vector(psi));

      EXPECT_CEQ(norm2(Hpsi1 - Hpsi2), 0.0);
    }
    {
      MPO mpo(3,2);
      add_interaction(mpo, mps::Pauli_z, 1, mps::Pauli_z);
      Tensor Hpsi1 = mps_to_vector(apply(mpo, psi));

      Tensor H = kron2(mps::Pauli_id, kron2(mps::Pauli_z, mps::Pauli_z));
      Tensor Hpsi2 = mmult(H, mps_to_vector(psi));

      EXPECT_CEQ(norm2(Hpsi1 - Hpsi2), 0.0);
    }
    {
      MPO mpo(3,2);
      add_interaction(mpo, mps::Pauli_z, 0, mps::Pauli_z);
      add_interaction(mpo, mps::Pauli_z, 1, mps::Pauli_z);
      Tensor Hpsi1 = mps_to_vector(apply(mpo, psi));

      Tensor H = kron2(mps::Pauli_id, kron2(mps::Pauli_z, mps::Pauli_z)) +
        kron2(mps::Pauli_z, kron2(mps::Pauli_z, mps::Pauli_id));
      Tensor Hpsi2 = mmult(H, mps_to_vector(psi));

      EXPECT_CEQ(norm2(Hpsi1 - Hpsi2), 0.0);
    }
  }

  ////////////////////////////////////////////////////////////

  TEST(RMPO, Zero) {
    test_over_integers(2, 10, test_zero_mpo<RMPO>);
  }

  TEST(RMPO, SmallLocalMPO) {
    test_small_local_mpo<RMPO>();
  }

  TEST(RMPO, SmallNNMPO) {
    test_small_nn_mpo<RMPO>();
  }

  ////////////////////////////////////////////////////////////

  TEST(CMPO, Zero) {
    test_over_integers(2, 10, test_zero_mpo<CMPO>);
  }

  TEST(CMPO, SmallLocalMPO) {
    test_small_local_mpo<CMPO>();
  }

  TEST(CMPO, SmallNNMPO) {
    test_small_nn_mpo<CMPO>();
  }

} // namespace test
