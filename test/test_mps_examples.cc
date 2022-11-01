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

TEST_P(TestOverSizes, GHZ) {
  RMPS ghz = ghz_state(size());
  EXPECT_EQ(ghz.size(), size());
  RTensor psi = mps_to_vector(ghz);
  double v = 1.0 / sqrt((double)2.0);
  for (index i = 0; i < psi.size(); i++) {
    double psi_i = ((i == 0) || (i == psi.size() - 1)) ? v : 0.0;
    EXPECT_DOUBLE_EQ(psi[i], psi_i);
  }
  EXPECT_DOUBLE_EQ(norm2(psi), 1.0);
}

const RMPS apply_cluster_state_stabilizer(RMPS state, int site) {
  state = apply_local_operator(state, mps::Pauli_x, site);
  if (site > 0) state = apply_local_operator(state, mps::Pauli_z, site - 1);
  if (site < state.last())
    state = apply_local_operator(state, mps::Pauli_z, site + 1);
  return state;
}

TEST_P(TestOverSizes, cluster_state) {
  RMPS cluster = cluster_state(size());
  RTensor psi = mps_to_vector(cluster);
  EXPECT_EQ(cluster.size(), size());
  EXPECT_EQ(psi.size(), 2 << (size() - 1));
  EXPECT_NEAR(norm2(psi), 1.0, STRICT_EPSILON);
  for (index i = 1; i < cluster.size(); i++) {
    RTensor psi2 = mps_to_vector(apply_cluster_state_stabilizer(cluster, i));
    if (!simeq(psi, psi2)) {
      RMPS aux = apply_cluster_state_stabilizer(cluster, i);
      std::cerr << "site=" << i << '\n';
      std::cerr << "psi2=" << psi2 << '\n';
      std::cerr << "psi1=" << psi << '\n';
    }
    EXPECT_CEQ(psi, psi2);
  }
}

INSTANTIATE_TEST_SUITE_OVER_SIZES(MPSExamples, 1, 10);

}  // namespace tensor_test