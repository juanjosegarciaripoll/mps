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
#include <mps/mps.h>
#include <mps/mps_algorithms.h>

namespace tensor_test {

using namespace tensor;
using namespace mps;
using tensor::index;

template <class MPS>
const MPS add_errors(const MPS &psi) {
  MPS output = psi;
  for (index i = 0; i < output.size(); i++) {
    typename MPS::elt_t x = output[i];
    output.at(i) = x + 0.01 * x.random(x.dimensions());
  }
  return output;
}

//
// Simplifying a state that does not require simplification.
//
template <class MPS, bool two_sites>
void trivial_simplify(int size) {
  MPS psi = cluster_state(size);
  MPS aux = psi;
  int sense;

  if (two_sites)
    mps::FLAGS.set(MPS_SIMPLIFY_ALGORITHM, MPS_TWO_SITE_ALGORITHM);
  else
    mps::FLAGS.set(MPS_SIMPLIFY_ALGORITHM, MPS_SINGLE_SITE_ALGORITHM);

  for (int sweeps = 1; sweeps < 3; sweeps++) {
    sense = +1;
    aux = canonical_form(psi, sense);
    simplify_obc(&aux, psi, &sense, sweeps, true);

    EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
    EXPECT_CEQ3(tensor::abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
    EXPECT_CEQ(mps_to_vector(psi), mps_to_vector(aux));

    sense = -1;
    aux = canonical_form(psi, sense);
    simplify_obc(&aux, psi, &sense, 1, true);

    EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
    EXPECT_CEQ3(tensor::abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
    EXPECT_CEQ(mps_to_vector(psi), mps_to_vector(aux));
  }
}

//
// Simplifying a state that does not require simplification,
// but adding some errors.
//
template <class MPS, bool two_sites>
void trivial_simplify_with_errors(int size) {
  MPS psi = cluster_state(size);
  MPS aux = psi;
  int sense;

  if (two_sites)
    mps::FLAGS.set(MPS_SIMPLIFY_ALGORITHM, MPS_TWO_SITE_ALGORITHM);
  else
    mps::FLAGS.set(MPS_SIMPLIFY_ALGORITHM, MPS_SINGLE_SITE_ALGORITHM);

  for (int sweeps = 1; sweeps < 3; sweeps++) {
    aux = add_errors(psi);
    sense = +1;
    simplify_obc(&aux, psi, &sense, sweeps, true);

    EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
    EXPECT_CEQ3(tensor::abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
    EXPECT_CEQ(mps_to_vector(psi), mps_to_vector(aux));

    aux = add_errors(psi);
    sense = -1;
    simplify_obc(&aux, psi, &sense, 1, true);

    EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
    EXPECT_CEQ3(tensor::abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
    EXPECT_CEQ(mps_to_vector(psi), mps_to_vector(aux));
  }
}

////////////////////////////////////////////////////////////
// SIMPLIFY RMPS
//

TEST(RMPSSimplify, Identity) {
  test_over_integers(2, 10, trivial_simplify<RMPS, false>);
}

TEST(RMPSSimplify, IdentityWithErrors) {
  test_over_integers(2, 10, trivial_simplify_with_errors<RMPS, false>);
}

TEST(RMPSSimplify, Identity2sites) {
  test_over_integers(2, 10, trivial_simplify<RMPS, true>);
}

TEST(RMPSSimplify, IdentityWithErrors2sites) {
  test_over_integers(2, 10, trivial_simplify_with_errors<RMPS, true>);
}

////////////////////////////////////////////////////////////
// SIMPLIFY CMPS
//

TEST(CMPSSimplify, Identity) {
  test_over_integers(2, 10, trivial_simplify<CMPS, false>);
}

TEST(CMPSSimplify, IdentityWithErrors) {
  test_over_integers(2, 10, trivial_simplify_with_errors<CMPS, false>);
}

TEST(CMPSSimplify, Identity2sites) {
  test_over_integers(2, 10, trivial_simplify<CMPS, true>);
}

TEST(CMPSSimplify, IdentityWithErrors2sites) {
  test_over_integers(2, 10, trivial_simplify_with_errors<CMPS, true>);
}

}  // namespace tensor_test
