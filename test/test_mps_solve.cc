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
#include <mps/hamiltonian.h>
#include <mps/quantum.h>
#include <mps/algorithms.h>

namespace tensor_test {

using namespace mps;

const CMPO build_MPO(void (*fH)(ConstantHamiltonian &H, const Indices &d),
                     const Indices &d, CSparse *Hsparse) {
  ConstantHamiltonian H(d.size());
  fH(H, d);
  if (Hsparse) {
    *Hsparse = sparse_hamiltonian(H);
  }
  return Hamiltonian_to<CMPO>(H);
}

// Given an input state, 'psi', it creates some random Hamiltonian
// using the function fH and solves the equation
//		H*xi = y
// where y = H*psi. Of course, the answer is xi = psi and it is
// therefore verifiable.
//
template <void (*fH)(ConstantHamiltonian &H, const Indices &d)>
void test_solve(CMPS psi) {
  /*
     * We create a random Hamiltonian H and solve
     *
     */
  int sense = -1;
  CSparse Hsp;
  CMPO H = build_MPO(fH, dimensions(psi), &Hsp);
  CMPS Hpsi = mps::apply(H, psi);
  CMPS xi = Hpsi;
  solve(H, &xi, Hpsi, &sense, 4);

  CTensor vxi = mps_to_vector(xi);
  CTensor vHxi = mmult(Hsp, vxi);
  CTensor vHpsi = mps_to_vector(Hpsi);
  double scp = tensor::abs(scprod(vHxi, vHxi)) / norm2(vHxi) / norm2(vHpsi);
  double err = norm2(vHxi - vHxi) / norm2(vHxi);

  EXPECT_CEQ3(err, 0.0, mps::FLAGS.get(MPS_SOLVE_TOLERANCE));
  EXPECT_CEQ3(scp, 1.0, mps::FLAGS.get(MPS_SOLVE_TOLERANCE));
}

// Create a non-negative operator made of \sigma_x*\sigma_x interactions or
// generalizations thereby to higher dimensions.
void xx_H(ConstantHamiltonian &H, const Indices &d) {
  index size = H.size();
  RTensor phases = linspace(0, 1.0, size + 2);
  for (index i = 0; i < size; i++) {
    H.set_local_term(i, CTensor::zeros(d[i], d[i]));
  }
  for (index i = 1; i < size; i++) {
    RTensor op = RTensor::zeros(d[i], d[i]);
    for (index j = 0; j < d[i]; j++) {
      op.at(j, d[i] - 1 - j) = 1.0;
      op.at(j, j) = 1.0;
    }
    H.set_interaction(i - 1, phases[i] * op, op);
  }
}

// Create a non-negative operator made of interactions between nearest
// neighbor sites, where the interaction Hamiltonian is built from diagonal,
// non-negative operators.
void zz_H(ConstantHamiltonian &H, const Indices &d) {
  index size = H.size();
  RTensor phases = linspace(0, 1.0, size + 2);
  for (index i = 0; i < size; i++) {
    H.set_local_term(i, CTensor::zeros(d[i], d[i]));
  }
  for (index i = 1; i < size; i++) {
    RTensor op = diag(linspace(0.1, 0.8, d[i]));
    H.set_interaction(i - 1, op, op);
  }
}

// Create a non-negative operator that is a sum of local operators, all
// diagonal, but all acting with different weights on different sites.
void diagonal_H(ConstantHamiltonian &H, const Indices &d) {
  index size = H.size();
  RTensor phases = linspace(0, 1.0, size + 2);
  for (index i = 0; i < size; i++) {
    RTensor op = diag(linspace(0.1, 0.8, d[i]));
    H.set_local_term(i, op * phases(i + 1));
  }
}

template <class MPS, void (*f)(MPS), bool two_sites>
void try_over_states(int size) {
  if (two_sites)
    mps::FLAGS.set(MPS_SOLVE_ALGORITHM, MPS_TWO_SITE_ALGORITHM);
  else
    mps::FLAGS.set(MPS_SOLVE_ALGORITHM, MPS_SINGLE_SITE_ALGORITHM);
  f(MPS(cluster_state(size)));
  f(MPS::random(size, 2, 1));
}

////////////////////////////////////////////////////////////
// RQFORM
//
/*
  TEST(CQForm, DiagonalLocal1site) {
    test_over_integers(2, 10, &try_over_states<CMPS,test_solve<diagonal_H>,false>);
  }

  TEST(CQForm, ZZInt1site) {
    test_over_integers(2, 10, &try_over_states<CMPS,test_solve<zz_H>,false>);
  }

  TEST(CQForm, XXInt1site) {
    test_over_integers(2, 10, &try_over_states<CMPS,test_solve<xx_H>,false>);
  }
  */
TEST(CQForm, DiagonalLocal2site) {
  test_over_integers(2, 10,
                     &try_over_states<CMPS, test_solve<diagonal_H>, true>);
}

TEST(CQForm, ZZInt2site) {
  test_over_integers(2, 10, &try_over_states<CMPS, test_solve<zz_H>, true>);
}

TEST(CQForm, XXInt2site) {
  test_over_integers(2, 10, &try_over_states<CMPS, test_solve<xx_H>, true>);
}

}  // namespace tensor_test
