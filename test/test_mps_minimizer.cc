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

#include <tensor/io.h>
#include "loops.h"
#include <gtest/gtest.h>
#include <mps/mps.h>
#include <mps/hamiltonian.h>
#include <mps/minimizer.h>

namespace tensor_test {

using namespace mps;
using tensor::index;

template <class MPO, class MPS>
double ground_state(const MPO &mpo, MPS *output) {
  typedef typename MPS::elt_t Tensor;

  index L = mpo.size();
  MPS psi(L);
  index D = 2;
  for (index i = 0; i < mpo.size(); i++) {
    Tensor P = RTensor::random(D, 2, D) - 0.5;
    psi.at(i) = P / norm2(P);
  }
  psi.at(0) = psi[0](range(0), range(), range());
  psi.at(L - 1) = psi[L - 1](range(), range(), range(0));

  MinimizerOptions opts;
  opts.Dmax = std::min(1 << (L / 2), 50);
  double E = minimize(mpo, &psi, opts);
  *output = psi;
  return E;
}

// Create a random Hamiltonian in MPO form and verify that it is giving
// the same expected values as the Quadratic form.
template <class MPO, int model>
void test_minimizer_model(index L) {
  typedef typename MPO::MPS MPS;
  typedef typename MPS::elt_t Tensor;
  typedef typename Tensor::elt_t number;
  // Random Hamiltonian of spin 1/2 model with the given
  TestHamiltonian H(model, 0.5, L, false, false);

  MPO mpo(H);
  MPS psi;
  double minE = ground_state(mpo, &psi);

  // The minimal energy returned by the algorithm must coincide with
  // the expectation value (we are using QuadraticForm, which
  // fulfills this)
  psi = canonical_form(psi, 0);
  //std::cout << "psi=" << mps_to_vector(psi) << std::endl;
  //std::cout << "H=" << matrix_form(sparse_hamiltonian(H)) << std::endl;
  double expectedE = expected(psi, H, 0.0);
  EXPECT_CEQ(minE, expectedE);

  // The state must be an eigenstate of the Hamiltonian with given
  // eigenvalue
  MPS psi2 = apply(mpo, psi);
  number E = scprod(psi, psi2);
  double angle = abs(E) / norm2(psi2);
  EXPECT_CEQ(minE, E);
  EXPECT_CEQ3(angle, 1.0, 1e-10);
}

////////////////////////////////////////////////////////////
// MINIMIZE RMPO
//

TEST(RMinimize, ZField) {
  test_over_integers(2, 10,
                     test_minimizer_model<RMPO, TestHamiltonian::Z_FIELD>);
}

TEST(RMinimize, Ising) {
  test_over_integers(2, 10, test_minimizer_model<RMPO, TestHamiltonian::ISING>);
}

TEST(RMinimize, IsingZField) {
  test_over_integers(
      2, 10, test_minimizer_model<RMPO, TestHamiltonian::ISING_Z_FIELD>);
}

TEST(RMinimize, IsingXField) {
  test_over_integers(
      2, 10, test_minimizer_model<RMPO, TestHamiltonian::ISING_X_FIELD>);
}

TEST(RMinimize, XY) {
  test_over_integers(2, 10, test_minimizer_model<RMPO, TestHamiltonian::XY>);
}

TEST(RMinimize, XXZ) {
  test_over_integers(2, 10, test_minimizer_model<RMPO, TestHamiltonian::XXZ>);
}

TEST(RMinimize, Heisenberg) {
  test_over_integers(2, 10,
                     test_minimizer_model<RMPO, TestHamiltonian::HEISENBERG>);
}

////////////////////////////////////////////////////////////
// MINIMIZE CMPO
//

TEST(CMinimize, ZField) {
  test_over_integers(2, 10,
                     test_minimizer_model<CMPO, TestHamiltonian::Z_FIELD>);
}

TEST(CMinimize, Ising) {
  test_over_integers(2, 10, test_minimizer_model<CMPO, TestHamiltonian::ISING>);
}

TEST(CMinimize, IsingZField) {
  test_over_integers(
      2, 10, test_minimizer_model<CMPO, TestHamiltonian::ISING_Z_FIELD>);
}

TEST(CMinimize, IsingXField) {
  test_over_integers(
      2, 10, test_minimizer_model<CMPO, TestHamiltonian::ISING_X_FIELD>);
}

TEST(CMinimize, XY) {
  test_over_integers(2, 10, test_minimizer_model<CMPO, TestHamiltonian::XY>);
}

TEST(CMinimize, XXZ) {
  test_over_integers(2, 10, test_minimizer_model<CMPO, TestHamiltonian::XXZ>);
}

TEST(CMinimize, Heisenberg) {
  test_over_integers(2, 10,
                     test_minimizer_model<CMPO, TestHamiltonian::HEISENBERG>);
}

}  // namespace tensor_test
