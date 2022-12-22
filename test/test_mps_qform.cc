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
#include <mps/qform.h>
#include <mps/quantum.h>
#include <mps/io.h>

namespace tensor_test {

using namespace mps;

// Create a MPO with a single, random operator acting on 'site'. All other
// sites get the identity.
template <class MPO>
const MPO random_local_MPO(index size, int d, index site) {
  typedef typename MPO::elt_t Tensor;
  MPO output(size, d);
  Tensor id = reshape(Tensor::eye(d, d), 1, d, d, 1);
  Tensor op = reshape(Tensor::random(d, d), 1, d, d, 1);
  for (index i = 0; i < size; i++) {
    output.at(i) = (i == site) ? op : id;
  }
  return output;
}

// When a state is in canonical form w.r.t. a given site, and we have a local
// MPO, the quadratic form is given by the operators on that site, times the
// indentity on the bond dimensions.
template <class Tensor>
void test_qform_canonical(const MPS<Tensor> &psi) {
  index L = psi.size();
  index d = psi[0].dimension(1);
  for (index i = 0; i < L; i++) {
    auto mpo = random_local_MPO<MPO<Tensor>>(L, d, i);
    auto aux = canonical_form_at(psi, i);
    QuadraticForm<Tensor> f(mpo, aux, aux, i);
    Tensor H = f.single_site_matrix();
    Tensor op = reshape(mpo[i](range(0), _, _, range(0)).copy(), d, d);
    Tensor left_matrix = Tensor::eye(psi[i].dimension(0));
    Tensor right_matrix = Tensor::eye(psi[i].dimension(2));
    op = kron(kron(right_matrix, op), left_matrix);
    ASSERT_CEQ(op, H);
  }
}

// Create a random Hamiltonian in MPO form and verify that it is giving
// Hamiltonians of the right size
template <class Tensor, int model>
void test_qform_shape(const MPS<Tensor> &psi) {
  // Random Hamiltonian of spin 1/2 model with the given
  index L = psi.size();
  TestHamiltonian H(model, 0.5, L, false, false);
  auto mpo = Hamiltonian_to<MPO<Tensor>>(H);
  if (psi[0].dimension(1) != mpo[0].dimension(1)) return;
  // We run over all sites
  for (index i = 0; i < L; i++) {
    QuadraticForm<Tensor> qf(mpo, psi, psi, i);
    Tensor Hqform = qf.single_site_matrix();
    ASSERT_EQ(Hqform.rows(), Hqform.columns());
    ASSERT_EQ(Hqform.columns(), psi[i].size());
    if (Hqform.columns() != psi[i].size()) {
      abort();
    }
  }
}

template <class Tensor, int model>
void test_qform2_shape(const MPS<Tensor> &psi) {
  // Random Hamiltonian of spin 1/2 model with the given
  index L = psi.size();
  TestHamiltonian H(model, 0.5, L, false, false);
  auto mpo = Hamiltonian_to<MPO<Tensor>>(H);
  if (psi[0].dimension(1) != mpo[0].dimension(1)) return;
  // We run over all sites
  for (index i = 1; i < L; i++) {
    QuadraticForm<Tensor> qf(mpo, psi, psi, i - 1);
    Tensor Hqform = qf.two_site_matrix(DIR_RIGHT);
    ASSERT_EQ(Hqform.rows(), Hqform.columns());
    index expected_size = psi[i - 1].dimension(0) * psi[i - 1].dimension(1) *
                          psi[i].dimension(1) * psi[i].dimension(2);
    ASSERT_EQ(Hqform.columns(), expected_size);
  }
}

// Create a random Hamiltonian in MPO form and verify that it is giving
// the same expected values as the Quadratic form.
template <class Tensor, int model>
void test_qform_expected(const MPS<Tensor> &psi0) {
  // Random Hamiltonian of spin 1/2 model with the given
  if (psi0[0].dimension(1) == 2) {
    auto psi = psi0;
    index L = psi.size();
    TestHamiltonian H(model, 0.5, L, false, false);
    auto mpo = Hamiltonian_to<MPO<Tensor>>(H);
    // We run over all sites
    for (index i = 0; i < L; i++) {
      QuadraticForm<Tensor> qf(mpo, psi, psi, i);
      Tensor Hqform = qf.single_site_matrix();
      // verify that we produce a single-site square matrix with
      // dimensions equals to the tensor it will act upon
      EXPECT_EQ(Hqform.rank(), 2);
      EXPECT_EQ(Hqform.rows(), Hqform.columns());
      EXPECT_EQ(Hqform.columns(), psi[i].size());
      // and on each site we try random matrices and verify
      // that the total expectation value is the same one.
      for (index j = 0; j < 10; j++) {
        Tensor psij = flatten(psi[i]);
        Tensor Hpsij = mmult(Hqform, psij);
        auto psijHpsij = scprod(psij, Hpsij);
        auto psiHpsi = scprod(psi, apply(mpo, psi));
        ASSERT_CEQ(psiHpsi, psijHpsij);
        // and then we use the light-weight application of the tensors
        Tensor Hpsij2 = qf.apply_single_site_matrix(psij);
        ASSERT_EQ(Hpsij2.size(), Hpsij.size());
        ASSERT_EQ(Hpsij2.rank(), 1);
        auto psijHpsij2 = scprod(psij, Hpsij2);
        ASSERT_CEQ(psiHpsi, psijHpsij2);
        ASSERT_CEQ(Hpsij, Hpsij2);
        psi.at(i).randomize();
        psi.at(i) /= norm2(flatten(psi[i]));
      }
    }
  }
}

// Create a random Hamiltonian in MPO form and verify that it is giving
// the same expected values as the Quadratic form.
template <class Tensor, int model>
void test_qform_expected2(const MPS<Tensor> &psi0) {
  // Random Hamiltonian of spin 1/2 model with the given
  if (psi0[0].dimension(1) == 2) {
    auto psi = psi0;
    index L = psi.size();
    auto testH = TestHamiltonian(model, 0.5, L, false, false);
    auto mpo = Hamiltonian_to<MPO<Tensor>>(testH);
    // We run over all sites
    for (index i = 1; i < L; i++) {
      QuadraticForm<Tensor> qf(mpo, psi, psi, i - 1);
      // and on each site we try random matrices and verify
      // that the total expectation value is the same one.
      for (index j = 0; j < 10; j++) {
        Tensor psij = flatten(fold(psi[i - 1], -1, psi[i], 0));
        Tensor Hqform = qf.two_site_matrix(DIR_RIGHT);
        EXPECT_EQ(Hqform.rank(), 2);
        EXPECT_EQ(Hqform.rows(), Hqform.columns());
        EXPECT_EQ(Hqform.columns(), psij.size());
        // for that we first use the two_site_matrix()
        Tensor Hpsij = mmult(Hqform, psij);
        auto psijHpsij = scprod(psij, Hpsij);
        auto psiHpsi = scprod(psi, apply(mpo, psi));
        EXPECT_CEQ(psiHpsi, psijHpsij);
        // and then we use the light-weight application of the tensors
        Tensor Hpsij2 = qf.apply_two_site_matrix(psij, DIR_RIGHT);
        EXPECT_EQ(Hpsij2.size(), Hpsij.size());
        EXPECT_EQ(Hpsij2.rank(), 1);
        auto psijHpsij2 = scprod(psij, Hpsij2);
        EXPECT_CEQ(psiHpsi, psijHpsij2);
        EXPECT_CEQ(Hpsij, Hpsij2);
        psi.at(i - 1).randomize();
        psi.at(i - 1) /= norm2(flatten(psi[i - 1]));
        psi.at(i).randomize();
        psi.at(i) /= norm2(flatten(psi[i]));
      }
    }
  }
}

template <class MPS>
auto try_over_states(void (*f)(const MPS &)) {
  return [=](int size) {
    f(MPS(cluster_state(size)));
    f(MPS::random(size, 2, 1));
    f(MPS::random(size, 3, 1));
    f(MPS::random(size, 4, 1));
  };
}

////////////////////////////////////////////////////////////
// RQFORM
//

TEST(RQForm, SingleSiteShape) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS>(test_qform_shape<RTensor, TestHamiltonian::ISING>));
}

TEST(RQForm, TwoSiteShape) {
  test_over_integers(2, 10,
                     try_over_states<RMPS>(
                         test_qform2_shape<RTensor, TestHamiltonian::ISING>));
}

TEST(RQForm, LocalCanonical) {
  test_over_integers(2, 10,
                     try_over_states<RMPS>(test_qform_canonical<RTensor>));
}

//--------------------------------------------------

TEST(RQForm, ExpectedZField) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS>(
          test_qform_expected<RTensor, TestHamiltonian::Z_FIELD>));
}

TEST(RQForm, ExpectedIsing) {
  test_over_integers(2, 10,
                     try_over_states<RMPS>(
                         test_qform_expected<RTensor, TestHamiltonian::ISING>));
}

TEST(RQForm, ExpectedIsingZField) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS>(
          test_qform_expected<RTensor, TestHamiltonian::ISING_Z_FIELD>));
}

TEST(RQForm, ExpectedIsingXField) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS>(
          test_qform_expected<RTensor, TestHamiltonian::ISING_X_FIELD>));
}

TEST(RQForm, ExpectedXY) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS>(test_qform_expected<RTensor, TestHamiltonian::XY>));
}

TEST(RQForm, ExpectedXXZ) {
  test_over_integers(2, 10,
                     try_over_states<RMPS>(
                         test_qform_expected<RTensor, TestHamiltonian::XXZ>));
}

TEST(RQForm, ExpectedHeisenberg) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS>(
          test_qform_expected<RTensor, TestHamiltonian::HEISENBERG>));
}

//--------------------------------------------------

TEST(RQForm, Expected2sitesIsing) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS>(
          test_qform_expected2<RTensor, TestHamiltonian::ISING_Z_FIELD>));
}

TEST(RQForm, Expected2sitesZField) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS>(
          test_qform_expected2<RTensor, TestHamiltonian::Z_FIELD>));
}

TEST(RQForm, Expected2sitesIsingZField) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS>(
          test_qform_expected2<RTensor, TestHamiltonian::ISING_Z_FIELD>));
}

TEST(RQForm, Expected2sitesIsingXField) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS>(
          test_qform_expected2<RTensor, TestHamiltonian::ISING_X_FIELD>));
}

TEST(RQForm, Expected2sitesXY) {
  test_over_integers(2, 10,
                     try_over_states<RMPS>(
                         test_qform_expected2<RTensor, TestHamiltonian::XY>));
}

TEST(RQForm, Expected2sitesXXZ) {
  test_over_integers(2, 10,
                     try_over_states<RMPS>(
                         test_qform_expected<RTensor, TestHamiltonian::XXZ>));
}

TEST(RQForm, Expected2sitesHeisenberg) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS>(
          test_qform_expected2<RTensor, TestHamiltonian::HEISENBERG>));
}

////////////////////////////////////////////////////////////
// CQFORM
//

TEST(CQForm, SingleSiteShape) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS>(test_qform_shape<CTensor, TestHamiltonian::ISING>));
}

TEST(CQForm, TwoSiteShape) {
  test_over_integers(2, 10,
                     try_over_states<CMPS>(
                         test_qform2_shape<CTensor, TestHamiltonian::ISING>));
}

TEST(CQForm, LocalCanonical) {
  test_over_integers(2, 10,
                     try_over_states<CMPS>(test_qform_canonical<CTensor>));
}

//--------------------------------------------------

TEST(CQForm, ExpectedZField) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS>(
          test_qform_expected<CTensor, TestHamiltonian::Z_FIELD>));
}

TEST(CQForm, ExpectedIsing) {
  test_over_integers(2, 10,
                     try_over_states<CMPS>(
                         test_qform_expected<CTensor, TestHamiltonian::ISING>));
}

TEST(CQForm, ExpectedIsingZField) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS>(
          test_qform_expected<CTensor, TestHamiltonian::ISING_Z_FIELD>));
}

TEST(CQForm, ExpectedIsingXField) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS>(
          test_qform_expected<CTensor, TestHamiltonian::ISING_X_FIELD>));
}

TEST(CQForm, ExpectedXY) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS>(test_qform_expected<CTensor, TestHamiltonian::XY>));
}

TEST(CQForm, ExpectedXXZ) {
  test_over_integers(2, 10,
                     try_over_states<CMPS>(
                         test_qform_expected<CTensor, TestHamiltonian::XXZ>));
}

TEST(CQForm, ExpectedHeisenberg) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS>(
          test_qform_expected<CTensor, TestHamiltonian::HEISENBERG>));
}

//--------------------------------------------------

TEST(CQForm, Expected2sitesIsing) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS>(
          test_qform_expected2<CTensor, TestHamiltonian::ISING_Z_FIELD>));
}

TEST(CQForm, Expected2sitesZField) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS>(
          test_qform_expected2<CTensor, TestHamiltonian::Z_FIELD>));
}

TEST(CQForm, Expected2sitesIsingZField) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS>(
          test_qform_expected2<CTensor, TestHamiltonian::ISING_Z_FIELD>));
}

TEST(CQForm, Expected2sitesIsingXField) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS>(
          test_qform_expected2<CTensor, TestHamiltonian::ISING_X_FIELD>));
}

TEST(CQForm, Expected2sitesXY) {
  test_over_integers(2, 10,
                     try_over_states<CMPS>(
                         test_qform_expected2<CTensor, TestHamiltonian::XY>));
}

TEST(CQForm, Expected2sitesXXZ) {
  test_over_integers(2, 10,
                     try_over_states<CMPS>(
                         test_qform_expected<CTensor, TestHamiltonian::XXZ>));
}

TEST(CQForm, Expected2sitesHeisenberg) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS>(
          test_qform_expected2<CTensor, TestHamiltonian::HEISENBERG>));
}

}  // namespace tensor_test
