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
template <class MPO>
void test_qform_canonical(const typename MPO::MPS &psi) {
  typedef typename MPO::MPS MPS;
  typedef mp_tensor_t<MPO> Tensor;
  index L = psi.size();
  index d = psi[0].dimension(1);
  for (index i = 0; i < L; i++) {
    MPO mpo = random_local_MPO<MPO>(L, d, i);
    MPS aux = canonical_form_at(psi, i);
    QuadraticForm<Tensor> f(mpo, aux, aux, i);
    Tensor H = f.single_site_matrix();
    Tensor op = reshape(mpo[i](range(0), _, _, range(0)).copy(), d, d);
    Tensor L = Tensor::eye(psi[i].dimension(0));
    Tensor R = Tensor::eye(psi[i].dimension(2));
    op = kron(kron(R, op), L);
    EXPECT_CEQ(op, H);
  }
}

// Create a random Hamiltonian in MPO form and verify that it is giving
// Hamiltonians of the right size
template <class Tensor, int model>
void test_qform_shape(const MPS<Tensor> &psi) {
  // Random Hamiltonian of spin 1/2 model with the given
  index L = psi.size();
  TestHamiltonian H(model, 0.5, L, false, false);
  MPO<Tensor> mpo(H);
  if (psi[0].dimension(1) != mpo[0].dimension(1)) return;
  // We run over all sites
  for (index i = 0; i < L; i++) {
    QuadraticForm<Tensor> qf(mpo, psi, psi, i);
    Tensor Hqform = qf.single_site_matrix();
    ASSERT_EQ(Hqform.rows(), Hqform.columns());
    EXPECT_EQ(Hqform.columns(), psi[i].size());
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
  MPO<Tensor> mpo(H);
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
template <class MPO, int model>
void test_qform_expected(typename MPO::MPS psi) {
  typedef typename MPO::MPS MPS;
  typedef typename MPS::elt_t Tensor;
  typedef tensor_scalar_t<Tensor> number;
  // Random Hamiltonian of spin 1/2 model with the given
  index L = psi.size();
  if (psi[0].dimension(1) == 2) {
    TestHamiltonian H(model, 0.5, L, false, false);
    MPO mpo(H);
    // We run over all sites
    for (index i = 0; i < L; i++) {
      QuadraticForm<Tensor> qf(mpo, psi, psi, i);
      Tensor Hqform = qf.single_site_matrix();
      // and on each site we try random matrices and verify
      // that the total expectation value is the same one.
      std::cerr << "Hqform = " << Hqform << '\n' << "psik = " << psi[i] << '\n';
      for (index j = 0; j < 10; j++) {
        Tensor psik = flatten(psi[i]);
        number psikHpsik = scprod(psik, mmult(Hqform, psik));
        number psiHpsi = scprod(psi, mps::apply(mpo, psi));
        EXPECT_CEQ(psiHpsi, psikHpsik);
        psi.at(i).randomize();
        psi.at(i) = psi[i] / norm2(psi[i]);
      }
    }
  }
}

// Create a random Hamiltonian in MPO form and verify that it is giving
// the same expected values as the Quadratic form.
template <class MPO, int model>
void test_qform_expected2sites(typename MPO::MPS psi) {
  typedef typename MPO::MPS MPS;
  typedef typename MPS::elt_t Tensor;
  typedef tensor_scalar_t<Tensor> number;
  // Random Hamiltonian of spin 1/2 model with the given
  index L = psi.size();
  if (psi[0].dimension(1) == 2) {
    TestHamiltonian H(model, 0.5, L, false, false);
    MPO mpo(H);
    // We run over all sites
    for (index i = 1; i < L; i++) {
      QuadraticForm<Tensor> qf(mpo, psi, psi, i - 1);
      Tensor H = qf.two_site_matrix(DIR_RIGHT);
      // and on each site we try random matrices and verify
      // that the total expectation value is the same one.
      for (index j = 0; j < 10; j++) {
        // for that we first use the two_site_matrix()
        {
          Tensor psik = flatten(fold(psi[i - 1], -1, psi[i], 0));
          number psikHpsik = scprod(psik, mmult(H, psik));
          number psiHpsi = scprod(psi, apply(mpo, psi));
          EXPECT_CEQ(psiHpsi, psikHpsik);
        }
        // and then we use the light-weight application of the tensors
        {
          Tensor psik = fold(psi[i - 1], -1, psi[i], 0);
          number psikHpsik = scprod(psik, qf.apply_two_site_matrix(psik, +1));
          number psiHpsi = scprod(psi, apply(mpo, psi));
          EXPECT_CEQ(psiHpsi, psikHpsik);
        }
        psi.at(i - 1).randomize();
        psi.at(i - 1) = psi[i - 1] / norm2(psi[i - 1]);
        psi.at(i).randomize();
        psi.at(i) = psi[i] / norm2(psi[i]);
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

#if 0

TEST(RQForm, LocalCanonical) {
  test_over_integers(2, 10, try_over_states<RMPS>(test_qform_canonical<RMPO>));
}

//--------------------------------------------------

TEST(RQForm, ExpectedIsing) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS,
                      test_qform_expected<RMPO, TestHamiltonian::ISING> >);
}

TEST(RQForm, ExpectedZField) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS,
                      test_qform_expected<RMPO, TestHamiltonian::Z_FIELD> >);
}

TEST(RQForm, ExpectedIsingZField) {
  test_over_integers(
      2, 10,
      try_over_states<
          RMPS, test_qform_expected<RMPO, TestHamiltonian::ISING_Z_FIELD> >);
}

TEST(RQForm, ExpectedIsingXField) {
  test_over_integers(
      2, 10,
      try_over_states<
          RMPS, test_qform_expected<RMPO, TestHamiltonian::ISING_X_FIELD> >);
}

TEST(RQForm, ExpectedXY) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS, test_qform_expected<RMPO, TestHamiltonian::XY> >);
}

TEST(RQForm, ExpectedXXZ) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS, test_qform_expected<RMPO, TestHamiltonian::XXZ> >);
}

TEST(RQForm, ExpectedHeisenberg) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS,
                      test_qform_expected<RMPO, TestHamiltonian::HEISENBERG> >);
}

//--------------------------------------------------

TEST(RQForm, Expected2sitesIsing) {
  test_over_integers(
      2, 10,
      try_over_states<
          RMPS, test_qform_expected2sites<RMPO, TestHamiltonian::ISING> >);
}

TEST(RQForm, Expected2sitesZField) {
  test_over_integers(
      2, 10,
      try_over_states<
          RMPS, test_qform_expected2sites<RMPO, TestHamiltonian::Z_FIELD> >);
}

TEST(RQForm, Expected2sitesIsingZField) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS, test_qform_expected2sites<
                                RMPO, TestHamiltonian::ISING_Z_FIELD> >);
}

TEST(RQForm, Expected2sitesIsingXField) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS, test_qform_expected2sites<
                                RMPO, TestHamiltonian::ISING_X_FIELD> >);
}

TEST(RQForm, Expected2sitesXY) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS,
                      test_qform_expected2sites<RMPO, TestHamiltonian::XY> >);
}

TEST(RQForm, Expected2sitesXXZ) {
  test_over_integers(
      2, 10,
      try_over_states<RMPS,
                      test_qform_expected2sites<RMPO, TestHamiltonian::XXZ> >);
}

TEST(RQForm, Expected2sitesHeisenberg) {
  test_over_integers(
      2, 10,
      try_over_states<
          RMPS, test_qform_expected2sites<RMPO, TestHamiltonian::HEISENBERG> >);
}

////////////////////////////////////////////////////////////
// CQFORM
//

TEST(CQForm, LocalCanonical) {
  test_over_integers(2, 10, try_over_states<CMPS, test_qform_canonical<CMPO> >);
}

//--------------------------------------------------

TEST(CQForm, ExpectedIsing) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS,
                      test_qform_expected<CMPO, TestHamiltonian::ISING> >);
}

TEST(CQForm, ExpectedZField) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS,
                      test_qform_expected<CMPO, TestHamiltonian::Z_FIELD> >);
}

TEST(CQForm, ExpectedIsingZField) {
  test_over_integers(
      2, 10,
      try_over_states<
          CMPS, test_qform_expected<CMPO, TestHamiltonian::ISING_Z_FIELD> >);
}

TEST(CQForm, ExpectedIsingXField) {
  test_over_integers(
      2, 10,
      try_over_states<
          CMPS, test_qform_expected<CMPO, TestHamiltonian::ISING_X_FIELD> >);
}

TEST(CQForm, ExpectedXY) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS, test_qform_expected<CMPO, TestHamiltonian::XY> >);
}

TEST(CQForm, ExpectedXXZ) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS, test_qform_expected<CMPO, TestHamiltonian::XXZ> >);
}

TEST(CQForm, ExpectedHeisenberg) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS,
                      test_qform_expected<CMPO, TestHamiltonian::HEISENBERG> >);
}

//--------------------------------------------------

TEST(CQForm, Expected2sitesIsing) {
  test_over_integers(
      2, 10,
      try_over_states<
          CMPS, test_qform_expected2sites<CMPO, TestHamiltonian::ISING> >);
}

TEST(CQForm, Expected2sitesZField) {
  test_over_integers(
      2, 10,
      try_over_states<
          CMPS, test_qform_expected2sites<CMPO, TestHamiltonian::Z_FIELD> >);
}

TEST(CQForm, Expected2sitesIsingZField) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS, test_qform_expected2sites<
                                CMPO, TestHamiltonian::ISING_Z_FIELD> >);
}

TEST(CQForm, Expected2sitesIsingXField) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS, test_qform_expected2sites<
                                CMPO, TestHamiltonian::ISING_X_FIELD> >);
}

TEST(CQForm, Expected2sitesXY) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS,
                      test_qform_expected2sites<CMPO, TestHamiltonian::XY> >);
}

TEST(CQForm, Expected2sitesXXZ) {
  test_over_integers(
      2, 10,
      try_over_states<CMPS,
                      test_qform_expected2sites<CMPO, TestHamiltonian::XXZ> >);
}

TEST(CQForm, Expected2sitesHeisenberg) {
  test_over_integers(
      2, 10,
      try_over_states<
          CMPS, test_qform_expected2sites<CMPO, TestHamiltonian::HEISENBERG> >);
}

#endif

}  // namespace tensor_test
