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
#include <mps/lform.h>

namespace tensor_test {

  using namespace mps;
  using tensor::index;

  // The norm of a Linear form associated to vector 'v' is the norm of the
  // vector itself
  template<class MPS>
  void test_lform_norm(MPS psi)
  {
    typename MPS::elt_t w(1);
    w.randomize();
    std::vector<MPS> v(1);
    v.at(0) = normal_form(psi);
    LinearForm<MPS> f(w, v, psi, 0);
    EXPECT_CEQ(f.norm2(), norm2(w));
  }

  // The norm of a Linear form associated to vector 'a*v + b*v' is the norm of the
  // vector itself
  template<class MPS>
  void test_lform_norm2(MPS psi)
  {
    typename MPS::elt_t w(2);
    w.randomize();
    std::vector<MPS> v(2);
    v.at(0) = v.at(1) = normal_form(psi);
    LinearForm<MPS> f(w, v, psi, 0);
    double n = sqrt(tensor::abs(sum(kron(conj(w), w))));
    EXPECT_CEQ(f.norm2(), n);
  }

  // When a state is in canonical form w.r.t. a given site, the linear form is
  // described just by the tensor on that site, because the transfer matrices
  // are the identity.
  template<class MPS>
  void test_lform_canonical(MPS psi)
  {
    for (index i = 0; i < psi.size(); i++) {
      MPS aux = canonical_form_at(psi, i);
      LinearForm<MPS> f(aux, aux, i);
      EXPECT_CEQ(aux[i], conj(f.single_site_vector()));
    }
  }

  // Same as before, but we use two or more vectors instead of one and assign
  // them weights.
  template<class MPS>
  void test_lform_canonical2(MPS psi)
  {
    std::vector<MPS> vs(2);
    typename MPS::elt_t w(2);
    for (index i = 0; i < psi.size(); i++) {
      vs.at(0) = vs.at(1) = canonical_form_at(psi, i);
      w.at(0) = 0.13; w.at(1) = -1.57;
      LinearForm<MPS> f(w, vs, vs[0], i);
      EXPECT_CEQ((w.at(0) + w.at(1)) * vs[0][i], conj(f.single_site_vector()));
    }
  }

  // When a state is in canonical form w.r.t. a given site, the linear form is
  // described just by the tensor on that site, because the transfer matrices
  // are the identity.
  template<class MPS>
  void test_lform_canonical_2_sites(MPS psi)
  {
    for (index i = 0; i < (psi.size()-1); i++) {
      MPS aux = canonical_form_at(psi, i);
      LinearForm<MPS> f(aux, aux, i);
      typename MPS::elt_t P12 = fold(aux[i],-1, aux[i+1],0);
      EXPECT_CEQ(P12, conj(f.two_site_vector(+1)));
    }
  }

  template<class MPS, void (*f)(MPS)>
  void try_over_states(int size) {
    f(cluster_state(size));
    f(MPS::random(size, 2, 1));
    f(MPS::random(size, 3, 1));
    f(MPS::random(size, 4, 1));
  }

  ////////////////////////////////////////////////////////////
  // SIMPLIFY RMPS
  //

  TEST(RLForm, Norm1State) {
    test_over_integers(2, 10, try_over_states<RMPS,test_lform_norm<RMPS> >);
  }

  TEST(RLForm, Norm2States) {
    test_over_integers(2, 10, try_over_states<RMPS,test_lform_norm2<RMPS> >);
  }

  TEST(RLForm, Canonical1State) {
    test_over_integers(2, 10, try_over_states<RMPS,test_lform_canonical<RMPS> >);
  }

  TEST(RLForm, Canonical2States) {
    test_over_integers(2, 10, try_over_states<RMPS,test_lform_canonical2<RMPS> >);
  }

  TEST(RLForm, Canonical1State2sites) {
    test_over_integers(2, 10, try_over_states<RMPS,test_lform_canonical_2_sites<RMPS> >);
  }

  ////////////////////////////////////////////////////////////
  // SIMPLIFY CMPS
  //

  TEST(CLForm, Norm1State) {
    test_over_integers(2, 10, try_over_states<CMPS,test_lform_norm<CMPS> >);
  }

  TEST(CLForm, Norm2States) {
    test_over_integers(2, 10, try_over_states<CMPS,test_lform_norm2<CMPS> >);
  }

  TEST(CLForm, Canonical1State) {
    test_over_integers(2, 10, try_over_states<CMPS,test_lform_canonical<CMPS> >);
  }

  TEST(CLForm, Canonical2States) {
    test_over_integers(2, 10, try_over_states<CMPS,test_lform_canonical2<CMPS> >);
  }

  TEST(CLForm, Canonical1State2sites) {
    test_over_integers(2, 10, try_over_states<CMPS,test_lform_canonical_2_sites<CMPS> >);
  }


} // namespace tensor_test
