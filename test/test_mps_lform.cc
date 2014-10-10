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

  template<class MPS>
  void test_lform_norm2(MPS psi)
  {
    {
      typename MPS::elt_t w(1);
      w.randomize();
      std::vector<MPS> v(1);
      v.at(0) = psi;
      LinearForm<MPS> f(w, v, psi, 0);
      EXPECT_CEQ(f.norm2(), norm2(w));
    }
    {
      typename MPS::elt_t w(2);
      w.randomize();
      std::vector<MPS> v(2);
      v.at(0) = v.at(1) = psi;
      LinearForm<MPS> f(w, v, psi, 0);
      double n = sqrt(tensor::abs(sum(kron(conj(w), w))));
      EXPECT_CEQ(f.norm2(), n);
    }
  }

  //
  // Canonical form of a state that does not require simplification.
  //
  template<class MPS>
  void test_lform_canonical(MPS psi)
  {
    // When a state is in canonical form w.r.t. a given site, the linear form is
    // described just by the tensor on that site, because the transfer matrices
    // are the identity.
    for (index i = 0; i < psi.size(); i++) {
      MPS aux = canonical_form_at(psi, i);
      LinearForm<MPS> f(aux, aux, i);
      EXPECT_CEQ(aux[i], f.single_site_vector());
    }
  }

  template<class MPS>
  void test_lform_canonical2(MPS psi)
  {
    // Same as before, but we use two or more vectors instead of one and assign
    // them weights.
    std::vector<MPS> vs(2);
    typename MPS::elt_t w(2);
    for (index i = 0; i < psi.size(); i++) {
      vs.at(0) = vs.at(1) = canonical_form_at(psi, i);
      w.at(0) = 0.13; w.at(1) = -1.57;
      LinearForm<MPS> f(w, vs, vs[0], i);
      EXPECT_CEQ((w.at(0) + w.at(1)) * vs[0][i], f.single_site_vector());
    }
  }

  template<class MPS, void (*f)(MPS)>
  void try_over_states(int size) {
    f(cluster_state(size));
  }

  ////////////////////////////////////////////////////////////
  // SIMPLIFY RMPS
  //

  TEST(RLForm, Norm2) {
    test_over_integers(2, 10, try_over_states<RMPS,test_lform_norm2<RMPS> >);
  }

  TEST(RLForm, Canonical1State) {
    test_over_integers(2, 10, try_over_states<RMPS,test_lform_canonical<RMPS> >);
  }

  TEST(RLForm, Canonical2States) {
    test_over_integers(2, 10, try_over_states<RMPS,test_lform_canonical2<RMPS> >);
  }

  ////////////////////////////////////////////////////////////
  // SIMPLIFY CMPS
  //

  TEST(CLForm, Norm2) {
    test_over_integers(2, 10, try_over_states<CMPS,test_lform_norm2<CMPS> >);
  }

  TEST(CLForm, Canonical1State) {
    test_over_integers(2, 10, try_over_states<CMPS,test_lform_canonical<CMPS> >);
  }

  TEST(CLForm, Canonical2States) {
    test_over_integers(2, 10, try_over_states<CMPS,test_lform_canonical2<CMPS> >);
  }


} // namespace tensor_test
