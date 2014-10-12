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


namespace tensor_test {

  using namespace mps;
  using tensor::index;

  template<class MPO>
  const MPO random_local_MPO(index size, int d)
  {
    typedef typename MPO::elt_t Tensor;
    MPO output(size, d);
    for (index i = 0; i < size; i++) {
      Tensor aux = Tensor::random(1,d,d,1);
      output.at(i) = aux + conj(permute(aux, 1,2));
    }
    return output;
  }

  // When a state is in canonical form w.r.t. a given site, and we have a local
  // MPO, the quadratic form is given by the operators on that site
  template<class MPO>
  void test_qform_canonical(typename MPO::MPS psi)
  {
    typedef typename MPO::MPS MPS;
    typedef typename MPS::elt_t Tensor;
    index L = psi.size();
    index d = psi[0].dimension(1);
    std::cout << "L=" << L << ", d=" << d << std::endl;
    MPO mpo = random_local_MPO<MPO>(L, d);
    std::cout << "d=" << largest_bond_dimension(mpo) << std::endl;
    for (index i = 0; i < L; i++) {
      MPS aux = canonical_form_at(psi, i);
      std::cout << "aux=" << aux[0] << std::endl;
      QuadraticForm<MPO> f(mpo, aux, aux, i);
      std::cout << "Qform built\n";
      Tensor H = f.single_site_matrix();
      Tensor op = reshape(mpo[i], d,d);
      EXPECT_CEQ(op, H);
    }
    abort();
  }


  template<class MPS, void (*f)(MPS)>
  void try_over_states(int size) {
    f(cluster_state(size));
    f(MPS::random(size, 2, 1));
    f(MPS::random(size, 3, 1));
    f(MPS::random(size, 4, 1));
  }

  ////////////////////////////////////////////////////////////
  // RQFORM
  //

  TEST(RQForm, LocalCanonical) {
    test_over_integers(2, 10, try_over_states<RMPS,test_qform_canonical<RMPO> >);
  }

  ////////////////////////////////////////////////////////////
  // CQFORM
  //

  TEST(CQForm, LocalCanonical) {
    test_over_integers(2, 10, try_over_states<CMPS,test_qform_canonical<CMPO> >);
  }

} // tensor_test

