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

namespace tensor_test {

  using namespace mps;
  using namespace tensor;

  template<class MPO>
  void test_zero_mpo(int size)
  {
    typedef typename MPO::elt_t Tensor;

    Tensor first = Tensor::zeros(1,2,2,2);

    Tensor last = Tensor::zeros(2,2,2,1);
    last.at(1,0,0,0) = 1.0;
    last.at(1,1,1,0) = 1.0;

    Tensor middle = Tensor::zeros(2,2,2,2);
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
  }


  ////////////////////////////////////////////////////////////

  TEST(RMPO, Zero) {
    test_over_integers(2, 10, test_zero_mpo<RMPO>);
  }

  ////////////////////////////////////////////////////////////

  TEST(CMPO, Zero) {
    test_over_integers(2, 10, test_zero_mpo<CMPO>);
  }


} // namespace test
