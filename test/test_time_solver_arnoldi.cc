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

#include <vector>
#include "loops.h"
#include <gtest/gtest.h>
#include <gtest/gtest-death-test.h>
#include <mps/mps.h>
#include <mps/time_evolve.h>
#include <mps/hamiltonian.h>
#include <mps/quantum.h>
#include <tensor/linalg.h>

#include "test_time_solver.cc"

#define ARNOLDI_EPSILON 1e-10

namespace tensor_test {

  template<class Ham, class Tensor>
  CTensor
  arnoldi_expm(const Ham &H, const Tensor &psi, cdouble idt, int max_states = 5)
  {
    typedef typename Tensor::elt_t number;
    double tolerance = 1e-10;

    if (max_states <= 0) {
      std::cerr << "In arnoldi_expm(), max_states must be a non-negative number.\n";
      abort();
    }

    Tensor N = CTensor::zeros(max_states, max_states);
    Tensor Heff = N;

    std::vector<Tensor> states;
    states.reserve(max_states);
    states.push_back(psi / norm2(psi));
    N.at(0,0) = to_complex(1.0);
    Heff.at(0,0) = scprod(psi, mmult(H, psi));

    for (int ndx = 1; ndx < max_states; ndx++) {
      const Tensor &last = states[ndx-1];
      //
      // 0) Estimate a new vector of the basis.
      //	current = H v[0] - <v[1]|H|v[0]> v[1] - <v[2]|H|v[0]> v[2]
      //    where
      //	v[0] = states[ndx-1]
      //	v[1] = states[ndx-2]
      //
      Tensor current = mmult(H, states.back());
      double n0 = norm2(current);

      current = current - scprod(states[ndx-1], current) * states[ndx-1];
      if (ndx > 1) {
        current = current - scprod(states[ndx-2], current) * states[ndx-2];
      }
      {
        double n = norm2(current);
        if (n < tolerance * std::max(n0, 1.0)) {
          N = N(range(0,ndx-1),range(0,ndx-1));
          Heff = Heff(range(0,ndx-1),range(0,ndx-1));
          break;
        }
        current = current / n;
      }

      //
      // 1) Add the matrices of the new vector to the whole set and, at the same time
      //    compute the scalar products of the old vectors with the new one.
      //    Also compute the matrix elements of the Hamiltonian in this new basis.
      //
      states.push_back(current);
      for (int n = 0; n <= ndx; n++) {
	cdouble aux;
	N.at(n, ndx) = aux = scprod(states[n], current);
	N.at(ndx, n) = tensor::conj(aux);
	Heff.at(n, ndx) = aux = scprod(states[n], mmult(H, current));
	Heff.at(ndx, n) = tensor::conj(aux);
      }
      N.at(ndx,ndx) = real(N(ndx,ndx));
      Heff.at(ndx,ndx) = real(Heff(ndx,ndx));
    }
    //
    // 2) Once we have the basis, we compute the exponential on it. Notice that, since
    //    our set of states is not orthonormal, we have to first orthogonalize it, then
    //    compute the exponential and finally move on to the original basis and build
    //    the approximate vector.
    //
    CTensor coef = CTensor::zeros(igen << Heff.rows());
    coef.at(0) = to_complex(1.0);
    coef = mmult(expm(idt * solve_with_svd(N, Heff)), coef);
    //
    // 4) Here is where we perform the truncation from our basis to a single MPS.
    //
    CTensor output = CTensor::zeros(psi.dimensions());
    for (int i = 0; i < coef.size(); i++)
      output = output + states[i] * coef[i];
    return output;
  }


  template<int Dmax, int nvectors>
  const CMPS apply_H_Arnoldi(const Hamiltonian &H, double dt, const CMPS &psi)
  {
    ArnoldiSolver solver(H, 0.1, nvectors);
    CMPS aux = psi;
    solver.one_step(&aux, Dmax);
    return aux;
  }

  template<int Dmax, int nvectors>
  void test_Arnoldi_truncated(const Hamiltonian &H, double dt, const CMPS &psi)
  {
    CMPS truncated_psi_t = psi;
    ArnoldiSolver solver(H, dt, nvectors);
    double err = solver.one_step(&truncated_psi_t, Dmax);
    EXPECT_CEQ(norm2(truncated_psi_t), 1.0);

    cdouble idt = to_complex(0, -dt);
    CTensor Hm = full(sparse_hamiltonian(H));
    CTensor psiv = mps_to_vector(psi);
    CTensor psi1 = mmult(expm(Hm * idt), psiv);
    CTensor psi2 = arnoldi_expm(Hm, psiv, idt, nvectors);
    CTensor psi3 = mps_to_vector(truncated_psi_t);

    std::cout << "|psi1 - psi2| = " << norm2(psi1 - psi2) << "\t"
              << "|psi1 - psi3| = " << norm2(psi3 - psi1) << std::endl;
    EXPECT_TRUE(norm2(psi3 - psi1) < std::max(ARNOLDI_EPSILON, 10 * norm2(psi1 - psi2)));
  }

  ////////////////////////////////////////////////////////////
  // EVOLVE WITH TROTTER METHODS
  //

  TEST(ArnoldiSolver, Identity) {
    test_over_integers(2, 7, evolve_identity, apply_H_Arnoldi<2,3>);
  }

  TEST(ArnoldiSolver, GlobalPhase) {
    test_over_integers(2, 7, evolve_global_phase, apply_H_Arnoldi<2,3>);
  }

  TEST(ArnoldiSolver, LocalOperatorSzTruncated) {
    test_over_integers(2, 7, evolve_local_operator_sz, test_Arnoldi_truncated<4,6>);
  }

  TEST(ArnoldiSolver, LocalOperatorSxTruncated) {
    test_over_integers(2, 7, evolve_local_operator_sx, test_Arnoldi_truncated<4,5>);
  }

  TEST(ArnoldiSolver, NearestNeighborSzSzTruncated) {
    test_over_integers(2, 7, evolve_interaction_zz, test_Arnoldi_truncated<7,6>);
  }

  TEST(ArnoldiSolver, NearestNeighborSxSxTruncated) {
    test_over_integers(2, 7, evolve_interaction_xx, test_Arnoldi_truncated<4,6>);
  }

} // namespace tensor_test
