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
#include <mps/mps.h>
#include <mps/time_evolve.h>
#include <mps/hamiltonian.h>
#include <mps/quantum.h>
#include <tensor/linalg.h>

#include "test_time_solver.cc"

namespace tensor_test {

  CTensor
  apply_Arnoldi(const Hamiltonian &H, cdouble idt, const CTensor &psi)
  {
    return mmult(expm(full(sparse_hamiltonian(H)) * idt), psi);
  }

  template<int nvectors>
  const CMPS apply_H_Arnoldi(const Hamiltonian &H, double dt, const CMPS &psi)
  {
    ArnoldiSolver solver(H, 0.1, nvectors);
    CMPS aux = psi;
    solver.one_step(&aux, 2);
    return aux;
  }

  template<int Dmax, int nvectors>
  void test_Arnoldi_truncated(const Hamiltonian &H, double dt, const CMPS &psi)
  {
    CMPS truncated_psi_t = psi;
    ArnoldiSolver solver(H, dt, nvectors);
    double err = solver.one_step(&truncated_psi_t, Dmax);
    EXPECT_CEQ(norm2(truncated_psi_t), 1.0);
    CTensor psi_t = apply_Arnoldi(H, to_complex(0.0,-dt), mps_to_vector(psi));
    EXPECT_CEQ(mps_to_vector(truncated_psi_t), psi_t);
  }

  ////////////////////////////////////////////////////////////
  // EVOLVE WITH TROTTER METHODS
  //

  TEST(ArnoldiSolver, Identity) {
    test_over_integers(2, 10, evolve_identity, apply_H_Arnoldi<3>);
  }

  TEST(ArnoldiSolver, GlobalPhase) {
    test_over_integers(2, 10, evolve_global_phase, apply_H_Arnoldi<3>);
  }

  TEST(ArnoldiSolver, LocalOperatorSzTruncated) {
    test_over_integers(2, 5, evolve_local_operator_sz, test_Arnoldi_truncated<2,5>);
  }

  TEST(ArnoldiSolver, LocalOperatorSxTruncated) {
    test_over_integers(2, 5, evolve_local_operator_sx, test_Arnoldi_truncated<2,5>);
  }

  TEST(ArnoldiSolver, NearestNeighborSzSzTruncated) {
    test_over_integers(2, 5, evolve_interaction_zz, test_Arnoldi_truncated<4,5>);
  }

  TEST(ArnoldiSolver, NearestNeighborSxSxTruncated) {
    test_over_integers(2, 5, evolve_interaction_xx, test_Arnoldi_truncated<4,5>);
  }

} // namespace tensor_test
