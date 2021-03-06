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
#include <mps/time_evolve.h>
#include <mps/hamiltonian.h>
#include <mps/quantum.h>
#include <tensor/linalg.h>

#include "test_time_solver.cc"

namespace tensor_test {

  CTensor
  apply_trotter3(const Hamiltonian &H, cdouble idt, const CTensor &psi)
  {
    Hamiltonian *pHeven, *pHodd;
    split_Hamiltonian(&pHeven, &pHodd, H);

    CTensor U1 = expm(full(sparse_hamiltonian(*pHodd)) * idt);
    CTensor U2 = expm(full(sparse_hamiltonian(*pHeven)) * (0.5*idt));
    CTensor new_psi = mmult(U2, mmult(U1, mmult(U2, psi)));

    delete pHeven;
    delete pHodd;
    return new_psi;
  }

  const CMPS apply_H_Trotter3(const Hamiltonian &H, double dt, const CMPS &psi)
  {
    Trotter3Solver solver(H, 0.1);
    CMPS aux = psi;
    solver.one_step(&aux, 2);
    return aux;
  }

  void test_Trotter3_no_truncation(const Hamiltonian &H, double dt, const CMPS &psi)
  {
    CMPS aux = psi;
    Trotter3Solver solver(H, dt);
    solver.strategy = Trotter2Solver::DO_NOT_TRUNCATE;
    double err = solver.one_step(&aux, 0);
    EXPECT_CEQ(err, 0.0);
    EXPECT_CEQ(norm2(aux), 1.0);
    CTensor aux2 = apply_trotter3(H, to_complex(0.0,-dt), mps_to_vector(psi));
    EXPECT_CEQ(mps_to_vector(aux), aux2);
  }

  template<int Dmax>
  void test_Trotter3_truncated(const Hamiltonian &H, double dt, const CMPS &psi)
  {
    CMPS truncated_psi_t = psi;
    Trotter3Solver solver(H, dt);
    solver.strategy = Trotter2Solver::TRUNCATE_EACH_LAYER;
    double err = solver.one_step(&truncated_psi_t, Dmax);
    EXPECT_CEQ(norm2(truncated_psi_t), 1.0);
    CTensor psi_t = apply_trotter3(H, to_complex(0.0,-dt), mps_to_vector(psi));
    EXPECT_CEQ(mps_to_vector(truncated_psi_t), psi_t);
  }

  ////////////////////////////////////////////////////////////
  // EVOLVE WITH TROTTER METHODS
  //

  TEST(Trotter3Solver, Identity) {
    test_over_integers(2, 10, evolve_identity, apply_H_Trotter3);
  }

  TEST(Trotter3Solver, GlobalPhase) {
    test_over_integers(2, 10, evolve_global_phase, apply_H_Trotter3);
  }

  TEST(Trotter3Solver, LocalOperatorSzFull) {
    test_over_integers(2, 5, evolve_local_operator_sz, test_Trotter3_no_truncation);
  }

  TEST(Trotter3Solver, LocalOperatorSzTruncated) {
    test_over_integers(2, 5, evolve_local_operator_sz, test_Trotter3_truncated<2>);
  }

  TEST(Trotter3Solver, LocalOperatorSxFull) {
    test_over_integers(2, 5, evolve_local_operator_sx, test_Trotter3_no_truncation);
  }

  TEST(Trotter3Solver, LocalOperatorSxTruncated) {
    test_over_integers(2, 5, evolve_local_operator_sx, test_Trotter3_truncated<2>);
  }

  TEST(Trotter3Solver, NearestNeighborSzSzFull) {
    test_over_integers(2, 5, evolve_interaction_zz, test_Trotter3_no_truncation);
  }

  TEST(Trotter3Solver, NearestNeighborSzSzTruncated) {
    test_over_integers(2, 5, evolve_interaction_zz, test_Trotter3_truncated<4>);
  }

  TEST(Trotter3Solver, NearestNeighborSxSxFull) {
    test_over_integers(2, 5, evolve_interaction_xx, test_Trotter3_no_truncation);
  }

  TEST(Trotter3Solver, NearestNeighborSxSxTruncated) {
    test_over_integers(2, 5, evolve_interaction_xx, test_Trotter3_truncated<4>);
  }

} // namespace tensor_test
