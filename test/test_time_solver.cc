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

namespace tensor_test {

using namespace mps;
using namespace linalg;

//////////////////////////////////////////////////////////////////////
// EXACT SOLVERS
//

void split_Hamiltonian(Hamiltonian **ppHeven, Hamiltonian **ppHodd,
                       const Hamiltonian &H) {
  ConstantHamiltonian *pHeven = new ConstantHamiltonian(H.size());
  ConstantHamiltonian *pHodd = new ConstantHamiltonian(H.size());
  for (int i = 0; i < H.size(); i++) {
    ConstantHamiltonian &pHok = (i & 1) ? (*pHodd) : (*pHeven);
    ConstantHamiltonian &pHno = (i & 1) ? (*pHeven) : (*pHodd);
    if (i + 1 < H.size()) {
      for (int j = 0; j < H.interaction_depth(i); j++) {
        pHok.add_interaction(i, H.interaction_left(i, j, 0.0),
                             H.interaction_right(i, j, 0.0));
        pHno.add_interaction(i, H.interaction_left(i, j, 0.0) * 0.0,
                             H.interaction_right(i, j, 0.0) * 0.0);
      }
    }
    pHok.set_local_term(i, H.local_term(i, 0.0) * 0.5);
    pHno.set_local_term(i, H.local_term(i, 0.0) * 0.5);
  }
  *ppHeven = pHeven;
  *ppHodd = pHodd;
}

//////////////////////////////////////////////////////////////////////

void evolve_identity(int size, const CMPS apply_U(const Hamiltonian &H,
                                                  double dt, const CMPS &psi)) {
  auto psi = CMPS(ghz_state(size));
  // Id is a zero operator that causes the evolution operator to
  // be the identity
  TIHamiltonian H(size, RTensor(), RTensor::zeros(2, 2));
  CMPS aux = apply_U(H, 0.1, psi);
  EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
  EXPECT_CEQ3(abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
  EXPECT_CEQ(mps_to_vector(psi), mps_to_vector(aux));
}

void evolve_global_phase(int size,
                         const CMPS apply_U(const Hamiltonian &H, double dt,
                                            const CMPS &psi)) {
  auto psi = CMPS(ghz_state(size));
  // H is a multiple of the identity, causing the evolution
  // operator to be just a global phase
  TIHamiltonian H(size, RTensor(), RTensor::eye(2, 2));
  CMPS aux = apply_U(H, 0.1, psi);
  EXPECT_CEQ3(norm2(aux), 1.0, 10 * EPSILON);
  EXPECT_CEQ3(abs(scprod(aux, psi)), 1.0, 10 * EPSILON);
}

void evolve_local_operator_sz(int size,
                              void do_test(const Hamiltonian &H, double dt,
                                           const CMPS &psi)) {
  double dphi = 1.3 / size;
  ConstantHamiltonian H(size);
  for (int i = 0; i < size; i++) {
    H.set_local_term(i, mps::Pauli_z * (dphi * i));
  }
  do_test(H, 0.1, CMPS(ghz_state(size)));
  do_test(H, 0.1, CMPS(cluster_state(size)));
}

void evolve_local_operator_sx(int size,
                              void do_test(const Hamiltonian &H, double dt,
                                           const CMPS &psi)) {
  double dphi = 0.3;
  ConstantHamiltonian H(size);
  for (int i = 0; i < size; i++) {
    H.set_local_term(i, mps::Pauli_x * dphi);
    dphi = -dphi;
    if (i > 0)
      H.add_interaction(i - 1, CTensor::zeros(2, 2), CTensor::zeros(2, 2));
  }
  do_test(H, 0.1, CMPS(ghz_state(size)));
  do_test(H, 0.1, CMPS(cluster_state(size)));
}

void evolve_interaction_zz(int size, void do_test(const Hamiltonian &H,
                                                  double dt, const CMPS &psi)) {
  double dphi = 1.3 / size;
  ConstantHamiltonian H(size);
  for (int i = 0; i < size; i++) {
    H.set_local_term(i, mps::Pauli_id * 0.0);
    if (i > 0) H.add_interaction(i - 1, mps::Pauli_z, mps::Pauli_z);
  }
  do_test(H, 0.1, CMPS(ghz_state(size)));
  do_test(H, 0.1, CMPS(cluster_state(size)));
}

void evolve_interaction_xx(int size, void do_test(const Hamiltonian &H,
                                                  double dt, const CMPS &psi)) {
  double dphi = 1.3 / size;
  ConstantHamiltonian H(size);
  for (int i = 0; i < size; i++) {
    H.set_local_term(i, mps::Pauli_id * 0.0);
    if (i > 0) H.add_interaction(i - 1, mps::Pauli_x, mps::Pauli_x);
  }
  do_test(H, 0.1, CMPS(ghz_state(size)));
  do_test(H, 0.1, CMPS(cluster_state(size)));
}

}  // namespace tensor_test
