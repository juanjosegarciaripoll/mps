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
#include <mps/hamiltonian.h>
#include <mps/quantum.h>
#include <mps/mps_algorithms.h>

namespace tensor_test {

  using namespace mps;
  using tensor::index;

  // Create a MPO with a single, random operator acting on 'site'. All other
  // sites get the identity.
  const CMPO diagonal_MPO(Indices d, CSparse *Hsparse)
  {
    index size = d.size();
    RTensor phases = linspace(0, 1.0, size + 2);
    ConstantHamiltonian H(size);
    if (0) {
      for (index i = 0; i < size; i++) {
        H.set_local_term(i, (i? 0.0 : 1.0) * mps::Pauli_z);
      }
      if (Hsparse) {
        *Hsparse = sparse_hamiltonian(H);
      }
      return H;
    }
    for (index i = 0; i < size; i++) {
      RTensor op = diag(linspace(0.1, 0.8, d[i]));
      H.set_local_term(i, op * phases(i+1));
    }
    if (Hsparse) {
      *Hsparse = sparse_hamiltonian(H);
    }
    return H;
  }

  // When a state is in canonical form w.r.t. a given site, and we have a local
  // MPO, the quadratic form is given by the operators on that site, times the
  // indentity on the bond dimensions.
  void test_solve_diagonal(CMPS psi)
  {
    int sense = -1;
    CMPS phi = psi;
    CSparse A;
    CMPO H = diagonal_MPO(dimensions(psi), &A);
    solve(H, &phi, psi, &sense, 1);
    
    CTensor vpsi = mps_to_vector(psi);
    CTensor vphi = mps_to_vector(phi);
    CTensor Aphi = mmult(A, vphi);

    std::cout << "In vectors:\n"
              << " psi=" << vpsi << std::endl
              << " phi=" << vphi << std::endl
              << " A*phi=" << Aphi << std::endl
              << "A=" << matrix_form(A) << std::endl;

    double scp = tensor::abs(scprod(Aphi, vpsi)) / norm2(vpsi) / norm2(Aphi);
    double err = norm2(Aphi - vpsi) / norm2(vpsi);
    std::cout << " scprod=" << scp << ", err=" << err << std::endl;
  }

  template<class MPS, void (*f)(MPS)>
  void try_over_states(int size) {
    //f(cluster_state(size));
    f(MPS::random(size, 2, 1));
  }

  ////////////////////////////////////////////////////////////
  // RQFORM
  //

  TEST(CQForm, Diagonal) {
    test_over_integers(2, 2, &try_over_states<CMPS,test_solve_diagonal>);
  }

} // tensor_test

