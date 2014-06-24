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

#include <algorithm>
#include "loops.h"
#include <tensor/numbers.h>
#include <tensor/linalg.h>
#include <tensor/arpack.h>
#include <mps/mps.h>
#include <mps/dmrg.h>
#include <mps/quantum.h>

using namespace tensor_test;
using namespace tensor;
using namespace mps;

//======================================================================

#if 1
#define DMRG RDMRG
#define CMPS RMPS
#endif

void test_dmrg_gap()
{
  //
  // We prepare a simple AKLT Hamiltonian, whose gap we control and
  // which is exactly solvable
  //
  CTensor sx, sy, sz;
  spin_operators(1, &sx, &sy, &sz);
  CTensor H1 = CTensor::zeros(sz.dimensions());
  CTensor H12 = real(kron(sx,sx) + kron(sy,sy) + kron(sz,sz));

  //
  // These matrices are used to build an antiferromagnetic state which
  // is similar to the ground state.
  //
  RTensor P[2];
  P[0] = reshape(rgen << 1.0 << 0.0 << 0.0, 1, 3, 1);
  P[1] = reshape(rgen << 0.0 << 0.0 << 1.0, 1, 3, 1);

  std::cout << "Computing energy gaps for s=1\n"
	    << "=============================\n";

  // Exact energies for Sz=1 subspace
  double E0[]= {
    //
    // AKLT
    //
    -1.33333333333333, -0.33333333333333, 0.33333333333333,
    -2.00000000000000, -1.10208826828127, -1.05808955154502,
    -2.66666666666667, -1.84018705476851, -1.77386736344380,
    -3.33333333333333, -2.53643086977244, -2.51477951935688,
    -3.99999999999999, -3.22680947203538, -3.20638672469596,
    -4.66666666666665, -3.90796840052703, -3.89612051086983,
    -5.33333333333335, -4.58566773737622, -4.57633012406655,
    -5.99999999999999, -5.26031105515244, -5.25361436082523,
    -6.66666666666668, -5.93315262962396, -5.92792387526474,
    //
    // Heisenberg
    //
    -3.00000000000000, -1.00000000000000, -1.00000000000000,
    -4.13658152134851, -2.79128784747792, -2.61803398874990,
    -5.83021252277083, -4.40526186008417, -4.39670975876054,
    -7.06248941077255, -6.01853339744504, -5.83601252972241,
    -8.63453198270616, -7.52385106298521, -7.50230573903891,
    -9.92275854832016, -9.04914106101226, -8.89788012440900,
    -11.43293164033019, -10.51108838743845, -10.48540537556920,
    -12.75622919691644, -11.99072610774712, -11.86911757196646,
    -14.23035896960049, -13.43143361113490, -13.40562630781231
  };
  RTensor Exact(igen << 3 << 9 << 2,
		Vector<double>(3*9*2, E0));

  for (int model = 0; model < 2; model++) {
    size_t Dmax = 80;
    double err = 0.0;
    std::cout << (model? "Heisenberg " : "AKLT       ");
    for (size_t L = 3; L <= 8/*11*/; L++) {
      // 1) Create the Hamiltonian (translationally invariant, OBC)
      TIHamiltonian H(L, model? H12 : (H12 + mmult(H12,H12)/3.0), H1);

      // 2) Create the estimate for the ground state (AF)
      CMPS psi0(L);
      for (size_t i = 0; i < L; i++) {
	psi0.at(i) = P[i&1];
      }
      CMPS psi1 = psi0;

      // 3) Solve with restrictions
      DMRG solver(H);
      solver.sweeps = 128;
      solver.debug = 0;
      solver.tolerance = 1e-10;
      solver.Q_operators = DMRG::elt_vector_t(1);
      solver.Q_operators.at(0) = real(sz);
      solver.Q_values = DMRG::elt_t(1);
      solver.Q_values.at(0) = 1;

      double E0 = solver.minimize(&psi0, Dmax);

      // 4) Repeat the procedure, but with an excited state
      solver.orthogonal_to(psi0);
      double E1 = solver.minimize(&psi1, Dmax);
      double diff = abs(Exact(0, L-3, model) - E0) + abs(Exact(1, L-3, model) - E1);
      err = std::max(err, diff);
      if (diff < 3e-9) {
	std::cout << '.';
      } else {
	std::cout << '!';
      }
      std::cout.flush();
    }
    std::cout << " Max. err. = " << err << std::endl;
  }
  std::cout << std::endl;
}

bool
test_dmrg_inner(const mps::Hamiltonian &H, double &err)
{
  // This value of Dmax guarantees that there are no truncation
  // errors due to the Matrix Product States ansatz.
  size_t L = H.size();
  size_t max_Dmax = (unsigned int)pow((double)H.dimension(0), (double)L/2);
  size_t small_Dmax = 3*max_Dmax/4;
  if (small_Dmax + 20 >= max_Dmax)
    small_Dmax = max_Dmax;

  err = 0.0;
  for (int iter = 1; iter < 7; iter++) {
    DMRG s(H);
    s.sweeps = 128;
    s.debug = 0;
    s.tolerance = 1e-9;
    //
    // We scan all combinations of
    //  - block and normal SVD
    // accurate_svd = (iter & 1) == 1;
    //  - single and two-sites algorithm
    bool twosites = (iter & 2) == 1;
    //  - exact or truncated spaces
    size_t Dmax;
    double tol;
    if (iter & 4) {
      tol = 1e-7;
      Dmax = small_Dmax;
    } else {
      tol = 10*s.tolerance;
      Dmax = max_Dmax;
    }

    if (twosites && (L == 1))
      continue;

    CMPS P0 = normal_form(RMPS::random(H.dimensions(), Dmax,
				       H.is_periodic()));
#if 1
    CTensor aux = linalg::eigs(sparse_hamiltonian(H),
			       linalg::SmallestAlgebraic, 1);
    double realE = std::min(real(aux));
#else
    RTensor aux = eig_sym(full(H.sparse_matrix(0)));
    double realE = std::min(aux);
#endif
    double E, e;
    if (twosites) {
      E = s.minimize(&P0, Dmax);
    } else {
      E = s.minimize(&P0);
    }
    // accurate_svd = false;
    e = abs(E - realE);
    err = std::max(e,err);
    if (e > tol)
      return false;
  }
  return true;
}

int main(int argc, char **argv)
{
  if (!mps_init(&argc, &argv)) {
    exit(-1);
  }
  test_dmrg_gap();

  std::cout << "Testing DMRG algorithm\n"
	    << "======================\n";

  test_over_H(test_dmrg_inner, 14);

  return 0;
}

/// Local variables:
/// mode: c++
/// fill-column: 80
/// c-basic-offset: 4
/// End:
