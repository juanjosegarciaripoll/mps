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

#include <tensor/linalg.h>
#include <tensor/io.h>
#include <mps/mps_algorithms.h>
#include <mps/time_evolve.h>
#include <mps/tools.h>
#include <mps/io.h>

namespace mps {

  using namespace linalg;

  ArnoldiSolver::ArnoldiSolver(const Hamiltonian &H, cdouble dt, int nvectors) :
    TimeSolver(dt), H_(H, 0.0), max_states_(nvectors), tolerance_(0)
  {
    if (max_states_ <= 0 || max_states_ >= 30) {
      std::cerr << "In ArnoldiSolver(...), the number of states exceeds the limits [1,30]"
		<< std::endl;
      abort();
    }
  }

  ArnoldiSolver::ArnoldiSolver(const CMPO &H, cdouble dt, int nvectors) :
    TimeSolver(dt), H_(H), max_states_(nvectors), tolerance_(0)
  {
    if (max_states_ <= 0 || max_states_ >= 30) {
      std::cerr << "In ArnoldiSolver(...), the number of states exceeds the limits [1,30]"
		<< std::endl;
      abort();
    }
  }

  double
  ArnoldiSolver::one_step(CMPS *psi, index Dmax)
  {
    CTensor N = CTensor::zeros(max_states_, max_states_);
    CTensor Heff = N;
    CMPS current = normal_form(*psi, -1);
    CMPS Hcurrent = apply(H_, current);
    int debug = mps::FLAGS.get(MPS_DEBUG_ARNOLDI);

    std::vector<CMPS> states;
    states.reserve(max_states_);
    states.push_back(current);
    N.at(0,0) = to_complex(1.0);
    Heff.at(0,0) = real(scprod(current, Hcurrent));

    std::vector<CMPS> vectors(3);
    std::vector<cdouble> coeffs(3);
    std::vector<double> errors;
    double err, n;
    int sense;
    for (int ndx = 1; ndx < max_states_; ndx++) {
      const CMPS &last = states[ndx-1];
      //
      // 0) Estimate a new vector of the basis.
      //	scurrent = H v[0] - <v[1]|H|v[0]> v[1] - <v[2]|H|v[0]> v[2]
      //    where
      //	v[0] = states[ndx-1]
      //	v[1] = states[ndx-2]
      //
      current = Hcurrent = canonical_form(Hcurrent, -1);
      {
	vectors.clear();
	coeffs.clear();

	vectors.push_back(current);
	coeffs.push_back(number_one<cdouble>());

	vectors.push_back(states[ndx-1]);
	//coeffs.push_back(-scprod(current, states[ndx-1]));
        coeffs.push_back(-Heff(ndx-1,ndx-1));

	if (ndx > 1) {
	  vectors.push_back(states[ndx-2]);
          //coeffs.push_back(-scprod(current, states[ndx-2]));
          coeffs.push_back(-Heff(ndx-1,ndx-2));
	}
        int sense = +1;
	err = simplify_obc(&current, coeffs, vectors, &sense, 2, true,
                           2*Dmax, -1, &n);
        if (sense < 0) {
          current = canonical_form(current, -1);
        }
        if (debug >= 2) {
          std::cout << "ndx=" << ndx << ", err=" << err
                    << ", n=" << norm2(current) << "=" << n << ", tol="
                    << tolerance_ << ", sense=" << sense << std::endl;
        }
        if (n < 1e-15 ||
            (tolerance_ && (n < tolerance_*std::max(norm2(Hcurrent), 1.0))))
        {
          if (debug >= 2) {
            std::cout << "Arnoldi method converged before tolerance\n";
          }
          N = N(range(0,ndx-1),range(0,ndx-1));
          Heff = Heff(range(0,ndx-1),range(0,ndx-1));
          break;
        }
        /* We ensure that the states are normalized and with a canonical
         * form opposite to the sense of the simplification above. This
         * improves stability and speed in the SVDs. */
        if (sense > 0) {
          current = normal_form(current, -1);
        }
        errors.push_back(err / n);
      }

      //
      // 1) Add the matrices of the new vector to the whole set and, at the same time
      //    compute the scalar products of the old vectors with the new one.
      //    Also compute the matrix elements of the Hamiltonian in this new basis.
      //
      states.push_back(current);
      Hcurrent = apply(H_, current);
      for (int n = 0; n < ndx; n++) {
	cdouble aux;
	N.at(n, ndx) = aux = scprod(states[n], current);
	N.at(ndx, n) = tensor::conj(aux);
	Heff.at(n, ndx) = aux = scprod(states[n], Hcurrent);
	Heff.at(ndx, n) = tensor::conj(aux);
      }
      N.at(ndx, ndx) = 1.0;
      Heff.at(ndx, ndx) = real(scprod(current, Hcurrent));
    }
    //
    // 2) Once we have the basis, we compute the exponential on it. Notice that, since
    //    our set of states is not orthonormal, we have to first orthogonalize it, then
    //    compute the exponential and finally move on to the original basis and build
    //    the approximate vector.
    //
    CTensor coef = CTensor::zeros(igen << Heff.rows());
    cdouble idt = to_complex(0, -1) * time_step();
    coef.at(0) = to_complex(1.0);
    coef = mmult(expm(idt * solve_with_svd(N, Heff)), coef);
    if (debug >= 2) {
      std::cout << "N=" << matrix_form(tensor::abs(N)) << std::endl
                << "H=" << matrix_form(tensor::abs(Heff)) << std::endl
                << "U=" << matrix_form(expm(idt * solve_with_svd(N, Heff)))
                << std::endl
                << "H/N=" << matrix_form(solve_with_svd(N, Heff))
                << std::endl
                << "v=" << matrix_form(coef) << std::endl
                << "|v|=" << norm2(coef) << std::endl
                << "|v|=" << scprod(coef, mmult(N, coef)) << std::endl
                << "idt=" << idt << std::endl;
    }

    //
    // 4) Here is where we perform the truncation from our basis to a single MPS.
    //
    sense = +1;
    err = simplify_obc(psi, coef, states, &sense, 12, true, Dmax, -1);
    err += scprod(RTensor(errors), square(abs(coef)));
    if (debug) {
      std::cout << "Arnoldi final truncation error " << err << std::endl;
    }
    return err;
  }

} // namespace mps
