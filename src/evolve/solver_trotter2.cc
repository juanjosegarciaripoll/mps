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

#include <mps/time_evolve.h>
#include <mps/mps_algorithms.h>

namespace mps {

  /**********************************************************************
   * Trotter's method with only two passes.
   *
   *	exp(-iHdt) = \prod_k={N-1}^1 exp(-iH_{kk+1} dt/2) \prod_k=1^{N-1} exp(-iH_{kk+1} dt/2)
   */

  Trotter2Solver::Trotter2Solver(const Hamiltonian &H, cdouble dt) :
    TrotterSolver(dt), U(H, 0, dt/2.0, false), sense(0)
  {
  }

  double
  Trotter2Solver::one_step(CMPS *P, index Dmax)
  {
    int debug = FLAGS.get(MPS_DEBUG_TROTTER);

    switch (strategy) {
    case TRUNCATE_EACH_UNITARY: {
      double err;
      if (debug) std::cout << "Trotter2 method: truncate unitaries\n"
                           << "Trotter2 Layer 1/2\n";
      err = U.apply(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cout << "Trotter2 Layer 2/2\n";
      err += U.apply(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax, normalize);
      return err;
    }
    case TRUNCATE_EACH_LAYER: {
      double err;
      if (debug) std::cout << "Trotter2 method: truncate layers\n"
                           << "Trotter2 Layer 1/2\n";
      err = U.apply_and_simplify(P, &sense, MPS_TRUNCATE_ZEROS, 0);
      if (debug) std::cout << "Trotter2 Layer 2/2\n";
      err += U.apply_and_simplify(P, &sense, MPS_TRUNCATE_ZEROS, 0,
                                  normalize);
      return err;
    }
    case DO_NOT_TRUNCATE: {
      if (debug) std::cout << "Trotter2 method: no truncation\n"
                           << "Trotter2 Layer 1/2\n";
      U.apply(P, &sense, MPS_TRUNCATE_ZEROS, 0);
      if (debug) std::cout << "Trotter2 Layer 2/2\n";
      U.apply(P, &sense, MPS_TRUNCATE_ZEROS, 0);
      return 0.0;
    }
    default: {
      if (debug) std::cout << "Trotter2 method: truncate group:\n"
                           << "Trotter2 Layer 1/2\n";
      U.apply(P, &sense, MPS_TRUNCATE_ZEROS, 0);
      if (debug) std::cout << "Trotter2 Layer 2/2\n";
      return U.apply_and_simplify(P, &sense, MPS_TRUNCATE_ZEROS, Dmax,
                                  normalize);
    }
    }
  }

} // namespace mps
