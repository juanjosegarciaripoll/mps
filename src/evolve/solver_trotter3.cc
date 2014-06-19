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

namespace mps {

  /**********************************************************************
   * Trotter's method with three passes
   *
   *	exp(-iHdt) = exp(-iH_even dt/2) exp(-iH_odd dt) exp(-iH_even dt/2)
   */

  Trotter3Solver::Trotter3Solver(const Hamiltonian &H, cdouble dt) :
  TrotterSolver(dt), U1(H, 1, dt), U2(H, 0, dt/2.0), sense(0)
  {
  }

  double
  Trotter3Solver::one_step(CMPS *P, index Dmax)
  {
    U1.debug = U2.debug = debug;
    switch (strategy) {
    case TRUNCATE_EACH_UNITARY: {
      double err;
      if (debug) std::cout << "Truncate unitaries:\nLayer 1/3\n";
      err = U2.apply(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cout << "Layer 2/3\n";
      err += U1.apply(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cout << "Layer 3/3\n";
      err += U2.apply(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax, normalize);
      return err;
    }
    case TRUNCATE_EACH_LAYER: {
      double err = 0.0;
      CMPS Pfull = *P;
      if (debug) std::cout << "Truncate layers:\nLayer 1/3\n";
      err = U2.apply_and_simplify(&Pfull, &sense, MPS_TRUNCATE_ZEROS, Dmax);
      if (debug) std::cout << "Layer 2/3\n";
      err += U1.apply_and_simplify(&Pfull, &sense, MPS_TRUNCATE_ZEROS, Dmax);
      if (debug) std::cout << "Layer 3/3\n";
      err += U2.apply_and_simplify(&Pfull, &sense, MPS_TRUNCATE_ZEROS, Dmax,
                                   normalize);
      return err;
    }
    case DO_NOT_TRUNCATE: {
      U2.apply(P, &sense, MPS_TRUNCATE_ZEROS, 0);
      U1.apply(P, &sense, MPS_TRUNCATE_ZEROS, 0);
      U2.apply(P, &sense, MPS_TRUNCATE_ZEROS, 0, normalize);
      return 0.0;
    }
    default: {
      CMPS Pfull = *P;
      if (debug) std::cout << "Truncate group:\nLayer 1/3\n";
      U2.apply(P, &sense, MPS_TRUNCATE_ZEROS, 0);
      if (debug) std::cout << "Layer 2/3\n";
      U1.apply(P, &sense, MPS_TRUNCATE_ZEROS, 0);
      if (debug) std::cout << "Layer 3/3\n";
      return U2.apply_and_simplify(P, &sense, MPS_TRUNCATE_ZEROS, Dmax,
                                   normalize);
    }
    }
  }

} // namespace mps
