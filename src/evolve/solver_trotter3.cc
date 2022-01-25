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

namespace mps
{

  /**********************************************************************
   * Trotter's method with three passes
   *
   *	exp(-iHdt) = exp(-iH_even dt/2) exp(-iH_odd dt) exp(-iH_even dt/2)
   */

  Trotter3Solver::Trotter3Solver(const Hamiltonian &H, cdouble dt) : TrotterSolver(dt), U1(H, 1, dt), U2(H, 0, dt / 2.0), sense(0)
  {
  }

  double
  Trotter3Solver::one_step(CMPS *P, index Dmax)
  {
    int debug = static_cast<int>(FLAGS.get(MPS_DEBUG_TROTTER));
    if (!Dmax)
    {
      if (strategy != DO_NOT_TRUNCATE)
      {
        std::cerr << "In TrotterSolver::one_step(), no maximum dimension provided\n";
        abort();
      }
    }

    U1.debug = U2.debug = debug;
    switch (strategy)
    {
    case TRUNCATE_EACH_UNITARY:
    {
      double err;
      if (debug)
        std::cout << "Trotter3 method: truncate unitaries:\n"
                  << "Trotter3 Layer 1/3\n";
      err = U2.apply(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug)
        std::cout << "Trotter3 Layer 2/3\n";
      err += U1.apply(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug)
        std::cout << "Trotter3 Layer 3/3\n";
      err += U2.apply(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax, normalize);
      return err;
    }
    case TRUNCATE_EACH_LAYER:
    {
      double err = 0.0;
      if (debug)
        std::cout << "Trotter3 method: truncate layers:\n"
                  << "Trotter3 Layer 1/3\n";
      err = U2.apply_and_simplify(P, &sense, MPS_TRUNCATE_EPSILON, Dmax);
      if (debug)
        std::cout << "Trotter3 Layer 2/3\n";
      err += U1.apply_and_simplify(P, &sense, MPS_TRUNCATE_EPSILON, Dmax);
      if (debug)
        std::cout << "Trotter3 Layer 3/3\n";
      err += U2.apply_and_simplify(P, &sense, MPS_TRUNCATE_EPSILON, Dmax,
                                   normalize);
      return err;
    }
    case DO_NOT_TRUNCATE:
    {
      if (debug)
        std::cout << "Trotter3 method: do not truncate:\n"
                  << "Trotter3 Layer 1/3\n";
      U2.apply(P, &sense, MPS_TRUNCATE_EPSILON, 0);
      if (debug)
        std::cout << "Trotter3 Layer 2/3\n";
      U1.apply(P, &sense, MPS_TRUNCATE_EPSILON, 0);
      if (debug)
        std::cout << "Trotter3 Layer 3/3\n";
      U2.apply(P, &sense, MPS_TRUNCATE_EPSILON, 0, normalize);
      return 0.0;
    }
    default:
    {
      CMPS Pfull = *P;
      if (debug)
        std::cout << "Trotter3 method: truncate group:\n"
                  << "Trotter3 Layer 1/3\n";
      U2.apply(P, &sense, MPS_TRUNCATE_EPSILON, 0);
      if (debug)
        std::cout << "Trotter3 Layer 2/3\n";
      U1.apply(P, &sense, MPS_TRUNCATE_EPSILON, 0);
      if (debug)
        std::cout << "Trotter3 Layer 3/3\n";
      return U2.apply_and_simplify(P, &sense, MPS_TRUNCATE_EPSILON, Dmax,
                                   normalize);
    }
    }
  }

} // namespace mps
