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

#include <tensor/exceptions.h>
#include <mps/time_evolve.h>
#include <mps/algorithms.h>

namespace mps {

/**********************************************************************
 * Trotter's method of fourth order.
 */

//static const double inv_theta = 0.74007895010513;
static const double FR_param[5] = {0.67560359597983, 1.35120719195966,
                                   -0.17560359597983, -1.70241438391932};

ForestRuthSolver::ForestRuthSolver(const Hamiltonian &H, cdouble dt)
    : TrotterSolver(dt),
      U1(H, 0, dt * FR_param[0]),
      U2(H, 1, dt * FR_param[1]),
      U3(H, 0, dt * FR_param[2]),
      U4(H, 1, dt * FR_param[3]),
      sense(0) {}

double ForestRuthSolver::one_step(CMPS *P, index_t Dmax) {
  int debug = tensor::narrow_cast<int>(FLAGS.get(MPS_DEBUG_TROTTER));
  if (!Dmax) {
    if (strategy != DO_NOT_TRUNCATE) {
      std::cerr
          << "In TrotterSolver::one_step(), no maximum dimension provided\n";
      abort();
    }
  }

  switch (strategy) {
    case TRUNCATE_EACH_UNITARY: {
      double err = 0.0;
      if (debug)
        std::cerr << "Trotter4 method: truncate unitaries:\n"
                  << "Trotter4 Layer 1/7\n";
      err += U1.apply(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cerr << "Trotter3 Layer 2/7\n";
      err += U2.apply(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cerr << "Trotter3 Layer 3/7\n";
      err += U3.apply(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cerr << "Trotter3 Layer 4/7\n";
      err += U4.apply(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cerr << "Trotter3 Layer 5/7\n";
      err += U3.apply(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cerr << "Trotter3 Layer 6/7\n";
      err += U2.apply(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cerr << "Trotter3 Layer 7/7\n";
      err += U1.apply(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax, normalize);
      return err;
    }
    case DO_NOT_TRUNCATE: {
      double err = 0.0;
      err += U1.apply(P, &sense, MPS_DEFAULT_TOLERANCE, 0);
      err += U2.apply(P, &sense, MPS_DEFAULT_TOLERANCE, 0);
      err += U3.apply(P, &sense, MPS_DEFAULT_TOLERANCE, 0);
      err += U4.apply(P, &sense, MPS_DEFAULT_TOLERANCE, 0);
      err += U3.apply(P, &sense, MPS_DEFAULT_TOLERANCE, 0);
      err += U2.apply(P, &sense, MPS_DEFAULT_TOLERANCE, 0);
      err += U1.apply(P, &sense, MPS_DEFAULT_TOLERANCE, 0, normalize);
      return err;
    }
    case TRUNCATE_EACH_LAYER: {
      double err = 0.0;
      if (debug)
        std::cerr << "Trotter4 method: truncate layers:\n"
                  << "Trotter4 Layer 1/7\n";
      err += U1.apply_and_simplify(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cerr << "Trotter3 Layer 2/7\n";
      err += U2.apply_and_simplify(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cerr << "Trotter3 Layer 3/7\n";
      err += U3.apply_and_simplify(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cerr << "Trotter3 Layer 4/7\n";
      err += U4.apply_and_simplify(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cerr << "Trotter3 Layer 5/7\n";
      err += U3.apply_and_simplify(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cerr << "Trotter3 Layer 6/7\n";
      err += U2.apply_and_simplify(P, &sense, MPS_DEFAULT_TOLERANCE, 0);
      if (debug) std::cerr << "Trotter3 Layer 7/7\n";
      err +=
          U1.apply_and_simplify(P, &sense, MPS_DEFAULT_TOLERANCE, 0, normalize);
      return err;
    }
    default: {
      double err = 0.0;
      if (debug)
        std::cerr << "Trotter4 method: truncate groups:\n"
                  << "Trotter4 Layer 1/7\n";
      err += U1.apply(P, &sense, MPS_DEFAULT_TOLERANCE, 0);
      if (debug) std::cerr << "Trotter3 Layer 2/7\n";
      err += U2.apply_and_simplify(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cerr << "Trotter3 Layer 3/7\n";
      err += U3.apply(P, &sense, MPS_DEFAULT_TOLERANCE, 0);
      if (debug) std::cerr << "Trotter3 Layer 4/7\n";
      err += U4.apply(P, &sense, MPS_DEFAULT_TOLERANCE, 0);
      if (debug) std::cerr << "Trotter3 Layer 5/7\n";
      err += U3.apply_and_simplify(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax);
      if (debug) std::cerr << "Trotter3 Layer 6/7\n";
      err += U2.apply(P, &sense, MPS_DEFAULT_TOLERANCE, 0);
      if (debug) std::cerr << "Trotter3 Layer 7/7\n";
      err += U1.apply_and_simplify(P, &sense, MPS_DEFAULT_TOLERANCE, Dmax,
                                   normalize);
      return err;
    }
  }
}

}  // namespace mps
