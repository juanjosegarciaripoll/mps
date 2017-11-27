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

#ifndef MPS_MINIMIZER_H
#define MPS_MINIMIZER_H

#include <mps/mps.h>
#include <mps/mpo.h>

namespace mps {

  struct MinimizerOptions {

    MinimizerOptions() :
      sweeps(32), display(false), debug(false), tolerance(1e-10),
      svd_tolerance(1e-11), allow_E_growth(1), Dmax(0),
      compute_gap(false), gap(0), constrained_gap(0)
    {}

    index sweeps;
    bool display;
    index debug;
    double tolerance;
    double svd_tolerance;
    int allow_E_growth;
    index Dmax;

    bool compute_gap;
    double gap, constrained_gap;
  };

  double minimize(const RMPO &H, RMPS *psi, const MinimizerOptions &opt,
                  const RMPO &constraint, double value);
  double minimize(const CMPO &H, CMPS *psi, const MinimizerOptions &opt,
                  const CMPO &constraint, cdouble value);

  double minimize(const RMPO &H, RMPS *psi, const MinimizerOptions &opt);
  double minimize(const CMPO &H, CMPS *psi, const MinimizerOptions &opt);

  double minimize(const RMPO &H, RMPS *psi);
  double minimize(const CMPO &H, CMPS *psi);

} // namespace dmrg

#endif /* !MPS_MINIMIZER_H */
