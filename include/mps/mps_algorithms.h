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

#ifndef MPS_MPS_ALGORITHM_H
#define MPS_MPS_ALGORITHM_H

#include <mps/mps.h>
#include <mps/mpo.h>

namespace mps {

  const RTensor prop_matrix_close(const RTensor &N);

  const RTensor prop_matrix_close(const RTensor &L, const RTensor &R);

  const RTensor prop_matrix(const RTensor &M0, int sense, const RTensor &Q,
			    const RTensor &P, const RTensor *op = NULL);

  const CTensor prop_matrix_close(const CTensor &N);

  const CTensor prop_matrix_close(const CTensor &L, const CTensor &R);

  const CTensor prop_matrix(const CTensor &M0, int sense, const CTensor &Q,
			    const CTensor &P, const CTensor *op = NULL);

  /** Given an MPS, produce another with bond dimension <= Dmax, by truncating it. */
  bool truncate(RMPS *P, const RMPS &Q, index Dmax, bool periodicbc, bool increase = false);

  /** Given an MPS, produce another with bond dimension <= Dmax, by truncating it. */
  bool truncate(CMPS *P, const CMPS &Q, index Dmax, bool periodicbc, bool increase = false);

  double simplify(RMPS *P, const RMPS &Q, int *sense, bool periodicbc,
		  index sweeps, bool normalize);

  double simplify(CMPS *P, const CMPS &Q, int *sense, bool periodicbc,
		  index sweeps, bool normalize);

  double simplify(RMPS *P, const std::vector<RMPS> &Q, const RTensor &weights,
                  int *sense, index sweeps, bool normalize);

  double simplify(CMPS *P, const std::vector<CMPS> &Q, const CTensor &weights,
                  int *sense, index sweeps, bool normalize);

  double simplify(RMPS *P, const std::vector<RMPS> &Q, const RTensor &weights,
                  index Dmax, double tol, int *sense, index sweeps, bool normalize);

  double simplify(CMPS *P, const std::vector<CMPS> &Q, const CTensor &weights,
                  index Dmax, double tol, int *sense, index sweeps, bool normalize);

  /* Open boundary condition algorithms that simplify a state, optimizing over one site */

  double simplify_obc(RMPS *P, const RMPS &Q, int *sense, index sweeps, bool normalize);

  double simplify_obc(CMPS *P, const CMPS &Q, int *sense, index sweeps, bool normalize);

  double simplify_obc(RMPS *P, const RTensor &weights, const std::vector<RMPS> &Q,
                      int *sense, index sweeps, bool normalize);

  double simplify_obc(CMPS *P, const CTensor &weights, const std::vector<CMPS> &Q,
                      int *sense, index sweeps, bool normalize);

  double simplify_obc_2_sites(RMPS *P, const RMPS &Q, int *sense, index sweeps,
                              bool normalize, index Dmax = 0, double tol = -1);

  double simplify_obc_2_sites(CMPS *P, const CMPS &Q, int *sense, index sweeps,
                              bool normalize, index Dmax = 0, double tol = -1);

  double simplify_obc_2_sites(RMPS *P, const RTensor &weights, const std::vector<RMPS> &Q,
                              int *sense, index sweeps, bool normalize, index Dmax = 0,
                              double tol = -1);

  double simplify_obc_2_sites(CMPS *P, const CTensor &weights, const std::vector<CMPS> &Q,
                              int *sense, index sweeps, bool normalize, index Dmax = 0,
                              double tol = -1);

  double solve(const RMPO &H, RMPS *ptrP, const RMPS &Q, int *sense, index sweeps,
               bool normalize = false);

  double solve(const CMPO &H, CMPS *ptrP, const CMPS &Q, int *sense, index sweeps,
               bool normalize = false);

} // namespace mps

#endif // !MPS_MPS_ALGORITHM_H
