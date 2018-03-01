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

#ifndef MPS_FLAGS_H
#define MPS_FLAGS_H

#include <tensor/flags.h>

namespace mps {

  extern tensor::Flags FLAGS;

  /**Do not truncate tensors.*/
  extern const double MPS_DO_NOT_TRUNCATE;
  /**Truncate tensors eliminating zero values from the SVD.*/
  extern const double MPS_TRUNCATE_ZEROS;
  /**Default relative tolerance of the singular values dropped.*/
  extern const double MPS_DEFAULT_TOLERANCE;

  /**Decide whether to use block_svd.*/
  extern const unsigned int MPS_USE_BLOCK_SVD;

  /**Algorithms acting on one site.*/
  extern const unsigned int MPS_SINGLE_SITE_ALGORITHM;
  /**Algorithms acting on two sites.*/
  extern const unsigned int MPS_TWO_SITE_ALGORITHM;

  /**Flag key for debugging the simplification routines.*/
  extern const unsigned int MPS_DEBUG_TRUNCATION;
  /**Flag key for the default relative truncation tolerance.*/
  extern const unsigned int MPS_TRUNCATION_TOLERANCE;

  /**Flag key for debugging the time evolution routines.*/
  extern const unsigned int MPS_DEBUG_TROTTER;

  /**Flag key for debugging the Arnoldi method.*/
  extern const unsigned int MPS_DEBUG_ARNOLDI;
  /**Flag key for number of Arnoldi simplification sweeps.*/
  extern const unsigned int MPS_ARNOLDI_SIMPLIFY_INTERNAL_SWEEPS;
  /**Flag key for number of Arnoldi final simplification sweeps.*/
  extern const unsigned int MPS_ARNOLDI_SIMPLIFY_FINAL_SWEEPS;

  /**Flag key for debugging the simplification routines.*/
  extern const unsigned int MPS_DEBUG_SIMPLIFY;
  /**Flag key indicating the sweeps in the simplification routines.*/
  extern const unsigned int MPS_SIMPLIFY_MAX_SWEEPS;
  /**Flag key indicating what relative error is acceptable when simplifying.*/
  extern const unsigned int MPS_SIMPLIFY_TOLERANCE;

  /**Flag keys for the simplify_obc() algorithms.*/
  extern const unsigned int MPS_SIMPLIFY_ALGORITHM;

  /**Flag key for debugging the mps::solve routines.*/
  extern const unsigned int MPS_DEBUG_SOLVE;
  /**Flag keys for the solve() algorithms.*/
  extern const unsigned int MPS_SOLVE_ALGORITHM;
  /**Flag key indicating what relative error is acceptable when inverting.*/
  extern const unsigned int MPS_SOLVE_TOLERANCE;

  /**iTEBD expectation values assuming canonical form.*/
  extern const unsigned int MPS_ITEBD_CANONICAL_EXPECTED;
  /**iTEBD expectation values computing powers of transfer matrices.*/
  extern const unsigned int MPS_ITEBD_SLOW_EXPECTED;
  /**iTEBD expectation values computing boundary states by power method.*/
  extern const unsigned int MPS_ITEBD_BDRY_EXPECTED;
  /**iTEBD expectation value method selector.*/
  extern const unsigned int MPS_ITEBD_EXPECTED_METHOD;

} // namespace mps

#endif // MPS_FLAGS_H
