// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2012 Juan Jose Garcia Ripoll

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

#include <float.h>
#include <mps/flags.h>

namespace mps {

  tensor::Flags FLAGS;

  const double MPS_DEFAULT = 12345678912345e78;

  const double MPS_DO_NOT_TRUNCATE = 2.0;
  const double MPS_TRUNCATE_ZEROS = 0.0;
  const double MPS_DEFAULT_TOLERANCE = -4.0;

  const unsigned MPS_TRUNCATION_TOLERANCE = FLAGS.create_key(DBL_EPSILON);

  const unsigned MPS_DEBUG_TROTTER = FLAGS.create_key(0);

  const unsigned MPS_DEBUG_SIMPLIFY = FLAGS.create_key(0);

  const unsigned MPS_SIMPLIFY_MAX_SWEEPS = FLAGS.create_key(12);

  const unsigned MPS_SIMPLIFY_TOLERANCE = FLAGS.create_key(1e-14);

  const unsigned MPS_SINGLE_SITE_ALGORITHM = 1;
  const unsigned MPS_TWO_SITE_ALGORITHM = 2;

  const unsigned MPS_SIMPLIFY_ALGORITHM = FLAGS.create_key(MPS_SINGLE_SITE_ALGORITHM);

  const unsigned MPS_DEBUG_SOLVE = FLAGS.create_key(0);

  const unsigned MPS_SOLVE_ALGORITHM = FLAGS.create_key(MPS_SINGLE_SITE_ALGORITHM);

  const unsigned MPS_SOLVE_TOLERANCE = FLAGS.create_key(1e-10);

  const unsigned MPS_ITEBD_CANONICAL_EXPECTED = 1;
  const unsigned MPS_ITEBD_SLOW_EXPECTED = 1;
  const unsigned MPS_ITEBD_EXPECTED_METHOD = FLAGS.create_key(MPS_ITEBD_SLOW_EXPECTED);

}

