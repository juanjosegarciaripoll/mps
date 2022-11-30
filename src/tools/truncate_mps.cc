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

#include <mps/mps_algorithms.h>
#include <tensor/io.h>

namespace mps {

template <class MPS>
static const Indices expected_dimensions(const MPS &P, index_t Dmax,
                                         bool periodic) {
  index_t l = P.ssize();
  Indices d(l + 1);
  if (periodic) {
    for (index_t i = 0; i <= l; i++) d.at(i) = Dmax;
  } else {
    d.at(0) = 1;
    d.at(l) = 1;
    for (index_t i = 1, c = 1; i < l; i++) {
      c *= P[i - 1].dimension(1);
      if (c > Dmax) c = Dmax;
      d.at(i) = c;
    }
    for (index_t i = l, c = 1; i--;) {
      c *= P[i].dimension(1);
      if (c > Dmax) c = Dmax;
      if (c < d[i]) d.at(i) = c;
    }
  }
  return d;
}

template <class MPS>
bool truncate_inner(MPS *Q, const MPS &P, index_t Dmax, bool periodic,
                    bool increase) {
  if (Dmax == 0) {
    *Q = P;
    return false;
  }
  Indices d = expected_dimensions<MPS>(P, Dmax, periodic);
  bool truncated = 0;
  index_t L = P.ssize();
  *Q = MPS(L);
  for (index_t k = 0; k < L; k++) {
    typename MPS::elt_t Qk = P[k];
    if (Qk.dimension(0) > d[k]) {
      truncated = 1;
      Qk = change_dimension(Qk, 0, d[k]);
    } else if (increase && (Qk.dimension(0) < d[k])) {
      Qk = change_dimension(Qk, 0, d[k]);
    }
    if (Qk.dimension(2) > d[k + 1]) {
      truncated = 1;
      Qk = change_dimension(Qk, 2, d[k + 1]);
    } else if (increase && (Qk.dimension(2) < d[k + 1])) {
      Qk = change_dimension(Qk, 2, d[k + 1]);
    }
    Q->at(k) = Qk;
  }
  return truncated;
}

}  // namespace mps
