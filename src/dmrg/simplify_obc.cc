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

#include <algorithm>
#include <tensor/linalg.h>
#include <tensor/tools.h>
#include <mps/flags.h>
#include <mps/mps.h>
#include <mps/mps_algorithms.h>
#include <mps/lform.h>

namespace mps {

  template<class MPS>
  cdouble
  xprod(const typename MPS::elt_t &w, const std::vector<MPS> &Q, const MPS &P)
  {
    return scprod(Q[0], P);
    cdouble x = number_zero<cdouble>();
    for (int i = 0; i < Q.size(); i++) {
      x += conj(w[i])*scprod(Q[i], P);
    }
    return x;
  }

  template<class MPS>
  double
  do_simplify(MPS *ptrP, const typename MPS::elt_t &w, const std::vector<MPS> &Q,
              int *sense, index sweeps, bool normalize)
  {
    assert(sweeps > 0);
    double tolerance = FLAGS.get(MPS_SIMPLIFY_TOLERANCE);
    typedef typename MPS::elt_t Tensor;
    MPS &P = *ptrP;

    // The distance between vectors is
    //	    |Q-P|^2 = |Q|^2 + |P|^2 - 2 re<Q|P>
    // With respect to each site, we can write this as
    //	    norm(Pk)^2 + normQ2 - 2 real(Qk^T * Pk)
    // where Qk is the vector associated to the linear form <Q|P>
    // The stationary state solution of this problem is Pk = conj(Qk)
    // and thus the minimal distance is
    //	    normQ2 - norm(Pk)^2
    // and the relative error
    //	    err^2 = 1 - (norm(Pk)^2/normQ2)
    index k, last = P.size() - 1;
    LinearForm<MPS> lf(w, Q, P, (*sense > 0) ? last : 0);
    double err = 1.0, olderr, normQ2 = square(lf.norm2()), normP2, scp;
    for (index sweep = 0; sweep < sweeps; sweep++) {
      *sense = -*sense;
      if (*sense < 0) {
        // Last iteration was left-to-right and state P is in canonical form with
        // respect to site (N-1)
        for (k = last; k > 0; k--) {
          set_canonical(P, k, conj(lf.single_site_vector()), -1);
          lf.propagate_left(P[k]);
        }
      } else {
        // Last iteration was left-to-right and state P is in canonical form with
        // respect to site (N-1)
        for (k = 0; k < last; k++) {
          set_canonical(P, k, conj(lf.single_site_vector()), +1);
          lf.propagate_right(P[k]);
        }
      }
      P.at(k) = conj(lf.single_site_vector());
      normP2 = tensor::abs(scprod(P[k], P[k]));
      olderr = err;
      err = sqrt(tensor::abs(1 - normP2/normQ2));
      if (normP2 > normQ2+0.5)
        abort();
      if ((olderr-err) < 1e-5*tensor::abs(olderr) || (err < tolerance)) {
	break;
      }
    }
    if (normalize) {
      P.at(k) = P[k] / sqrt(normP2);
    }
    return err;
  }

} // namespace mps
