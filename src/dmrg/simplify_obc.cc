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
#include <tensor/io.h>
#include <mps/flags.h>
#include <mps/mps.h>
#include <mps/algorithms.h>
#include <mps/lform.h>

namespace mps {

template <class MPS>
SimplificationOutput do_simplify(MPS *ptrP, const typename MPS::elt_t &w,
                                 const vector<MPS> &Q,
                                 const SimplificationStrategy &strategy) {
  bool single_site = strategy.single_site_simplification();
  int debug = FLAGS.get_int(MPS_DEBUG_SIMPLIFY);
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
  Sweeper s = P.sweeper(strategy.direction());
  LinearForm<mp_tensor_t<MPS>> lf(w, Q, P, s.site());
  double err = 1.0, normQ2 = square(lf.norm2()), normP2 = 0.0;
  if (debug) {
    std::cerr << "simplify_obc: "
              << (strategy.single_site_simplification() ? "single_site"
                                                        : "two-sites")
              << ", dmax=" << strategy.maximum_dimension()
              << ", truncate_tol=" << strategy.truncation_relative_tolerance()
              << ", stop_tol=" << strategy.stop_relative_tolerance() << '\n'
              << "\tweights=" << w << '\n';
  }
  for (index_t i = 0; i < strategy.sweeps(); ++i) {
    if (single_site) {
      do {
        set_canonical(P, s.site(), conj(lf.single_site_vector()), s.sense());
        lf.propagate(P[s.site()], s.sense());
      } while (--s);
    } else {
      do {
        set_canonical_2_sites(P, conj(lf.two_site_vector(s.sense())), s.site(),
                              s.sense(), false, strategy);
        lf.propagate(P[s.site()], s.sense());
        --s;
      } while (!s.is_last());
    }
    normP2 = abs(scprod(P[s.site()], P[s.site()]));
    double olderr = err;
    err = sqrt(abs(1 - normP2 / normQ2));
    if (debug) {
      std::cerr << "\terr=" << err << ", sense=" << s.sense() << '\n';
    }
    s.flip();
    if ((olderr - err) < 1e-5 * abs(olderr) ||
        (err < strategy.stop_relative_tolerance())) {
      break;
    }
  }
  if (strategy.normalize()) {
    P.at(s.site()) /= sqrt(normP2);
  }
  return {err, normP2, -s.sense()};
}

}  // namespace mps
