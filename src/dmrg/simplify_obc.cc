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
#include <mps/mps_algorithms.h>
#include <mps/lform.h>

namespace mps {

  template<class MPS>
  double
  do_simplify(MPS *ptrP, const typename MPS::elt_t &w, const std::vector<MPS> &Q,
              int *sense, index sweeps, bool normalize, index Dmax, double tol,
              double *norm)
  {
    assert(sweeps > 0);
    bool single_site = !Dmax && (FLAGS.get(MPS_SIMPLIFY_ALGORITHM) == MPS_SINGLE_SITE_ALGORITHM);
    double tolerance = FLAGS.get(MPS_SIMPLIFY_TOLERANCE);
    int debug = FLAGS.get(MPS_DEBUG_SIMPLIFY);
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
    Sweeper s = P.sweeper(-*sense);
    LinearForm<MPS> lf(w, Q, P, s.site());
    double err = 1.0, olderr, normQ2 = square(lf.norm2()), normP2, scp;
    if (debug) {
      std::cout << "simplify_obc: "
                << (single_site? "single_site" : "two-sites")
                << ", dmax=" << Dmax << ", tol=" << tol
                << std::endl
                << "\tweights=" << w << std::endl;
    }
    while (sweeps--) {
      if (single_site) {
        do {
          set_canonical(P, s.site(), conj(lf.single_site_vector()), s.sense());
          lf.propagate(P[s.site()], s.sense());
        } while (--s);
      } else {
        do {
          set_canonical_2_sites(P, conj(lf.two_site_vector(s.sense())),
                                s.site(), s.sense(), Dmax, tol,
                                false);
          lf.propagate(P[s.site()], s.sense());
          --s;
        } while (!s.is_last());
      }
      normP2 = abs(scprod(P[s.site()], P[s.site()]));
      olderr = err;
      err = sqrt(abs(1 - normP2/normQ2));
      if (debug) {
        std::cout << "\terr=" << err << ", sense=" << s.sense()
                  << std::endl;
      }
      s.flip();
      if ((olderr-err) < 1e-5*abs(olderr) || (err < tolerance)) {
	break;
      }
    }
    if (norm) {
      *norm = normP2;
    }
    if (normalize) {
      P.at(s.site()) /= sqrt(normP2);
    }
    *sense = -s.sense();
    return err;
  }

} // namespace mps
