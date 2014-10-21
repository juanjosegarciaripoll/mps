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
#include <mps/mpo.h>
#include <mps/qform.h>
#include <mps/lform.h>
#include <tensor/io.h>

namespace mps {

  template<class Tensor>
  static const Tensor to_vector(const Tensor &v) { return reshape(v, v.size()); }

  template<class Tensor, class MPS>
  static const Tensor
  new_tensor(const Tensor &H2, const Tensor &vHQ, const MPS &psi, index k,
             double normQ2)
  {
    Tensor vP = linalg::solve_with_svd(H2, to_vector(vHQ));
    std::cout << "     H=" << H2.dimensions() << "\n";
    return reshape(vP, vHQ.dimensions());
  }
  
  /*
   * We solve the problem
   *	H * P = Q
   * by minimizing
   *	|H*P - Q|^2 = <P|H^+ H|P> + <Q|Q> - 2 * Re<Q|H^+ |P>
   */
  template<class MPO, class MPS>
  double
  do_solve(const MPO &H, MPS *ptrP, const MPS &oQ, int *sense, index sweeps, bool normalize,
           index Dmax, double tol)
  {
    MPS Q = canonical_form(oQ, -1);
    assert(sweeps > 0);
    double tolerance = FLAGS.get(MPS_SIMPLIFY_TOLERANCE);
    typedef typename MPS::elt_t Tensor;
    MPS &P = *ptrP;
    if (!P.size()) P = Q;

    double normQ2 = tensor::abs(scprod(Q, Q));
    if (normQ2 < 1e-16) {
      std::cerr << "Right-hand side in solve(MPO, MPS, ...) is zero\n";
      abort();
    }

    double normHP, olderr, err = 0.0;
    typename Tensor::elt_t scp;

    MPS HQ = canonical_form(apply(H, Q), -1);
    MPO HH = mmult(adjoint(H), H);

    index k, last = P.size() - 1;
    LinearForm<MPS> lf(HQ, P, (*sense > 0) ? last : 0);
    QuadraticForm<MPO> qf(HH, P, P, (*sense > 0) ? last : 0);
    
    Tensor vP, Heff, vHQ;
    std::cout << "psi.size()=" << P.size() << ", last=" << last << std::endl;
    for (index sweep = 0; sweep < sweeps; sweep++) {
      *sense = -*sense;
      if (*sense < 0) {
        // Last iteration was left-to-right and state P is in canonical form with
        // respect to site (N-1)
        for (k = last; k > 0; k--) {
          std::cout << "site k=" << k << "\n";
          vP = new_tensor(Heff = qf.two_site_matrix(-1),
                          vHQ = conj(lf.two_site_vector(-1)),
                          P, k, normQ2);
          set_canonical_2_sites(P, vP, k-1, -1, Dmax, tol);
          lf.propagate_left(P[k]);
          qf.propagate_left(P[k],P[k]);
        }
      } else {
        // Last iteration was left-to-right and state P is in canonical form with
        // respect to site (N-1)
        for (k = 0; k < last; k++) {
          std::cout << "site k=" << k << "\n";
          vP = new_tensor(Heff = qf.two_site_matrix(+1),
                          vHQ = conj(lf.two_site_vector(+1)),
                          P, k, normQ2);
          set_canonical_2_sites(P, vP, k, +1, Dmax, tol);
          lf.propagate_right(P[k]);
          qf.propagate_right(P[k],P[k]);
        }
      }
      {
	vP = to_vector(vP);
	normHP = real(scprod(vP, mmult(Heff, vP)));
	scp = scprod(vHQ, vP);
        std::cout << "err = " << normHP + normQ2 - 2*real(scp) << std::endl;
      }
      olderr = err;
      err = normHP + normQ2 - 2*real(scp);
      if (sweep) {
	if ((olderr-err) < 1e-5*tensor::abs(olderr) || (err < tolerance)) {
	  break;
	}
      }
    }
    if (normalize) {
      P.at(k) = P[k] / norm2(P[k]);
    }
    return err;
  }

} // namespace mps
