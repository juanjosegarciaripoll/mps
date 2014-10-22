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
    Tensor vP = linalg::solve_with_svd(H2, vHQ);
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
  do_solve(const MPO &H, MPS *ptrP, const MPS &oQ, int *sense, index sweeps, bool normalize)
  {
    double tolerance = FLAGS.get(MPS_SIMPLIFY_TOLERANCE);
    typedef typename MPS::elt_t Tensor;

    assert(ptrP);
    MPS &P = *ptrP;
    if (!P.size()) P = Q;
    int aux = +1;
    if (!sense) sense = &aux;
    assert(sweeps > 0);

    MPS Q = canonical_form(oQ, -1);
    double normQ2 = abs(scprod(Q[0], Q[0]));
    if (normQ2 < 1e-16) {
      std::cerr << "Right-hand side in solve(MPO, MPS, ...) is zero\n";
      abort();
    }

    double normHP, olderr = 0.0, err = 0.0;
    typename Tensor::elt_t scp;

    MPS HQ = canonical_form(apply(H, Q), -1);
    MPO HH = mmult(adjoint(H), H);

    Sweeper s = P.sweeper();
    LinearForm<MPS> lf(HQ, P, s.site());
    QuadraticForm<MPO> qf(HH, P, P, s.site());

    Tensor Heff, vHQ, vP;
    while (sweeps--) {
      for (s.flip(); !s.is_last(); ++s) {
        Heff = qf.single_site_matrix();
        vHQ = conj(lf.single_site_vector());
        vP = linalg::solve_with_svd(Heff, to_vector(vHQ));
        set_canonical(P, s.site(), reshape(vP, vHQ.dimensions()), s.sense());

        const Tensor &newP = P[s.site()];
        lf.propagate(newP, s.sense());
        qf.propagate(newP, newP, s.sense());
      }
      normHP = real(scprod(vP, mmult(Heff, vP)));
      scp = scprod(to_vector(vHQ), vP);
      olderr = err;
      err = normHP + normQ2 - 2*real(scp);
      if (olderr) {
	if ((olderr-err) < 1e-5*tensor::abs(olderr) || (err < tolerance)) {
	  break;
	}
      }
    }
    if (normalize) {
      P.at(k) = P[k] / norm2(P[k]);
    }
    *sense = s.sense();
    return err;
  }

} // namespace mps
