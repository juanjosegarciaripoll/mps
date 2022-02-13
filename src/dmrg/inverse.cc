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

template <class QuadraticForm>
auto apply(const typename QuadraticForm::elt_t &x, QuadraticForm &qf,
           int sense) {
  return qf.apply_two_site_matrix(x, sense);
}

/*
   * We solve the problem
   *	H * P = Q
   * by minimizing
   *	|H*P - Q|^2 = <P|H^+ H|P> + <Q|Q> - 2 * Re<Q|H^+ |P>
   */
template <class MPO, class MPS>
double do_solve(const MPO &H, MPS *ptrP, const MPS &oQ, int *sense,
                index sweeps, bool normalize, index Dmax, double tol) {
  bool single_site =
      FLAGS.get(MPS_SOLVE_ALGORITHM) == MPS_SINGLE_SITE_ALGORITHM;
  double tolerance = FLAGS.get(MPS_SIMPLIFY_TOLERANCE);
  bool debug = FLAGS.get(MPS_DEBUG_SOLVE);
  typedef typename MPS::elt_t Tensor;

  if (tol <= 0) {
    tol = mps::FLAGS.get(MPS_SOLVE_TOLERANCE);
  }

  // We set Q in canonical form to "stabilize" all the transfer matrices.
  MPS Q = canonical_form(oQ, -1);

  // We need an initial state. We assume that if P is an empty matrix product state
  // we can start directly with 'Q'. Note that in this single-site algorithm that
  // leads to poor results.
  assert(ptrP);
  MPS &P = *ptrP;
  if (!P.size()) P = Q;

  // 'sense' can be nullptr.
  int aux = -1;
  if (!sense) sense = &aux;
  assert(sweeps > 0);

  double olderr = 0.0, err = 0.0;  // err = <P|H^2|P> + <Q|Q> - 2re<Q|H|P>
  double normHP;                   // <P|H^2|P>
  tensor_scalar_t<Tensor> scp;     // <Q|H|P>
  double normQ2 = abs(scprod(Q[0], Q[0]));  // <Q|Q>
  if (normQ2 < 1e-16) {
    std::cerr << "Right-hand side in solve(MPO, MPS, ...) is zero\n";
    abort();
  }

  // Iterator object to run over the MPS
  Sweeper s = P.sweeper(-*sense);

  // LinearForm object implementing <Q|H|P>
  LinearForm<MPS> lf(canonical_form(apply(H, Q), -1), P, s.site());

  // QuadraticForm object implementing <P|H^2|P>
  QuadraticForm<MPO> qf(mmult(adjoint(H), H), P, P, s.site());

  Tensor Heff, vHQ, vP;
  while (sweeps--) {
    if (single_site) {
      do {
        //
        // Single-site algorithm. We solve directly for the state on a given
        // site, constructing the full quadratic form matrix.
        //
        Heff = qf.single_site_matrix();
        vHQ = conj(lf.single_site_vector());
        vP = linalg::solve_with_svd(Heff, flatten(vHQ));
        set_canonical(P, s.site(), reshape(vP, vHQ.dimensions()), s.sense());

        // Update quadratic and linear form with the new site
        const Tensor &newP = P[s.site()];
        lf.propagate(newP, s.sense());
        qf.propagate(newP, newP, s.sense());
      } while (--s);
      s.flip();
      normHP = real(scprod(vP, mmult(Heff, vP)));
    } else {
      do {
        //
        // Two-site algorithm. We solve iteratively, using a conjugate
        // gradient algorithm with a map that applies the quadratic form
        // without actually constructing any matrices.
        //
        vHQ = conj(lf.two_site_vector(s.sense()));
        if (s.sense() > 0) {
          vP = fold(P[s.site()], -1, P[s.site() + 1], 0);
        } else {
          vP = fold(P[s.site() - 1], -1, P[s.site()], 0);
        }
        vP = linalg::cgs(
            [&](const Tensor &v) -> Tensor {
              return apply<QuadraticForm<MPO>>(v, qf, s.sense());
            },
            vHQ, &vP, 2 * vHQ.size(), tol);
        set_canonical_2_sites(P, vP, s.site(), s.sense(), Dmax, tol);

        // Update quadratic and linear form with the new site
        const Tensor &newP = P[s.site()];
        lf.propagate(newP, s.sense());
        qf.propagate(newP, newP, s.sense());
        --s;
      } while (!s.is_last());
      s.flip();
      normHP = real(scprod(vP, qf.apply_two_site_matrix(vP, s.sense())));
    }

    // Compute stop criteria.
    scp = scprod(vHQ, vP);
    olderr = err;
    err = normHP + normQ2 - 2 * real(scp);
    if (debug) {
      std::cerr << "[mps::solve] sweeps=" << sweeps << ", err=" << err
                << ", derr=" << olderr - err << '\n';
    }
    if (olderr) {
      if ((olderr - err) < 1e-5 * tensor::abs(olderr) || (err < tolerance)) {
        break;
      }
    }
  }
  if (normalize) {
    P.at(s.site()) /= norm2(P[s.site()]);
  }
  *sense = -s.sense();
  return err;
}

}  // namespace mps
