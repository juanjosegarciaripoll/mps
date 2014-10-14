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
  new_tensor(const Tensor &Heff, const Tensor &rhand, const MPS &psi, index k)
  {
    Tensor output = reshape(linalg::solve_with_svd(Heff, rhand), psi[k].dimensions());
    std::cout << "new_tensor=" << output << std::endl;
    return output;
  }
  
  /*
   * We solve the problem
   *	H * P = Q
   * by minimizing
   *	|H*P - Q|^2 = <P|H^+ H|P> + <Q|Q> - 2 * Re<Q|H^+ |P>
   */
  template<class MPO, class MPS>
  double
  do_solve(const MPO &H, MPS *ptrP, const MPS &Q, int *sense, index sweeps, bool normalize)
  {
    assert(sweeps > 0);
    double tolerance = FLAGS.get(MPS_SIMPLIFY_TOLERANCE);
    typedef typename MPS::elt_t Tensor;
    MPS &P = *ptrP;
    if (!P.size()) P = Q;

    std::cout << "do_solve\n";

    std::cout << "H=" << H.size() << std::endl;

    double normQ2 = tensor::abs(scprod(Q, Q));
    std::cout << "|Q2|=" << normQ2 << std::endl;
    double normHP, scp, olderr, err = 0.0;

    MPS HQ = apply(H, Q);
    std::cout << "|HQ|=" << tensor::abs(scprod(HQ,HQ)) << std::endl;
    std::cout << "H+=" << adjoint(H).size() << std::endl;
    MPO HH = mmult(adjoint(H), H);
    std::cout << "HH=" << HH.size() << std::endl;

    std::cout << "HQ=" << H.size() << std::endl;

    index k, last = P.size() - 1;
    LinearForm<MPS> lf(HQ, P, (*sense > 0) ? last : 0);
    QuadraticForm<MPO> qf(HH, P, P, (*sense > 0) ? last : 0);

    std::cout << "Forms constructed\n";

    for (index sweep = 0; sweep < sweeps; sweep++) {
      std::cout << "sweep=" << sweep << std::endl;
      *sense = -*sense;
      if (*sense < 0) {
        // Last iteration was left-to-right and state P is in canonical form with
        // respect to site (N-1)
        for (k = last; k > 0; k--) {
	  Tensor Heff = qf.single_site_matrix();
	  Tensor vHQ = to_vector(conj(lf.single_site_vector()));
	  std::cout << "Heff=" << Heff << std::endl
		    << "vHQ=" << vHQ << std::endl;
          set_canonical(P, k, new_tensor(Heff, vHQ, P, k), -1);
          lf.propagate_left(P[k]);
        }
      } else {
        // Last iteration was left-to-right and state P is in canonical form with
        // respect to site (N-1)
        for (k = 0; k < last; k++) {
	  Tensor Heff = qf.single_site_matrix();
	  Tensor vHQ = to_vector(conj(lf.single_site_vector()));
	  std::cout << "Heff=" << Heff << std::endl
		    << "vHQ=" << vHQ << std::endl;
          set_canonical(P, k, new_tensor(Heff, vHQ, P, k), +1);
          lf.propagate_right(P[k]);
        }
      }
      {
	Tensor Heff = qf.single_site_matrix();
	Tensor vHQ = to_vector(conj(lf.single_site_vector()));
	Tensor vP = new_tensor(Heff, vHQ, P, k);
	P.at(k) = vP;
	normHP = square(norm2(mmult(Heff, to_vector(vP))));
	scp = real(scprod(vHQ, vP)); 
      }
      olderr = err;
      err = normHP + normQ2 - 2*scp;
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
