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

#include <tensor/linalg.h>
#include <tensor/io.h>
#include <tensor/tools.h>
#include <mps/time_evolve.h>
#include <mps/mps_algorithms.h>

namespace mps {

  TrotterSolver::Unitary::Unitary(const Hamiltonian &H, index k, cdouble dt,
                                  int do_debug) :
    debug(do_debug), k0(k), kN(H.size()), U(H.size())
  {
    /*
     * When we do 'Trotter' evolution, the Hamiltonian is split into
     * 'even' and 'odd' contributions made of mutually commuting terms.
     * This object splits those contributions:
     *
     *     H_even = \sum_n H_{2n,2n+1} + 0.5*Hloc_{L}
     *     H_odd  = 0.5*Hloc_{0} + \sum_n H_{2n+1,2n+2}
     *
     * where the pairwise terms include part of the local operators
     *
     *     H_{kk+1} = Hint_{kk+1} + 0.5*Hloc_{k} + 0.5*Hloc_{k+1}
     *
     * and where the extra Hloc_{0} and Hloc_{L} ensure that all local
     * operators appear both in the even and odd splittings.
     */
    if (k > 1) {
      std::cerr << "In TrotterSolver::Unitary::Unitary(H, k, ...), "
        "the initial site k was neither 0 nor 1";
      abort();
    }
    if ((k & 1) ^ (kN & 1)) {
      // We look for the nearest site which has the same parity.
      kN--;
    }
    if ((kN - k0) < 2) {
      // No sites!
      kN = k0;
    }
    if (debug) std::cout << "computing: ";
    dt = to_complex(-tensor::abs(imag(dt)), -real(dt));
    for (int di, i = 0; i < (int)H.size(); i += di) {
      CTensor Hi;
      if (i < k0 || i >= kN) {
	// Local operator
	Hi = H.local_term(i,0.0) / 2.0;
	if (debug) std::cout << "[" << i << "]";
	di = 1;
      } else {
	CTensor i1 = CTensor::eye(H.dimension(i));
	CTensor i2 = CTensor::eye(H.dimension(i+1));
	Hi = H.interaction(i,0.0)
	  + kron2(0.5 * H.local_term(i,0.0), i2)
	  + kron2(i1, 0.5 * H.local_term(i+1,0.0));
	if (debug) std::cout << "[" << i << "," << i+1 << "]";
        di = 2;
      }
      U.at(i) = linalg::expm(Hi * dt);
    }
    if (debug) {
      std::cout << std::endl;
      std::cout << "Unitaries running from " << k0 << " to " << kN << std::endl;
    }
  }

  void
  TrotterSolver::Unitary::apply_onto_one_site(CMPS &P, const CTensor &Uloc,
                                              index k, int dk, index max_a2) const
  {
    CTensor P1 = P[k];
    if (!Uloc.is_empty()) {
      index a1,i1,a2;
      P1.get_dimensions(&a1, &i1, &a2);
      P1 = foldin(Uloc, -1, P1, 1);
    }
    if (max_a2) {
      set_canonical(P, k, P1, dk);
    } else {
      P.at(k) = P1;
    }
  }

  double
  TrotterSolver::Unitary::apply_onto_two_sites(CMPS &P, const CTensor &U12,
					       index k1, index k2, int dk,
					       double tolerance, index max_a2)
    const
  {
    index a1, i1, a2, i2, a3;

    CTensor P1 = P[k1];
    P1.get_dimensions(&a1, &i1, &a2);
    CTensor P2 = P[k2];
    P2.get_dimensions(&a2, &i2, &a3);

    double err = 0.0;

    if (debug > 1) {
      std::cout << "Applying two-site unitary on (" << k1 << "," << k2 << ")"
                << ", max bond dimension " << max_a2 << ", tolerance " << tolerance
                << std::endl
                << "From dimensions "
                << P1.dimensions() << "," << P2.dimensions();
    }

    /* Apply the unitary onto two neighboring sites. This creates a
     * larger tensor that we have to split into two new tensors, Pout[k1]
     * and Pout[k2], that represent the sites */
    P1 = reshape(fold(P1, -1, P2, 0), a1,i1*i2,a3);
    if (!U12.is_empty()) {
      P1 = foldin(U12, -1, P1, 1);
    }
    RTensor s = linalg::svd(reshape(P1,a1*i1,i2*a3), &P1, &P2,
                            SVD_ECONOMIC);
    if (dk > 0) {
      scale_inplace(P2, 0, s);
    } else {
      scale_inplace(P1, -1, s);
    }
    a2 = s.size();
    index new_a2 = where_to_truncate(s, tolerance, max_a2? max_a2 : a2);
    if (new_a2 != a2) {
      P1 = change_dimension(P1, -1, new_a2);
      P2 = change_dimension(P2, 0, new_a2);
      a2 = new_a2;
      for (index i = a2; i < s.size(); i++)
        err += square(s[i]);
    }
    if (max_a2) {
      /* If we impose a truncation at this stage, we are using
       * Guifre's original TEBD algorithm and we split and
       * orthogonalize as we move on. */
      if (dk > 0) {
        P.at(k1) = reshape(P1, a1,i1,a2);
        set_canonical(P, k2, reshape(P2, a2,i2,a3), dk);
      } else {
        P.at(k2) = reshape(P2, a2,i2,a3);
        set_canonical(P, k1, reshape(P1, a1,i1,a2), dk);
      }
    } else {
      /*
       * Otherwise we are not truncating (except for eliminating zero singular
       * values) and we just keep the result of applying the
       * unitaries. Truncation will be done at a later stage.
       */
      P.at(k1) = reshape(P1, a1,i1,a2);
      P.at(k2) = reshape(P2, a2,i2,a3);
    }
    if (debug > 1) {
      std::cout << " to "
                << P1.dimensions() << "," << P2.dimensions()
                << " error " << err
                << std::endl;
    }

    return err;
  }

  double
  TrotterSolver::Unitary::apply_and_simplify(CMPS *psi, int *sense,
                                             double tolerance,
                                             index Dmax, bool normalize) const
  {
    /*
     * In this version we first apply all unitaries. The state is not
     * truncated, except for eliminating zero singular values, what does
     * not introduce errors.
     */
    *sense = +1;
    double err = apply(psi, sense, tolerance, 0, false);
    /*
     * After this initial phase, we now simplify the state to have the
     * right bond dimension.
     */
    int simplify_sense = -1;
    *psi = canonical_form(*psi, simplify_sense);
    if (largest_bond_dimension(*psi) > Dmax) {
      index sweeps = 12;
      err += simplify_obc(psi, *psi, &simplify_sense, sweeps, normalize, Dmax);
    }
    return err;
  }

  double
  TrotterSolver::Unitary::apply(CMPS *psi, int *sense, double tolerance,
                                index Dmax, bool normalize) const
  {
    if (*sense == 0) {
      *sense = +1;
    }
    tic();

    index L = psi->size();
    double err = 0;
    int dk = 2;
    if (*sense > 0) {
      for (int k = 0; k < k0; k++) {
        apply_onto_one_site(*psi, U[k], k, *sense, Dmax);
      }
      for (int k = k0; k < kN; k += dk) {
	err += apply_onto_two_sites(*psi, U[k], k, k+1, *sense, tolerance, Dmax);
      }
      for (int k = kN; k < (int)L; k++) {
        apply_onto_one_site(*psi, U[k], k, *sense, Dmax);
      }
    } else {
      for (int k = L-1; k >= kN; k--) {
        apply_onto_one_site(*psi, U[k], k, *sense, Dmax);
      }
      for (int k = kN - dk; k >= k0; k -= dk) {
	err += apply_onto_two_sites(*psi, U[k], k, k+1, *sense, tolerance, Dmax);
      }
      for (int k = k0 - 1; k >= 0; k--) {
        apply_onto_one_site(*psi, U[k], k, *sense, Dmax);
      }
    }
    if (debug) {
      std::cout << "Unitary: bond dimension = " << largest_bond_dimension(*psi)
                << "\t[" << toc() << "s]\n";
    }
    if (normalize) {
      *psi = normal_form(*psi, *sense);
    }

    *sense = -*sense;
    return err;
  }

} // namespace mps

