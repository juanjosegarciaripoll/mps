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

#include <cmath>
#include <tensor/linalg.h>
#include <tensor/io.h>
#include <mps/minimizer.h>
#include <mps/lform.h>
#include <mps/qform.h>

namespace mps {

  /*----------------------------------------------------------------------
   * SEARCH OF GROUND STATES:
   * ========================
   *
   * With respect to a single site, the energy functional of a MPS is the quotient
   * of two quadratic polynomials. If Pk are the matrices of the MPS,
   *	E(Pk) = fH(Pk)/fN(Pk)	fH = <psi|H|psi>, fN = <psi|psi>
   * which means that finding the optimal Pk reduces to solving the eigenvalue
   * problem
   *	fH'' Pk = E fN'' Pk,
   * the colon ('') denoting second order derivative (fH(0)=fN(0)=0).
   */

  template<class MPS>
  Minimizer<MPS>::DMRG(const mpo_t &H) :
    H_(H),
    error(false),
    sweeps(32), display(true), debug(0),
    tolerance(1e-6), svd_tolerance(1e-8),
    allow_E_growth(0), Hqform_(0)
  {
    if (size() < 2) {
      std::cerr << "The DMRG solver only solves problems with more than two sites";
      abort();
    }
    if (H.is_periodic()) {
      std::cerr << "The DMRG class does not work with periodic boundary conditions";
      abort();
    }
  }

  /**********************************************************************
   * SINGLE-SITE DMRG ALGORITHM
   */

  template<class MPS>
  double
  Minimizer<MPS>::minimize_single_site(MPS &P, index k, int dk)
  {
    /* First we build the components of the Hamiltonian:
     *		Op = Opl + Opli + Opi + Opir + Opr
     * which are the Hamiltonian for the left and right block (Opl,Opr) the
     * intearction between block and site (Opli,Opir) and the local terms
     * (Opi). In the case of periodic b.c. we also will need the norm
     * matrices. Notice that, for economy, we join Opl, Opli on one side, and
     * Opr and Opr on the other. Opi is either added to Opli or to Opri
     * depending on which operator we will use in update_matrices_left/right.
     */
    index a1,i1,b1;
    elt_t Pk = P[k];
    Pk.get_dimensions(&a1, &i1, &b1);

    elt_t Opli = block_site_interaction_left(P, k);
    elt_t Opir = block_site_interaction_right(P, k);
    if (dk > 0) {
      Opli = Opli + kron2(elt_t::eye(a1), local_term(k));
    } else {
      Opir = Opir + kron2(local_term(k), elt_t::eye(b1));
    }

    /*
     * Now we find the minimal energy and optimal projector, Pk.  We use an
     * iterative algorithm that does not require us to build explicitely the
     * matrix.  NOTE: For small sizes, our iterative algorithm (ARPACK) fails
     * and we have to resort to a full diagonalization.
     */
    index neig = std::max<int>(1, neigenvalues);
    elt_t aux;
    if (a1*i1*b1 <= 10) {
      elt_t Heff = kron2(Opli, elt_t::eye(b1)) + kron2(elt_t::eye(a1), Opir);
      aux = eigs(Heff, linalg::SmallestAlgebraic, neig, &Pk, Pk.begin());
    } else {
      linalg::Arpack<elt_t> eigs(Pk.size(), linalg::SmallestAlgebraic, neig);
      eigs.set_maxiter(Pk.size());
      eigs.set_start_vector(Pk.begin());
      while (eigs.update() < eigs.Finished) {
	Pk = eigs.get_x();
	Pk = fold(reshape(Opli, a1*i1,a1*i1), -1, reshape(Pk, a1*i1,b1), 0)
	  + fold(reshape(Pk, a1,i1*b1), -1, reshape(Opir, i1*b1,i1*b1), -1);
	eigs.set_y(Pk);
      }
      if (eigs.get_status() != eigs.Finished) {
	std::cerr << "DMRG: Diagonalization routine did not converge.\n"
		  << eigs.error_message();
	abort();
      }
      aux = eigs.get_data(&Pk);
    }
    /*
     * And finally we update the matrices.
     */
    if (neig > 1) Pk = Pk(range(), range(0));
    set_canonical(P, k, reshape(Pk, a1,i1,b1), dk);
    if (dk > 0) {
      update_matrices_left(P, k);
    } else {
      update_matrices_right(P, k);
    }
    eigenvalues = tensor::real(aux);
    return eigenvalues[0];
  }

  /**********************************************************************
   * COMMON LOOP
   */

  template<class MPS>
  void
  Minimizer<MPS>::show_state_info(const MPS &P, index iter,
                             index k, double newE)
  {
    std::cout << "k=" << k << "; iteration=" << iter << "; E=" << newE
	      << "; E'=" << expected(P, *H_, 0);
  }

  template<class MPS>
  double
  Minimizer<MPS>::minimize(MPS *Pptr, index Dmax)
  {
    MPS &P = *Pptr;
    int dk=+1, k0, kN;

    //SpecialVar<bool> old_accurate_svd(accurate_svd, true);

    P = canonical_form(P);

    double E = 100000;
    int failures = 0;
    for (index L = size(), iter = 0; iter < sweeps; iter++, dk=-dk) {
      double newE;
      /*
       * We sweep from the left to the right or viceversa. We also try to
       * avoid minimizing twice the same site, and for that reason the backward
       * sweeps begin one site to the left of the last optimized site.
       * NOTE: If we use a two-sites algorithm, we have to be careful because
       * the pathological case of L=2 leads to an iteration being empty [1]
       */
      if (Dmax) {
	if (dk > 0) {
	  k0 = 0; kN = L-1; dk = +1;
	} else if (L < 3) {
	  continue;
	} else {
	  k0 = L-2; kN = -1; dk = -1;
	}
      } else {
	if (dk > 0) {
	  k0 = 0; kN = L; dk = +1;
	} else {
	  k0 = L-1; kN = 0; dk = -1;
	}
      }
      for (int k = k0; k != kN; k += dk) {
	error = false;
	if (Dmax) {
	  newE = DMRG::minimize_two_sites(P, k, dk, Dmax);
	} else {
	  newE = DMRG::minimize_single_site(P, k, dk);
	}
	if (error) {
	  P = canonical_form(P);
	  return E;
	}
	if (debug > 2) {
	  show_state_info(P, iter, k, newE);
	}
      }
      if (debug) {
	if (debug > 1) {
	  show_state_info(P, iter, 0, newE);
	} else {
	  std::cout << "k=" << 0 << "; iteration=" << iter << "; E=" << newE << "; ";
	}
	std::cout << "dE=" << newE - E << '\n' << std::flush;
      }
      /*
       * Check the convergence by seeing how much the energy changed between
       * iterations.
       */
      if (iter) {
	if (tensor::abs(newE-E) < tolerance) {
	  if (debug) {
	    std::cout << "Reached tolerance dE=" << newE-E
		      << "<=" << tolerance << '\n' << std::flush;
	  }
	  E = newE;
	  break;
	}
	if ((newE - E) > 1e-14*tensor::abs(newE)) {
	  if (debug) {
	    std::cout << "Energy does not decrease!\n" << std::flush;
	  }
	  if (failures >= allow_E_growth) {
	    E = newE;
	    break;
	  }
	  failures++;
	}
      }
      E = newE;
    }
    return E;
  }


} // namespace mps
