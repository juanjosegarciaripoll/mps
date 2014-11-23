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

#include <tensor/io.h>
#include <tensor/linalg.h>
#include <mps/minimizer.h>
#include <mps/qform.h>

namespace mps {

  template<class Tensor, class QForm>
  const Tensor apply_qform1(const Tensor &P, const Indices *d, const QForm *qform)
  {
    return reshape(qform->apply_one_site_matrix(reshape(P, *d)), P.size());
  }

  template<class Tensor, class QForm>
  const Tensor apply_qform2(const Tensor &P, int sense, const Indices *d, const QForm *qform)
  {
    return reshape(qform->apply_two_site_matrix(reshape(P, *d), sense), P.size());
  }

  template<class MPO>
  struct Minimizer : public MinimizerOptions {
    typedef MPO mpo_t;
    typedef typename MPO::MPS mps_t;
    typedef typename mps_t::elt_t tensor_t;
    typedef typename tensor_t::elt_t number_t;
    typedef QuadraticForm<mpo_t> qform_t;

    mps_t psi;
    qform_t Hqform;
    index site;
    int step;
    bool converged;

    Minimizer(const MinimizerOptions &opt, const mpo_t &H, const mps_t &state) :
      MinimizerOptions(opt),
      psi(canonical_form(state, 0)),
      Hqform(H, psi, psi, 0),
      site(0),
      step(+1)
    {}

    ~Minimizer()
    {}

    index size() {
      return psi.size();
    }

    const mps_t &state() {
      return psi;
    }

    double single_site_step() {
      tensor_t P = psi[site];
      const Indices d = P.dimensions();
      tensor_t E = linalg::eigs(with_args(apply_qform1<tensor_t,qform_t>, &d, &Hqform),
                                P.size(), linalg::SmallestAlgebraic, 1, &P, &converged);
      if (converged) {
        set_canonical(psi, site, reshape(P, psi[site].dimensions()), step, false);
        Hqform.propagate(psi[site], psi[site], step);
        if (debug > 1) {
          std::cout << "site=" << site << ", E=" << real(E[0])
                    << ", P.dimensions()" << psi[site].dimensions()
                    << std::endl;
        }
      }
      return real(E[0]);
    }

    double single_site_sweep() {
      double E;
      if (step > 0) {
	for (site = 0; site < size(); site++) {
	  E = single_site_step();
	}
	step = -1;
      } else {
	site = size();
	do {
	  site--;
	  E = single_site_step();
	} while (site);
	step = +1;
      }
      return E;
    }

    double two_site_step() {
      tensor_t P12 =
        (step > 0) ?
        fold(psi[site], -1, psi[site+1], 0) :
        fold(psi[site-1], -1, psi[site], 0);
      const Indices d = P12.dimensions();
      tensor_t E = linalg::eigs(with_args(apply_qform2<tensor_t,qform_t>, step, &d, &Hqform),
                                P12.size(), linalg::SmallestAlgebraic, 1, &P12,
                                &converged);
      if (converged) {
        set_canonical_2_sites(psi, reshape(P12, d), site, step, Dmax, svd_tolerance);
      }
      Hqform.propagate(psi[site], psi[site], step);
      if (debug > 1) {
        std::cout << "site=" << site << ", E=" << real(E[0])
                  << ", P1.dimensions()" << psi[site].dimensions()
                  << ", P2.dimensions()" << psi[site+1].dimensions()
                  << std::endl;
      }
      return real(E[0]);
    }

    double two_site_sweep() {
      double E;
      if (step > 0) {
	for (site = 0; site+1 < size(); site++) {
	  E = two_site_step();
	}
	step = -1;
      } else {
        for (site = size()-1; site; site--) {
	  E = two_site_step();
	}
	step = +1;
      }
      return E;
    }

    bool single_site() {
      return !Dmax;
    }

    double full_sweep(mps_t *psi) {
      double E = 1e28;
      if (debug) {
        std::cout << "***\n*** Algorithm with " << size() << " sites, "
                  << "two-sites = " << !single_site() << std::endl;
      }
      for (index failures = 0, i = 0; i < sweeps; i++) {
        double newE = single_site()? single_site_sweep() : two_site_sweep();
        if (debug) {
          std::cout << "iteration=" << i << "; E=" << newE
                    << "; dE=" << newE - E << "; tol=" << tolerance
                    << (converged? "" : "; did not converge!")
                    << std::endl;
        }
        if (!converged) {
          break;
        }
        if (i) {
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
      *psi = state();
      return E;
    }
  };


} // namespace mps
