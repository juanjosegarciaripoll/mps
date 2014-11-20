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
#include <mps/minimizer.h>
#include <mps/qform.h>

namespace mps {

  template<class Tensor, class QForm>
  const Tensor apply_qform1(const Tensor &P, const QForm *qform)
  {
    return qform->apply_one_site_matrix(P);
  }

  template<class Tensor, class QForm>
  const Tensor apply_qform2(const Tensor &P, const QForm *qform)
  {
    return qform->apply_two_site_matrix(P);
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
      P = reshape(P, P.size());
      tensor_t E = linalg::eigs(with_args(apply_qform1<tensor_t,qform_t>, &Hqform),
                                P.size(), linalg::SmallestAlgebraic, 1, &P);
      set_canonical(psi, site, reshape(P, psi[site].dimensions()), step);
      Hqform.propagate(psi[site], psi[site], step);
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
      tensor_t P12 = fold(psi[site], -1, psi[site+1], 0);
      Indices dims = P12.dimensions();
      P12 = reshape(P12, P12.size());
      tensor_t E = linalg::eigs(with_args(apply_qform2<tensor_t,qform_t>, &Hqform),
                                P12.size(), linalg::SmallestAlgebraic, 1, &P12);
      set_canonical_2_sites(psi, reshape(P12, dims), site, step, Dmax, svd_tolerance);
      Hqform.propagate(psi[site], psi[site], step);
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
	site = size()-1;
	do {
	  site--;
	  E = two_site_step();
	} while (site);
	step = +1;
      }
      return E;
    }

    bool single_site() {
      return !Dmax;
    }

    double full_sweep(mps_t *psi) {
      double E = 1e28;
      for (index failures = 0, i = 0; i < sweeps; i++) {
        double newE = single_site()? single_site_sweep() : two_site_sweep();
        if (debug > 1) {
          std::cout << "iteration=" << i << "; E=" << newE << std::endl;
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
      }
      *psi = state();
      return E;
    }
  };


} // namespace mps
