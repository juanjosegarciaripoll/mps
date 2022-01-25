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

#include <list>
#include <tensor/tools.h>
#include <tensor/io.h>
#include <tensor/linalg.h>
#include <mps/minimizer.h>
#include <mps/lform.h>
#include <mps/qform.h>

namespace mps {

template <class Tensor>
Tensor orthogonalize(Tensor Psi, const std::list<Tensor> &ortho) {
  for (auto other : ortho) {
    Psi -= scprod(other, Psi) * (other);
  }
  return Psi;
}

template <class Tensor, class QForm>
Tensor apply_qform1(const Tensor &P, const Indices &d, const QForm &qform,
                    const std::list<Tensor> &ortho) {
  Tensor v = orthogonalize(reshape(P, d), ortho);
  v = qform.apply_one_site_matrix(v);
  return reshape(orthogonalize(v, ortho), P.size());
}

template <class Tensor, class QForm>
Tensor apply_qform2_with_subspace(const Tensor &P, int sense, Tensor &P12,
                                  const QForm &qform, const Indices &subspace,
                                  const std::list<Tensor> &ortho) {
  // This is a bit fishy, because we reuse P12 as a buffer where to
  // do the computation of H * psi. However, we rely on the fact that
  // ARPACK is not multithreaded and thus apply_qform2_... is never
  // concurrent on the same tensor.
  P12.fill_with_zeros();
  P12.at(range(subspace)) = P;
  P12 = orthogonalize(P12, ortho);
  return orthogonalize(qform.apply_two_site_matrix(P12, sense),
                       ortho)(range(subspace));
}

template <class Tensor, class QForm>
Tensor apply_qform2(const Tensor &P, int sense, const Indices &d,
                    const QForm &qform, const std::list<Tensor> &ortho) {
  Tensor v = orthogonalize(reshape(P, d), ortho);
  v = qform.apply_two_site_matrix(v, sense);
  return orthogonalize(reshape(v, v.size()), ortho);
}

template <class MPO>
struct Minimizer : public MinimizerOptions {
  typedef MPO mpo_t;
  typedef typename MPO::MPS mps_t;
  typedef typename mps_t::elt_t tensor_t;
  typedef typename tensor_t::elt_t number_t;
  typedef QuadraticForm<mpo_t> qform_t;
  typedef LinearForm<mps_t> lform_t;
  typedef std::list<tensor_t> tensor_list_t;

  mps_t psi;
  qform_t Hqform, *Nqform;
  std::list<mps_t> OrthoMPS;
  std::list<lform_t> OrthoLform;
  number_t Nvalue;
  double Ntol;
  index site;
  int step;
  bool converged;

  Minimizer(const MinimizerOptions &opt, const mpo_t &H, const mps_t &state)
      : MinimizerOptions(opt),
        psi(canonical_form(state, -1)),
        Hqform(H, psi, psi, 0),
        Nqform(0),
        Nvalue(0),
        Ntol(1e-6),
        site(0),
        step(+1),
        converged(true) {}

  ~Minimizer() {
    if (Nqform) delete Nqform;
  }

  index size() { return psi.size(); }

  const mps_t &state() { return psi; }

  void add_states(const std::list<mps_t> &psi_orthogonal) {
    for (typename std::list<mps_t>::const_iterator it = psi_orthogonal.begin();
         it != psi_orthogonal.end(); ++it) {
      OrthoMPS.push_front(*it);
      OrthoLform.push_front(lform_t(*it, psi, 0));
    }
  }

  tensor_list_t orthogonal_projectors(index site, int sense) {
    tensor_list_t output;
    for (typename std::list<lform_t>::iterator it = OrthoLform.begin();
         it != OrthoLform.end(); ++it) {
      if (site != it->here()) {
        std::cout << "DMRG algorithm place at " << site
                  << " while projectors at " << it->here() << std::endl;
        abort();
      }
      tensor_t other =
          sense ? it->two_site_vector(sense) : it->single_site_vector();
      output.push_front(other / sqrt(norm2(other)));
    }
    return output;
  }

  void propagate(const tensor_t &P, index site, int step) {
    // Update Hamiltonian, constraints and linear forms for orthogonal
    // states with the new tensors that have just been introduced into
    // 'psi'.
    Hqform.propagate(psi[site], psi[site], step);
    if (Nqform) Nqform->propagate(psi[site], psi[site], step);
    for (typename std::list<lform_t>::iterator it = OrthoLform.begin();
         it != OrthoLform.end(); ++it) {
      it->propagate(psi[site], step);
    }
  }

  void add_constraint(const mpo_t &constraint, number_t value) {
    if (Nqform) delete Nqform;
    Nqform = new qform_t(constraint, psi, psi, 0);
    Nvalue = value;
  }

  double single_site_step() {
    tensor_list_t orthogonal_to = orthogonal_projectors(site, 0);
    tensor_t P = psi[site];
    const Indices d = P.dimensions();
    tensor_t E = linalg::eigs(
        [&](const tensor_t &v) {
          return apply_qform1<tensor_t, qform_t>(v, d, Hqform, orthogonal_to);
        },
        P.size(), linalg::SmallestAlgebraic, 1, &P, &converged);
    if (site == psi.size() / 2 && compute_gap) {
      tensor_t newP = P;
      tensor_t E = linalg::eigs(
          [&](const tensor_t &v) -> tensor_t {
            return apply_qform1<tensor_t, qform_t>(v, d, Hqform, orthogonal_to);
          },
          newP.size(), linalg::SmallestAlgebraic, 2, &newP, &converged);
      if (debug) {
        std::cout << "\thalf-size gap " << E[1] - E[0] << std::endl;
      }
      constrained_gap = gap = real(E[1] - E[0]);
    }
    if (converged) {
      set_canonical(psi, site, reshape(P, d), step, false);
      propagate(psi[site], site, step);
      if (debug > 1) {
        std::cout << "\tsite=" << site << ", E=" << real(E[0]) << ", P.d"
                  << psi[site].dimensions() << std::endl;
      }
    }
    return real(E[0]);
  }

  double single_site_sweep() {
    double E;
    if (step > 0) {
      for (site = 0; (site < size()) && converged; site++) {
        E = single_site_step();
      }
      step = -1;
    } else {
      site = size();
      do {
        site--;
        E = single_site_step();
      } while (site && converged);
      step = +1;
    }
    return E;
  }

  double two_site_step() {
    tensor_t E;
    if (debug > 1) {
      if (step > 0) {
        std::cout << "\tsite=" << site
                  << ", dimensions=" << psi[site].dimensions() << ","
                  << psi[site + 1].dimensions() << std::endl;
      } else {
        std::cout << "\tsite=" << site
                  << ", dimensions=" << psi[site - 1].dimensions() << ","
                  << psi[site].dimensions() << std::endl;
      }
    }
    tensor_t P12 = (step > 0) ? fold(psi[site], -1, psi[site + 1], 0)
                              : fold(psi[site - 1], -1, psi[site], 0);
    tensor_list_t orthogonal_to = orthogonal_projectors(site, step);
    if (site == psi.size() / 2 && compute_gap) {
      tensor_t newP12 = P12;
      auto fn = [&](const tensor_t &v) {
        return apply_qform2<tensor_t, qform_t>(v, step, newP12.dimensions(),
                                               Hqform, orthogonal_to);
      };
      tensor_t E = linalg::eigs(fn, newP12.size(), linalg::SmallestAlgebraic, 2,
                                &newP12);
      if (debug) {
        std::cout << "\thalf-size gap " << E[1] - E[0] << std::endl;
      }
      constrained_gap = gap = real(E[1] - E[0]);
    }
    if (Nqform) {
      Indices subspace;
      {
        tensor_t aux = Nqform->take_two_site_matrix_diag(step);
        subspace = which(abs(aux - Nvalue) < Ntol);
        if (debug > 1) {
          std::cout << "\tsite=" << site << ", constraints=" << subspace.size()
                    << "/" << aux.size() << std::endl;
        }
        if (subspace.size() == 0) {
          std::cout << "Unable to satisfy constraint " << Nvalue
                    << " with tolerance " << Ntol << std::endl;
          std::cout << "Values:\n" << matrix_form(aux) << std::endl;
          std::cout << abs(aux - Nvalue) << std::endl;
          std::cout << (abs(aux - Nvalue) < Ntol) << std::endl;
          converged = false;
          return 0.0;
        }
      }
      tensor_t subP12 = P12(range(subspace));
      P12.fill_with_zeros();
      E = linalg::eigs(
          [&](const tensor_t &v) -> tensor_t {
            return apply_qform2_with_subspace(v, step, P12, Hqform, subspace,
                                              orthogonal_to);
          },
          subP12.size(), linalg::SmallestAlgebraic, 1, &subP12, &converged);
      if (site == psi.size() / 2 && compute_gap) {
        tensor_t newP12 = subP12;
        auto fn = [&](const tensor_t &v) -> tensor_t {
          return apply_qform2_with_subspace(v, step, P12, Hqform, subspace,
                                            orthogonal_to);
        };
        tensor_t E = linalg::eigs(fn, newP12.size(), linalg::SmallestAlgebraic,
                                  2, &newP12, &converged);
        if (debug) {
          std::cout << "\tconstrained half-size gap " << (E[1] - E[0])
                    << std::endl;
        }
        constrained_gap = real(E[1] - E[0]);
      }
      P12.fill_with_zeros();
      P12.at(range(subspace)) = subP12;
    } else {
      const Indices d = P12.dimensions();
      auto fn = [&](const tensor_t &v) -> tensor_t {
        return apply_qform2<tensor_t, qform_t>(v, step, d, Hqform,
                                               orthogonal_to);
      };
      E = linalg::eigs(fn, P12.size(), linalg::SmallestAlgebraic, 1, &P12,
                       &converged);
      P12 = reshape(P12, d);
    }
    if (converged) {
      set_canonical_2_sites(psi, P12, site, step, Dmax, svd_tolerance, false);
    }
    propagate(psi[site], site, step);
    if (debug > 1) {
      std::cout << "\tsite=" << site << ", E=" << real(E[0])
                << ", P1.d=" << psi[site].dimensions()
                << ", P2.d=" << psi[site + step].dimensions()
                << ", converged=" << converged << std::endl;
    }
    return real(E[0]);
  }

  double two_site_sweep() {
    double E;
    if (step > 0) {
      for (site = 0; (site + 1 < size()) && converged; ++site) {
        E = two_site_step();
      }
      step = -1;
    } else {
      for (site = size() - 1; site && converged; --site) {
        E = two_site_step();
      }
      step = +1;
    }
    return E;
  }

  bool single_site() { return !Dmax; }

  double full_sweep(mps_t *psi) {
    double E = 1e28;
    if (debug) {
      tic();
      std::cout << "***\n*** Algorithm with " << size() << " sites, "
                << "two-sites = " << !single_site()
                << (Nqform ? ", constrained" : ", unconstrained") << std::endl;
    }
    for (index failures = 0, i = 0; i < sweeps; i++) {
      double newE = single_site() ? single_site_sweep() : two_site_sweep();
      if (debug) {
        std::cout << "iteration=" << i << "; E=" << newE << "; dE=" << newE - E
                  << "; tol=" << tolerance
                  << (converged ? "" : "; did not converge!") << "; t=" << toc()
                  << "s" << std::endl;
      }
      if (!converged) {
        *psi = mps_t();
        return E;
      }
      if (i) {
        if (tensor::abs(newE - E) < tolerance) {
          if (debug) {
            std::cout << "Reached tolerance dE=" << newE - E
                      << "<=" << tolerance << '\n'
                      << std::flush;
          }
          E = newE;
          break;
        }
        if ((newE - E) > 1e-14 * tensor::abs(newE)) {
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

}  // namespace mps
