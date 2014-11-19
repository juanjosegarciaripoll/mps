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

#ifndef MPS_DMRG_H
#define MPS_DMRG_H

#include <list>
#include <mps/mps.h>
#include <mps/mpo.h>
#include <mps/qform.h>
#include <mps/lform.h>

namespace mps {

  template<class MPS>
  class Minimizer {
  public:
    typedef MPS mps_t;
    typedef typename MPS::elt_t elt_t;
    typedef MPO<elt_t> mpo_t;
    typedef typename std::list<mpo_t> mpo_vector_t;
    typedef typename std::listr<mps_t> mps_vector_t;
    typedef QuadraticForm<mpo_t> qform_t;

    bool error;
    index sweeps;
    bool display;
    index debug;
    double tolerance;
    double svd_tolerance;
    int allow_E_growth;

    DMRG(const mpo_t &H);

    double minimize(mps_t *P, index Dmax = 0);

  private:
    const mpo_t H_;
    qform_t *Hqform_;

    double minimize_single_site(mps_t &P, index k, int dk);
    double minimize_two_sites(mps_t &P, index k, int dk, index Dmax);

    virtual void show_state_info(const mps_t &Pk, index iter, index k, double newE);

  private:
    DMRG(const DMRG &c); // Hidden, not allowed
  };

  extern template class DMRG<RMPS>;
  typedef DMRG<RMPS> RDMRG;

  extern template class DMRG<CMPS>;
  typedef DMRG<CMPS> CDMRG;

} // namespace dmrg

#endif /* !MPS_DMRG_H */
