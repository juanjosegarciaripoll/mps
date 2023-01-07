#pragma once
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

#ifndef MPS_MINIMIZER_H
#define MPS_MINIMIZER_H

#include <list>
#include <functional>
#include <optional>
#include <mps/mps.h>
#include <mps/mpo.h>
#include <mps/algorithms/truncation.h>

namespace mps {

template<class MPO>
struct MinimizerOptions {
  using mps_t = typename MPO::MPS;
  using constraints_t = std::tuple<const MPO &, double>;
  using callback_t = std::function<void(double E, const mps_t &state)>;

  index_t sweeps{32};
  int allow_E_growth{1};
  double tolerance{1e-10};

  int debug{0};
  bool single_site{false};
  bool display{false};
  bool compute_gap{false};

  TruncationStrategy truncation_strategy{};

  std::optional<callback_t> callback;
  std::optional<constraints_t> constraints;
  std::list<mps_t> orthogonal_states{};
};

double minimize(const RMPO &H, RMPS *psi, const MinimizerOptions<RMPO> &opt);
double minimize(const CMPO &H, CMPS *psi, const MinimizerOptions<CMPO> &opt);

double minimize(const RMPO &H, RMPS *psi);
double minimize(const CMPO &H, CMPS *psi);

}  // namespace mps

#endif /* !MPS_MINIMIZER_H */
