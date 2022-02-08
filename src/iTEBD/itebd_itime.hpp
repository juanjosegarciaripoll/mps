// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2012 Juan Jose Garcia Ripoll

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

#include <numeric>
#include <tensor/linalg.h>
#include <tensor/io.h>
#include <mps/itebd.h>

namespace mps {

template <class Tensor>
const iTEBD<Tensor> evolve_itime(iTEBD<Tensor> psi, const Tensor &H12,
                                 double dt, tensor::index nsteps,
                                 double tolerance, tensor::index max_dim,
                                 tensor::index deltan, int method,
                                 std::vector<double> *energies,
                                 std::vector<double> *entropies) {
  static const double FR_param[5] = {0.67560359597983, 1.35120719195966,
                                     -0.17560359597983, -1.70241438391932};

  Tensor eH12[4];
  //int method = 2;
  switch (method) {
    case 1:
      /* Second order Trotter expansion */
      eH12[1] = linalg::expm((-dt / 2) * H12);
    case 0:
      /* First order Trotter expansion */
      eH12[0] = linalg::expm((-dt) * H12);
      break;
    default:
      /* Fourth order Trotter expansion */
      for (int i = 0; i < 4; i++) {
        eH12[i] = linalg::expm((-dt * FR_param[i]) * H12);
      }
  }
  Tensor Id = Tensor::eye(H12.rows());
  double time = 0;
  psi = psi.canonical_form();
  double E = energy(psi, H12), S = psi.entropy();
  if (energies) energies->push_back(E);
  if (entropies) entropies->push_back(S);

  if (!deltan) {
    deltan = 1;
  }
  std::cout.precision(5);
  std::cout << nsteps << ", " << dt << " x " << deltan << " = " << dt * deltan
            << std::endl;
  std::cout << "t=" << time << ";\tE=" << E << "; dE=" << 0.0 << ";\tS=" << S
            << "; dS=" << 0.0 << ";\tl="
            << std::max(psi.left_dimension(0), psi.right_dimension(0))
            << std::endl
            << "l = " << matrix_form(real(psi.left_vector(0))) << std::endl;
  for (index phases = (nsteps + deltan - 1) / deltan; phases; phases--) {
    for (index i = 0; (i < deltan); i++) {
      switch (method) {
        case 0:
          psi = psi.apply_operator(eH12[0], 0, tolerance, max_dim);
          psi = psi.apply_operator(eH12[0], 1, tolerance, max_dim);
          break;
        case 1:
          psi = psi.apply_operator(eH12[1], 0, tolerance, max_dim);
          psi = psi.apply_operator(eH12[0], 1, tolerance, max_dim);
          psi = psi.apply_operator(eH12[1], 0, tolerance, max_dim);
          break;
        default:
          psi = psi.apply_operator(eH12[0], 0, tolerance, max_dim);
          psi = psi.apply_operator(eH12[1], 1, tolerance, max_dim);
          psi = psi.apply_operator(eH12[2], 0, tolerance, max_dim);
          psi = psi.apply_operator(eH12[3], 1, tolerance, max_dim);
          psi = psi.apply_operator(eH12[2], 0, tolerance, max_dim);
          psi = psi.apply_operator(eH12[1], 1, tolerance, max_dim);
          psi = psi.apply_operator(eH12[0], 0, tolerance, max_dim);
      }
      time += dt;
    }
    psi = psi.canonical_form();
    double newE = energy(psi, H12);
    double newS = psi.entropy();
    if (energies) energies->push_back(newE);
    if (entropies) entropies->push_back(newS);
    std::cout << "t=" << time << ";\tE=" << newE << "; dE=" << newE - E
              << ";\tS=" << newS << "; dS=" << newS - S << ";\tl="
              << std::max(psi.left_dimension(0), psi.right_dimension(0))
              << std::endl
              << "l = " << matrix_form(real(psi.left_vector(0))) << std::endl;
    E = newE;
    S = newS;
  }
  return psi;
}

}  // namespace mps
