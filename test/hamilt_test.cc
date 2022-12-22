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

#include <tensor/rand.h>
#include <mps/quantum.h>
#include "loops.h"

namespace tensor_test {

using namespace mps;

TestHamiltonian::TestHamiltonian(index amodel, double spin, index size, bool ti,
                                 bool pbc)
    : ConstantHamiltonian(size, pbc),
      model(amodel),
      coefs(),
      periodic(pbc),
      translational_invariance(ti) {
  double data[][4] = {{1, 0, 0, 0},   // Magnetic field
                      {0, 0, 0, 1},   // Ising
                      {1, 0, 0, 1},   // Ising long. field
                      {1, 1, 0, 0},   // Ising transv. field
                      {1, 1, 1, 0},   // XX
                      {1, 1, 1, 2},   // XXY
                      {1, 1, 1, 1},   // Heisenberg
                      {0, 0, 0, 0}};  // Random

  RTensor aux;
  if (model < last_model()) {
    aux = Vector<double>(4, data[model]);
    aux.at(0) = aux(0) * rand<double>(1.0);
  } else {
    aux = RTensor::random(4);
  }
  coefs = RTensor::zeros(4, size);
  if (ti) {
    for (index j = 0; j < size; j++) {
      coefs.at(_, range(j)) = aux;
    }
  } else {
    coefs.randomize();
    for (index j = 0; j < size; j++) {
      coefs.at(_, range(j)) = aux * coefs(0, j);
    }
  }
  if (!is_periodic()) coefs.at(range(1, -1), range(-1)) = 0.0;

  spin_operators(spin, &sx, &sy, &sz);
  CTensor op[4] = {sz, sx, imag(sy), sz};
  double sgn[4] = {1, 1, -1, 1};

  for (index i = 0; i < size; i++) {
    set_local_term(i, coefs(0, i) * op[0]);
  }
  for (index i = 1; i < size; i++) {
    for (int n = 1; n < 4; n++) {
      if (coefs(n, i)) {
        add_interaction(i - 1, sgn[n] * coefs(n, i) * op[n], op[n]);
      }
    }
  }
}

const char *TestHamiltonian::model_name(index model) {
  const char *name[] = {"Mag. field", "Ising", "Ising long.", "Ising perp.",
                        "XX",         "XXY",   "Heisenberg",  "Random"};
  return name[model];
}

index TestHamiltonian::last_model() { return 6; }

void test_over_H(bool test(const mps::Hamiltonian &, double &), int max_spins,
                 bool pbc) {
  for (int periodic = 0; periodic < (pbc ? 2 : 1); periodic++) {
    for (unsigned model = 0; model <= TestHamiltonian::last_model(); model++) {
      for (int ti = 1; ti >= 0; ti--) {
        double err = 0.0;
        std::cerr << TestHamiltonian::model_name(model)
                  << (periodic ? ",pbc" : ",obc") << (ti ? ",t.i " : ",inh ");
        tic();
        for (unsigned nspins = 2; nspins < max_spins; nspins++) {
          for (int times = 1; times < 5; times++) {
            TestHamiltonian H(model, 0.5, nspins, ti, periodic);
            double e = 0.0;
            if (test(H, e)) {
              std::cerr << '.';
            } else {
              std::cerr << '!';
            }
            std::cerr.flush();
            err = std::max(e, err);
          }
        }
        std::cerr << "\n\t\tMax. Err: " << err << ", Time: " << toc() << '\n';
      }
    }
  }
  std::cerr << '\n';
}

}  // namespace tensor_test
