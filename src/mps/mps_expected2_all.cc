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

#include <mps/mps.h>
#include <mps/mps_algorithms.h>

namespace mps {

  using namespace tensor;

  /* TWO-SITE CORRELATION FUNCTION */

  template <class MPS, class Tensor>
  Tensor
  all_correlations_fast(const MPS &a,
                        const std::vector<Tensor> &op1,
                        const std::vector<Tensor> &op2,
                        const MPS &b,
                        bool symmetric = false,
                        const Tensor *jordan_wigner_op = 0)
  {
    size_t L = a.size();
    if (b.size() != L) {
      std::cerr << "In expected(MPS, Tensor, Tensor, MPS), two MPS of different size were passed";
      abort();
    }
    if (op1.size() != L) {
      std::cerr << "In expected(MPS, std::vector<Tensor>, std::vector<Tensor>, MPS), the 1st argument differs from the MPS size.";
      abort();
    }
    if (op2.size() != L) {
      std::cerr << "In expected(MPS, std::vector<Tensor>, std::vector<Tensor>, MPS), the 2n argument differs from the MPS size.";
      abort();
    }
    Tensor *auxRight = new Tensor[L];
    Tensor *auxLeft = new Tensor[L];
    {
      Tensor aux;
      for (size_t i = 1; i < L; i++) {
        auxLeft[i] = aux = prop_matrix(aux, +1, a[i-1], b[i-1], 0);
      }
      aux = Tensor();
      for (size_t i = 2; i < L; i++) {
        size_t ndx = L - i;
        auxRight[ndx] = aux = prop_matrix(aux, -1, a[ndx+1], b[ndx+1], 0);
      }
    }
    Tensor output = Tensor::zeros(L, L);
    for (size_t i = 0; i < L; i++) {
      {
        Tensor op12 = mmult(op1[i], op2[i]);
        Tensor aux = prop_matrix(auxLeft[i], +1, a[i], b[i], &op12);
        output.at(i,i) = prop_matrix_close(aux, auxRight[i])[0];
      }
      {
        Tensor aux = prop_matrix(auxLeft[i], +1, a[i], b[i], &op1[i]);
        for (size_t j = i+1; j < L; j++) {
          Tensor aux2 = prop_matrix(aux, +1, a[j], b[j], &op2[j]);
          output.at(i,j) = prop_matrix_close(aux2, auxRight[j])[0];
          aux = prop_matrix(aux, +1, a[j], b[j], jordan_wigner_op);
          output.at(j,i) = conj(output.at(i,j));
        }
      }
      if (!symmetric) {
        Tensor aux = prop_matrix(auxLeft[i], +1, a[i], b[i], &op2[i]);
        for (size_t j = 0; j < i; j++) {
          Tensor aux2 = prop_matrix(aux, +1, a[j], b[j], &op1[j]);
          output.at(i,j) = prop_matrix_close(aux2, auxRight[j])[0];
          aux = prop_matrix(aux, +1, a[j], b[j], 0);
        }
      }
    }
    delete[] auxRight;
    delete[] auxLeft;
    return output;
  }


} // namespace mps
