// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
#pragma once
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
#ifndef MPS_ALGORITHMS_LINALG_H
#define MPS_ALGORITHMS_LINALG_H

#include <mps/algorithms/environments.h>
#include <tensor/io.h>

namespace mps {

template <typename Tensor>
class MPS;

/**Norm of a RMPS.*/
template <class Tensor>
inline double norm2(const MPS<Tensor> &a) {
  return sqrt(tensor::abs(scprod(a, a)));
}

/**Scalar product between MPS.*/
template <class T1, class T2>
inline tensor_scalar_t<tensor_common_t<T1, T2>> scprod(
    const MPS<T1> &a, const MPS<T2> &b, int direction = DIR_RIGHT) {
  Environment<tensor_common_t<T1, T2>> env(direction);
  if (a.size() != b.size()) {
    throw std::invalid_argument(
        "In scprod(), mismatch in the sizes of both MPS");
  }
  if (direction == DIR_RIGHT) {
    for (index k = 0; k < a.ssize(); k++) {
      env.propagate(a[k], b[k]);
    }
  } else {
    for (index k = a.ssize(); k--;) {
      env.propagate(a[k], b[k]);
    }
  }
  return env.close();
}

/**Compute a single-site expected value.*/
template <class T1, class T2>
inline tensor_scalar_t<tensor_common_t<T1, T2>> expected(
    const MPS<T1> &a, const T2 &op, index k1, int direction = DIR_RIGHT) {
  if (direction != DIR_RIGHT) {
    throw std::domain_error("DIM_LEFT not implemented in expected()");
  }
  Environment<tensor_common_t<T1, T2>> env(DIR_RIGHT);
  for (index k = 0, target = a.normal_index(k1); k < a.ssize(); k++) {
    auto Pk = a[k];
    if (k == target) {
      env.propagate(Pk, Pk, op);
    } else {
      env.propagate(Pk, Pk);
    }
  }
  return env.close();
}

/**Compute a single-site expected value.*/
template <class T1, class T2>
inline tensor_scalar_t<tensor_common_t<T1, T2>> expected(
    const MPS<T1> &a, const T2 &op1, index k1, const T2 &op2, index k2,
    int direction = DIR_RIGHT) {
  if (k1 == k2) {
    return expected(a, mmult(op1, op2), k1, direction);
  }
  if (direction != DIR_RIGHT) {
    throw std::domain_error("DIM_LEFT not implemented in expected()");
  }
  Environment<tensor_common_t<T1, T2>> env(DIR_RIGHT);
  auto target1 = a.normal_index(k1);
  auto target2 = a.normal_index(k2);
  for (index k = 0; k < a.ssize(); k++) {
    auto Pk = a[k];
    if (k == target1) {
      env.propagate(Pk, Pk, op1);
    } else if (k == target2) {
      env.propagate(Pk, Pk, op2);
    } else {
      env.propagate(Pk, Pk);
    }
  }
  return env.close();
}

/**Compute a single-site expected value, summed over all sites.*/
template <class T1, class T2>
inline tensor_scalar_t<tensor_common_t<T1, T2>> expected(const MPS<T1> &a,
                                                         const T2 &op) {
  tensor_common_t<T1, T2> adapted_op = op;
  tensor_scalar_t<tensor_common_t<T1, T2>> output = 0.0;
  for (index i = 0; i < a.ssize(); i++) {
    output += expected(a, adapted_op, i);
  }
  return output;
}

template <class Tensor>
Tensor expected_vector(const MPS<Tensor> &a, const std::vector<Tensor> &op,
                       const MPS<Tensor> &b) {
  index L = a.ssize();
  if (b.ssize() != L) {
    throw std::invalid_argument(
        "In expected_vector(), MPS have different sizes");
  }
  if (ssize(op) != L) {
    throw std::invalid_argument(
        "In expected_vector(), number of operators does not match MPS size");
  }
  std::vector<Environment<Tensor>> auxLeft(L, Environment<Tensor>(DIR_RIGHT));
  {
    Environment<Tensor> left(DIR_RIGHT);
    for (index i = 1; i < L; i++) {
      auxLeft[i] = left.propagate(a[i - 1], b[i - 1]);
    }
  }

  Environment<Tensor> right(DIR_LEFT);
  auto output = Tensor::empty(L);
  for (index i = L; i--;) {
    output.at(i) = auxLeft[i].propagate(a[i], b[i], op[i]) * right;
    right.propagate(a[i], b[i]);
  }
  return output;
}

/**Compute a vector of single-site expected values, reusing environments.*/
template <class Tensor>
Tensor expected_vector(const MPS<Tensor> &a, const std::vector<Tensor> &op) {
  return expected_vector(a, op, a);
}

/**Compute a vector of single-site expected values, reusing environments.*/
template <class Tensor>
Tensor expected_vector(const MPS<Tensor> &a, const Tensor &op) {
  return expected_vector(a, std::vector<Tensor>(a.size(), op), a);
}

template <class Tensor>
Tensor all_correlations_fast(const MPS<Tensor> &a,
                             const std::vector<Tensor> &op1,
                             const std::vector<Tensor> &op2,
                             const MPS<Tensor> &b, bool symmetric = false,
                             const Tensor *jordan_wigner_op = 0) {
  index L = a.ssize();
  if (b.ssize() != L) {
    throw std::invalid_argument(
        "In expected(MPS, Tensor, Tensor, MPS), two MPS of different size were "
        "passed");
  }
  if (ssize(op1) != L || ssize(op2) != L) {
    throw std::invalid_argument(
        "In expected(MPS, Tensor, Tensor, MPS), the list of operators differs "
        "from the MPS size");
  }

  Environment<Tensor> aux(DIR_RIGHT);
  std::vector<Environment<Tensor>> auxLeft(L, aux);
  for (index i = 1; i < L; i++) {
    auxLeft[i] = aux.propagate(a[i - 1], b[i - 1]);
  }
  aux = Environment<Tensor>(DIR_LEFT);
  std::vector<Environment<Tensor>> auxRight(L, aux);
  for (index i = L - 1; i; --i) {
    auxRight[i - 1] = aux.propagate(a[i], b[i]);
  }
  Tensor output = Tensor::zeros(L, L);
  for (index i = 0; i < L; i++) {
    {
      Tensor op12 = mmult(op1[i], op2[i]);
      output.at(i, i) = auxLeft[i].propagate(a[i], b[i], op12) * auxRight[i];
    }
    {
      aux = auxLeft[i].propagate(a[i], b[i], op1[i]);
      for (index j = i + 1; j < L; j++) {
        auto aux2 = aux;
        output.at(i, j) = aux.propagate(a[j], b[j], op2[j]) * auxRight[j];
        if (jordan_wigner_op) {
          aux.propagate(a[j], b[j], *jordan_wigner_op);
        } else {
          aux.propagate(a[j], b[j]);
        }
        output.at(j, i) = tensor::conj(output.at(i, j));
      }
    }
  }
  if (!symmetric) {
    for (index i = 0; i < L; i++) {
      aux = auxLeft[i].propagate(a[i], b[i], op2[i]);
      for (index j = i + 1; j < L; j++) {
        auto aux2 = aux;
        output.at(j, i) = aux2.propagate(a[j], b[j], op1[j]) * auxRight[j];
        if (jordan_wigner_op) {
          aux.propagate(a[j], b[j], *jordan_wigner_op);
        } else {
          aux.propagate(a[j], b[j]);
        }
      }
    }
  }
  return output;
}

/**Compute all two-site correlations.*/
template <class Tensor>
Tensor expected(const MPS<Tensor> &a, const std::vector<Tensor> &op1,
                const std::vector<Tensor> &op2) {
  return all_correlations_fast(a, op1, op2, a);
}

/**Compute all two-site correlations.*/
template <class Tensor>
Tensor expected(const MPS<Tensor> &a, const Tensor &op1, const Tensor &op2) {
  index L = a.size();
  std::vector<Tensor> vec1(L, op1);
  std::vector<Tensor> vec2(L, op2);
  return all_correlations_fast(a, vec1, vec2, a);
}

}  // namespace mps

#endif  // MPS_ALGORITHMS_LINALG_H