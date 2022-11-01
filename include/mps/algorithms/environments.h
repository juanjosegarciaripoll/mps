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
#ifndef MPS_ALGORITHMS_ENVIRONMENTS_H
#define MPS_ALGORITHMS_ENVIRONMENTS_H

#include <stdexcept>
#include <tensor/tensor.h>
#include <mps/mps/types.h>

namespace mps {

using namespace tensor;

/*TODO: use enum class Dir*/
#if 0
enum class Dir { RIGHT = +1, LEFT = -1 };
#else
enum { DIR_RIGHT = +1, DIR_LEFT = -1 };
typedef int Dir;
#endif

template <class Tensor>
class Environment {
 public:
  typedef typename Tensor::elt_t scalar_t;

  Environment(Dir direction, Tensor t = Tensor())
      : env_{std::move(t)}, direction_{direction} {
    if (direction != DIR_RIGHT && direction != DIR_LEFT) {
      throw std::invalid_argument(
          "Invalid direction supplied to Environment()");
    }
  }

  Environment propagate(const Tensor &bra, const Tensor &ket) const {
    if (direction_ == DIR_RIGHT) {
      return Environment(direction(),
                         propagate_right(tensor(), bra, ket, nullptr));
    } else {
      return Environment(direction(),
                         propagate_left(tensor(), bra, ket, nullptr));
    }
  }

  Environment propagate(const Tensor &bra, const Tensor &ket,
                        const Tensor &op) const {
    if (direction_ == DIR_RIGHT) {
      return Environment(direction(), propagate_right(tensor(), bra, ket, &op));
    } else {
      return Environment(direction(), propagate_left(tensor(), bra, ket, &op));
    }
    return *this;
  }

  scalar_t close() const { return close_tensor(env_); }

  scalar_t operator*(Environment &other) {
    if (this->direction_ != DIR_RIGHT || other.direction_ != DIR_LEFT) {
      throw std::invalid_argument(
          "Incompatible directions when contracting mps::Environment's");
    }
    return combine_environments(this->tensor(), other.tensor());
  }

  Environment operator+(const Environment &other) const {
    tensor_assert(direction() == other.direction());
    return Environment(direction(), tensor() + other.tensor());
  }

  Environment &operator+=(const Environment &other) {
    tensor_assert(direction() == other.direction());
    if (this->is_empty()) {
      env_ = other.tensor();
    } else {
      env_ += other.tensor();
    }
    return *this;
  }

  Tensor tensor() const {
    if (this->is_empty()) {
      return Tensor::ones(1, 1, 1, 1);
    } else {
      return env_;
    }
  }

  constexpr Dir direction() const { return direction_; }

  bool is_empty() const { return env_.size() == 0; }

 private:
  Tensor env_;
  Dir direction_;

  static inline scalar_t close_tensor(const Tensor &env) {
    index a1, b1, a2, b2;
    env.get_dimensions(&a1, &b1, &a2, &b2);
    return trace(reshape(env, a1 * b1, a2 * b2), 0, -1)[0];
  }

  static inline Tensor init_environment(const Tensor &Q, const Tensor &P,
                                        const Tensor *op) {
    // M(a1,a2,b1,b2) = Q'(a1,i,a2) P(b1,i,b2)
    Tensor env = op ? foldc(Q, 1, fold(*op, 1, P, 1), 0) : foldc(Q, 1, P, 1);
    // M(a1,a2,b1,b2) -> M(a1,b1,a2,b2)
    return permute(env, 1, 2);
  }

  static inline Tensor propagate_right(const Tensor &env, const Tensor &Q,
                                       const Tensor &P, const Tensor *op) {
    if (env.is_empty()) {
      return init_environment(Q, P, op);
    }
    index a1, b1, a2, b2, i2, a3, b3;
    env.get_dimensions(&a1, &b1, &a2, &b2);
    Q.get_dimensions(&a2, &i2, &a3);
    P.get_dimensions(&b2, &i2, &b3);
    Tensor M = op ?
                  // M(a1,b1,a2,b2) Op(j2,i2) Q'(a2,j2,a3) -> M(a1,b1,b2,i2,a3)
                   fold(env, 2, fold(*op, 0, tensor::conj(Q), 1), 1)
                  :
                  // M(a1,b1,a2,b2) Q'(a2,i2,a3) -> M(a1,b1,b2,i2,a3)
                   fold(env, 2, tensor::conj(Q), 0);
    // M(a1,b1,[b2,i2],a3) P([b2,i2],b3) -> M(a1,b1,a3,b3)
    return fold(reshape(M, a1, b1, b2 * i2, a3), 2, reshape(P, b2 * i2, b3), 0);
  }

  static inline Tensor propagate_left(const Tensor &env, const Tensor &Q,
                                      const Tensor &P, const Tensor *op) {
    if (env.is_empty()) {
      return init_environment(Q, P, op);
    }
    index a1, b1, a2, b2, a0, b0, i0;
    env.get_dimensions(&a1, &b1, &a2, &b2);
    Q.get_dimensions(&a0, &i0, &a1);
    P.get_dimensions(&b0, &i0, &b1);
    Tensor M = op ?
                  // P(b0,j0,b1) Op(i0,j0) M(a1,b1,a2,b2) -> M(b0,i0,a1,a2,b2)
                   fold(fold(P, 1, *op, -1), 1, env, 1)
                  :
                  // P(b0,i0,b1) M(a1,b1,a2,b2) -> M(b0,i0,a1,a2,b2)
                   fold(P, -1, env, 1);
    // Q'(a0,[i0,a1]) M(b0,[i0,a1],a2,b2) -> M(a0,b0,a2,b2)
    return foldc(reshape(Q, a0, i0 * a1), -1, reshape(M, b0, i0 * a1, a2, b2),
                 1);
  }

  static inline scalar_t combine_environments(const Tensor &L,
                                              const Tensor &R) {
    if (L.is_empty()) {
      return close_tensor(R);
    }
    if (R.is_empty()) {
      return close_tensor(L);
    }
    index a1, a2, b1, b2;
    L.get_dimensions(&a1, &a2, &b1, &b2);
    R.get_dimensions(&b1, &b2, &a1, &a2);
    return mmult(reshape(L, 1, a1 * a2 * b1 * b2),
                 reshape(transpose(reshape(R, b1 * b2, a1 * a2)),
                         a1 * a2 * b1 * b2, 1))[0];
  }
};

template <class Tensor>
inline Tensor qform_matrix(const Environment<Tensor> &Lenv,
                           const Environment<Tensor> &Renv) {
  const Tensor &L = Lenv.tensor();
  const Tensor &R = Renv.tensor();
  //
  // We have some quadratic function expressed as
  //	E = L([a3,b3],a1,b1) P'(a1,i,a2) Op(i,j) P(b1,j,b2) R(a2,b2,[a3,b3])
  //	E = P'(i,a1,a2) Q([i,a1,a2],[j,b1,b2]) P(j,b1,b2)
  // where
  //	Q([a1,i,a2],[b1,j,b2]) = Op(i,j) (R*L)
  //
  // Notice that in L(a1,[a3,b3],b1), 'a1' is the index associated with the
  // complex conjugate of P, while in R(b2,[a3,b3],a2) it is 'a2', the last
  // one. IT IS CRITICAL THAT WE GET THE ORDER OF THE a's AND b's IN Q RIGHT.
  // Use effective_Hamiltonian_test(), by setting opts.debug=1 in
  // ground_state() to detect bugs in these functions.
  //
  if (L.is_empty()) {
    // P is the matrix corresponding to the first site, and thus there is no
    // "L" matrix or rather L(a1,b1,a3,b3) = delta(a1,a3)\delta(b1,b3).
    if (R.is_empty()) {
      // This state has a single site.
      return Tensor::eye(1);
    } else {
      index a1, a2, b1, b2;
      R.get_dimensions(&a2, &b2, &a1, &b1);
      // R(a2,b2,a1,b1) -> R(b1,b2,a1,a2) -> R(a1,a2,b1,b2) =: Q
      return transpose(reshape(permute(R, 0, 3), b1 * b2, a1 * a2));
    }
  } else if (R.is_empty()) {
    index a1, a2, b1, b2;
    // Similar as before, but P is the matrix is the one of the last site
    // and R(a2,b2,a3,b3) = delta(a2,a3)delta(b2,b3)
    L.get_dimensions(&a2, &b2, &a1, &b1);
    // L(a2,b2,a1,b1) -> L(b1,b2,a1,a2) -> L(a1*a2,b1*b2) =: Q
    return transpose(reshape(permute(L, 0, 3), b1 * b2, a1 * a2));
  } else {
    index a1, a2, a3, b1, b2, b3;
    L.get_dimensions(&a3, &b3, &a1, &b1);
    R.get_dimensions(&a2, &b2, &a3, &b3);
    // L(a3,b3,a1,b1)R(a2,b2,a3,b3) -> Q(a1,b1,a2,b2)
    Tensor Q =
        fold(reshape(L, a3 * b3, a1, b1), 0, reshape(R, a2, b2, a3 * b3), 2);
    // Q(a1,b1,a2,b2) -> Q(a1,a2,b1,b2)
    Q = reshape(permute(Q, 1, 2), a1 * a2, b1 * b2);
  }
}

template <class Tensor>
Tensor compose(const Environment<Tensor> &Lenv, const Tensor &op,
               const Environment<Tensor> &Renv) {
  // std::cerr << "Compose\n"
  //           << " L=" << L << '\n'
  //           << " R=" << R << '\n'
  //           << " op=" << op << '\n';
  auto L = Lenv.tensor();
  auto R = Renv.tensor();
  index a1, a2, b1, b2, a3, b3;
  // L(a1,b1,a2,b2) op(i,j) R(a3,b3,a1,b1) -> H([a2,i,a3],[b2,j,b3])
  L.get_dimensions(&a1, &b1, &a2, &b2);
  R.get_dimensions(&a3, &b3, &a1, &b1);
  tensor_assert(a1 == 1 && b1 == 1);
  // Remember that kron(A(i,j),B(k,l)) -> C([k,i],[l,j])
  return kron(kron(reshape(R, a3, b3), op), reshape(L, a2, b2));
}

template <class Tensor>
Tensor compose(const Environment<Tensor> &Lenv, const Tensor &op1,
               const Tensor &op2, const Environment<Tensor> &Renv) {
  // std::cerr << "Compose\n"
  //           << " L=" << L << '\n'
  //           << " R=" << R << '\n'
  //           << " op1=" << op1 << '\n'
  //           << " op2=" << op2 << '\n';
  auto L = Lenv.tensor();
  auto R = Renv.tensor();
  index a1, a2, b1, b2, a3, b3;
  // L(a1,b1,a2,b2) op(i,j) R(a3,b3,a1,b1) -> H([a2,i,a3],[b2,j,b3])
  L.get_dimensions(&a1, &b1, &a2, &b2);
  R.get_dimensions(&a3, &b3, &a1, &b1);
  tensor_assert(a1 == 1 && b1 == 1);
  // Remember that kron(A(i,j),B(k,l)) -> C([k,i],[l,j])
  return kron(kron(kron(reshape(R, a3, b3), op2), op1), reshape(L, a2, b2));
}

template <class Tensor>
Tensor apply_environments(const Environment<Tensor> &Lenv,
                          const Environment<Tensor> &Renv, const Tensor &P) {
  const auto &Ltensor = Lenv.tensor();
  const auto &Rtensor = Renv.tensor();
  index a2 = Ltensor.dimension(2);
  index b2 = Ltensor.dimension(3);
  index a3 = Rtensor.dimension(0);
  index b3 = Rtensor.dimension(1);
  return fold(fold(reshape(Ltensor, a2, b2), 1, P, 0), -1,
              reshape(Rtensor, a3, b3), 1);
}

}  // namespace mps

#endif  // MPS_ALGORITHMS_ENVIRONMENTS_H