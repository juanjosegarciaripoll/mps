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
#ifndef MPS_ALGORITHMS_MPO_MPOEnvironmentS_H
#define MPS_ALGORITHMS_MPO_MPOEnvironmentS_H

#include <stdexcept>
#include <unordered_map>
#include <tensor/tensor.h>
#include <tensor/linalg.h>
#include <mps/mps/types.h>
#include <mps/mpo/types.h>
#include <mps/algorithms/environments.h>

namespace mps {

using namespace tensor;

template <class Tensor>
class Sparse4Tensor {
 public:
  struct subtensor_t {
    index left_index, right_index;
    Tensor matrix;
    subtensor_t(index left, index right, Tensor &&m)
        : left_index{left}, right_index{right}, matrix{std::move(m)} {}
  };

  Sparse4Tensor(const Tensor &t)
      : dimensions_(t.dimensions()), subtensors_{make_subtensors(t)} {}

  const Dimensions &dimensions() const { return dimensions_; }

  index dimension(index which) const { return dimensions_[which]; }

  Tensor zero_matrix() const {
    return Tensor::zeros(dimensions_[1], dimensions_[2]);
  }

  auto begin() const { return subtensors_.begin(); }
  auto end() const { return subtensors_.end(); }

 private:
  typedef std::list<subtensor_t> data_t;
  Dimensions dimensions_;
  data_t subtensors_;

  static data_t make_subtensors(const Tensor &t) {
    data_t output;
    index M = t.dimension(0);
    index N = t.dimension(3);
    for (index left_index = 0; left_index < M; ++left_index) {
      for (index right_index = 0; right_index < N; ++right_index) {
        Tensor matrix = t(range(left_index), _, _, range(right_index));
        if (norm2(matrix) != 0.0) {
          output.emplace_back(left_index, right_index, std::move(matrix));
        }
      }
    }
    return output;
  }
};

template <class Tensor>
class SparseMPO {
 public:
  typedef Sparse4Tensor<Tensor> value_type;

  SparseMPO(const MPO<Tensor> &mpo) : tensors_{make_sparse_tensors(mpo)} {}

  auto begin() const { return tensors_.begin(); }
  auto end() const { return tensors_.end(); }
  index size() const { return tensors_.size(); }
  index ssize() const { return tensor::ssize(tensors_); }
  const value_type &operator[](index i) const { return tensors_[i]; }

 private:
  typedef std::vector<Sparse4Tensor<Tensor>> tensor_list_t;
  tensor_list_t tensors_;

  static tensor_list_t make_sparse_tensors(const MPO<Tensor> &mpo) {
    tensor_list_t output;
    output.reserve(mpo.size());
    for (const auto &P : mpo) {
      output.emplace_back(P);
    }
    return output;
  }
};

template <class Tensor>
class MPOEnvironment {
 public:
  typedef Environment<Tensor> single_env_t;
  typedef std::unordered_map<index, single_env_t> env_t;

  explicit MPOEnvironment(Dir direction) : direction_{direction} {
    tensor_assert2(direction == DIR_RIGHT || direction == DIR_LEFT,
                   std::invalid_argument(
                       "Invalid direction supplied to MPOEnvironment()"));
    envs_.emplace(0, single_env_t(direction));
  }

  MPOEnvironment(Dir direction, env_t env, Dimensions dims)
      : envs_{std::move(env)},
        direction_{direction},
        dimensions_(std::move(dims)) {
    tensor_assert2(direction == DIR_RIGHT || direction == DIR_LEFT,
                   std::invalid_argument(
                       "Invalid direction supplied to MPOEnvironment()"));
  }

  MPOEnvironment propagate(const Tensor &bra, const Tensor &ket,
                           const Tensor &op) const {
    return propagate(bra, ket, Sparse4Tensor<Tensor>(op));
  }

  MPOEnvironment propagate(const Tensor &bra, const Tensor &ket,
                           const Sparse4Tensor<Tensor> &op) const {
    if (direction_ == DIR_RIGHT) {
      Dimensions dims = {index(1), index(1), bra.dimension(2),
                         ket.dimension(2)};
      return MPOEnvironment(direction(), propagate_right(envs_, bra, ket, op),
                            dims);
    } else {
      Dimensions dims = {bra.dimension(0), ket.dimension(0), index(1),
                         index(1)};
      return MPOEnvironment(direction(), propagate_left(envs_, bra, ket, op),
                            dims);
    }
  }

  bool has_environment_at(index i) const {
    return has_environment_at(envs_, i);
  }

  const single_env_t &operator[](index i) const { return envs_.at(i); }

  size_t size() const { return envs_.size(); }

  const env_t &tensors() const { return envs_; }

  constexpr Dir direction() const { return direction_; }

  Environment<Tensor> zero_environment() const {
    return Environment<Tensor>(direction(), Tensor::zeros(dimensions_));
  }

  const Dimensions &dimensions() const { return dimensions_; }

 private:
  env_t envs_{};
  Dir direction_{DIR_RIGHT};
  Dimensions dimensions_{1, 1, 1, 1};

  static bool has_environment_at(const env_t &env, index n) {
    return env.find(n) != env.end();
  }

  static inline env_t propagate_right(const env_t &env, const Tensor &Q,
                                      const Tensor &P,
                                      const Sparse4Tensor<Tensor> &op) {
    env_t new_env;
    for (auto &subtensor : op) {
      if (has_environment_at(env, subtensor.left_index)) {
        new_env.try_emplace(subtensor.right_index, DIR_RIGHT);
        new_env.at(subtensor.right_index) +=
            env.at(subtensor.left_index).propagate(Q, P, subtensor.matrix);
      }
    }
    return new_env;
  }

  static inline env_t propagate_left(const env_t &env, const Tensor &Q,
                                     const Tensor &P,
                                     const Sparse4Tensor<Tensor> &op) {
    env_t new_env;
    for (auto &subtensor : op) {
      if (has_environment_at(env, subtensor.right_index)) {
        new_env.try_emplace(subtensor.left_index, DIR_LEFT);
        new_env.at(subtensor.left_index) +=
            env.at(subtensor.right_index).propagate(Q, P, subtensor.matrix);
      }
    }
    return new_env;
  }
};

template <class Tensor>
Tensor maybe_add(const Tensor &a, const Tensor &b) {
  if (a.is_empty()) {
    return b;
  } else if (b.is_empty()) {
    return a;
  } else {
    return a + b;
  }
}

template <class Tensor>
Tensor compose(const MPOEnvironment<Tensor> &Lenv,
               const Sparse4Tensor<Tensor> &op,
               const MPOEnvironment<Tensor> &Renv) {
  Tensor output;
  for (const auto &opn : op) {
    if (Lenv.has_environment_at(opn.left_index) &&
        Renv.has_environment_at(opn.right_index)) {
      output = maybe_add(output, compose(Lenv[opn.left_index], opn.matrix,
                                         Renv[opn.right_index]));
    }
  }
  if (output.is_empty()) {
    return compose(Lenv.zero_environment(), op.zero_matrix(),
                   Renv.zero_environment());
  }
  return output;
}

template <class Tensor>
Tensor compose(const MPOEnvironment<Tensor> &Lenv,
               const Sparse4Tensor<Tensor> &op1,
               const Sparse4Tensor<Tensor> &op2,
               const MPOEnvironment<Tensor> &Renv) {
  Tensor output;
  for (const auto &op1n : op1) {
    for (const auto &op2n : op2) {
      if (op1n.right_index == op2n.left_index &&
          Lenv.has_environment_at(op1n.left_index) &&
          Renv.has_environment_at(op2n.right_index)) {
        output =
            maybe_add(output, compose(Lenv[op1n.left_index], op1n.matrix,
                                      op2n.matrix, Renv[op2n.right_index]));
      }
    }
  }
  if (output.is_empty()) {
    return compose(Lenv.zero_environment(), op1.zero_matrix(),
                   op2.zero_matrix(), Renv.zero_environment());
  }
  return output;
}

template <class Tensor>
linalg::LinearMap<Tensor> single_site_linear_map(
    const MPOEnvironment<Tensor> &Lenv, const Sparse4Tensor<Tensor> &op,
    const MPOEnvironment<Tensor> &Renv) {
  // We implement this
  // Q(a2,i,a3) = L(a1,b1,a2,b2) O1(i,k) P(b2,k,b3) R(a3,b3,a1,b1)
  // where a1=b1 = 1, because of open boundary conditions
  return [&](const Tensor &Pflat) {
    Tensor output;
    const Tensor P = reshape(Pflat, Lenv.dimensions()[3], op.dimension(2),
                             Renv.dimensions()[1]);
    for (const auto &opn : op) {
      if (Lenv.has_environment_at(opn.left_index) &&
          Renv.has_environment_at(opn.right_index)) {
        output =
            maybe_add(output, apply_environments(Lenv[opn.left_index],
                                                 Renv[opn.right_index],
                                                 foldin(opn.matrix, -1, P, 1)));
      }
    }
    if (output.is_empty()) {
      output =
          apply_environments(Lenv.zero_environment(), Renv.zero_environment(),
                             foldin(op.zero_matrix(), -1, P, 1));
    }
    return flatten(output);
  };
}

template <class Tensor>
linalg::LinearMap<Tensor> two_site_linear_map(
    const MPOEnvironment<Tensor> &Lenv, const Sparse4Tensor<Tensor> &op1,
    const Sparse4Tensor<Tensor> &op2, const MPOEnvironment<Tensor> &Renv) {
  // We implement this
  // Q(a2,i,a3) = L(a1,b1,a2,b2) O1(i,k) O2(j,l) P(b2,k,l,b3) R(a3,b3,a1,b1)
  // where a1=b1 = 1, because of open boundary conditions
  return [&](const Tensor &P12flat) {
    Tensor output;
    const Tensor P12 = reshape(P12flat, Lenv.dimensions()[3], op1.dimension(2),
                               op2.dimension(2), Renv.dimensions()[1]);
    for (const auto &op1n : op1) {
      for (const auto &op2n : op2) {
        if (op1n.right_index == op2n.left_index &&
            Lenv.has_environment_at(op1n.left_index) &&
            Renv.has_environment_at(op2n.right_index)) {
          output = maybe_add(
              output,
              apply_environments(
                  Lenv[op1n.left_index], Renv[op2n.right_index],
                  foldin(op1n.matrix, -1, foldin(op2n.matrix, -1, P12, 2), 1)));
        }
      }
    }
    if (output.is_empty()) {
      output =
          apply_environments(Lenv.zero_environment(), Renv.zero_environment(),
                             foldin(op1.zero_matrix(), -1,
                                    foldin(op2.zero_matrix(), -1, P12, 2), 1));
    }
    return flatten(output);
  };
}

}  // namespace mps

#endif  // MPS_ALGORITHMS_MPOEnvironmentS_H