// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
#pragma once
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#ifndef TENSOR_TEST_STATES_H
#define TENSOR_TEST_STATES_H

#include <mps/vector.h>
#include <tensor/tensor.h>

namespace tensor_test {

using mps::vector;

template <typename Tensor>
inline vector<Tensor> random_product_state(index size, index dim = 2) {
  vector<Tensor> states(size);

  for (index i = 0; i < size; i++) {
    states[i] = Tensor::random(2);
    states[i] = states[i] / norm2(states[i]);
  }
  return states;
}

}  // namespace tensor_test

#endif  // TENSOR_TEST_STATES_H
