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
#ifndef MPS_IMATH_H
#define MPS_IMATH_H

#include <type_traits>
#include <tensor/exceptions.h>

namespace mps {

namespace imath {

template <typename base, typename power,
          typename = std::enable_if<std::is_integral<power>::value>>
static inline base ipow(base b, power p) {
  if (p < 0) {
    // Negative powers means 1/b**|p| < 1, which rounds to zero
    return 0;
  } else {
    base x = b, output = 1;
    while (p) {
#ifdef TENSOR_DEBUG
      if (p & 1) {
        base new_output = output * x;
        tensor_assert2(new_output / x == output,
                       std::overflow_error("Integer overflow in ipow()"));
      }
#endif
      p >>= 1;
      x *= x;
    }
    return output;
  }
}

template <typename base,
          typename = std::enable_if<std::is_integral<base>::value>>
static inline base isqrt(base b) {
  tensor_assert2(b >= 0, std::invalid_argument("Negative argument to isqrt()"));
  if (b < 2) {
    return b;
  }
  base x0 = static_cast<base>(std::sqrt(static_cast<double>(b)));
  // Find new estimate
  base x1 = (x0 + b / x0) / 2;
  while (x1 < x0)  // This also checks for cycle
  {
    x0 = x1;
    x1 = (x0 + b / x0) / 2;
  }
  return x0;
}

}  // namespace imath

}  // namespace mps

#endif  // !MPS_IMATH_H
