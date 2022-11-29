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

#ifndef MPS_IO_H
#define MPS_IO_H

#include <tensor/io.h>
#include <tensor/sdf.h>
#include <mps/vector.h>
#include <mps/mps.h>
#include <mps/mpo.h>

namespace mps {

namespace impl {

template <class Tensor>
inline std::ostream &text_dump(std::ostream &s, const MP<Tensor> &mp,
                               const char *name) {
  index n = 0;
  for (const auto &P : mp) {
    s << name << '[' << n++ << "]=" << P << '\n';
  }
  return s;
}

template <class Tensor>
inline std::ostream &text_dump(std::ostream &s, const MPO<Tensor> &mpo,
                               const char *name) {
  index n = 0;
  for (const auto &P : mpo) {
    index r = P.dimension(1);
    index c = P.dimension(2);
    for (index i = 0; i < P.dimension(0); i++) {
      for (index j = 0; j < P.dimension(3); j++) {
        s << name << '[' << n << "](" << i << ",:,:," << j << ")=\n"
          << matrix_form(reshape(P(range(i), _, _, range(j)).copy(), r, c))
          << '\n';
      }
    }
  }
  return s;
}

extern template std::ostream &text_dump(std::ostream &s, const MP<RTensor> &mpo,
                                        const char *name);
extern template std::ostream &text_dump(std::ostream &s, const MP<CTensor> &mpo,
                                        const char *name);
extern template std::ostream &text_dump(std::ostream &s,
                                        const MPO<RTensor> &mpo,
                                        const char *name);
extern template std::ostream &text_dump(std::ostream &s,
                                        const MPO<CTensor> &mpo,
                                        const char *name);

template <class Tensor>
vector<Tensor> load_tensors(sdf::InDataFile &d, const std::string &name) {
  vector<Tensor> aux;
  d.load(&aux, name);
  return aux;
}

}  // namespace impl

inline std::ostream &operator<<(std::ostream &s, const RMPS &mpo) {
  return impl::text_dump(s, mpo, "RMPS");
}

inline std::ostream &operator<<(std::ostream &s, const CMPS &mpo) {
  return impl::text_dump(s, mpo, "CMPS");
}

inline std::ostream &operator<<(std::ostream &s, const RMPO &mpo) {
  return impl::text_dump(s, mpo, "RMPO");
}

inline std::ostream &operator<<(std::ostream &s, const CMPO &mpo) {
  return impl::text_dump(s, mpo, "CMPO");
}

template <class Tensor>
inline void dump(sdf::OutDataFile &d, const MPS<Tensor> &mps,
                 const std::string &name) {
  d.dump(mps.to_vector(), name);
}

inline RMPS load_rmps(sdf::InDataFile &d, const std::string &name) {
  return RMPS(impl::load_tensors<RTensor>(d, name));
}

inline CMPS load_cmps(sdf::InDataFile &d, const std::string &name) {
  return CMPS(impl::load_tensors<CTensor>(d, name));
}

}  // namespace mps

#endif /* MPS_IO_H */
