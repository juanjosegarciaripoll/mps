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

#include <tensor/io.h>
#include <mps/lattice.h>

namespace mps {

const Indices Lattice::states_in_particle_range(index_t sites, index_t nmin,
                                                index_t nmax) {
  if (sizeof(word) == 4) {
    if (sites >= 32) {
      std::cerr << "In this architecture with 32-bit words, Lattice can only "
                   "handle up to 31 sites"
                << '\n';
      abort();
    }
  } else {
    if (sites > 34) {
      std::cerr << "In this architecture with " << sizeof(word) * 8
                << "bit words, Lattice can only handle up to 34 sites" << '\n';
      abort();
    }
  }
  word problem_size = 0;
  const word end = word(1) << sites;
  for (word configuration = 0; configuration < end; ++configuration) {
    const index_t total_particles = count_bits(configuration);
    if (total_particles >= nmin && total_particles <= nmax) {
      ++problem_size;
    }
  }
  Indices output(problem_size);
  for (word n = 0, configuration = 0; configuration < end; ++configuration) {
    const index_t total_particles = count_bits(configuration);
    if (total_particles >= nmin && total_particles <= nmax) {
      output.at(n++) = configuration;
    }
  }
  return output;
}

static Lattice::word find_configuration(Lattice::word w,
                                        const Indices &configurations) {
  Lattice::word j = configurations.ssize() - 1;
  Lattice::word i = 0;
  if (w == configurations[i]) return i;
  if (w == configurations[j]) return j;
  while (i != j) {
    Lattice::word k = (i + j) / 2;
    Lattice::word neww = configurations[k];
    if (neww > w) {
      j = k;
    } else if (neww == w) {
      return k;
    } else {
      i = k;
    }
  }
  return i;
}

void Lattice::bipartition(index_t sites_left, Indices *left_states,
                          Indices *right_states,
                          Indices *matrix_indices) const {
  index_t sites_right = size() - sites_left;
  if (sites_left <= 0 || sites_right <= 0) {
    std::cerr << "In Lattice::bipartition(), the number of sites "
                 "in any bipartition cannot be less than one."
              << '\n'
              << "sites_left = " << sites_left << '\n'
              << "total sites = " << size() << '\n';
    abort();
  }

  // The lowest number of particles in the left bipartition is
  // achieved when all are in the right half.
  index_t left_nmin = std::max<index_t>(0, particles() - sites_right);
  index_t right_nmin = std::max<index_t>(0, particles() - sites_left);
  // The highest number of particles in a half of the lattice is
  // achieved by filling with the most possible number of particles
  index_t left_nmax = std::min<index_t>(sites_left, particles());
  index_t right_nmax = std::min<index_t>(sites_right, particles());

  *left_states = states_in_particle_range(sites_left, left_nmin, left_nmax);
  *right_states = states_in_particle_range(sites_right, right_nmin, right_nmax);
  *matrix_indices = Indices(dimension());

  word one = 1;
  word right_mask = (one << sites_right) - 1;
  for (word i = 0; i < static_cast<word>(configurations.size()); i++) {
    word w = configurations[i];
    word wl = w >> sites_right;
    word wr = w & right_mask;
    word il = find_configuration(wl, *left_states);
    word ir = find_configuration(wr, *right_states);
    //
    // Since elements in the states are ordered as 'w', this means
    // that 'ir' is the fastest running index and, in our convention,
    // it corresponds to the 'rows' (i.e. the first index in a tensor)
    //
    word ndx = ir + il * static_cast<word>(right_states->size());
    // std::cerr << "(w,wl,wr,ndx)=" << w << ',' << wl << ',' << wr << ',' << ndx << '\n';
    matrix_indices->at(i) = ndx;
  }
}

}  // namespace mps
