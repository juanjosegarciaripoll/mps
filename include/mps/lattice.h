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

#ifndef MPS_LATTICE_H
#define MPS_LATTICE_H

#include <tensor/linalg.h>
#include <tensor/sparse.h>

namespace mps {

using namespace tensor;
using linalg::EigType;
using linalg::LinearMap;

/** Class representing fermionic or hard-core-bosons particles hopping in a
   * finite lattice. The lattice constructs operators representing the motion of
   * particles, their density, their interactions, assuming that there is a
   * fixed (beforehand) number of particles at all times.
   */
class Lattice {
 public:
  typedef index_t word;

  enum particle_kind_t {
    /** The lattice will contain impenetrable bosonic particles. */
    HARD_CORE_BOSONS = 0,
    /** The lattice will contain fermions (in Jordan-Wigner representation).*/
    FERMIONS = 1
  };

  /** Construct the internal representation for a lattice with N particles in
        those 'sites'*/
  Lattice(index_t sites, index_t N);

  /** Maximum number of sites that a lattice can have */
  static constexpr index_t max_sites() { return (sizeof(word) == 4) ? 31 : 34; }

  /** Hopping operator for a particle between two sites. Returns the
        equivalent of \$a^\dagger_{to}a_{from}\$. */
  const RSparse hopping_operator(index_t to, index_t from,
                                 particle_kind_t kind = FERMIONS) const;
  /** Number operator for the given lattice site.*/
  const RSparse number_operator(index_t site) const;
  /** Hubbard interaction between different lattice site. It implements
        operator \$ n_{site1} n_{site2} \$.*/
  const RSparse interaction_operator(index_t site1, index_t site2) const;

  /** Full Hamiltonian containing hopping of kind and
        interactions. Matrix J(i,j) is nonzero when there is hopping between
        sites 'i' and 'j', and interactions U(i,j) among those sites
        too. Entries in these matrices can be zero. */
  const RSparse Hamiltonian(const RTensor &J, const RTensor &interactions,
                            double mu, particle_kind_t kind = FERMIONS) const;
  /** Full Hamiltonian containing hopping of kind and
        interactions. Matrix J(i,j) is nonzero when there is hopping between
        sites 'i' and 'j', and interactions U(i,j) among those sites
        too. Entries in these matrices can be zero. */
  const CSparse Hamiltonian(const CTensor &J, const CTensor &interactions,
                            double mu, particle_kind_t kind = FERMIONS) const;

  /** Bipartition of the lattice. We regard lattice sites 0 to N-1 as one
        half, and N to size() as the other half. We construct two vectors of all
        physical configurations that result from splitting particles on each of
        the sublattices, having respective sizes L1 and L2. In addition to this
        we also construct a vector of indices, ndx, such that the element psi[i]
        of a wavefunction is mapped to the ndx[i] entry of a reduced density
        matrix, of size L1 * L2.
    */
  void bipartition(index_t sites_left, Indices *left_states,
                   Indices *right_states, Indices *matrix_indices) const;

  /** Number of sites in the lattice. */
  index_t size() const;
  /** Preconfigured number of particles. */
  index_t particles() const;
  /** Dimensionality of the constrained Hilbert space. */
  index_t dimension() const;

  template <class Tensor>
  Tensor apply(const Tensor &psi, const Tensor &J, const Tensor &U,
               particle_kind_t kind = FERMIONS) const {
    if (psi.rank() > 1) {
      index_t M = psi.dimension(0);
      index_t L = psi.ssize() / M;
      Tensor output = reshape(psi, M, L);
      for (index_t i = 0; i < L; ++i) {
        output.at(_, range(i)) = apply(Tensor(output(_, range(i))), J, U, kind);
      }
      return reshape(output, psi.dimensions());
    } else {
      Tensor output = Tensor::zeros(psi.dimensions());
      RTensor values;
      Indices ndx;
      for (index_t i = 0; i < J.rows(); ++i) {
        for (index_t j = 0; j < J.columns(); ++j) {
          if (abs(J(i, j)) != 0) {
            /* TODO: Avoid sort_indices() by doing the adjoint of the hopping */
            hopping_inner(&values, &ndx, i, j, kind);
            output += J(i, j) * values * psi(range(sort_indices(ndx)));
          }
          if (j >= i && abs(U(i, j)) != 0) {
            values = interaction_inner(i, j);
            output += (U(i, j) + U(j, i)) * values * psi;
          }
        }
      }
      return output;
    }
  }

  template <class Tensor>
  LinearMap<Tensor> map(const Tensor &J, const Tensor &U,
                        particle_kind_t kind = FERMIONS) const {
    return [J, U, L = *this, kind](const Tensor &psi) {
      return L.apply(psi, J, U, kind);
    };
  }

  RTensor eigs(const RTensor &J, const RTensor &U, EigType eig_type,
               size_t neig, RTensor *vectors = nullptr,
               bool *converged = nullptr,
               particle_kind_t kind = FERMIONS) const {
    return linalg::eigs(this->map(J, U, kind), dimension(), eig_type, neig,
                        vectors, converged);
  }

  CTensor eigs(const CTensor &J, const CTensor &U, EigType eig_type,
               size_t neig, CTensor *vectors = nullptr,
               bool *converged = nullptr,
               particle_kind_t kind = FERMIONS) const {
    return linalg::eigs(this->map(J, U, kind), dimension(), eig_type, neig,
                        vectors, converged);
  }

  void hopping_inner(RTensor *values, Indices *ndx, index_t to_site,
                     index_t from_site, particle_kind_t kind) const;
  const RTensor interaction_inner(index_t site1, index_t site2) const;

 private:
  const index_t number_of_sites;
  const index_t number_of_particles;
  const Indices configurations;

  static index_t count_bits(Lattice::word w);

  static const Indices states_with_n_particles(index_t sites,
                                               index_t number_of_particles);
  static const Indices states_in_particle_range(index_t sites, index_t nmin,
                                                index_t nmax);
};

}  // namespace mps

#endif /* !MPS_LATTICE_H */
