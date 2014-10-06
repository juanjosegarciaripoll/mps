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

#ifndef MPS_QFORM_H
#define MPS_QFORM_H

#include <vector>
#include <forward_list>
#include <mps/mpo.h>
#include <mps/hamiltonian.h>

namespace mps {

  /** Internal representation of a MPO as a quadratic form.
      This object works with a scalar product $$\langle\psi|O|\phi\rangle$$,
      where $O$ is a MPO, and the bra and ked, $\psi$ and $\phi$, are two
      MPS. This scalar product, from the point of view of the k-th site, it
      may be seen as a quadratic form $w^{T} \bar{O} v$, where $w$ and $v$
      are the elements of the tensors $\psi$[k] and $\phi$[k] of the MPS
      on that given site.
  */
  template<class MPO>
  class QuadraticForm {
  public:
    typedef typename MPO::MPS MPS;
    typedef typename MPS::elt_t elt_t;

    /** Initialize with the given MPO and bra and ket states. This function
	assumes that we are inspecting site 'start', which may be at the
	beginning or the end of the chain.*/
    QuadraticForm(const MPO &mpo, const MPS &bra, const MPS &ket, int start = 0);
    /** Update the form changing the tensors of the bra and ket states. The
	function updates the $\psi$ and $\phi$ states in $\langle\psi|O|\phi\rangle$
	changing the tensor of those states at the site here(), and moving to
	the next site, here()+1.*/
    void propagate_right(const elt_t &braP, const elt_t &ketP);
    /** Update the form changing the tensors of the bra and ket states. The
	function updates the $\psi$ and $\phi$ states in $\langle\psi|O|\phi\rangle$
	changing the tensor of those states at the site here(), and moving to
	the next site, here()-1.*/
    void propagate_left(const elt_t &braP, const elt_t &ketP);
    /** The site at which the quadratic form is defined. */
    int here() const { return current_site_; }
    /** Left bond dimension of the MPO at the present site. */
    index left_dimension(index site) const { return bond_dimensions_[site]; }
    /** Right bond dimension of the MPO at the present site. */
    index right_dimension(index site) const { return bond_dimensions_[site+1]; }
    /** Number of sites in the lattice. */
    index size() const { return left_matrix_.size(); }

    /** Matrix representation of the quadratic form with respect to site here().*/
    const elt_t single_site_matrix() const;
    /** Matrix representation of the quadratic form with respect to
	sites here() and here()+1.*/
    const elt_t two_site_matrix() const;

  private:

    struct Pair {
      int left_ndx, right_ndx; // inner dimensions in the MPO
      elt_t op; // operator

      Pair(int i, int j, const elt_t &tensor) :
	left_ndx(i),
	right_ndx(j),
	op(reshape(tensor(range(i), range(),range(), range(j)),
		   tensor.dimension(1), tensor.dimension(2)))
      {}

      bool is_empty() const { return norm2(op) == 0; }
    };

    typedef typename std::vector<elt_t> matrix_array_t;
    typedef typename std::vector<matrix_array_t> matrix_database_t;
    typedef typename std::forward_list<Pair> pair_list_t;
    typedef typename std::vector<pair_list_t> pair_tree_t;
    typedef typename pair_list_t::const_iterator pair_iterator_t;

    int current_site_;
    Indices bond_dimensions_;
    matrix_database_t left_matrix_, right_matrix_;
    pair_tree_t pairs_;

    static Indices mpo_inner_dimensions(const MPO &mpo);
    static matrix_database_t make_matrix_database(const Indices &dimensions, bool left);
    static pair_tree_t make_pairs(const MPO &mpo);
  };

  extern template class QuadraticForm<RMPO>;
  typedef QuadraticForm<RMPO> RQForm;

  extern template class QuadraticForm<CMPO>;
  typedef QuadraticForm<CMPO> CQForm;

} // namespace dmrg

#endif /* !MPS_DMRG_H */