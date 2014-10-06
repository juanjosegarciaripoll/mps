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

#ifndef MPS_LFORM_H
#define MPS_LFORM_H

#include <vector>
#include <mps/mps.h>

namespace mps {

  /** Internal representation of a MPS as a linear form.  This object
      works with a scalar product $$\langle\psi|\phi\rangle$$, where
      the bra and ked, $\psi$ and $\phi$, are two MPS. This scalar
      product, from the point of view of the k-th site, it may be seen
      as a linear form $w^{T} v$, where $w$ and $v$ are the
      elements of the tensors $\psi$[k] and $\phi$[k] of the MPS on
      that given site.
  */
  template<class MPS>
  class LinearForm {
  public:
    typedef MPS MPS;
    typedef typename MPS::elt_t elt_t;

    LinearForm(const MPS &bra, const MPS &ket, int start = 0);

    void propagate_right(const elt_t &braP, const elt_t &ketP);
    void propagate_left(const elt_t &braP, const elt_t &ketP);

    /** The site at which the quadratic form is defined. */
    int here() const { return current_site_; }
    /** Left bond dimension of the MPS at the present site. */
    index left_dimension(index site) const { return bond_dimensions_[site]; }
    /** Right bond dimension of the MPS at the present site. */
    index right_dimension(index site) const { return bond_dimensions_[site+1]; }
    /** Number of sites in the lattice. */
    index size() const { return bra_.size(); }

    /** Vector representation of the linear form with respect to site here().*/
    const elt_t single_site_vector() const;
    /** Vector representation of the linear form with respect to
	sites here() and here()+1.*/
    const elt_t two_site_vector() const;

  private:

    typedef typename std::vector<elt_t> matrix_array_t;

    int current_site_;
    Indices bond_dimensions_;
    matrix_array_t left_matrix_, right_matrix_;
    MPS bra_, ket_;

    static Indices mps_inner_dimensions(const MPS &mps);
    static matrix_array_t make_matrix_array(const Indices &dimensions, bool left);
  };

  extern template class LinearForm<RMPS>;
  typedef LinearForm<RMPS> RLForm;

  extern template class LinearForm<CMPS>;
  typedef LinearForm<CMPS> CLForm;

} // namespace mps

#endif // MPS_LFORM_H
