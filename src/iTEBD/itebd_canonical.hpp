// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2012 Juan Jose Garcia Ripoll

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

#include <tensor/linalg.h>
#include <tensor/io.h>
#include <mps/itebd.h>
#include <mps/mps_algorithms.h>

namespace mps {

  template<class Tensor>
  static const Tensor ensure_3_indices(const Tensor &G)
  {
    if (G.rank() == 3) {
      return G;
    } else if (G.rank() == 4) {
      tensor::index a, i,j, b;
      G.get_dimensions(&a,&i,&j,&b);
      return reshape(G, a,i*j,b);
    } else {
      std::cerr << "Tensor with wrong dimensions in iTEBD" << std::endl;
      abort();
      return Tensor();
    }
  }

  template<class Tensor>
  static void
  ortho_basis(const Tensor V, Tensor *U, Tensor *Uinv)
  {
    /*
     * We have a basis |b> such that
     *		V(b,b') = <b'|b>
     * We decompose v into its eigenvalues, so that
     *		V = R * diag(s) * R'
     * where R is a unitary matrix R' R = R R' = identity.
     * These matrices give us a new orthogonal basis. Given
     *		|b> = U(b,c) |c>
     * we obtain, given <c|c'>=delta(c,c')
     *		V(b,b') = U(b,c) * U*(b',c') <c|c'> = U * U'
     * so that U = R * sqrt(s) and V = U * U';
     */
    Tensor R;
    Tensor s = abs(linalg::eig_sym(V, &R));
    s = sqrt(s);
    *U = scale(R, -1, s);
    *Uinv = scale(adjoint(R), 0, 1.0/s);
  }

  /**********************************************************************
   * CANONICAL FORM FOR A SINGLE TENSOR
   */

  template<class Tensor>
  const Tensor px(const Tensor &v, int sense, const Tensor *G)
  {
    return prop_matrix(v, sense, *G, *G);
  }

  template<class Tensor>
  static Tensor
  itebd_power_eig(const Tensor G, int sense, tensor::index iter = 0, double tol = 1e-11)
  {
    typedef typename Tensor::elt_t elt_t;
    tensor::index a, i;
    G.get_dimensions(&a, &i, &a);
    Tensor v = ((sense > 0)?
                reshape(Tensor::eye(a), 1,1,a,a) :
                reshape(Tensor::eye(a), a,a,1,1)) / (double)a;
    linalg::eig_power(with_args(px<Tensor>, sense, &G), v.size(), &v);
    // v(a1,a2) is associated index 'a1' with the conjugate tensor
    // and 'a2' with the tensor itself, (see prop_init) hence we have to
    // transpose.
    v = reshape(v, a, a);
    v = (transpose(v) + conj(v)) / 2.0;
    return v;
  }

  template<class Tensor>
  static void
  ortho_right(const Tensor &G, const Tensor &l, Tensor *X, Tensor *Xinv)
  {
    Tensor v = itebd_power_eig(scale(ensure_3_indices<Tensor>(G), -1, l), -1);
    ortho_basis<Tensor>(v, X, Xinv);
  }

  template<class Tensor>
  static void
  ortho_left(const Tensor &G, const Tensor &l, Tensor *Y, Tensor *Yinv)
  {
    Tensor v = itebd_power_eig(scale(ensure_3_indices<Tensor>(G), 0, l), +1);
    ortho_basis<Tensor>(v, Y, Yinv);
  }

  template<class Tensor>
  static void
  canonical_form(Tensor G, Tensor l, Tensor *pGout, Tensor *plout,
                 double tolerance, tensor::index max_dim)
  {
    Tensor X;		/* X(b,c) */
    Tensor Xinv;	/* Xinv(c,b) */
    ortho_right<Tensor>(G, l, &X, &Xinv);

    Tensor Y;		/* Y(a,d) */
    Tensor Yinv;	/* Yinv(d,a) */
    ortho_left<Tensor>(G, l, &Y, &Yinv);

    /* We implement http://arxiv.org/pdf/0711.3960v4.pdf */

    /* Xinv(c,a) G(a,i,b) Yinv(d,b) -> G(c,i,d) */
    G = fold(Xinv, -1, fold(G, -1, Yinv, -1), 0);

    /* Y(a,d) l(b) X(b,c) = aux(d,c) */
    Tensor aux = fold(Y, 0, scale(X, 0, l), 0);

    /* aux(d,c) = U(d,x) l(x) V(x,c) */
    Tensor U, V;
    *plout = limited_svd(aux, &U, &V, tolerance, max_dim);
    /* V(x,c) G(c,i,d) U(d,x') -> G(x,i,x') */
    *pGout = fold(fold(V, -1, G, 0), -1, U, 0);
  }

  template<class Tensor>
  static void
  split_tensor(Tensor GAB, Tensor lAB, Tensor *pA, Tensor *plA, Tensor *pB,
               Tensor *plB, double tolerance, tensor::index max_dim)
  {
    tensor::index a, i, j, b;
    /*
     * GAB is a coarse grain tensor that spans two sites. lB are the Schmidt
     * coefficients associated to this tensor. We first find the canonical form
     * so that the left and right basis are orthogonalized.
     */
    GAB.get_dimensions(&a,&i,&j,&b);
    canonical_form<Tensor>(reshape(GAB, a,i*j,b), lAB, &GAB, &lAB, tolerance, max_dim);
    a = b = lAB.size();
    /*
     * Now the state is given by
     *		|psi> = lB(a) GAB(a,i,j,b) lB(b) |a>|i,j>|b>
     * where |a> and |b> are orthonormal basis. We thus combine
     *		lB GAB lB -> GAB
     */
    GAB = scale(GAB, 0, lAB);
    GAB = scale(GAB, -1, lAB);
    /*
     * ...and perform the Schmidt decomposition of this tensor to obtain
     * the optimal "A" and "B".
     */
    *plB = lAB;
    *plA = Tensor(limited_svd(reshape(GAB, a*i, j*b), pA, pB, tolerance, max_dim));
    *pA = reshape(*pA, a, i, plA->size());
    *pB = reshape(*pB, plA->size(), i, b);
    *pA = scale(*pA, 0, 1.0/(*plB));
    *pB = scale(*pB, -1, 1.0/(*plB));
  }

  template<class Tensor>
  iTEBD<Tensor>::iTEBD(const Tensor &AB, const Tensor &lAB,
                       double tolerance, tensor::index max_dim)
  : canonical_(false)
  {
    split_tensor<Tensor>(AB, lAB, &A_, &lA_, &B_, &lB_, tolerance, max_dim);
    AlA_ = scale(A_, -1, lA_);
    BlB_ = scale(B_, -1, lB_);
  }

  /**********************************************************************
   * CANONICAL FORM TWO SITES
   */

  template<class Tensor>
  const Tensor px4(const Tensor &v, int sense, const Tensor *A, const Tensor *B)
  {
    return prop_matrix(prop_matrix(v, sense, *A, *A), sense, *B, *B);
  }

  template<class Tensor>
  static Tensor
  itebd_power_eig(const Tensor &A, const Tensor &lA, const Tensor &B, const Tensor &lB,
                  int sense)
  {
    typedef typename Tensor::elt_t elt_t;
    tensor::index a = lB.size();
    Tensor v = Tensor::eye(a,a) / sqrt((double)a);
    if (sense > 0) {
      Tensor A1 = scale(A, 0, lB);
      Tensor A2 = scale(B, 0, lA);
      linalg::eig_power(with_args(px4<Tensor>, sense, &A1, &A2), v.size(), &v);
    } else {
      Tensor A2 = scale(A, -1, lA);
      Tensor A1 = scale(B, -1, lB);
      linalg::eig_power(with_args(px4<Tensor>, sense, &A1, &A2), v.size(), &v);
    }
    // v(a1,a2) is associated index 'a1' with the conjugate tensor
    // and 'a2' with the tensor itself, (see prop_init) hence we have to
    // transpose.
    v = reshape(v, a, a);
    v = (transpose(v) + conj(v)) / 2.0;
    return v;
  }

  template<class Tensor>
  static void
  ortho_right(const Tensor &A, const Tensor &lA, const Tensor &B, const Tensor &lB,
              Tensor *X, Tensor *Xinv)
  {
    Tensor v = itebd_power_eig(A, lA, B, lB, -1);
    ortho_basis<Tensor>(v, X, Xinv);
  }

  template<class Tensor>
  static void
  ortho_left(const Tensor &A, const Tensor &lA, const Tensor &B, const Tensor &lB,
             Tensor *Y, Tensor *Yinv)
  {
    Tensor v = itebd_power_eig(A, lA, B, lB, +1);
    ortho_basis<Tensor>(v, Y, Yinv);
  }

  template<class Tensor>
  const iTEBD<Tensor>
  iTEBD<Tensor>::canonical_form(double tolerance, tensor::index max_dim) const
  {
    Tensor X;		/* X(b,c) */
    Tensor Xinv;	/* Xinv(c,b) */
    ortho_right<Tensor>(A_, lA_, B_, lB_, &X, &Xinv);

    Tensor Y;		/* Y(a,d) */
    Tensor Yinv;	/* Yinv(d,a) */
    ortho_left<Tensor>(A_, lA_, B_, lB_, &Y, &Yinv);

    /* We implement http://arxiv.org/pdf/0711.3960v4.pdf */

    /* Xinv(c,a) (A*lA*B)(a,i,b) Yinv(d,b) -> (A'*lA*B')(c,i,d) */
    Tensor newA = fold(Xinv, -1, A_, 0);
    Tensor newB = fold(B_, -1, Yinv, -1);

    /* Y(a,d) lB(b) X(b,c) = aux(d,c) */
    Tensor aux = fold(Y, 0, scale(X, 0, lB_), 0);

    /* aux(d,c) = U(d,x) l(x) V(x,c) */
    Tensor U, V;
    // There is no need to truncate because U 'aux' already has the
    // size of the bond dimension
    Tensor newlB = linalg::svd(aux, &U, &V, SVD_ECONOMIC);
    
    /* V(x,c) (A'*lA*B')(c,i,d) U(d,x') -> (A''*lA*B'')(x,i,x') */
    newA = fold(V, -1, newA, 0);
    newB = fold(newB, -1, U, 0);

#if 0    
    std::cout << "Testing canonical form\n";
    {
      Tensor A2 = scale(newA, -1, lA_);
      Tensor A1 = scale(newB, -1, newlB);
      Tensor v = Tensor::eye(A1.dimension(0));
      Tensor w = px4<Tensor>(v, -1, &A1, &A2);
      std::cout << "--- scprod(1,(AlABlB)*1) = " << scprod(w, v)/norm2(w)/norm2(v)
                << std::endl;
    }

    {
      Tensor A1 = scale(newA, 0, newlB);
      Tensor A2 = scale(newB, 0, lA_);
      Tensor v = Tensor::eye(A1.dimension(0));
      Tensor w = px4<Tensor>(v, +1, &A1, &A2);
      std::cout << "+++ scprod(1,1*(lBAlAB)) = " << scprod(w, v)/norm2(w)/norm2(v)
                << std::endl;
    }
#endif
    return iTEBD<Tensor>(newA, lA_, newB, newlB, true);
  }

  template<class Tensor>
  const iTEBD<Tensor>
  iTEBD<Tensor>::apply_operator(const Tensor &U, int site, double tolerance,
                                tensor::index max_dim) const
  {
    Tensor A, lA, B, lB;
    if (site & 1) {
      Tensor GBA = fold(BlB_, -1, A_, 0);
      tensor::index a, i, j, b;
      GBA.get_dimensions(&a, &i, &j, &b);
      GBA = reshape(foldin(U, -1, reshape(GBA, a, i*j, b), 1), a, i, j, b);
      split_tensor(GBA, lA_, &B, &lB, &A, &lA, tolerance, max_dim);
      return iTEBD<Tensor>(A, lA, B, lB, false);
    } else {
      Tensor GAB = fold(AlA_, -1, B_, 0);
      tensor::index a, i, j, b;
      GAB.get_dimensions(&a, &i, &j, &b);
      GAB = reshape(foldin(U, -1, reshape(GAB, a, i*j, b), 1), a, i, j, b);
      split_tensor(GAB, lB_, &A, &lA, &B, &lB, tolerance, max_dim);
      return iTEBD<Tensor>(A, lA, B, lB, false);
    }
  }

}
