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

#include <mps/vector.h>
#include <algorithm>
#include <tensor/linalg.h>
#include <mps/algorithms.h>
#include <mps/flags.h>
#include <tensor/io.h>
#include "mps_prop_matrix.cc"

namespace mps {

/*----------------------------------------------------------------------
   * THE MATHEMATICS OF THE SIMPLIFICATION
   *
   * We have a matrix product state
   *	Q1(a1,i1,a2) Q1(a2,i2,a3) ... |i1...iN> = |Q> = |Q>
   * and we want to approximate it by a simpler one
   *	P1(b1,i1,b2) P1(b2,i2,b3) ... |i1...iN> = |P> = |P>
   * where the indices bk run over a smaller range and thus the projectors "P"
   * are much smaller.
   *
   * We do it variationally, optimizing Pk and leaving the rest invariant. For
   * the optimization of Pk we minimize
   *	||P-Q||^2 = <P|Q> + <P|Q> - <P|Q> - <Q|P>
   * Since the vectors "P" are orthogonalized (see orthogonalize_mps)
   *	<P|P> = P*(b1,i1,b2) P(b1,i1,b2)
   *	<P|Q> = Ml(b1,a1) Q(a1,i1,a2) Mr(a2,b2) P*(b1,i1,b2)
   * from where the solution to this problem follows
   *	P = Ml(b1,a1) Q(a1,i1,j1,a2) Mr(a2,b2).
   * and afterwards we need to orthonormalize P again.
   *
   * We will need the following "propagators":
   *
   * Mr{k}(a1,b1) = Qk(a1,i1,a2) Mr{k+1}(a2,b2) Pk*(b1,i1,b2)
   * Mr{N+1} = 1
   * Ml{k}(b2,a2) = Ml{k-1}(b1,a1) Qk(a1,i1,a2) Pk*(b1,i1,b2)
   * Ml{0} = 1
   *
   * We store them so that, when we are sweeping and optimizing the k-th
   * site, then
   *	A{k} = Ml{k-1}
   *	A{k+2} = Mr{k+1}
   * For the first round we only need to compute the Mr{2}..Mr{N}.
   */
template <class Tensor>
static void normalize_this(Tensor &Pk, int sense) {
  index_t a1, i1, a2;
  Tensor U;
  if (sense > 0) {
    Pk.get_dimensions(&a1, &i1, &a2);
    linalg::block_svd(reshape(Pk, a1 * i1, a2), &U, nullptr, SVD_ECONOMIC);
    a2 = std::min(a1 * i1, a2);
  } else {
    Pk.get_dimensions(&a1, &i1, &a2);
    linalg::block_svd(reshape(Pk, a1, i1 * a2), nullptr, &U, SVD_ECONOMIC);
    a1 = std::min(a1, i1 * a2);
  }
  Pk = reshape(U, a1, i1, a2);
}

template <class MPS>
struct MPSManySimplifier {
  typedef typename MPS::elt_t Tensor;
  typedef tensor_scalar_t<Tensor> number;
  typedef vector<Tensor> tensor_vector_t;
  typedef vector<tensor_vector_t> matrices_t;
  typedef vector<MPS> mps_vector_t;

  index_t L;
  const index_t nvectors;
  const mps_vector_t &Q;
  const Tensor &weights;
  matrices_t A;
  double normQ2;

  MPSManySimplifier(const mps_vector_t &aQ, const Tensor &aweights)
      : nvectors(aQ.ssize()), Q(aQ), weights(aweights) {
    if (nvectors == 0) {
      std::cerr << "In mps::simplify(), at least one vector has to be provided."
                << '\n';
      abort();
    }
    L = Q[0].ssize();
    if (L == 1) {
      std::cerr << "The mps::simplify() function is designed to "
                   "work with states that have more than one site."
                << '\n';
      abort();
    }

    A = matrices_t(L + 2, tensor_vector_t(nvectors));
    dump_matrices("Const");

    number x = number_zero<number>();
    for (int i = 0; i < nvectors; i++) {
      for (int j = i; j < nvectors; j++) {
        number y = scprod(Q[i], Q[j]) * tensor::conj(weights[i]) * weights[j];
        if (i == j)
          x = x + y;
        else
          x = x + 2 * real(y);
      }
    }
    normQ2 = real(x);
  }

  Tensor &matrix(index_t site, index_t vector) { return A[site + 1].at(vector); }

#if 0
  void dump_matrices(const char *context = "foo") {
      std::cerr << context << '\n';
      for (int j = 0; j < A.size(); j++) {
        for (int i = 0; i < A[j].size(); i++) {
          std::cerr << "A[" << j << "][" << i << "] = " << A[j][i].dimensions()
                    << '\n';
        }
      }
  }
#else
  void dump_matrices(const char *) {}
#endif

  void update_matrices(index_t site, const Tensor &Pk, int sense) {
    if (sense > 0) {
      for (int i = 0; i < nvectors; i++) {
        matrix(site, i) = prop_matrix(matrix(site - 1, i), +1, Pk, Q[i][site]);
      }
    } else {
      for (int i = 0; i < nvectors; i++) {
        matrix(site, i) = prop_matrix(matrix(site + 1, i), -1, Pk, Q[i][site]);
      }
    }
  }

  number true_scprod(const MPS &P) {
    number x = number_zero<number>();
    for (int i = 0; i < nvectors; i++) {
      x = x + scprod(P, Q[i]) * weights[i];
    }
    return x;
  }

  void initialize_matrices(const MPS &P, int sense) {
    if (sense > 0) {
      for (index_t k = L; k--;) {
        update_matrices(k, P[k], -1);
      }
    } else if (sense < 0) {
      for (index_t k = 0; k < L; k++) {
        update_matrices(k, P[k], +1);
      }
    } else {
      std::cerr << "In simplify(MPS &,...): sense=0 is not a valid direction";
      abort();
    }
    dump_matrices("initialization");
  }

  number scalar_product(index_t site) {
    number M = number_zero<number>();
    dump_matrices("scprod");
    for (int i = 0; i < nvectors; i++) {
      M += prop_matrix_close(matrix(site, i))[0] * weights[i];
    }
    return M;
  }

  const Tensor next_projector(Tensor Ml, Tensor Mr, const Tensor &Qk) {
    index_t a1, a2, b1, i1, b2, a3, b3;

    if (Mr.is_empty()) {
      if (Ml.is_empty()) {
        return Qk;
      } else {
        Ml.get_dimensions(&a1, &b1, &a2, &b2);
        Qk.get_dimensions(&b2, &i1, &b1);
        // Ml(a1,b1,a2,b2) -> Ml([b2,b1],a2,a1)
        Ml = reshape(permute(Ml, 0, 3), b2 * b1, a2, a1);
        // Qk(b2,i1,b1) -> Pk([b2,b1],i1)
        Tensor Pk = reshape(permute(Qk, 1, 2), b2 * b1, i1);
        // Pk(a2,i1,a1) = Ml([b2,b1],a2,a1) Pk([b2,b1],i1)
        return permute(reshape(fold(Ml, 0, Pk, 0), a2, a1, i1), 1, 2);
      }
    } else if (Ml.is_empty()) {
      Mr.get_dimensions(&a1, &b1, &a2, &b2);
      Qk.get_dimensions(&b2, &i1, &b1);
      // Mr(a1,b1,a2,b2) -> Mr([b2,b1],a2,a1)
      Mr = reshape(permute(Mr, 0, 3), b2 * b1, a2, a1);
      // Qk(b2,i1,b1) -> Pk([b2,b1],i1)
      Tensor Pk = reshape(permute(Qk, 1, 2), b2 * b1, i1);
      // Pk(a2,i1,a1) = Mr([b2,b1],a2,a1) Pk([b2,b1],i1)
      return permute(reshape(fold(Mr, 0, Pk, 0), a2, a1, i1), 1, 2);
    } else {
      Ml.get_dimensions(&a1, &b1, &a2, &b2);
      Qk.get_dimensions(&b2, &i1, &b3);
      Mr.get_dimensions(&a3, &b3, &a1, &b1);
      // Qk(b2,i1,b3) Ml(a1,b1,a2,b2) -> Pk(i1,[b3,a1,b1],a2)
      Tensor Pk = reshape(fold(Qk, 0, Ml, 3), i1, b3 * a1 * b1, a2);
      // Mr(a1,b1,a3,b3) -> Mr([a1,b1,b3],a3)
      Mr = reshape(permute(permute(Mr, 2, 3), 0, 2), b3 * a1 * b1, a3);
      // Pk(i1,[b3,a1,b1],a2) Mr([a1,b1,b3],a3) -> Pk(a2,i1,a3)
      return permute(reshape(fold(Pk, 1, Mr, 0), i1, a2, a3), 0, 1);
    }
  }

  void dump_state(const MPS &P) {
#if 0
      for (int i = 0; i < P.size(); i++)
        std::cerr << "dP[" << i << "]=" << P[i].dimensions() << '\n';
#endif
  }

  const Tensor next_projector(index_t site) {
    Tensor output;
    for (int i = 0; i < nvectors; i++) {
      Tensor new_Pk =
          weights[i] *
          next_projector(matrix(site - 1, i), matrix(site + 1, i), Q[i][site]);
      if (i)
        output = output + new_Pk;
      else
        output = new_Pk;
    }
    return output;
  }

  /*
     * This routine takes a state Q with large dimensionality and obtains another
     * matrix product state P which is smaller.
     *
     * Input:	Q = MPS of large dimension
     *		P = MPS of smaller dimensions (initial guess)
     *		sense = +1/-1 depending on wheter we move to the right or left
     *
     * Output:	P = Most accurate approximation to Q
     *		sense = +1/-1 depending on the orthonormality of P
     *		double : the error |P-Q|^2
     *
     * 1) If sense = +1 initially, we move from left to right, and the states Q and
     *    P are assumed to be orthogonal on the right.
     *
     * 2) If the last iteration was from left to right, then sense = -1, meaning that
     *    the state is orthogonal on the left.
     *
     * 3) The value of SENSE can be passed further to apply_unitary(), simplify(), etc.
     */
  double simplify(MPS &P, int *sense, index_t sweeps, bool normalize) {
    bool debug = mps::FLAGS.get(MPS_DEBUG_SIMPLIFY);
    double tolerance = FLAGS.get(MPS_SIMPLIFY_TOLERANCE);

    int aux_sense = 1;
    if (!sense) {
      sense = &aux_sense;
    }
    if (sweeps < 1) sweeps = 1;

    initialize_matrices(P, *sense);

    if (debug) {
      std::cerr << "----------\nSimplifying " << Q.size()
                << " states with norm " << normQ2 << '\n';
    }
    double err = normQ2 * 10, scp, normP2;
    for (index_t sweep = 0; sweep < sweeps; sweep++) {
      Tensor Pk;
      if (*sense > 0) {
        for (index_t k = 0; k < L; k++) {
          Pk = next_projector(k);
          set_canonical(P, k, Pk, +1, false);
          Pk = P.at(k);
          update_matrices(k, Pk, +1);
        }
        scp = real(scalar_product(L - 1));
        normP2 = real(scprod(Pk, Pk));
      } else {
        for (index_t k = L; k--;) {
          Pk = next_projector(k);
          set_canonical(P, k, Pk, -1, false);
          Pk = P.at(k);
          update_matrices(k, Pk, -1);
        }
        scp = real(scalar_product(0));
        normP2 = real(scprod(Pk, Pk));
      }
      double olderr = err;
      err = tensor::abs(normQ2 + normP2 - 2 * scp);
      if (debug) {
        std::cerr << "error = " << err << ",\tnorm2(P)=" << normP2 << '\n';
      }
      if (tensor::abs(olderr - err) < 1e-5 * tensor::abs(normQ2) ||
          (err < tolerance * normQ2) || (err < 1e-14)) {
        if (normalize) {
          index_t ndx = (*sense > 0) ? L - 1 : 0;
          P.at(ndx) = Pk / sqrt(normP2);
          *sense = -*sense;
        }
        break;
      }
      *sense = -*sense;
    }
    return err;
  }

  const Tensor next_projector_2_sites(index_t site) {
    Tensor output;
    index_t a1, i1 = 0, i2 = 0, a2;
    for (int i = 0; i < nvectors; i++) {
      Tensor P = fold(Q[i][site], -1, Q[i][site + 1], 0);
      P.get_dimensions(&a1, &i1, &i2, &a2);

      Tensor new_Pk =
          weights[i] * next_projector(matrix(site - 1, i), matrix(site + 2, i),
                                      reshape(P, a1, i1 * i2, a2));
      if (i)
        output = output + new_Pk;
      else
        output = new_Pk;
    }
    return reshape(output, output.dimension(0), i1, i2, output.dimension(2));
  }

  /*
     * This routine takes a state Q with large dimensionality and obtains another
     * matrix product state P which is smaller.
     *
     * Input:	Q = MPS of large dimension
     *		P = MPS of smaller dimensions (initial guess)
     *		sense = +1/-1 depending on wheter we move to the right or left
     *
     * Output:	P = Most accurate approximation to Q
     *		sense = +1/-1 depending on the orthonormality of P
     *		double : the error |P-Q|^2
     *
     * 1) If sense = +1 initially, we move from left to right, and the states Q and
     *    P are assumed to be orthogonal on the right.
     *
     * 2) If the last iteration was from left to right, then sense = -1, meaning that
     *    the state is orthogonal on the left.
     *
     * 3) The value of SENSE can be passed further to apply_unitary(), simplify(), etc.
     */
  double simplify_2_sites(MPS &P, index_t Dmax, double tol, int *sense,
                          index_t sweeps, bool normalize) {
    bool debug = mps::FLAGS.get(MPS_DEBUG_SIMPLIFY);
    double tolerance = FLAGS.get(MPS_SIMPLIFY_TOLERANCE);

    int aux_sense = 1;
    if (!sense) {
      sense = &aux_sense;
    }
    if (sweeps < 1) sweeps = 1;

    initialize_matrices(P, *sense);

    if (debug) {
      std::cerr << "----------\nSimplifying " << Q.size()
                << " states with norm " << normQ2 << '\n';
    }

    double err = normQ2 * 10, scp, normP2;
    for (index_t sweep = 0; sweep < sweeps; sweep++) {
      Tensor Pk;
      if (*sense > 0) {
        for (index_t k = 0; k < (L - 1); k++) {
          Pk = next_projector_2_sites(k);
          set_canonical_2_sites(P, Pk, k, +1, Dmax, tol);
          update_matrices(k, P[k], +1);
        }
        update_matrices(L - 1, Pk = P[L - 1], +1);
        scp = real(scalar_product(L - 1));
        normP2 = real(scprod(Pk, Pk));
      } else {
        for (index_t k = L - 1; k > 0; k--) {
          Pk = next_projector_2_sites(k - 1);
          set_canonical_2_sites(P, Pk, k, -1, Dmax, tol);
          update_matrices(k, P[k], -1);
        }
        update_matrices(0, Pk = P[0], -1);
        scp = real(scalar_product(0));
        normP2 = real(scprod(Pk, Pk));
      }
      double olderr = err;
      err = tensor::abs(normQ2 + normP2 - 2 * scp);
      if (debug) {
        std::cerr << "error = " << err << ",\tnorm2(P)=" << normP2 << '\n';
      }
      if (tensor::abs(olderr - err) < 1e-5 * tensor::abs(normQ2) ||
          (err < tolerance * normQ2) || (err < 1e-14)) {
        if (normalize) {
          index_t ndx = (*sense > 0) ? L - 1 : 0;
          P.at(ndx) = Pk / sqrt(normP2);
          *sense = -*sense;
        }
        break;
      }
      *sense = -*sense;
    }
    return err;
  }
};

}  // namespace mps
