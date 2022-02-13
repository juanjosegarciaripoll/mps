// MPS library
//
// (c) 2004 Juan Jose Garcia Ripoll
//
// TESTING PROGRAM OF REAL AND IMAGINARY TIME EVOLUTION
//

#include <linalg.h>
#include <mps.h>
#include <arpack.h>
#include <qm.h>
#ifdef PARALLEL
#include <pminimizer.h>
using namespace parallel;
#else
#include <minimizer.h>
#endif

//======================================================================

#if 1
#define CMinimizer RMinimizer
#define CMPS RMPS
#endif

#ifdef PARALLEL
#define HAVE_TTY Context::have_tty()
#define COUT Context::my_cout()
#else
#define HAVE_TTY 1
#define COUT std::cerr
#endif

//**********************************************************************
// COMPUTE GAP USING MINIMIZER
//

void test_minimizer_gap() {
  accurate_svd = true;

  //
  // We prepare a simple AKLT Hamiltonian, whose gap we control and
  // which is exactly solvable
  //
  CTensor sx, sy, sz;
  spin_operators(1, sx, sy, sz);
  CTensor H1 = 0 * sz;
  CTensor H12 = to_real(kron(sx, sx) + kron(sy, sy) + kron(sz, sz));

  //
  // These matrices are used to build an antiferromagnetic state which
  // is similar to the ground state.
  //
  RTensor P[2];
  P[0] = RTensor(1, 3, 1);
  P[0].set_to_zero();
  P[1] = P[0];
  P[0].at(0, 0, 0) = 1.0;
  P[1].at(0, 2, 0) = 1.0;

  std::cerr << "Computing energy gaps for s=1\n"
            << "=============================\n";

  // Exact energies for Sz=1 subspace
  double E0[] = {//
                 // AKLT
                 //
                 -1.33333333333333, -0.33333333333333, 0.33333333333333,
                 -2.00000000000000, -1.10208826828127, -1.05808955154502,
                 -2.66666666666667, -1.84018705476851, -1.77386736344380,
                 -3.33333333333333, -2.53643086977244, -2.51477951935688,
                 -3.99999999999999, -3.22680947203538, -3.20638672469596,
                 -4.66666666666665, -3.90796840052703, -3.89612051086983,
                 -5.33333333333335, -4.58566773737622, -4.57633012406655,
                 -5.99999999999999, -5.26031105515244, -5.25361436082523,
                 -6.66666666666668, -5.93315262962396, -5.92792387526474,
                 //
                 // Heisenberg
                 //
                 -3.00000000000000, -1.00000000000000, -1.00000000000000,
                 -4.13658152134851, -2.79128784747792, -2.61803398874990,
                 -5.83021252277083, -4.40526186008417, -4.39670975876054,
                 -7.06248941077255, -6.01853339744504, -5.83601252972241,
                 -8.63453198270616, -7.52385106298521, -7.50230573903891,
                 -9.92275854832016, -9.04914106101226, -8.89788012440900,
                 -11.43293164033019, -10.51108838743845, -10.48540537556920,
                 -12.75622919691644, -11.99072610774712, -11.86911757196646,
                 -14.23035896960049, -13.43143361113490, -13.40562630781231};
  size_t d[] = {3, 9, 2};
  RTensor Exact(UIVector(3, d), RVector(3 * 9 * 2, E0));

  for (int model = 0; model < 2; model++) {
    size_t Dmax = 80;
    double err = 0.0;
    std::cerr << (model ? "Heisenberg " : "AKLT       ");
    for (size_t L = 3; L <= 8 /*11*/; L++) {
      // 1) Create the Hamiltonian (translationally invariant, OBC)
      TIHamiltonian H(L, model ? H12 : (H12 + mmult(H12, H12) / 3.0), H1);

      // 2) Create the estimate for the ground state (AF)
      CMPS psi0(L);
      for (size_t i = 0; i < L; i++) {
        psi0.set(i, P[i & 1]);
      }
      CMPS psi1 = psi0.copy();

      // 3) Solve with restrictions
      CMinimizer solver;
      solver.sweeps = 128;
      solver.debug = 0;
      solver.tolerance = 1e-10;
      solver.Q_operators = new_gc RTensor[1];
      solver.Q_operators[0] = to_real(sz);
      solver.Q_values = RTensor(1);
      solver.Q_values.at(0) = 1;

      double E0 = solver.minimize(H, psi0, Dmax);

      // 4) Repeat the procedure, but with an excited state
      solver.orthogonal_to(psi0);
      double E1 = solver.minimize(H, psi1, Dmax);
      double diff =
          abs(Exact(0, L - 3, model) - E0) + abs(Exact(1, L - 3, model) - E1);
      err = max(err, diff);
      if (diff < 3e-9) {
        std::cerr << '.';
      } else {
        std::cerr << '!';
      }
      std::cerr.flush();
    }
    std::cerr << " Max. err. = " << err << '\n';
  }
  std::cerr << '\n';
}

//**********************************************************************
// APPLY MINIMIZER ON N-NEIGH. HAMILTONIANS, NO CONSTRAINTS
//

void test_minimizer(bool twosites, bool block_svd, bool approximate) {
  accurate_svd = block_svd;

  CMinimizer s;
  s.sweeps = 64;
  s.debug = 0;
  s.tolerance = 1e-5;

  //
  // Spin operators for the problems we are solving.
  //
  const double spin = 0.5;

  // This loop is for when we support periodic boundary conditions
  COUT << "Testing minimizer ("
       << (approximate ? "w. truncation" : "no truncation")
       << (block_svd ? ",block svd" : ",norml svd")
       << (twosites ? ",2-site" : ",1-site")
       << "):\n===================================================\n";
  for (int periodic = 0; periodic < 2; periodic++) {
    if (twosites && periodic) return;
    for (unsigned model = 0; model <= TestHamiltonian::last_model(); model++) {
      for (int ti = 1; ti >= 0; ti--) {
        double e, err = 0.0;
        COUT << TestHamiltonian::model_name(model)
             << (periodic ? ",pbc" : ",obc") << (ti ? ",t.i " : ",inh ")
             << std::flush;
        tic();
        for (int nspins = 2; nspins < 8; nspins++) {
          //
          // This value of Dmax guarantees that there are no truncation
          // errors due to the Matrix Product States ansatz.
          //
          size_t Dmax = (unsigned int)pow(2 * spin + 1, nspins / 2);
          //
          // ... but sometimes we want to take a smaller dimension and
          // see how good the algorithm works in the approximate case.
          //
          if (approximate && (Dmax > (2 * spin + 1))) {
            Dmax /= 2;
          }
          CMPS P0(nspins);
          //
          // For each size, we compute up to 10 different problems
          //
          for (int times = 0; times < 10; times++) {
            //
            // Create random initial Hamiltonian and state
            //
            TestHamiltonian H(model, spin, nspins, ti, periodic);
            P0.randomize(H.physical_dimension(), Dmax, periodic);
            P0.orthonormalize();
            //
            // Solve the problem, collecting errors in OUT().
            //
            CTensor Haux = full(H.sparse_matrix(0));
            CTensor realE = eigs(Haux, CArpack::SmallestAlgebraic, 1);
            if (twosites) {
              e = abs(s.minimize(H, P0, Dmax) - re_part(realE[0]));
            } else {
              e = abs(s.minimize(H, P0) - re_part(realE[0]));
            }
            if (e > s.tolerance * 10) {
              COUT << '!' << std::flush;
#if 0
			    COUT << H.expected_value(P0, 0.0) << ' '
				 << P0.norm();
			    myabort();
#endif
            } else {
              COUT << '.' << std::flush;
            }
            err = max(e, err);
          }
        }
        COUT << "\n\t\tMax. Err: " << err << ", Time: " << toc() << '\n';
      }
    }
  }

  // Back to default value
  accurate_svd = 0;
}

int main(int argc, char **argv) {
  if (!mps_init(&argc, &argv)) {
    exit(-1);
  }

  test_minimizer_gap();

  test_minimizer(0, 0, 0);

  test_minimizer(0, 0, 1);

  test_minimizer(1, 0, 0);

  test_minimizer(1, 0, 1);

#ifdef PARALLEL
  mps_close_mp();
#endif
  return 0;
}

/// Local variables:
/// mode: c++
/// fill-column: 80
/// c-basic-offset: 4
