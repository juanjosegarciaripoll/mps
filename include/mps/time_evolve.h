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

#ifndef MPS_TIME_EVOLVE_H
#define MPS_TIME_EVOLVE_H

#include <vector>
#include <mps/hamiltonian.h>
#include <mps/mps.h>
#include <mps/mpo.h>

namespace mps {

  #define MPS_TIME_EVOLVE_TOLERANCE -1.0

  /**General class for time evolution. A Solver is a class that
     approximates time evolution with MPS only for a short time.
  */
  class TimeSolver {
  public:
    /**Create Solver with fixed time step.*/
    TimeSolver(cdouble new_dt) : dt_(new_dt) {};
    virtual ~TimeSolver();

    /**Compute next time step. Given the state \f$\psi(0)\f$ represented
       by P, estimate the state at \f$\psi(\Delta t)\f$ within the space
       of MPS with dimension <= Dmax. P contains the output.*/
    virtual double one_step(CMPS *P, index Dmax) = 0;

    /**How long in time this solver advances.*/
    cdouble time_step() const { return dt_; }

  private:
    const cdouble dt_;
  };


  class TrotterSolver : public TimeSolver {
  public:
    enum {
      TRUNCATE_GROUPS = 0,
      TRUNCATE_EACH_LAYER = 1,
      TRUNCATE_EACH_UNITARY = 2,
      DO_NOT_TRUNCATE = 3
    } strategy;

    int sweeps;
    bool normalize;
    int sense;

    TrotterSolver(cdouble new_dt) :
      TimeSolver(new_dt),
      strategy(TRUNCATE_EACH_LAYER),
      sweeps(8),
      normalize(true)
    {};

    virtual ~TrotterSolver();

  protected:
    /*Unitary arising from a Trotter decomposition.

      The unitary arises from a Trotter decomposition and contains thus
      only nearest neighbor and local operators. This operator is used
      internally by Trotter2Solver, Trotter3Solver, ForestRuthSolver, but
      unless you want to implement your own algorithms, it is of little
      use.
    */
    struct Unitary {
      bool debug;

      /*Construct the unitary operator.*/
      Unitary(const Hamiltonian &H, index k, cdouble dt,
	      bool apply_pairwise = true, bool do_debug = false);

      /*Apply the unitary on a MPS.*/
      double apply(CMPS *psi, int *dk, double tolerance, index Dmax,
                   bool normalize = false) const;

      /*Apply the unitary on a MPS and optimize the output.*/
      double apply_and_simplify(CMPS *psi, int *dk, double tolerance, index Dmax,
				bool normalize = false) const;

    private:
      bool pairwise;
      int k0, kN;
      std::vector<CTensor> U;
      void apply_onto_one_site(CMPS &P, const CTensor &Uloc, index k, int dk) const;
      double apply_onto_two_sites(CMPS &P, const CTensor &U12,
				  index k1, index k2, int dk,
				  double tolerance, index max_a2) const;
    };
  };

  /**Trotter method with only two passes. This solver uses the second order Trotter approximation:
     \f[
     exp(-iH\Delta t) = exp(-iH_{even} \Delta t/2) exp(-iH_{odd} \Delta t/2)\f]
  */
  class Trotter2Solver : public TrotterSolver {
    Unitary U;
    bool optimize;
    int sense;
  public:
    /**Create a solver for the given nearest neighbor Hamiltonian and time step.*/
    Trotter2Solver(const Hamiltonian &H, cdouble dt);
    
    virtual double one_step(CMPS *P, index Dmax);
  };

  /**Trotter method with three passes. This method uses the second order
     formula (slightly more accurate than Trotter2Solver):

     \f[exp(-iH\Delta t) =
     exp(-iH_{even} \Delta t/2)
     exp(-iH_{odd} \Delta t)
     exp(-iH_{even} \Delta t/2)\f]
  */
  class Trotter3Solver : public TrotterSolver {
    Unitary U1, U2;
    int sense;
  public:
    /**Create a solver for the given nearest neighbor Hamiltonian and time step.*/
    Trotter3Solver(const Hamiltonian &H, cdouble dt);

    virtual double one_step(CMPS *P, index Dmax);
  };

  /**Forest-Ruth method. This method uses a fourth order Forest-Ruth decomposition
     (see http://xxx.arxiv.org/cond-mat/0610210)*/
  class ForestRuthSolver : public TrotterSolver {
    Unitary U1, U2, U3, U4;
    int sense;
  public:
    int sweeps;
    bool normalize;
    /**Create a solver for the given nearest neighbor Hamiltonian and time step.*/
    ForestRuthSolver(const Hamiltonian &H, cdouble dt);
    
    virtual double one_step(CMPS *P, index Dmax);
  };

  /**Time evolution with the Arnoldi method.
  */
  class ArnoldiSolver : public TimeSolver {
  public:
    /**Create Solver with fixed time step.*/
    ArnoldiSolver(const Hamiltonian &H, cdouble dt, int nvectors);

    /**Create Solver with fixed time step.*/
    ArnoldiSolver(const CMPO &H, cdouble dt, int nvectors);

    /**Compute next time step. Given the state \f$\psi(0)\f$ represented
       by P, estimate the state at \f$\psi(\Delta t)\f$ within the space
       of MPS with dimension <= Dmax. P contains the output.*/
    virtual double one_step(CMPS *P, index Dmax);

  private:
    const cdouble dt_;
    const CMPO H_;
    const int max_states_;
    const double tolerance_;
  };



} // namespace mps

#endif // MPS_TIME_EVOLVE_H
