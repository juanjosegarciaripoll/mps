# We define the global AM_CPPFLAGS as everything we compile includes from these
# directories.
AM_CPPFLAGS = \
	-I$(top_srcdir) -I$(top_srcdir)/include \
	-I$(top_builddir)/include

#
# Main library
#
lib_LTLIBRARIES = libmps.la

itebd_expected_slow = \
	iTEBD/itebd_slow_expected_d.cc \
	iTEBD/itebd_slow_expected_z.cc \
	iTEBD/itebd_slow_expected2_d.cc \
	iTEBD/itebd_slow_expected2_z.cc \
	iTEBD/itebd_slow_string_order_d.cc \
	iTEBD/itebd_slow_string_order_z.cc \
	iTEBD/itebd_slow_expected12_d.cc \
	iTEBD/itebd_slow_expected12_z.cc
itebd_expected_wrong = \
	iTEBD/itebd_expected1_d.cc \
	iTEBD/itebd_expected1_z.cc \
	iTEBD/itebd_expected2_d.cc \
	iTEBD/itebd_expected2_z.cc \
	iTEBD/itebd_string_order_d.cc \
	iTEBD/itebd_string_order_z.cc \
	iTEBD/itebd_expected12_d.cc \
	iTEBD/itebd_expected12_z.cc

libmps_la_SOURCES = \
	quantum/spin.cc \
	quantum/fock.cc \
	quantum/decompose_operator_d.cc \
	quantum/decompose_operator_z.cc \
	quantum/sparse_1d_hamiltonian_d.cc \
	quantum/sparse_1d_hamiltonian_z.cc \
	quantum/sparse_1d_hamiltonian_many_d.cc \
	quantum/sparse_1d_hamiltonian_many_z.cc \
	hamiltonian/hamiltonian.cc \
	hamiltonian/hamiltonian_expected_d.cc \
	hamiltonian/hamiltonian_expected_z.cc \
	hamiltonian/sparse_hamiltonian.cc \
	hamiltonian/hamiltonian_ti.cc \
	hamiltonian/hamiltonian_constant.cc \
	hamiltonian/split_interaction.cc \
	tools/entropy.cc \
	tools/truncate.cc \
	tools/limited_svd_d.cc \
	tools/limited_svd_z.cc \
	tools/build_E_matrix_d.cc \
	tools/build_E_matrix_z.cc \
	tools/propagate_mps_d.cc \
	tools/propagate_mps_z.cc \
	tools/truncate_rmps.cc \
	tools/truncate_cmps.cc \
	iTEBD/itebd_d.cc \
	iTEBD/itebd_z.cc \
	iTEBD/itebd_itime_d.cc \
	iTEBD/itebd_itime_z.cc \
	iTEBD/itebd_ghz_state.cc \
	iTEBD/itebd_cluster_state.cc \
	iTEBD/itebd_aklt_state.cc \
	$(itebd_expected_slow) \
	iTEBD/itebd_energy_d.cc \
	iTEBD/itebd_energy_z.cc \
	mps/rmps.cc \
	mps/rmps_random.cc \
	mps/rmps_product_state.cc \
	mps/rmps_ghz_state.cc \
	mps/rmps_cluster_state.cc \
	mps/rmps_to_vector.cc \
	mps/cmps.cc \
	mps/cmps_random.cc \
	mps/cmps_from_rmps.cc \
	mps/cmps_product_state.cc \
	mps/cmps_to_vector.cc \
	mps/mps_prop_matrix_d.cc \
	mps/mps_prop_matrix_z.cc \
	mps/mps_norm_d.cc \
	mps/mps_norm_z.cc \
	mps/mps_scprod_d.cc \
	mps/mps_scprod_z.cc \
	mps/mps_expected1_d.cc \
	mps/mps_expected1_dz.cc \
	mps/mps_expected1_z.cc \
	mps/mps_expected2_d.cc \
	mps/mps_expected2_dz.cc \
	mps/mps_expected2_z.cc \
	mps/mps_simplify_d.cc \
	mps/mps_simplify_z.cc \
	mps/mps_simplify_many_d.cc \
	mps/mps_simplify_many_z.cc \
	mps/mps_simplify_many2_d.cc \
	mps/mps_simplify_many2_z.cc \
	mps/mps_canonical_d.cc \
	mps/mps_canonical_z.cc \
	mps/mps_canonical2_d.cc \
	mps/mps_canonical2_z.cc \
	mps/mps_normal_d.cc \
	mps/mps_normal_z.cc \
	mps/apply_local_operator_d.cc \
	mps/apply_local_operator_z.cc \
	mps/rmpo.cc \
	mps/rmpo_from_hamiltonian.cc \
	mps/cmpo.cc \
	mps/cmpo_from_hamiltonian.cc \
	mps/mpo_add_term_d.cc \
	mps/mpo_add_term_z.cc \
	mps/mpo_apply_d.cc \
	mps/mpo_apply_z.cc \
	mps/mpo_expected_d.cc \
	mps/mpo_expected_z.cc \
	mps/mpo_dump_d.cc \
	mps/mpo_dump_z.cc \
	evolve/solver_base.cc \
	evolve/trotter_unitary.cc \
	evolve/solver_trotter2.cc \
	evolve/solver_trotter3.cc \
	evolve/solver_trotter4.cc \
	evolve/arnoldi.cc \
	dmrg/rdmrg.cc \
	dmrg/cdmrg.cc
