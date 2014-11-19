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
	iTEBD/slow_expected/expected_d.cc \
	iTEBD/slow_expected/expected_z.cc \
	iTEBD/slow_expected/expected2_d.cc \
	iTEBD/slow_expected/expected2_z.cc \
	iTEBD/slow_expected/string_order_d.cc \
	iTEBD/slow_expected/string_order_z.cc \
	iTEBD/slow_expected/expected12_d.cc \
	iTEBD/slow_expected/expected12_z.cc
itebd_expected = \
	iTEBD/expected/expected1_d.cc \
	iTEBD/expected/expected1_z.cc \
	iTEBD/expected/expected2_d.cc \
	iTEBD/expected/expected2_z.cc \
	iTEBD/expected/string_order_d.cc \
	iTEBD/expected/string_order_z.cc \
	iTEBD/expected/string_order_many_d.cc \
	iTEBD/expected/string_order_many_z.cc \
	iTEBD/expected/expected12_d.cc \
	iTEBD/expected/expected12_z.cc

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
	tools/bond_dimension_d.cc \
	tools/bond_dimension_z.cc \
	tools/debug.cc \
	tools/entropy.cc \
	tools/flags.cc \
	tools/truncate.cc \
	tools/limited_svd_d.cc \
	tools/limited_svd_z.cc \
	tools/split_tensor_d.cc \
	tools/split_tensor_z.cc \
	tools/build_E_matrix_d.cc \
	tools/build_E_matrix_z.cc \
	tools/propagate_mps_d.cc \
	tools/propagate_mps_z.cc \
	tools/truncate_rmps.cc \
	tools/truncate_cmps.cc \
	tools/dump_rmps.cc \
	tools/dump_cmps.cc \
	tools/load_rmps.cc \
	tools/load_cmps.cc \
	tools/sweeper.cc \
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
	mps/mps_dimensions_d.cc \
	mps/mps_dimensions_z.cc \
	mps/mps_dty_matrix_d.cc \
	mps/mps_dty_matrix_z.cc \
	mps/mps_expected1_d.cc \
	mps/mps_expected1_dz.cc \
	mps/mps_expected1_z.cc \
	mps/mps_expected2_d.cc \
	mps/mps_expected2_dz.cc \
	mps/mps_expected2_z.cc \
	mps/mps_expected2_all_d.cc \
	mps/mps_expected2_all_z.cc \
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
	mps/apply_local_operator_d.cc \
	mps/apply_local_operator_z.cc \
	mpo/rmpo.cc \
	mpo/rmpo_from_hamiltonian.cc \
	mpo/cmpo.cc \
	mpo/cmpo_from_hamiltonian.cc \
	mpo/mpo_add_term_d.cc \
	mpo/mpo_add_term_z.cc \
	mpo/mpo_apply_d.cc \
	mpo/mpo_apply_z.cc \
	mpo/mpo_expected_d.cc \
	mpo/mpo_expected_z.cc \
	mpo/mpo_dump_d.cc \
	mpo/mpo_dump_z.cc \
	mpo/mpo_adjoint_d.cc \
	mpo/mpo_adjoint_z.cc \
	mpo/mpo_mmult_d.cc \
	mpo/mpo_mmult_z.cc \
	evolve/solver_base.cc \
	evolve/trotter_unitary.cc \
	evolve/solver_trotter2.cc \
	evolve/solver_trotter3.cc \
	evolve/solver_trotter4.cc \
	evolve/arnoldi.cc \
	dmrg/qform_d.cc \
	dmrg/qform_z.cc \
	dmrg/lform_d.cc \
	dmrg/lform_z.cc \
	dmrg/simplify_obc_d.cc \
	dmrg/simplify_obc_z.cc \
	dmrg/rdmrg.cc \
	dmrg/cdmrg.cc \
	dmrg/inverse_d.cc \
	dmrg/inverse_z.cc
