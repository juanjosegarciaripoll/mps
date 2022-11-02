# TODO: Use FetchContent as in https://sourceforge.net/p/wavepacket/cpp/git/ci/master/tree/dependencies/CMakeLists.txt
# for sites where GTest might not be available, and to fix a certain version.

include(GoogleTest)
enable_testing()

mark_as_advanced(
  BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
  gtest_build_samples gtest_build_tests
  gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
  )

add_library(testmain
  test_main.cc random_unitary_d.cc random_unitary_z.cc random_permutation.cc
  hamilt_test.cc random_product_RMPS.cc random_product_CMPS.cc)
target_link_libraries(testmain
  PUBLIC mps tensor::tensor ${GTEST_COMPONENT})

macro(package_add_test TESTNAME)
    # create an exectuable in which the tests will be stored
    add_executable(${TESTNAME} ${ARGN})
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME} PUBLIC testmain ${GTEST_COMPONENT})
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

package_add_test(test_quantum_spin test_quantum_spin.cc)
package_add_test(test_quantum_fock test_quantum_fock.cc)
package_add_test(test_quantum_lattice test_quantum_lattice.cc)
package_add_test(test_quantum_lattice_bipartition test_quantum_lattice_bipartition.cc)
package_add_test(test_truncation test_truncation.cc)
package_add_test(test_mps_base test_mps_base.cc)
package_add_test(test_mps_norm test_mps_norm.cc)
package_add_test(test_mps_examples test_mps_examples.cc)
package_add_test(test_mps_expected test_mps_expected.cc)
package_add_test(test_mps_correlation test_mps_correlation.cc)
package_add_test(test_mps_canonical test_mps_canonical.cc)
package_add_test(test_mps_simplify test_mps_simplify.cc)
package_add_test(test_mps_simplify_many test_mps_simplify_many.cc)
package_add_test(test_mps_simplify_many2 test_mps_simplify_many2.cc)
package_add_test(test_mps_lform test_mps_lform.cc)
package_add_test(test_mps_simplify_obc test_mps_simplify_obc.cc)
package_add_test(test_mpo_base test_mpo_base.cc)
package_add_test(test_mpo_matrix test_mpo_matrix.cc)
package_add_test(test_mps_qform test_mps_qform.cc)
package_add_test(test_mps_solve test_mps_solve.cc)
package_add_test(test_itebd_base test_itebd_base.cc)
package_add_test(test_itebd_expected test_itebd_expected.cc)
package_add_test(test_itebd_expected_slow test_itebd_expected_slow.cc)
package_add_test(test_itebd_itime test_itebd_itime.cc)
package_add_test(test_time_solver_trotter2 test_time_solver_trotter2.cc)
package_add_test(test_time_solver_trotter3 test_time_solver_trotter3.cc)
#package_add_test(test_time_solver_arnoldi test_time_solver_arnoldi.cc)
package_add_test(test_mps_minimizer test_mps_minimizer.cc)
#package_add_test(test_dmrg test_dmrg.cc)
