# We define the global AM_CPPFLAGS as everything we compile includes from these
# directories.
AM_CPPFLAGS = -I$(top_srcdir) -I$(top_srcdir)/include -I$(top_srcdir)/src \
	-I$(top_builddir)/include -I"@GTEST_DIR@/include" -I"@GTEST_DIR@" \
	-DGTEST_HAS_TR1_TUPLE=0

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_LDFLAGS = libgtest_main.a $(LDFLAGS) -lpthread

#
# No rules here yet
#
TESTS =
check_PROGRAMS =
noinst_LIBRARIES =

AM_TESTS_FD_REDIRECT = --gtest_death_test_style=threadsafe

#####################################################################
# MPS LIBRARY TESTS
#

TESTS += test_quantum_spin
check_PROGRAMS += test_quantum_spin
test_quantum_spin_SOURCES = test_quantum_spin.cc
test_quantum_spin_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_quantum_fock
check_PROGRAMS += test_quantum_fock
test_quantum_fock_SOURCES = test_quantum_fock.cc
test_quantum_fock_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_mps_mps
check_PROGRAMS += test_mps_mps
test_mps_mps_SOURCES = test_mps_mps.cc
test_mps_mps_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_mps_norm
check_PROGRAMS += test_mps_norm
test_mps_norm_SOURCES = test_mps_norm.cc
test_mps_norm_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_mps_expected
check_PROGRAMS += test_mps_expected
test_mps_expected_SOURCES = test_mps_expected.cc
test_mps_expected_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_mps_correlation
check_PROGRAMS += test_mps_correlation
test_mps_correlation_SOURCES = test_mps_correlation.cc
test_mps_correlation_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_mps_canonical
check_PROGRAMS += test_mps_canonical
test_mps_canonical_SOURCES = test_mps_canonical.cc
test_mps_canonical_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_mps_simplify
check_PROGRAMS += test_mps_simplify
test_mps_simplify_SOURCES = test_mps_simplify.cc
test_mps_simplify_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_mps_simplify_many
check_PROGRAMS += test_mps_simplify_many
test_mps_simplify_many_SOURCES = test_mps_simplify_many.cc
test_mps_simplify_many_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_mps_simplify_many2
check_PROGRAMS += test_mps_simplify_many2
test_mps_simplify_many2_SOURCES = test_mps_simplify_many2.cc
test_mps_simplify_many2_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_mpo_base
check_PROGRAMS += test_mpo_base
test_mpo_base_SOURCES = test_mpo_base.cc
test_mpo_base_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_itebd_base
check_PROGRAMS += test_itebd_base
test_itebd_base_SOURCES = test_itebd_base.cc
test_itebd_base_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_itebd_expected
check_PROGRAMS += test_itebd_expected
test_itebd_expected_SOURCES = test_itebd_expected.cc
test_itebd_expected_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_itebd_expected_slow
check_PROGRAMS += test_itebd_expected_slow
test_itebd_expected_slow_SOURCES = test_itebd_expected_slow.cc
test_itebd_expected_slow_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_itebd_itime
check_PROGRAMS += test_itebd_itime
test_itebd_itime_SOURCES = test_itebd_itime.cc
test_itebd_itime_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_time_solver_trotter2
check_PROGRAMS += test_time_solver_trotter2
test_time_solver_trotter2_SOURCES = test_time_solver_trotter2.cc
test_time_solver_trotter2_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_time_solver_trotter3
check_PROGRAMS += test_time_solver_trotter3
test_time_solver_trotter3_SOURCES = test_time_solver_trotter3.cc
test_time_solver_trotter3_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

TESTS += test_time_solver_arnoldi
check_PROGRAMS += test_time_solver_arnoldi
test_time_solver_arnoldi_SOURCES = test_time_solver_arnoldi.cc
test_time_solver_arnoldi_LDADD = libtestmain.a ../src/libmps.la $(GTEST_LDFLAGS) #-lstdc++

noinst_LIBRARIES += libtestmain.a
libtestmain_a_SOURCES = test_main.cc \
	 random_unitary_d.cc random_unitary_z.cc random_permutation.cc

#
# Build the Google Test library. The way this library has changed: we now have to
# compile it for every project. However, since we have downloaded the library, we
# cannot rely on automake and we have to generate the sources.
#

if HAVE_GTEST
noinst_LIBRARIES += libgtest.a libgtest_main.a
libgtest_a_SOURCES = gtest-all.cc
libgtest_main_a_SOURCES = gtest-all.cc gtest_main.cc
endif

gtest-all.cc gtest_main.cc :
	cp @GTEST_DIR@/src/$@ .

OUTPUT_FLAGS = CXX CXXFLAGS LDFLAGS \
	TENSOR_USE_VECLIB TENSOR_USE_MKL TENSOR_USE_ATLAS \
	host_cpu host_os host_vendor

report.xml: $(check_PROGRAMS) Makefile
	echo '<?xml version="1.0" encoding="UTF-8"?>' > $@
	echo '<testframe>' >> $@
	echo '<config field="hostname" value="'`hostname`'"/>' >> $@
	echo '<config field="date" value="'`date +"%Y-%m-%d-%H:%M"`'"/>' >> $@
	for i in $(OUTPUT_FLAGS); do \
	 k=`sed -e '/[DS]\[.'$${i}'.\]/!d;s/.*="\(.*\)"$$/\1/' ../config.status`; \
	 if [ "x$$k" = "x" ]; then k="0"; fi; \
	 echo "<config field='$$i' value='$${k}'/>" >> $@ ; \
	 echo "$$i <config field='$$i' value='$${k}'/>" ; \
	done
	-shared=`sed '/^enable_shared=/!d;s,enable_shared=\(.*\)$,\1,'`; \
	echo '<config field="SHARED" value="'$$shared'"/>' >> $@
	for i in $(check_PROGRAMS); do \
	 xmlfile=$${i}.xml; \
	 test -f $$xmlfile || GTEST_OUTPUT=xml:$${xmlfile} ./$$i; \
	 test -f $$xmlfile || echo '<globalfailure name="'$$i'"/>' >> $$xmlfile; \
	 sed -e '/<\?xml/d;s/AllTests/'$$i'/g' $${xmlfile} >> $@; \
	done
	echo '</testframe>' >> $@

report.html: report.xml $(top_srcdir)/scripts/report.xsl
	xsltproc $(top_srcdir)/scripts/report.xsl report.xml > $@

report_nfailures: report.xml
	k=0; \
	for i in test*.xml; do \
	  n=`sed -e '/AllTests/!d;s/.*failures="\(.[^"]*\)".*$$/\1/g' $$i`; \
	  if [ "x$$n" = "x" ]; then n=1; fi; k=$$(($$k + $$n)); \
	done; \
	echo $$k > $@

upload_tests: report.xml report_nfailures
	if [ "x$(NAME)" != "x" ]; then \
	  host=$(HOST); \
	  if [ "x$${host}" = "x" ]; then host="master:"; fi; \
	  if [ ! -d $(NAME) ]; then mkdir $(NAME) ; fi;  \
	  cp report.xml report_nfailures test*.xml $(NAME) && \
	  rsync -rauvz $(NAME) $${host}src/tensor_logs/; \
	fi

CLEANFILES = *.xml report_nfailures
