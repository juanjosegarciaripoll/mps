dnl -*- Autoconf -*-
AC_INIT([C++ MPS Library],
	[0.0.0],
	[juanjose.garciaripoll@gmail.com],
	[mps])

#
# Initialize Autoconf. Note that all files in config_aux are created by
# the automatic tools.
#
AC_PREREQ(2.59)
AC_CONFIG_SRCDIR([./README])		dnl Unique file where sources are
AC_CONFIG_MACRO_DIR([m4])		dnl Additional macro directories
AC_CONFIG_AUX_DIR([config_aux])		dnl Where to put autoconf files
AC_CONFIG_HEADERS([config_aux/config.h])	dnl Automatically generated
AC_CONFIG_FILES([Makefile src/Makefile include/Makefile
                 scripts/Makefile test/Makefile www/Makefile])

#
# Initialize Automake, minimal version 1.9, with options to make it less
# pedantic, various distribution targets and placing object files together
# inside source dirs.
#
AM_INIT_AUTOMAKE([1.9 foreign dist-bzip2 dist-zip subdir-objects])

#
# Command line arguments
#
#AC_ARG_ENABLE([rpath],
# [AS_HELP_STRING([--enable-rpath],
#     [hard-code the location of the libraries in the binaries])],
# [enable_rpath=yes],
# [enable_rpath=no],
# [enable_rpath=yes])

# Programs used to build the library
AC_PROG_CC
AC_PROG_CXX
AM_PROG_CC_C_O # needed to build in subdirs
AC_LANG([C++])
AC_PROG_LIBTOOL

CPPFLAGS=`tensor-config --cppflags`
LDFLAGS=`tensor-config --ldflags`
CXXFLAGS=`tensor-config --cxxflags`

# Unit testing with google
MPS_GTEST

# Doxygen support

DX_HTML_FEATURE(ON)
DX_CHM_FEATURE(OFF)
DX_CHI_FEATURE(OFF)
DX_MAN_FEATURE(OFF)
DX_RTF_FEATURE(OFF)
DX_XML_FEATURE(OFF)
DX_PDF_FEATURE(OFF)
DX_PS_FEATURE(OFF)
DX_INIT_DOXYGEN(mps, doxygen.cfg, doc)

# Missing: headers and library checks

AC_OUTPUT
