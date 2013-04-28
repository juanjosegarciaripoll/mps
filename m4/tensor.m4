dnl -*- Autoconf -*-
dnl ----------------------------------------------------------------------
dnl Find the Google Test library
dnl
AC_DEFUN([MPS_GTEST],[
here=`pwd`
GTEST_URL=http://googletest.googlecode.com/files/gtest-1.6.0.zip
GTEST_NAME=`basename $GTEST_URL .zip`
GTEST_TMP="${here}/test/tmp.zip"
test -d "${here}/test" || mkdir "${here}/test"
AC_MSG_CHECKING([for googletest-read-only in ${ac_confdir}/test])
if test -d "${ac_confdir}/test/googletest-read-only"; then
  AC_MSG_RESULT([yes])
  GTEST_DIR="${ac_confdir}/test/googletest-read-only"
else
  AC_MSG_RESULT([no])
fi
if test "X$GTEST_DIR" = X; then
  AC_MSG_CHECKING([for $GTEST_NAME in ${ac_confdir}/test])	
  if test -d "${ac_confdir}/test/$GTEST_NAME" ; then
    AC_MSG_RESULT([yes])
    GTEST_DIR="${ac_confdir}/test/$GTEST_NAME"
  else
    AC_MSG_RESULT([no])
  fi
fi
if test "X$GTEST_DIR" = X ; then
  AC_MSG_CHECKING([for $GTEST_NAME in build directory]);
  if test -d "${here}/test/$GTEST_NAME" ; then
    AC_MSG_RESULT([yes])
    GTEST_DIR="${here}/test/$GTEST_NAME"
  fi
  if test -d "${here}/$GTEST_NAME" ; then
    GTEST_DIR="${here}/$GTEST_NAME"
  fi
  if test "X$GTEST_DIR" = X ; then
    AC_MSG_RESULT([yes])
  else
    AC_MSG_RESULT([no])
  fi
fi
if test "X$GTEST_DIR" = X ; then
  AC_MSG_CHECKING([trying to download Google Test library])
  if (which unzip && which curl && \
      curl $GTEST_URL > "${GTEST_TMP}" 2>/dev/null && \
      unzip -x "${GTEST_TMP}" -d "${here}/test/" && \
      rm "${GTEST_TMP}") >&AS_MESSAGE_LOG_FD; then
    AC_MSG_RESULT([done])
    GTEST_DIR="${here}/test/$GTEST_NAME"
  else
    if (which unzip && which wget && \
        wget --output-document="${GTEST_TMP}" $GTEST_URL >/dev/null 2>&1 && \
        unzip -x "${GTEST_TMP}" -d "${here}/test/" && \
        rm "${GTEST_TMP}") >&AS_MESSAGE_LOG_FD; then
      AC_MSG_RESULT([done])
      GTEST_DIR="${here}/test/$GTEST_NAME"
    else
      AC_MSG_RESULT([failed])
      GTEST_DIR=""
      AC_MSG_WARN([For testing, please download and unpack google test library]
	          [ $GTEST_URL ]
	          [in ${ac_confdir}/test/]
		  [before configuring tensor])
    fi
  fi
fi
AM_CONDITIONAL([HAVE_GTEST], [test "x${GTEST_DIR}" != x])
AC_SUBST(GTEST_DIR)
])

