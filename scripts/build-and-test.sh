#!/bin/bash

function clean () {
    if [ "$do_clean" = yes ]; then
        if [ -d "$builddir" ]; then
            rm -rf "$builddir"
        fi
    fi
}

function configure () {
    if [ "$do_configure" = yes ]; then
        test -d "$builddir" || mkdir "$builddir"
        CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DMPS_OPTIMIZED_BUILD=ON"
        CMAKE_FLAGS="${CMAKE_FLAGS} -DCMAKE_EXPORT_COMPILE_COMMANDS=1"
        if [ "$do_sanitize" = yes ]; then
            CMAKE_FLAGS="${CMAKE_FLAGS} -DMPS_ADD_SANITIZERS=ON"
        fi
        if [ "$do_cppcheck" = yes ]; then
            CMAKE_FLAGS="${CMAKE_FLAGS} -DMPS_CPPCHECK=ON"
        fi
        if [ "$do_clang_tidy" = yes ]; then
            CMAKE_FLAGS="${CMAKE_FLAGS} -DMPS_CLANG_TIDY=ON"
        fi
        if [ "$do_coverage" = yes ]; then
            CMAKE_FLAGS="${CMAKE_FLAGS} -DMPS_COVERAGE=ON"
        fi
        if [ "$do_install" = yes ]; then
            if [ -z "$CMAKE_INSTALL_PREFIX" ]; then
                echo Option --install selected but no installation directory supplied
                exit 1
            fi
            set -x
            if [ ! -d "$CMAKE_INSTALL_PREFIX" ]; then
                echo Option --install selected but installation directory missing
                exit 1
            fi
            CMAKE_FLAGS="${CMAKE_FLAGS} -DCMAKE_INSTALL_PREFIX='$CMAKE_INSTALL_PREFIX'"
        fi
        cmake -S"$sourcedir" -B"$builddir" $CMAKE_FLAGS -G "$generator" 2>&1 | tee -a "$logfile"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            echo CMake configuration failed
            exit 1
        fi
    fi
}

function build () {
    if [ "$do_build" = yes ]; then
        cmake --build "$builddir" -j $threads -- 2>&1 | tee -a "$logfile"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            echo CMake build failed
            exit 1
        fi
    fi
}

function docs () {
    if [ "$do_docs" = yes ]; then
        cmake --build "$builddir" --target doxygen -- 2>&1 | tee -a "$logfile"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            echo CMake documentation build failed
            exit 1
        fi
    fi
}

function profile () {
    if [ "$do_profile" = yes ]; then
        "$builddir/profile/profile" "$sourcedir/profile/benchmark_$os.json" 2>&1 | tee -a "$logfile"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            echo CMake profile failed
            exit 1
        fi
    fi
}

function check () {
    if [ "$do_check" = yes ]; then
        cd "$builddir"/tests
        ctest -j $threads --rerun-failed --output-on-failure | tee -a "$logfile"
        if [ "${PIPESTATUS[0]}" -ne 0 ]; then
            echo CMake test failed
            exit -1
        fi
    fi
}

function report_coverage() {
    if [ "$do_coverage" = yes ]; then
        cd "$builddir"
        lcov --capture --directory . --output-file coverage.info
        genhtml coverage.info --output-directory html-coverage
    fi
}

function install_library() {
    if [ "$do_install" = yes ]; then
        cmake --install "$builddir"
    fi
}

os=`uname -o`
if [ -f /etc/os-release ]; then
   os=`(. /etc/os-release; echo $ID)`
fi
threads=10
sourcedir=`pwd`
builddir="$sourcedir/build-$os"
OPENBLAS_NUM_THREADS=4
export OPENBLAS_NUM_THREADS
CMAKE_BUILD_TYPE=Release
if test -n `which ninja`; then
    generator="Ninja"
else
    generator="Unix Makefiles"
fi

do_clean=no
do_configure=no
do_build=no
do_profile=no
do_check=no
do_docs=no
do_sanitize=no
do_analyze=no
do_fftw=yes
do_arpack=yes
do_coverage=false
for arg in $*; do
    case $arg in
        --threads=*) threads=${arg:10};;
        --clean) do_clean=yes;;
        --configure) do_configure=yes;;
        --build) do_build=yes;;
        --profile) do_profile=yes;;
        --no-fftw) do_fftw=no;;
        --no-arpack) do_arpack=no;;
        --test) do_check=yes;;
        --docs) do_docs=yes;;
        --sanitize) do_sanitize=yes;;
        --analyze) do_cppcheck=yes; do_clang_tidy=yes;;
        --cppcheck) do_cppcheck=yes;;
        --clang-tidy) do_clang_tidy=yes;;
        --coverage) do_coverage=yes; CMAKE_BUILD_TYPE=Debug;;
        --all) do_clean=yes; do_configure=yes; do_build=yes; do_profile=yes; do_check=yes;;
        --debug) CMAKE_BUILD_TYPE=Debug;;
        --install=*) do_install=yes
            export CMAKE_INSTALL_PREFIX="${arg:10}"
            echo Installing in "$CMAKE_INSTALL_PREFIX";;
        --release) CMAKE_BUILD_TYPE=Release;;
        --*) echo Unknown option ${arg}; exit 1;;
        *) builddir=$arg;;
    esac
done
logfile="$builddir/log"
echo "Build directory $builddir"
echo "Logging into $logfile"

clean || exit 1
configure || exit 1
build || exit 1
docs || exit 1
check || exit 1
profile || exit 1
report_coverage || exit 1
install_library || exit 1