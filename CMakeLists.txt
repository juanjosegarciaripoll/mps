cmake_minimum_required(VERSION 3.18)

# If not set by the user, make a release build.
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# Source files are named relative but transformed into absolute paths
if(POLICY CMP0076)
  cmake_policy(SET CMP0076 NEW)
endif()
if(POLICY CMP0111)
  cmake_policy(SET CMP0111 OLD)
endif()

# set the project name and version
project(mps
  VERSION 0.2.0
  DESCRIPTION "C++ library for tensor network states"
  HOMEPAGE_URL https://github.com/juanjosegarciaripoll/mps)

# specify the C++ standard.
# We have to build Tensor with the same version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Used modules
include(FetchContent)
include(GNUInstallDirs)

# Find our extra modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(MPSOptions)

############################################################
# SYTEM CHECKS AND CONFIGURATION
#
# Configuration options
option(MPS_TEST "Enable support for tests with GoogleTest" ON)

# Compiler settings and other tooling
make_mps_options()

find_package(tensor REQUIRED)
set(MPS_DEPENDENCIES "find_package(tensor)")

############################################################
# LIBRARY
#

add_library(mps "")
target_link_libraries(mps PRIVATE mps_options)
target_include_directories(mps
  PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  )
target_link_libraries(mps PUBLIC tensor::tensor)
add_subdirectory(include)
add_subdirectory(src)

############################################################
# TESTING
#
if(MPS_TEST)
  find_package(GTest)
  set(GTEST_COMPONENT GTest::gtest)
  if(NOT GTest_FOUND)
    message(STATUS "GTest not found")
	  FetchContent_Declare(
	    googletest
	    URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
	    )
	  # For Windows: Prevent overriding the parent project's compiler/linker settings
	  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    message(STATUS "Downloading googletest")
	  FetchContent_MakeAvailable(googletest)
	  set(GTEST_COMPONENT "gtest_main")
  endif()
  add_subdirectory(test)
endif()

############################################################
# INSTALLATION
#
install(TARGETS mps mps_options
  EXPORT mpsTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

include(CMakePackageConfigHelpers)
message(STATUS "MPS installed dependencies:\n${MPS_DEPENDENCIES}")
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/mpsConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion
  )
install(EXPORT mpsTargets
  NAMESPACE mps::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mps"
  )
configure_file(cmake/mpsConfig.cmake.in mpsConfig.cmake @ONLY)
install(FILES "${PROJECT_BINARY_DIR}/mpsConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/mpsConfig.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mps"
  )

