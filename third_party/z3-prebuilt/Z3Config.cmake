# Minimal Z3 CMake package configuration for the official release binaries.
#
# AutoCellGen's CMakeLists locates Z3 with
# `find_package(Z3 CONFIG NO_DEFAULT_PATH)`, but the official Z3 release zip does
# not contain a Z3Config.cmake — that file is only produced by a CMake build from
# source. Building 4.8.11 from source takes 20-40 minutes and can fail on gcc 13,
# so this file is laid over the release binaries to provide the same interface.
#
# Usage: cmake -DZ3_DIR=<repo>/third_party/z3-prebuilt ...
#
# Provides: Z3_FOUND, Z3_VERSION_STRING, Z3_LIBRARIES, Z3_CXX_INCLUDE_DIRS

get_filename_component(_Z3_PREFIX "${CMAKE_CURRENT_LIST_DIR}/z3-4.8.11-x64-glibc-2.31" ABSOLUTE)

set(Z3_VERSION_STRING "4.8.11")
set(Z3_CXX_INCLUDE_DIRS "${_Z3_PREFIX}/include")

find_library(Z3_LIBRARY
  NAMES z3
  PATHS "${_Z3_PREFIX}/bin"
  NO_DEFAULT_PATH
)

if(NOT Z3_LIBRARY)
  set(Z3_FOUND FALSE)
  message(FATAL_ERROR "libz3 not found in ${_Z3_PREFIX}/bin. "
                      "Check that scripts/build_backend.sh downloaded the Z3 release.")
endif()

if(NOT EXISTS "${Z3_CXX_INCLUDE_DIRS}/z3++.h")
  set(Z3_FOUND FALSE)
  message(FATAL_ERROR "z3++.h not found in ${Z3_CXX_INCLUDE_DIRS}.")
endif()

add_library(z3::libz3 SHARED IMPORTED)
set_target_properties(z3::libz3 PROPERTIES
  IMPORTED_LOCATION "${Z3_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${Z3_CXX_INCLUDE_DIRS}"
)

set(Z3_LIBRARIES "${Z3_LIBRARY}")
set(Z3_FOUND TRUE)

message(STATUS "Z3 (prebuilt) ${Z3_VERSION_STRING}: ${Z3_LIBRARY}")
