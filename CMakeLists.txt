# ====================================
# == BBHX project base configuration ==
# ====================================

# ---- CMake related definitions ----
cmake_minimum_required(VERSION 3.23...3.31)

# ---- Main project definition ----
project(bbhx VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

# ---- Find required dependencies ----
find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# Phase 3M (2026-06-04): sprint-wide pybind11 -> nanobind migration.
# nanobind ships its CMake config under `$(python -m nanobind --cmake_dir)`;
# resolve here once so cutils/CMakeLists.txt can call `nanobind_add_module`.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED)

# This line defines the install directory variables
include(GNUInstallDirs)

# ---- Import project-specific CMake functions ----
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---- Define a list of BBHX specific options ----
set_property(GLOBAL PROPERTY BBHX_OPTION_NAMES)

# ---- Define project specific options ----

# BBHX_WITH_GPU controls the activation of both GPU and CPU backend compilations.
# Its allowed values are:
#
# * AUTO: enable CPU backend, and enable GPU backend if CUDA toolchain is found
#   in environment, otherwise disable it
# * ON: enable CPU backend, and enable GPU backend, fail if CUDA toolchain is
#   not available
# * OFF: enable CPU backend and disable GPU backend
# * ONLY: disable CPU backend and enable GPU backend (used in plugin wheel build
#   process)
# * BARE: disable both CPU and GPU backends
set(BBHX_WITH_GPU "AUTO" CACHE STRING "Whether to compile GPU backend")
set_property(CACHE BBHX_WITH_GPU PROPERTY STRINGS "AUTO" "ON" "OFF" "ONLY"
                                         "BARE")
set_property(GLOBAL APPEND PROPERTY BBHX_OPTION_NAMES BBHX_WITH_GPU)

# BBHX_CUDA_ARCH will be passed as the CUDA_ARCHITECTURES property for the CUDA
# backend if it is compiled. See the documentation of CUDA_ARCHITECTURES:
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
#
# For cuda13x and upwards cuda_arch <= 70 (V100) is not supported anymore. The
# default value is computed dynamically against the detected CUDAToolkit
# version (the find_package(CUDAToolkit) call lives in src/bbhx/cutils/
# CMakeLists.txt and runs before this default is read by the CUDA target),
# but to keep this option declaration self-contained we list the full default
# here and apply the filter at use-site. Mirrors LISAanalysistools'
# FILTERED_ARCHS pattern.
set(BBHX_CUDA_ARCH "70;75;80;86;90"
    CACHE STRING "CUDA Architecture targetted for BBHX compilation (see doc of \
          CMAKE_CUDA_ARCHITECTURES). On CUDA >= 13 the value 70 is filtered \
          out automatically since V100 is no longer supported by nvcc.")
set_property(GLOBAL APPEND PROPERTY BBHX_OPTION_NAMES BBHX_CUDA_ARCH)

set(BBHX_MARCH "native"
    CACHE STRING "Value of the -march compiler option if supported by compiler")
set_property(GLOBAL APPEND PROPERTY BBHX_OPTION_NAMES BBHX_MARCH)

# GBT_LAPACKE_DETECT_WITH sets the tool used to try to detect LAPACKE locally.
# The GBT_ prefix is deliberate: detection is delegated to GBT's
# get_lapacke() (src/bbhx/cutils/CMakeLists.txt includes GBT's
# cmake_functions.cmake), so the same pip
# --config-settings=cmake.define.GBT_LAPACKE_*=... flags work across GBT,
# GBGPU and BBHx. Possible values are:
#
# * AUTO: attempts PKGCONFIG and, in case of failure, CMAKE
# * CMAKE: use CMake find_package() mechanism (try to find
#   'lapacke-config.cmake' in CMAKE_PREFIX_PATH or in standard system locations)
# * PKGCONFIG: use pkgconfig (which must be installed), try to find 'lapacke.pc'
#   and its dependencies in standard system locations or in PKG_CONFIG_PATH
# * DISABLE (or any other value): disable detection of local lapacke
#   installation
set(GBT_LAPACKE_DETECT_WITH "AUTO" CACHE STRING "Tool used to locate LAPACKE.")
set_property(CACHE GBT_LAPACKE_DETECT_WITH PROPERTY STRINGS "AUTO" "CMAKE"
                                                    "PKGCONFIG" "DISABLE")
set_property(GLOBAL APPEND PROPERTY BBHX_OPTION_NAMES GBT_LAPACKE_DETECT_WITH)

# GBT_LAPACKE_FETCH controls whether LAPACKE sources will be downloaded and
# compiled with BBHX backends. Possible values are:
#
# * AUTO: will try to locate LAPACKE locally and, in case of failure, will
#   enable the fetch mechanism
# * ON: will always fetch LAPACKE sources and compile them, even if already
#   present locally
# * OFF: will disable LAPACKE fetching mechanism
set(GBT_LAPACKE_FETCH "AUTO" CACHE STRING
                                   "Whether to download and compile LAPACK(E)")
set_property(CACHE GBT_LAPACKE_FETCH PROPERTY STRINGS "AUTO" "ON" "OFF")
set_property(GLOBAL APPEND PROPERTY BBHX_OPTION_NAMES GBT_LAPACKE_FETCH)

# GBT_LAPACKE_EXTRA_LIBS sets extra libraries that must be linked when linking
# LAPACKE. For example, if importing the CPU backend fails and complains about
# missing "gfortran" symbols, set GBT_LAPACKE_EXTRA_LIBS=gfortran. Note that if
# this option is unset, it will be automatically set to "gfortran" if that
# library in linkable.
include(CheckLinkerFlag)
check_linker_flag(CXX "-lgfortran" GFORTRAN_AVAILABLE)
if(GFORTRAN_AVAILABLE)
  set(DEFAULT_EXTRA_LIBS "gfortran")
else()
  set(DEFAULT_EXTRA_LIBS "")
endif()
set(GBT_LAPACKE_EXTRA_LIBS "${DEFAULT_EXTRA_LIBS}"
    CACHE STRING "Extra libs to link to when\
linking against LAPACKE.")
set_property(GLOBAL APPEND PROPERTY BBHX_OPTION_NAMES GBT_LAPACKE_EXTRA_LIBS)
unset(GFORTRAN_AVAILABLE)
unset(DEFAULT_EXTRA_LIBS)

# ---- Phony target for project specific properties ----
add_library(bbhx INTERFACE)

# ---- Enable building the CPU version of backends by default ----
set_target_properties(bbhx PROPERTIES WITH_CPU ON)

# ---- Test whether the BBHX_MARCH option is supported by CXX compiler ----
include(CheckCXXCompilerFlag)
set(BBHX_MARCH_CXX_OPT "-march=${BBHX_MARCH}")
check_cxx_compiler_flag("${BBHX_MARCH_CXX_OPT}" CXX_COMPILER_SUPPORTS_BBHX_MARCH)
if(CXX_COMPILER_SUPPORTS_BBHX_MARCH)
  set_property(TARGET bbhx PROPERTY CXX_MARCH
                                                 "${BBHX_MARCH_CXX_OPT}")
  message(STATUS "The CXX compiler supports option '${BBHX_MARCH_CXX_OPT}'.")
else()
  message(
    WARNING "The CXX compiler does not support option '${BBHX_MARCH_CXX_OPT}'. \
      It will be ignored.")
endif()

# ---- Optionnally check if GPU is supported ----
if(BBHX_WITH_GPU STREQUAL "AUTO")
  if(DEFINED ENV{READTHEDOCS})
    message(
      STATUS
        "BBHX GPU backend generation is disabled in ReadTheDocs build context.")
    set(BBHX_WITH_GPU "OFF")
  endif()
endif()
if(BBHX_WITH_GPU STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit)
  endif()
  if(CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
    message(
      STATUS
        "BBHX GPU support was set to AUTO and will be turned ON as CUDA and \
        CUDA Toolkit are available with CUDA version \
        ${CUDAToolkit_VERSION_MAJOR}.")
    set_target_properties(bbhx PROPERTIES WITH_GPU ON)
  else()
    message(
      STATUS
        "BBHX GPU support was set to AUTO and will be turned OFF as CUDA and \
        CUDA Toolkit are not found (CMAKE_CUDA_COMPILER:${CMAKE_CUDA_COMPILER} \
        CUDAToolkit_FOUND:${CUDAToolkit_FOUND}).")
    set_target_properties(bbhx PROPERTIES WITH_GPU OFF)
  endif()
elseif(BBHX_WITH_GPU STREQUAL "ONLY")
  message(
    STATUS
      "BBHX GPU support is set to ON and CPU support is set to OFF (use only to \
      build a standalone plugin).")
  set_target_properties(bbhx PROPERTIES WITH_GPU ON)
  set_target_properties(bbhx PROPERTIES WITH_CPU OFF)
elseif(BBHX_WITH_GPU STREQUAL "BARE")
  message(STATUS "BBHX GPU and CPU support are disabled (use only to build a \
      non-functional pure-python release).")
  set_target_properties(bbhx PROPERTIES WITH_GPU OFF)
  set_target_properties(bbhx PROPERTIES WITH_CPU OFF)
else()
  message(STATUS "BBHX GPU support is set to ${BBHX_WITH_GPU}.")
  set_target_properties(bbhx PROPERTIES WITH_GPU ${BBHX_WITH_GPU})
endif()

# ---- Handle GPU support ----
get_target_property(_BBHX_WITH_GPU bbhx WITH_GPU)
if(_BBHX_WITH_GPU)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # For cuda13x and upwards cuda_arch <= 70 (V100) is not supported anymore.
  # The code below takes that into consideration, while still being agnostic
  # to the cuda version usage. Mirrors the LISAanalysistools FILTERED_ARCHS
  # pattern (LAT CMakeLists.txt:131-150). If a user passed an explicit
  # -DBBHX_CUDA_ARCH=... on the CMake command line, that cache value was
  # populated before this block runs, and the no-FORCE set() below leaves it
  # untouched (CMake CACHE semantics).
  set(DEFAULT_ARCHS "70;75;80;86;90")
  set(FILTERED_ARCHS "")
  foreach(_arch IN LISTS DEFAULT_ARCHS)
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0" AND _arch LESS_EQUAL 70)
      message(STATUS "CUDA Toolkit ${CUDAToolkit_VERSION} detected: Dropping unsupported architecture '${_arch}'")
    else()
      list(APPEND FILTERED_ARCHS "${_arch}")
    endif()
  endforeach()

  # BBHx (unlike LAT) declares BBHX_CUDA_ARCH up top so it can register the
  # var with BBHX_OPTION_NAMES for the build summary. That upfront set()
  # populates the cache to the unfiltered default before find_package
  # runs, so a no-FORCE set() here would be a no-op. Force-overwrite only
  # if the cache still equals that exact unfiltered default -- preserving a
  # user's -DBBHX_CUDA_ARCH=... override (which differs from the default by
  # construction).
  if(BBHX_CUDA_ARCH STREQUAL "${DEFAULT_ARCHS}")
    set(
      BBHX_CUDA_ARCH "${FILTERED_ARCHS}"
      CACHE STRING "CUDA Architectures targeted for BBHX compilation."
      FORCE
    )
  endif()

  set(CMAKE_CUDA_ARCHITECTURES ${BBHX_CUDA_ARCH})
  message(STATUS "BBHX CUDA Architectures set to: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ---- Include sources ----
add_subdirectory(src)

# ---- Let CMake handle copying CITATION.cff into wheel ----
if(SKBUILD_STATE STREQUAL "sdist" OR SKBUILD_STATE STREQUAL "wheel")
  if(_BBHX_WITH_CPU)
    # A bare install does not need to embed CITATION.cff
    install(FILES CITATION.cff DESTINATION bbhx)
  endif()
endif()

# ---- Print the list of options used for this build ----
get_property(ALL_BBHX_OPTION_NAMES GLOBAL PROPERTY BBHX_OPTION_NAMES)

# We assume here that:
#
# 1. The list of options is not empty (at least one BBHX option was registered)
# 2. All registered options exist as cache variables (since they are all defined
#    in this same file before being registered)
# 3. All cache variables have values (CMake cache variables always have a value,
#    even if it's an empty string)
message(STATUS "BBHX options used for this build:")
foreach(bbhx_option_name ${ALL_BBHX_OPTION_NAMES})
  get_property(BBHX_OPTION_VALUE CACHE ${bbhx_option_name} PROPERTY VALUE)
  message(STATUS "  ${bbhx_option_name}: ${BBHX_OPTION_VALUE}")
endforeach()
