# =============================
# ==== BACKEND COMPILATION ====
# =============================

# * * * * * * * * * * * * * * * * * * * * *
# * * Helper functions to detect LAPACK * *
# * * * * * * * * * * * * * * * * * * * * *

# bbhx_try_get_lapacke_with_cmake
# ------------------------------
#
# This method will try to locate LAPACKE using CMake find_package() mechanism.
#
# This functions sets the following variables in the parent scope:
#
# * In case of success:
#
#   * LAPACKE_WITH_CMAKE_SUCCESS to ON
#   * LAPACKE_WITH_CMAKE_LIBS to targets to link against lapack
#
# * In case of failure:
#
#   * LAPACKE_WITH_CMAKE_SUCCESS to OFF
function(bbhx_try_get_lapacke_with_cmake)

  message(CHECK_START "Trying with find_package()")
  find_package(LAPACKE)

  if(LAPACKE_FOUND)
    set(LAPACKE_WITH_CMAKE_SUCCESS ON PARENT_SCOPE)
    set(LAPACKE_WITH_CMAKE_LIBS lapacke PARENT_SCOPE)
    message(CHECK_PASS "success! Found LAPACKE ${LAPACKE_VERSION}")
  else()
    set(LAPACKE_WITH_CMAKE_SUCCESS OFF PARENT_SCOPE)
    message(CHECK_FAIL "not found")
  endif()
endfunction()

# bbhx_try_get_lapacke_with_pkgconfig
# ----------------------------------
#
# This method will try to locate LAPACKE using PkgConfig.
#
# This functions sets the following variables in the parent scope:
#
# * In case of success:
#
#   * LAPACKE_WITH_PKGCONFIG_SUCCESS to ON
#   * LAPACKE_WITH_PKGCONFIG_LIBS to targets to link against lapack
#
# * In case of failure:
#
#   * LAPACKE_WITH_PKGCONFIG_SUCCESS to OFF
#   * LAPACKE_WITH_PKGCONFIG_REASON to
#
#     * "MISSING_PKGCONFIG" if PKGCONFIG is not available
#     * "MISSING_LAPACKE" if lapacke.pc is not found
function(bbhx_try_get_lapacke_with_pkgconfig)
  message(CHECK_START "Trying with PkgConfig")
  find_package(PkgConfig)
  if(NOT PkgConfig_FOUND)
    message(CHECK_FAIL "PkgConfig not available")
    set(LAPACKE_WITH_PKGCONFIG_SUCCESS OFF PARENT_SCOPE)
    set(LAPACKE_WITH_PKGCONFIG_REASON "MISSING_PKGCONFIG" PARENT_SCOPE)
    return()
  endif()

  pkg_check_modules(lapacke IMPORTED_TARGET lapacke lapack blas)
  if(NOT lapacke_FOUND)
    message(CHECK_FAIL "not found")
    set(LAPACKE_WITH_PKGCONFIG_SUCCESS OFF PARENT_SCOPE)
    set(LAPACKE_WITH_PKGCONFIG_REASON "MISSING_LAPACKE" PARENT_SCOPE)
    return()
  endif()

  message(CHECK_PASS
          "success! Found LAPACKE ${lapacke_VERSION} in ${lapacke_LIBDIR}")
  set(LAPACKE_WITH_PKGCONFIG_SUCCESS ON PARENT_SCOPE)
  set(LAPACKE_WITH_PKGCONFIG_LIBS PkgConfig::lapacke PARENT_SCOPE)
endfunction()

# bbhx_try_get_lapacke_with_cpm
# ----------------------------
#
# This method will use CPM (CMake Package Manager) to fetch LAPACK sources and
# add them to the build tree with enabled LAPACKE support.
#
# This function sets to following variables in the parent scope:
#
# * LAPACKE_WITH_CPM_SUCCESS to ON
# * LAPACKE_WITH_CPM_LIBS to the link targets
function(bbhx_try_get_lapacke_with_cpm)
  include(CPM)
  message(CHECK_START "Trying with automatic fetching of Reference LAPACK")
  enable_language(Fortran)
  CPMAddPackage(
    NAME lapack
    GITHUB_REPOSITORY Reference-LAPACK/lapack
    GIT_TAG 6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca # v3.12.1
    OPTIONS "LAPACKE"
            "ON"
            "CMAKE_POSITION_INDEPENDENT_CODE"
            "ON"
            "CMAKE_UNITY_BUILD"
            "ON"
            "CMAKE_UNITY_BUILD_BATCH_SIZE"
            64)
  set(LAPACKE_WITH_CPM_SUCCESS ON PARENT_SCOPE)
  set(LAPACKE_WITH_CPM_LIBS lapacke lapack PARENT_SCOPE)
  message(CHECK_PASS "done")
endfunction()

# bbhx_get_lapacke
# ---------------
#
# This method will try to make LAPACKE available using the following strategies:
#
# * PKGCONFIG: use the function "bbhx_try_get_lapacke_with_pkgconfig"
# * CMAKE: use the function "bbhx_try_get_lapacke_with_cmake"
# * FETCH: use the function "bbhx_try_get_lapacke_with_cpm"
#
# The following rules are applied to select the attempted strategies:
#
# * The PKGCONFIG strategy is only attempted if BBHX_LAPACKE_DETECT_WITH is
#   "AUTO" or "PKGCONFIG"
# * The CMAKE strategy is only attempted if BBHX_LAPACKE_DETECT_WITH is "AUTO" or
#   "CMAKE".
# * The FETCH strategy is only attempted if BBHX_LAPACKE_FETCH is "AUTO" or "ON".
# * The CMAKE and PKGCONFIG strategies are forcefully disabled if
#   BBHX_LAPACKE_FETCH is "ON"
#
# If no strategy is attempted, the function raises a warning and does not set
# any variable in parent scope. In this case, it is the user responsibility to
# set BBHX_LAPACKE_LIBS to any relevant value.
#
# If at least one strategy is attempted and if no strategy succeeds, the method
# fails with a fatal error explaining how to help attempted strategies to work.
#
# If one strategy succeeds, the following variables are set in parent scope:
#
# * BBHX_LAPACKE_LIBS: list of libraries to link against to use LAPACKE
# * BBHX_LAPACKE_GET_SUCCESS: ON
# * BBHX_LAPACKE_GET_STRATEGY: strategy that succeeded (PKGCONFIG|CMAKE|FETCH )
function(bbhx_get_lapacke)
  # cmake-lint: disable=R0912,R0915
  message(CHECK_START "Locating LAPACKE")

  # I. Detect enabled strategies
  if(BBHX_LAPACKE_DETECT_WITH STREQUAL "AUTO" OR BBHX_LAPACKE_DETECT_WITH
                                                STREQUAL "PKGCONFIG")
    set(pkgconfig_strategy_enabled ON)
  else()
    set(pkgconfig_strategy_enabled OFF)
  endif()
  if(BBHX_LAPACKE_DETECT_WITH STREQUAL "AUTO" OR BBHX_LAPACKE_DETECT_WITH
                                                STREQUAL "CMAKE")
    set(cmake_strategy_enabled ON)
  else()
    set(cmake_strategy_enabled OFF)
  endif()
  if(BBHX_LAPACKE_FETCH STREQUAL "AUTO" OR BBHX_LAPACKE_FETCH STREQUAL "ON")
    set(fetch_strategy_enabled ON)
  else()
    set(fetch_strategy_enabled OFF)
  endif()
  if(BBHX_LAPACKE_FETCH STREQUAL "ON")
    set(pkgconfig_strategy_enabled OFF)
    set(cmake_strategy_enabled OFF)
  endif()

  if(pkgconfig_strategy_enabled OR cmake_strategy_enabled
     OR fetch_strategy_enabled)
    set(any_strategy_enabled ON)
  else()
    set(any_strategy_enabled OFF)
  endif()

  # II. Apply the PkgConfig strategy if enabled
  if(pkgconfig_strategy_enabled)
    bbhx_try_get_lapacke_with_pkgconfig()
    if(LAPACKE_WITH_PKGCONFIG_SUCCESS)
      set(BBHX_LAPACKE_LIBS "${LAPACKE_WITH_PKGCONFIG_LIBS}" PARENT_SCOPE)
      set(BBHX_LAPACKE_GET_SUCCESS ON PARENT_SCOPE)
      set(BBHX_LAPACKE_GET_STRATEGY "PKGCONFIG" PARENT_SCOPE)
      message(CHECK_PASS "found with pkgconfig")
      return()
    endif()
  endif()

  # III. Apply the CMake strategy if enabled
  if(cmake_strategy_enabled)
    bbhx_try_get_lapacke_with_cmake()
    if(LAPACKE_WITH_CMAKE_SUCCESS)
      set(BBHX_LAPACKE_LIBS "${LAPACKE_WITH_CMAKE_LIBS}" PARENT_SCOPE)
      set(BBHX_LAPACKE_GET_SUCCESS ON PARENT_SCOPE)
      set(BBHX_LAPACKE_GET_STRATEGY "CMAKE" PARENT_SCOPE)
      message(CHECK_PASS "found with find_package()")
      return()
    endif()
  endif()

  # IV. Apply the Fetch strategy if enabled
  if(fetch_strategy_enabled)
    bbhx_try_get_lapacke_with_cpm()
    if(LAPACKE_WITH_CPM_SUCCESS)
      set(BBHX_LAPACKE_LIBS "${LAPACKE_WITH_CPM_LIBS}" PARENT_SCOPE)
      set(BBHX_LAPACKE_GET_SUCCESS ON PARENT_SCOPE)
      set(BBHX_LAPACKE_GET_STRATEGY "FETCH" PARENT_SCOPE)
      message(CHECK_PASS "added to build tree with automatic fetching")
      return()
    endif()
  endif()

  # V. Fail if any strategy was applied
  if(any_strategy_enabled)
    message(CHECK_FAIL "not found")
    if(pkgconfig_strategy_enabled)
      message(WARNING "LAPACKE could not be located with PKGCONFIG.")
      if(LAPACKE_WITH_PKGCONFIG_REASON STREQUAL "MISSING_PKGCONFIG")
        message(
          WARNING "Make sure that pkg-config executable is installed in your \
          environment.\n"
                  "On Ubuntu, it can be installed with:\n"
                  "  $ sudo apt install pkg-config\n"
                  "On mac OS, it can be installed with Homebrew with:\n"
                  "  $ brew install pkgconf\n"
                  "In conda environment, it can be installed with:\n"
                  "  $ conda install pkgconfig")
      elseif(LAPACKE_WITH_PKGCONFIG_REASON STREQUAL "MISSING_LAPACKE")
        message(
          WARNING
            "PkgConfig could not locate the file 'lapacke.pc'.\n"
            "If your LAPACK installation provides it, add its directory to \
            the PKG_CONFIG_PATH environment variable.\n"
            "It is usually located in the library install path, in the \
            'lib/pkgconfig' subdirectory.")
      endif()
    endif()
    if(cmake_strategy_enabled)
      message(
        WARNING
          "LAPACKE could not be located with CMake find_package() mechanism.\n"
          "If your LAPACK installation provides a 'lapacke-config.cmake' file \
          (or similar installed target file), add its path to the \
          CMAKE_PREFIX_PATH environment variable.\n"
          "It is usually located in the library install path, in the \
          'lib/cmake' subdirectory.")
    endif()
    if(fetch_strategy_enabled)
      message(
        WARNING "LAPACKE automatic fetching was enabled but somehow failed.\n"
                "CMake processing should have stop much sooner with a detailed \
                explanation of the failure.\nSee previous error messages.")
    endif()
    message(
      FATAL_ERROR
        "LAPACKE support is required but could not be satisfied.\n"
        "READ CAREFULLY PREVIOUS WARNINGS - \
        THEY SHOULD HELP YOU TO FIX THE ISSUE.")
  endif()

  # VI. Add message if lapacke detection is ignored
  message(CHECK_PASS "ignored \
   (BBHX_LAPACKE_DETECT_WITH=${BBHX_LAPACKE_DETECT_WITH} \
   and BBHX_LAPACKE_FETCH=${BBHX_LAPACKE_FETCH})")
  message(
    WARNING
      "LAPACKE detection strategies were disabled.\n"
      "Manually define BBHX_LAPACKE_LIBS using pip "
      "--config-settings=cmake.define.BBHX_LAPACKE_LIBS=value\n\n"
      "Make sure that the compiler can locate lapacke libraries (usually by "
      "adding their directory to "
      "the LIBRARY_PATH environment variable) and the header 'lapacke.h' "
      "(usually done by adding its directory to the CPATH environment "
      "variable).")
endfunction()

# * * * * * * * * * * * * * * * * * * * * * * * * *
# * * Helper functions to define backend option * *
# * * * * * * * * * * * * * * * * * * * * * * * * *

# In the project root CMakeLists.txt, we defined a "bbhx" interface
# target with properties WITH_CPU and WITH_GPU defining whether the CPU and a
# GPU backend need to be compiled. Let's retrieve these information here:
get_target_property(BBHX_WITH_CPU bbhx WITH_CPU)
get_target_property(BBHX_WITH_GPU bbhx WITH_GPU)

# Locate the gpubackendtools cutils directory so we can find gbt_global.h /
# common helpers that the LISAanalysistools Detector.cu pulls in.
# Mirrors how GBGPU/src/gbgpu/cutils/CMakeLists.txt does it.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "if True:
    import gpubackendtools
    print(gpubackendtools.__file__.split('__init__.py')[0] + 'cutils')"
  OUTPUT_VARIABLE GBT_CUTILS
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Locate the LISAanalysistools cutils directory so we can copy Detector.cu /
# Detector.hpp / global.hpp from the installed package (modern version) rather
# than downloading them from GitHub main.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "if True:
    import lisatools
    print(lisatools.__file__.split('__init__.py')[0] + 'cutils')"
  OUTPUT_VARIABLE LISATOOLS_CUTILS
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Adapter to let inplace editable install work: the compiled backend will be
# placed into the source-tree in the 'src' directory:
if(SKBUILD_STATE STREQUAL "editable")
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${bbhx_BINARY_DIR}/src")
else()
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# apply_cpu_backend_common_options
# --------------------------------
#
# This method applies some common directive to CPU backend targets. It:
#
# * Expects a single "libname" argument
# * Expects the target to be named "bbhx_cpu_${libname}"
# * Defines the LIBRARY_OUTPUT_DIRECTORY property
# * Defines the OUTPUT_NAME property
# * Installs the target in the CPU backend directory
# * Ensures the target includes the NumPy header directory
# * Disable NumPy deprecated API
#
# Usage example: apply_cpu_backend_common_options(pymatmul)
function(apply_cpu_backend_common_options libname)
  set(target_name "bbhx_cpu_${libname}")
  set_property(
    TARGET ${target_name}
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/bbhx_backend_cpu")
  set_property(TARGET ${target_name} PROPERTY OUTPUT_NAME ${libname})

  install(TARGETS ${target_name} DESTINATION bbhx_backend_cpu)

  get_target_property(BBHX_CXX_MARCH_OPT bbhx CXX_MARCH)
  if(BBHX_CXX_MARCH_OPT)
    target_compile_options(${target_name} PRIVATE "${BBHX_CXX_MARCH_OPT}")
  endif()

  target_include_directories(${target_name} PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_compile_definitions(${target_name}
                             PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)
endfunction()

# apply_gpu_backend_common_options
# --------------------------------
#
# This method applies some common directive to GPU backend targets. It:
#
# * Expects a single "libname" argument
# * Expects the target to be named "bbhx_gpu_${libname}"
# * Defines the LIBRARY_OUTPUT_DIRECTORY property
# * Defines the OUTPUT_NAME property
# * Installs the target in the GPU backend directory (e.g.
#   'bbhx_backend_cuda12x')
# * Ensures the target includes the NumPy header directory
# * Disable NumPy deprecated API
# * Ensures the target links against CUDA libraries (cuBLAS, cuSPARSE, ...)
# * Defines the CUDA_ARCHITECTURE property
#
# Usage example: apply_gpu_backend_common_options(pymatmul)
function(apply_gpu_backend_common_options libname)
  set(target_name "bbhx_gpu_${libname}")
  set(backend_name "bbhx_backend_cuda${CUDAToolkit_VERSION_MAJOR}x")
  set_property(
    TARGET ${target_name}
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/${backend_name}")
  set_property(TARGET ${target_name} PROPERTY OUTPUT_NAME ${libname})

  install(TARGETS ${target_name} DESTINATION ${backend_name})

  target_include_directories(${target_name} PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_compile_definitions(${target_name}
                             PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)
  # __CUDA_COMPILATION__ is the sprint-wide CPU/GPU class-name alias toggle
  # (see lat_tdi_on_the_fly.hh / sobbh_tdi_on_the_fly.hh / binding_bbhx.hpp).
  # Without it, .cxx TUs in the GPU build (binding_bbhx.cxx etc.) compile
  # against the CPU-suffixed class names (SOBBHTDIonTheFlyCPU,
  # LISATDIonTheFlyCPU, ...) and the .so fails to load with
  # ``undefined symbol: _ZTV*CPU`` -- the CPU vtables are never emitted in
  # the GPU build because the .cu sources DO get __CUDACC__ from nvcc and
  # alias to *GPU instead. PUBLIC so the define propagates to every TU
  # compiled as part of this target. Mirrors what GBT's canonical
  # apply_gpu_backend_common_options does in cmake_functions.cmake (the
  # local override below was missing this; same gap GBGPU had pre-2026-06-06).
  target_compile_definitions(${target_name} PUBLIC __CUDA_COMPILATION__)
  target_link_libraries(${target_name} PUBLIC CUDA::cudart CUDA::cublas
                                              CUDA::cusparse)
  set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES
                                              ${BBHX_CUDA_ARCH})
endfunction()

# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *

# ----------------
# --- phenomhm ---
# ----------------

# Phase BBHx.pybind.{phenomhm,interp,bbhlikelihood,lisaresponse,bbhwaveformbuild,newhdynlike,gpuonlywaveformbuild}
# (2026-06-03): six Cython modules retired -- their free functions now
# live as methods on BBHxComputationWrap inside the cbbhx pybind11 module.
# The underlying .cu sources (PhenomHMWaveform, Interpolate, Likelihood,
# Response, WaveformBuild, SpecialLikelihood + LAT's Detector) compile
# into one .so per backend wheel, mirroring LAT's
# `lisaanalysistools_{cpu,gpu}_pycppdetector` pattern
# (one binding TU + directly-compiled .cu/.cxx files).

# CPU build: copy .cu sources to .cxx and pull LAT's Detector.{cu,hpp} +
# global.hpp from the installed lisatools package.
if(BBHX_WITH_CPU)
  add_custom_command(
    OUTPUT "PhenomHMWaveform.cxx"
    COMMENT "Copy PhenomHMWaveform.cu to PhenomHMWaveform.cxx (for cbbhx CPU)"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/PhenomHMWaveform.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/PhenomHMWaveform.cxx"
    DEPENDS "PhenomHMWaveform.cu"
    VERBATIM)

  # GPUBackendTools owns the cubic-spline solver (Phase 3.dedup-followup,
  # 2026-06-05): BBHx's local Interpolate.{hh,cu} were retired in favor
  # of GBT's Interpolate.cu directly compiled into cbbhx. Same
  # copy-then-compile pattern as LAT's Detector.cu below.
  add_custom_command(
    OUTPUT "Interpolate.cxx"
    COMMENT "Copy GBT's Interpolate.cu to Interpolate.cxx (for cbbhx CPU)"
    COMMAND ${CMAKE_COMMAND} -E copy "${GBT_CUTILS}/Interpolate.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Interpolate.cxx"
    DEPENDS "${GBT_CUTILS}/Interpolate.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "Likelihood.cxx"
    COMMENT "Copy Likelihood.cu to Likelihood.cxx (for cbbhx CPU)"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Likelihood.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Likelihood.cxx"
    DEPENDS "Likelihood.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "Response.cxx"
    COMMENT "Copy Response.cu to Response.cxx (for cbbhx CPU)"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Response.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Response.cxx"
    DEPENDS "Response.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "WaveformBuild.cxx"
    COMMENT "Copy WaveformBuild.cu to WaveformBuild.cxx (for cbbhx CPU)"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/WaveformBuild.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/WaveformBuild.cxx"
    DEPENDS "WaveformBuild.cu"
    VERBATIM)

  # SpecialLikelihood.cu is GPU only -- its `cmplx trans_complex1 = 0.0;`
  # syntax doesn't compile under the CPU std::complex<double> typedef.
  # Compiled into the GPU cbbhx target below; CPU build skips it entirely
  # and the cbbhx::speciallike method is `#ifdef __CUDACC__`-guarded in
  # binding_bbhx.hpp.

  # LAT's Detector.cu is the source of truth; copy it via the
  # LISATOOLS_CUTILS path detected above (mirrors how lisa-on-gpu does it).
  add_custom_command(
    OUTPUT "Detector.cxx"
    COMMENT "Copy LAT's Detector.cu to Detector.cxx (for cbbhx CPU)"
    COMMAND ${CMAKE_COMMAND} -E copy "${LISATOOLS_CUTILS}/Detector.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Detector.cxx"
    DEPENDS "${LISATOOLS_CUTILS}/Detector.cu"
    VERBATIM)

  # Phase 3L.8 (2026-06-04): SOBBH-specific TDI-on-the-fly bodies
  # carved out of lisa-on-gpu's TDIonTheFly.cu. Copy to .cxx for the
  # host C++ compiler (CPU backend), same pattern as the GB carve-out
  # in GBGPU's CMakeLists.txt at Phase 3L.7c.
  add_custom_command(
    OUTPUT "sobbh_tdi_on_the_fly.cxx"
    COMMENT "Copy sobbh_tdi_on_the_fly.cu to .cxx (for cbbhx CPU)"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/sobbh_tdi_on_the_fly.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/sobbh_tdi_on_the_fly.cxx"
    DEPENDS "sobbh_tdi_on_the_fly.cu"
    VERBATIM)

  # Phase 3L.8 (2026-06-04): SOBBHTDIonTheFly inherits from
  # LISATDIonTheFly. Copy-compile LAT's lat_tdi_on_the_fly.cu into
  # cbbhx so the virtual-class typeinfo lives in the cbbhx .so
  # (mirrors GBGPU's Phase 3L.7d pattern for gb_tdi_on_the_fly).
  add_custom_command(
    OUTPUT "lat_tdi_on_the_fly.cxx"
    COMMENT "Copy LAT's lat_tdi_on_the_fly.cu to .cxx (for cbbhx CPU)"
    COMMAND ${CMAKE_COMMAND} -E copy "${LISATOOLS_CUTILS}/lat_tdi_on_the_fly.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/lat_tdi_on_the_fly.cxx"
    DEPENDS "${LISATOOLS_CUTILS}/lat_tdi_on_the_fly.cu"
    VERBATIM)

  # Phase 3L.8 chunked-het pipeline (SOBBHComputationGroupWrap::
  # sobbh_wdm_het_*) needs GBT's fit_cubic_spline_thomas tridiagonal
  # cubic-spline solver. Pre-2026-06-05 this required a rename-shim TU
  # (lat_interpolate_thomas.cu) because BBHx had its own local
  # Interpolate.cu whose `prep_splines` / `fill_B` / `interpolate` etc.
  # collided at link time with GBT's. After the GBT-dedup pass, BBHx's
  # local copy is gone -- GBT's Interpolate.cu is the only one in
  # cbbhx, so the shim was deleted and the GBT-direct copy above
  # (Interpolate.cxx) is enough.
endif()

# LAPACKE: Interpolate + Likelihood need it. Detect once (idempotent).
bbhx_get_lapacke()


# (Cython sections for interp / likelihood / response / waveformbuild /
# speciallike / newhdyn all removed at Phase BBHx.pybind.bulk -- their
# functions are now methods on BBHxComputationWrap inside the cbbhx
# pybind11 module. The .cu sources they used to compile against now
# compile into cbbhx directly via the add_custom_command rules above.)


# ----------------
# --- cbbhx (pybind11 module skeleton) ---
# ----------------
#
# Phase BBHx.pybind (2026-06-03): adapted from origin/pybind branch
# (commits 7c383e3, 0543508, bd9e127), modernized against the
# post-Phase-3L LAT setup. The pybind11 module is reserved for
# BBH-specific wrappers (PhenomHM/Response/WaveformBuild/Likelihood/
# Interpolate as they migrate from the .pyx modules above) and the
# SOBBHTDIonTheFly + SOBBHComputationGroup carve-out scheduled for
# Phase 3L.8. The skeleton commit ships an empty BBHxComputationWrap;
# subsequent commits populate it. The single-registrant rule is
# enforced via static_assert(!LISATOOLS_IS_WRAPPER_OWNER) in
# binding_bbhx.cxx -- BBHx must NOT register OrbitsWrap,
# TDIConfigWrap, LISAResponseWrap, WDM*/FD*/Spline* (those are LAT's).

if(BBHX_WITH_CPU)
  # cbbhx CPU build: one .so per backend wheel containing the pybind11
  # binding + every BBHx-local .cu + LAT's Detector.cu, mirroring LAT's
  # `lisaanalysistools_cpu_pycppdetector` layout.
  nanobind_add_module(bbhx_cpu_cbbhx
    binding_bbhx.cxx
    PhenomHMWaveform.cxx
    Interpolate.cxx     # GBT's Interpolate.cu, copied via the rule above
    Likelihood.cxx
    Response.cxx
    WaveformBuild.cxx
    Detector.cxx
    sobbh_tdi_on_the_fly.cxx
    lat_tdi_on_the_fly.cxx)  # SpecialLikelihood.cxx omitted: GPU only
  apply_cpu_backend_common_options(cbbhx)
  target_include_directories(bbhx_cpu_cbbhx PRIVATE
    "${LISATOOLS_CUTILS}"  # ReturnPointerBase + lisatools_header_abi.hpp + LAT Detector.hpp
    "${GBT_CUTILS}"        # Interpolate.hh / InterpolateDevice.hh / gbt_global.h /
                            # cuda_complex.hpp / pybind11_cuda_array_interface.hpp
  )
  target_link_libraries(bbhx_cpu_cbbhx LINK_PUBLIC ${BBHX_LAPACKE_LIBS}
                        ${BBHX_LAPACKE_EXTRA_LIBS})
  target_sources(bbhx_cpu_cbbhx PUBLIC FILE_SET HEADERS FILES
                                       binding_bbhx.hpp
                                       PhenomHMWaveform.hh
                                       Likelihood.hh Response.hh
                                       WaveformBuild.hh
                                       constants.h global.h)
endif()

if(BBHX_WITH_GPU)
  # cbbhx GPU build: same layout, .cu sources compiled directly by nvcc.
  nanobind_add_module(bbhx_gpu_cbbhx
    binding_bbhx.cxx
    PhenomHMWaveform.cu
    "${GBT_CUTILS}/Interpolate.cu"       # GBT-owned cubic-spline solver
    Likelihood.cu
    Response.cu
    WaveformBuild.cu
    SpecialLikelihood.cu
    "${LISATOOLS_CUTILS}/Detector.cu"
    "${LISATOOLS_CUTILS}/lat_tdi_on_the_fly.cu"
    sobbh_tdi_on_the_fly.cu)
  apply_gpu_backend_common_options(cbbhx)
  target_include_directories(bbhx_gpu_cbbhx PRIVATE
    "${LISATOOLS_CUTILS}"
    "${GBT_CUTILS}"
  )
  target_link_libraries(bbhx_gpu_cbbhx LINK_PUBLIC ${BBHX_LAPACKE_LIBS}
                        ${BBHX_LAPACKE_EXTRA_LIBS})
  set_property(TARGET bbhx_gpu_cbbhx PROPERTY CUDA_ARCHITECTURES ${BBHX_CUDA_ARCH})
  set_property(TARGET bbhx_gpu_cbbhx PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET bbhx_gpu_cbbhx PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_property(TARGET bbhx_gpu_cbbhx PROPERTY POSITION_INDEPENDENT_CODE ON)
  # NB: cuda_complex.hpp + Interpolate.hh + InterpolateDevice.hh deliberately
  # omitted from local sources -- resolved via ${GBT_CUTILS}
  # (GPUBackendTools owns the canonical copies).
  target_sources(bbhx_gpu_cbbhx PUBLIC FILE_SET HEADERS FILES
                                       binding_bbhx.hpp
                                       PhenomHMWaveform.hh
                                       Likelihood.hh Response.hh
                                       WaveformBuild.hh SpecialLikelihood.hh
                                       constants.h global.h)
endif()