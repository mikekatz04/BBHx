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

# I. Process phenomhm.pyx into a C++ file
add_custom_command(
  OUTPUT "phenomhm.cxx"
  COMMENT "Cythonize phenomhm.pyx into phenomhm.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/phenomhm.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/phenomhm.cxx" -3 -+ --module-name
    "phenomhm" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "phenomhm.pyx"
  VERBATIM)

# II. Declare the CPU backend
if(BBHX_WITH_CPU)
  add_custom_command(
    OUTPUT "PhenomHMWaveform.cxx"
    COMMENT "Copy PhenomHM.cu to PhenomHMWaveform.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/PhenomHMWaveform.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/PhenomHMWaveform.cxx"
    DEPENDS "PhenomHMWaveform.cu"
    VERBATIM)

  python_add_library(bbhx_cpu_phenomhm MODULE WITH_SOABI phenomhm.cxx PhenomHMWaveform.cxx)
  apply_cpu_backend_common_options(phenomhm)

  target_sources(bbhx_cpu_phenomhm PUBLIC FILE_SET HEADERS FILES
                                        constants.h global.h PhenomHMWaveform.hh)
endif()

# III. Declare the GPU backend
if(BBHX_WITH_GPU)
  python_add_library(bbhx_gpu_phenomhm MODULE WITH_SOABI phenomhm.cxx PhenomHMWaveform.cu)
  apply_gpu_backend_common_options(phenomhm)
  target_sources(bbhx_gpu_phenomhm PUBLIC FILE_SET HEADERS FILES
                                         cuda_complex.hpp global.h PhenomHMWaveform.hh)
endif()


# ----------------
# --- likelihood ---
# ----------------

# I. Process bbhlikelihood.pyx into a C++ file
add_custom_command(
  OUTPUT "bbhlikelihood.cxx"
  COMMENT "Cythonize bbhlikelihood.pyx into bbhlikelihood.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/bbhlikelihood.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/bbhlikelihood.cxx" -3 -+ --module-name
    "likelihood" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "bbhlikelihood.pyx"
  VERBATIM)

# II. Declare the CPU backend
if(BBHX_WITH_CPU)
  add_custom_command(
    OUTPUT "Likelihood.cxx"
    COMMENT "Copy Likelihood.cu to Likelihood.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Likelihood.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Likelihood.cxx"
    DEPENDS "Likelihood.cu"
    VERBATIM)

  python_add_library(bbhx_cpu_likelihood MODULE WITH_SOABI bbhlikelihood.cxx Likelihood.cxx)
  apply_cpu_backend_common_options(likelihood)

  target_sources(bbhx_cpu_likelihood PUBLIC FILE_SET HEADERS FILES
                                        constants.h global.h Likelihood.hh)

  # likelihood needs to link against LAPACKE. USe the following method to detect
  # it and obtain ${BBHX_LAPACKE_LIBS}. In case of failure, it will stop the
  # processing execution.
  bbhx_get_lapacke()
  # Now, link against ${BBHX_LAPACKE_LIBS} to link against liblapacke.so and
  # include lapacke.h
  target_link_libraries(bbhx_cpu_likelihood LINK_PUBLIC ${BBHX_LAPACKE_LIBS}
                        ${BBHX_LAPACKE_EXTRA_LIBS})
endif()

# III. Declare the GPU backend
if(BBHX_WITH_GPU)
  python_add_library(bbhx_gpu_likelihood MODULE WITH_SOABI bbhlikelihood.cxx Likelihood.cu)
  apply_gpu_backend_common_options(likelihood)
  target_sources(bbhx_gpu_likelihood PUBLIC FILE_SET HEADERS FILES
                                       constants.h global.h Likelihood.hh)

endif()



# 0. Download lisa analysis tools c files.


execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import lisatools
    _tmp = lisatools.__file__
    print(_tmp.split('/__init__.py')[0])"
  OUTPUT_VARIABLE LISATOOLS_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import lisatools
    print(lisatools._version.version)"
  OUTPUT_VARIABLE LISATOOLS_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(FILES_TO_COPY 
  "${LISATOOLS_DIR}/cutils/global.hpp"
  "${LISATOOLS_DIR}/cutils/Detector.hpp"
  "${LISATOOLS_DIR}/cutils/Detector.cu"
)

foreach(FILE_TO_COPY IN LISTS FILES_TO_COPY)
  get_filename_component(NEW_FILE "${FILE_TO_COPY}" NAME)
  file(COPY_FILE "${FILE_TO_COPY}" "${CMAKE_CURRENT_SOURCE_DIR}/${NEW_FILE}"
     RESULT copy_result)
  if(NOT copy_result EQUAL 0)
      message(FATAL_ERROR "Failed to copy file: ${copy_result} (${FILE_TO_COPY})")
  endif()
endforeach()

# ----------------
# --- response ---
# ----------------

# I. Process lisaresponse.pyx into a C++ file
add_custom_command(
  OUTPUT "lisaresponse.cxx"
  COMMENT "Cythonize lisaresponse.pyx into lisaresponse.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/lisaresponse.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/lisaresponse.cxx" -3 -+ --module-name
    "response" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "lisaresponse.pyx"
  VERBATIM)

# II. Declare the CPU backend
if(BBHX_WITH_CPU)
  add_custom_command(
    OUTPUT "Response.cxx"
    COMMENT "Copy Response.cu to Response.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Response.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Response.cxx"
    DEPENDS "Response.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "Detector.cxx"
    COMMENT "Copy Detector.cu to Detector.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Detector.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Detector.cxx"
    DEPENDS "Detector.cu"
    VERBATIM)

  python_add_library(bbhx_cpu_response MODULE WITH_SOABI Detector.cxx lisaresponse.cxx Response.cxx)
  apply_cpu_backend_common_options(response)

  target_sources(bbhx_cpu_response PUBLIC FILE_SET HEADERS FILES
                                        global.hpp Detector.hpp constants.h global.h Response.hh)

endif()

# III. Declare the GPU backend
if(BBHX_WITH_GPU)
  python_add_library(bbhx_gpu_response MODULE WITH_SOABI Detector.cu lisaresponse.cxx Response.cu)
  apply_gpu_backend_common_options(response)
  target_sources(bbhx_gpu_response PUBLIC FILE_SET HEADERS FILES
                                       global.hpp Detector.hpp constants.h global.h Response.hh)

  set_property(TARGET bbhx_gpu_response PROPERTY CUDA_ARCHITECTURES ${BBHX_CUDA_ARCH})
  set_property(TARGET bbhx_gpu_response PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET bbhx_gpu_response PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_property(TARGET bbhx_gpu_response PROPERTY POSITION_INDEPENDENT_CODE ON)
  
  install(TARGETS bbhx_gpu_response
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/bbhx)        

endif()

# IV. Remove downloaded files. 
file(REMOVE "./src/bbhx/cutils/Detector.cu")
file(REMOVE "./src/bbhx/cutils/Detector.hpp")
file(REMOVE "./src/bbhx/cutils/global.hpp")



# ----------------
# --- waveformbuild ---
# ----------------

# I. Process bbhwaveformbuild.pyx into a C++ file
add_custom_command(
  OUTPUT "bbhwaveformbuild.cxx"
  COMMENT "Cythonize bbhwaveformbuild.pyx into bbhwaveformbuild.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/bbhwaveformbuild.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/bbhwaveformbuild.cxx" -3 -+ --module-name
    "waveformbuild" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "bbhwaveformbuild.pyx"
  VERBATIM)

# II. Declare the CPU backend
if(BBHX_WITH_CPU)
  add_custom_command(
    OUTPUT "WaveformBuild.cxx"
    COMMENT "Copy WaveformBuild.cu to WaveformBuild.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/WaveformBuild.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/WaveformBuild.cxx"
    DEPENDS "WaveformBuild.cu"
    VERBATIM)

  python_add_library(bbhx_cpu_waveformbuild MODULE WITH_SOABI bbhwaveformbuild.cxx WaveformBuild.cxx)
  apply_cpu_backend_common_options(waveformbuild)

  target_sources(bbhx_cpu_waveformbuild PUBLIC FILE_SET HEADERS FILES
                                        constants.h global.h Likelihood.hh)

endif()

# III. Declare the GPU backend
if(BBHX_WITH_GPU)
  python_add_library(bbhx_gpu_waveformbuild MODULE WITH_SOABI bbhwaveformbuild.cxx WaveformBuild.cu)
  apply_gpu_backend_common_options(waveformbuild)
  target_sources(bbhx_gpu_waveformbuild PUBLIC FILE_SET HEADERS FILES
                                       constants.h global.h WaveformBuild.hh)

endif()

