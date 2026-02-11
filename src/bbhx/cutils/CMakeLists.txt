# =============================
# ==== BACKEND COMPILATION ====
# =============================

# In the project root CMakeLists.txt, we defined a "bbhx" interface
# target with properties WITH_CPU and WITH_GPU defining whether the CPU and a
# GPU backend need to be compiled. Let's retrieve these information here:
get_target_property(BBHX_WITH_CPU bbhx WITH_CPU)
get_target_property(BBHX_WITH_GPU bbhx WITH_GPU)

# Adapter to let inplace editable install work: the compiled backend will be
# placed into the source-tree in the 'src' directory:
# TODO: is this needed ?
if(SKBUILD_STATE STREQUAL "editable")
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${bbhx_BINARY_DIR}/src")
else()
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import gpubackendtools
    print(gpubackendtools.__file__.split('__init__.py')[0] + 'cutils')"
  OUTPUT_VARIABLE GBT_CUTILS
  OUTPUT_STRIP_TRAILING_WHITESPACE)

set(GBT_CMAKE_TOOLS "${GBT_CUTILS}/cmake_functions.cmake")

# Check if the file exists
if(EXISTS "${GBT_CMAKE_TOOLS}")
    message("INFO: ${GBT_CMAKE_TOOLS} exists.")
else()
    message(FATAL_ERROR "${GBT_CMAKE_TOOLS} not found.")
    # Handle the case where the file is missing
endif()

include(${GBT_CMAKE_TOOLS})
set(HERE_CUDA_ARCH ${BBHX_CUDA_ARCH})
set(GBT_LAPACK_DETECT_WITH ${BBHX_LAPACK_DETECT_WITH})

# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *

# ----------------
# --- cbbhx ---
# ----------------


execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import lisatools
    _tmp = lisatools.__file__
    print(_tmp.split('/__init__.py')[0])"
  OUTPUT_VARIABLE LISATOOLS_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import gpubackendtools
    _tmp = gpubackendtools.__file__
    print(_tmp.split('/__init__.py')[0])"
  OUTPUT_VARIABLE GBT_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import lisatools
    print(lisatools._version.version)"
  OUTPUT_VARIABLE LISATOOLS_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "LISATOOLS location is ${LISATOOLS_DIR}.")
file(GLOB MY_FILES "${LISATOOLS_DIR}/cutils/*")
message(STATUS "LISATOOLS Source files: ${MY_FILES}")
message(STATUS "LISATOOLS version: ${LISATOOLS_VERSION}")

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    import gpubackendtools
    print(gpubackendtools._version.version)"
  OUTPUT_VARIABLE GBT_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE)


execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    from lisatools_backend_cpu import pycppdetector
    print(pycppdetector.__file__)"
  OUTPUT_VARIABLE LISATOOLS_DETECTOR_CPU
  OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    from gbt_backend_cpu import interp
    print(interp.__file__)"
  OUTPUT_VARIABLE GBT_INTERP_CPU
  OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "TEST CHECK 1: ${GBT_INTERP_CPU}")

message(STATUS "GBT version: ${GBT_VERSION}")

message(STATUS "LISATOOLS location is ${LISATOOLS_DIR}.")
message(STATUS "GBT location is ${GBT_DIR}.")
file(GLOB MY_FILES "${LISATOOLS_DIR}/cutils/*" "${GBT_DIR}/cutils/*")
message(STATUS "Source files: ${MY_FILES}")

# interp needs to link against LAPACKE. USe the following method to detect
# it and obtain ${GBT_LAPACKE_LIBS}. In case of failure, it will stop the
# processing execution.
enable_language(Fortran)
get_lapacke()

# II. Declare the CPU backend
if(BBHX_WITH_CPU)

  add_custom_command(
    OUTPUT "PhenomHMWaveform.cxx"
    COMMENT "Copy PhenomHMWaveform.cu to PhenomHMWaveform.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/PhenomHMWaveform.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/PhenomHMWaveform.cxx"
    DEPENDS "PhenomHMWaveform.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "Response.cxx"
    COMMENT "Copy Response.cu to Response.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Response.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Response.cxx"
    DEPENDS "Response.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "WaveformBuild.cxx"
    COMMENT "Copy WaveformBuild.cu to WaveformBuild.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/WaveformBuild.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/WaveformBuild.cxx"
    DEPENDS "WaveformBuild.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "Interpolate.cxx"
    COMMENT "Copy Interpolate.cu to Interpolate.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Interpolate.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Interpolate.cxx"
    DEPENDS "Interpolate.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "Likelihood.cxx"
    COMMENT "Copy Likelihood.cu to Likelihood.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Likelihood.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Likelihood.cxx"
    DEPENDS "Likelihood.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "Detector.cxx"
    COMMENT "Copy Detector.cu to Detector.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${LISATOOLS_DIR}/cutils/Detector.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Detector.cxx"
    DEPENDS "${LISATOOLS_DIR}/cutils/Detector.cu"
    VERBATIM)

  add_library(bbhx_cpu_cbbhx_static STATIC Detector.cxx PhenomHMWaveform.cxx Response.cxx WaveformBuild.cxx Likelihood.cxx Interpolate.cxx)
  apply_cpu_backend_common_options(cbbhx bbhx bbhx true )
  target_include_directories(bbhx_cpu_cbbhx_static PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})  
  target_sources(bbhx_cpu_cbbhx_static PUBLIC FILE_SET HEADERS FILES
                                        PhenomHMWaveform.hh Response.hh WaveformBuild.hh Likelihood.hh Interpolate.hh)

  target_link_libraries(bbhx_cpu_cbbhx_static LINK_PUBLIC ${GBT_LAPACKE_LIBS}
                        ${GBT_LAPACKE_EXTRA_LIBS})

  # pybind11_add_module(bbhx_cpu_cbbhx "${LISATOOLS_DIR}/cutils/binding.cxx" binding_bbhx.cxx)
  pybind11_add_module(bbhx_cpu_cbbhx binding_bbhx.cxx)
  apply_cpu_backend_common_options(cbbhx bbhx bbhx false)
  
  # Link your target to the imported target name
  target_link_libraries(bbhx_cpu_cbbhx PUBLIC bbhx_cpu_cbbhx_static)
  target_include_directories(bbhx_cpu_cbbhx PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})  
  target_sources(bbhx_cpu_cbbhx PUBLIC FILE_SET HEADERS FILES
                                          binding_bbhx.hpp)
  target_link_libraries(bbhx_cpu_cbbhx LINK_PUBLIC ${GBT_LAPACKE_LIBS}
                        ${GBT_LAPACKE_EXTRA_LIBS})

endif()

# III. Declare the GPU backend
if(BBHX_WITH_GPU)

  add_library(bbhx_gpu_cbbhx_gpu_static STATIC "${LISATOOLS_DIR}/cutils/Detector.cu" PhenomHMWaveform.cu PhenomHMWaveform.cu Response.cu WaveformBuild.cu Likelihood.cu Interpolate.cu)
  apply_gpu_backend_common_options(cbbhx_gpu bbhx bbhx true)
  target_sources(bbhx_gpu_cbbhx_gpu_static PUBLIC FILE_SET HEADERS FILES
                                         PhenomHMWaveform.hh Response.hh WaveformBuild.hh Likelihood.hh Interpolate.hh)

  target_include_directories(bbhx_gpu_cbbhx_gpu_static PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})
  pybind11_add_module(bbhx_gpu_cbbhx binding_bbhx.cxx)
  apply_gpu_backend_common_options(cbbhx bbhx bbhx false)
  target_sources(bbhx_gpu_cbbhx PUBLIC FILE_SET HEADERS FILES
                                         binding_bbhx.hpp
                                         )
  target_link_libraries(bbhx_gpu_cbbhx PUBLIC bbhx_gpu_cbbhx_gpu_static)
  target_include_directories(bbhx_gpu_cbbhx PUBLIC "${LISATOOLS_DIR}/cutils/" ${GBT_CUTILS})  
  
endif()
