cmake_minimum_required(VERSION 3.21)
project(dustppd VERSION 0.1.0 LANGUAGES CXX
        DESCRIPTION "ODE solver library and protoplanetary dust-dynamics model")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(DPD_BUILD_TESTS "Build C++ tests" ON)
option(DPD_BUILD_BENCH "Build benchmarks" ON)
option(DPD_BUILD_APPS "Build command-line application" ON)
option(DPD_BUILD_PYTHON "Build pybind11 Python module" OFF)
option(DPD_ENABLE_SIMD "Build explicit SIMD kernels (NEON / AVX2 runtime dispatch)" ON)
option(DPD_STRICT_FP "Disable floating-point contraction outside explicit FMA kernels" ON)
option(DPD_NATIVE "Compile for the host CPU (-march=native); not portable" OFF)
option(DPD_WERROR "Treat warnings as errors" OFF)
option(DPD_BUILD_GPU_EXPERIMENTS "Build experimental Metal/CUDA ensemble benchmarks" OFF)
set(DPD_SANITIZE "" CACHE STRING "Sanitizers: '', 'address,undefined', 'thread'")

include(cmake/Dependencies.cmake)

# Common compile settings ------------------------------------------------------
add_library(dpd_options INTERFACE)
if(MSVC)
  target_compile_options(dpd_options INTERFACE /W4 /permissive- /utf-8 /fp:precise)
  if(DPD_WERROR)
    target_compile_options(dpd_options INTERFACE /WX)
  endif()
else()
  target_compile_options(dpd_options INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wconversion
                         -Wno-sign-conversion)
  if(DPD_STRICT_FP)
    target_compile_options(dpd_options INTERFACE -ffp-contract=off)
  endif()
  if(DPD_NATIVE)
    target_compile_options(dpd_options INTERFACE -march=native)
  endif()
  if(DPD_WERROR)
    target_compile_options(dpd_options INTERFACE -Werror)
  endif()
  if(DPD_SANITIZE)
    target_compile_options(dpd_options INTERFACE -fsanitize=${DPD_SANITIZE} -fno-omit-frame-pointer -g)
    target_link_options(dpd_options INTERFACE -fsanitize=${DPD_SANITIZE})
  endif()
endif()
if(NOT DPD_ENABLE_SIMD)
  target_compile_definitions(dpd_options INTERFACE DPD_DISABLE_AVX2 DPD_DISABLE_NEON)
endif()

# Numerical core (no physics, no I/O) -----------------------------------------
add_library(dpd_ode
  src/ode/kernels.cpp src/ode/kernels_neon.cpp src/ode/kernels_avx2.cpp
  src/ode/linalg.cpp src/ode/jacobian.cpp src/ode/explicit.cpp src/ode/bdf.cpp
  src/ode/integrate.cpp src/ode/test_problems.cpp src/ode/sde.cpp)
add_library(dpd::ode ALIAS dpd_ode)
target_include_directories(dpd_ode PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                           $<INSTALL_INTERFACE:include>)
target_link_libraries(dpd_ode PRIVATE dpd_options)
set_target_properties(dpd_ode PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Physical model + configuration ----------------------------------------------
add_library(dpd_dust
  src/dust/disk.cpp src/dust/model.cpp src/dust/config.cpp src/dust/run.cpp src/dust/io.cpp
  src/dust/traps.cpp src/dust/diffusion.cpp)
add_library(dpd::dust ALIAS dpd_dust)
target_link_libraries(dpd_dust PUBLIC dpd_ode PRIVATE dpd_options tomlplusplus::tomlplusplus Threads::Threads)
set_target_properties(dpd_dust PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(DPD_BUILD_APPS)
  add_executable(dustppd apps/dustppd_cli.cpp)
  target_link_libraries(dustppd PRIVATE dpd_dust dpd_options)
endif()

if(DPD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

if(DPD_BUILD_BENCH)
  add_subdirectory(bench)
endif()

if(DPD_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(DPD_BUILD_GPU_EXPERIMENTS)
  add_subdirectory(experiments/gpu)
endif()

include(GNUInstallDirs)
install(TARGETS dpd_ode dpd_dust ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(DPD_BUILD_APPS)
  install(TARGETS dustppd RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
