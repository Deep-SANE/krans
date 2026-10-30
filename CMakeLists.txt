cmake_minimum_required(VERSION 3.14)
project(SparseNullSpaceSolver CXX C)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Default to Release ----
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)

# ---- Eigen3 ----
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
  message(STATUS "Eigen3 not found, fetching...")
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
  )
  FetchContent_MakeAvailable(eigen)
  if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})
  endif()
endif()

# ---- Optional Dependencies ----
option(ENABLE_SPECTRA "Enable Spectra integration" OFF)
option(ENABLE_PRIMME "Enable PRIMME integration" OFF)
option(ENABLE_SLEPC "Enable SLEPc integration" OFF)
option(ENABLE_SUITESPARSE "Enable SuiteSparse integration" OFF)
option(BUILD_SNS_CLI "Build the sns_nullspace CLI tool" ON)

if(ENABLE_SPECTRA)
  FetchContent_Declare(
    spectra
    GIT_REPOSITORY https://github.com/yixuan/spectra.git
    GIT_TAG        v1.0.1
  )
  FetchContent_MakeAvailable(spectra)
  set(Spectra_FOUND TRUE)
endif()

# ---- PRIMME ----
if(ENABLE_PRIMME)
  find_program(MAKE_EXE NAMES make gmake mingw32-make)
  if(MAKE_EXE)
    FetchContent_Declare(
      primme
      GIT_REPOSITORY https://github.com/primme/primme.git
      GIT_TAG        v3.2
    )
    FetchContent_GetProperties(primme)
    if(NOT primme_POPULATED)
      FetchContent_MakeAvailable(primme)
      
      set(PRIMME_INCLUDE_DIRS "${primme_SOURCE_DIR}/include" "${primme_SOURCE_DIR}/src/headers")
      set(PRIMME_LIB_FILE "${primme_SOURCE_DIR}/lib/libprimme.a")
      
      if(APPLE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        set(PRIMME_LIBS ${PRIMME_LIB_FILE} ${ACCELERATE_FRAMEWORK})
      else()
        find_package(BLAS QUIET)
        find_package(LAPACK QUIET)
        if(BLAS_FOUND AND LAPACK_FOUND)
          set(PRIMME_LIBS ${PRIMME_LIB_FILE} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
        else()
          set(PRIMME_LIBS ${PRIMME_LIB_FILE})
	    endif()
      endif()

      add_custom_command(
        OUTPUT ${PRIMME_LIB_FILE}
        COMMAND ${MAKE_EXE} lib
        WORKING_DIRECTORY ${primme_SOURCE_DIR}
        COMMENT "Building PRIMME library..."
        VERBATIM
      )
      add_custom_target(primme_build_target DEPENDS ${PRIMME_LIB_FILE})
      
      add_library(primme_lib INTERFACE)
      target_include_directories(primme_lib INTERFACE ${PRIMME_INCLUDE_DIRS})
      target_link_libraries(primme_lib INTERFACE ${PRIMME_LIBS})
      add_dependencies(primme_lib primme_build_target)
      
      set(PRIMME_FOUND TRUE)
    endif()
  else()
    message(STATUS "make not found, skipping PRIMME build.")
  endif()
endif()

# ---- SLEPc / PETSc ----
if(ENABLE_SLEPC)
  find_package(MPI QUIET)
  find_package(PkgConfig QUIET)
  if(MPI_FOUND AND PKG_CONFIG_FOUND)
    pkg_check_modules(SLEPC QUIET IMPORTED_TARGET slepc)
    pkg_check_modules(PETSC QUIET IMPORTED_TARGET petsc)
    
    if(SLEPC_FOUND AND PETSC_FOUND)
      add_library(slepc_lib INTERFACE)
      target_link_libraries(slepc_lib INTERFACE PkgConfig::SLEPC PkgConfig::PETSC MPI::MPI_CXX)
      set(SLEPc_FOUND TRUE)
    endif()
  endif()
endif()

# ---- SuiteSparse ----
if(ENABLE_SUITESPARSE)
  find_path(SUITESPARSE_INCLUDE_DIR SuiteSparseQR.hpp PATH_SUFFIXES suitesparse)
  find_library(SPQR_LIB spqr)
  find_library(CHOLMOD_LIB cholmod)
  find_library(SUITESPARSECONFIG_LIB suitesparseconfig)
  
  if(SUITESPARSE_INCLUDE_DIR AND SPQR_LIB AND CHOLMOD_LIB)
    set(SuiteSparse_FOUND TRUE)
    add_library(suitesparse_lib INTERFACE)
    target_include_directories(suitesparse_lib INTERFACE ${SUITESPARSE_INCLUDE_DIR})
    target_link_libraries(suitesparse_lib INTERFACE ${SPQR_LIB} ${CHOLMOD_LIB} ${SUITESPARSECONFIG_LIB})
  endif()
endif()

# ---- IPO/LTO ----
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_msg)
if(ipo_supported)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ---- Compile Options ----
add_library(sns_compile_options INTERFACE)
if(MSVC)
  target_compile_options(sns_compile_options INTERFACE /O2 /DNDEBUG)
else()
  target_compile_options(sns_compile_options INTERFACE -O3 -DNDEBUG)
endif()
target_compile_definitions(sns_compile_options INTERFACE
  $<$<CONFIG:Release>:EIGEN_NO_DEBUG>
)

# ---- Copy test data into the build dir ----
add_custom_target(copy_data ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_SOURCE_DIR}/data
  ${CMAKE_BINARY_DIR}/data
)

# ---------- Libraries ----------
add_library(sns_core STATIC
  src/Utils/GramSchmidt.cpp
  src/Utils/Projection.cpp
)
target_include_directories(sns_core PUBLIC include)
target_link_libraries(sns_core PUBLIC sns_compile_options)

add_library(sns_eigen STATIC
  src/IO/MatrixLoader.cpp
  src/Solvers/EigenFullPivLUSolver.cpp
  src/Solvers/EigenSVDSolver.cpp
  src/Solvers/EigenCODSolver.cpp
  src/Solvers/EigenQRCPSolver.cpp
  src/Utils/EigenSmallSVD.cpp
  src/Solvers/CyclicLanczosSolver.cpp
  src/Solvers/LOBPCGSolver.cpp
)

target_include_directories(sns_eigen PUBLIC include)
target_link_libraries(sns_eigen PUBLIC sns_core Eigen3::Eigen sns_compile_options)

if(Spectra_FOUND)
  target_sources(sns_eigen PRIVATE src/Solvers/SpectraSolver.cpp)
  target_link_libraries(sns_eigen PUBLIC Spectra)
  target_compile_definitions(sns_eigen PUBLIC SNS_USE_SPECTRA)
endif()

if(PRIMME_FOUND)
  target_sources(sns_eigen PRIVATE src/Solvers/PrimmeSolver.cpp)
  target_link_libraries(sns_eigen PUBLIC primme_lib)
  target_compile_definitions(sns_eigen PUBLIC SNS_USE_PRIMME)
endif()

if(SLEPc_FOUND)
  target_sources(sns_eigen PRIVATE src/Solvers/SlepcSolver.cpp)
  target_link_libraries(sns_eigen PUBLIC slepc_lib)
  target_compile_definitions(sns_eigen PUBLIC SNS_USE_SLEPC)
endif()

if(SuiteSparse_FOUND)
  target_sources(sns_eigen PRIVATE src/Solvers/SuiteSparseQRSolver.cpp)
  target_link_libraries(sns_eigen PUBLIC suitesparse_lib)
  target_compile_definitions(sns_eigen PUBLIC SNS_USE_SUITESPARSE)
endif()

# ---------- CLI ----------
if(BUILD_SNS_CLI)
  add_executable(sns_nullspace apps/sns_nullspace.cpp)
  target_link_libraries(sns_nullspace PRIVATE sns_eigen)
endif()

# ---------- Tests ----------
add_executable(test_vector_adapter tests/test_vector_adapter.cpp)
target_link_libraries(test_vector_adapter PRIVATE sns_eigen)

add_executable(test_dense_adapter tests/test_dense_adapter.cpp)
target_link_libraries(test_dense_adapter PRIVATE sns_eigen)

add_executable(test_sparse_adapter tests/test_sparse_adapter.cpp)
target_link_libraries(test_sparse_adapter PRIVATE sns_eigen)

add_executable(test_sparse_ref_adapter tests/test_sparse_ref_adapter.cpp)
target_link_libraries(test_sparse_ref_adapter PRIVATE sns_eigen)

add_executable(test_vector_factory tests/test_vector_factory.cpp)
target_link_libraries(test_vector_factory PRIVATE sns_eigen)

add_executable(test_adjoint_property tests/test_adjoint_property.cpp)
target_link_libraries(test_adjoint_property PRIVATE sns_eigen)

add_executable(test_ija_loader tests/test_ija_loader.cpp)
target_link_libraries(test_ija_loader PRIVATE sns_eigen)
add_dependencies(test_ija_loader copy_data)

add_executable(test_gram_schmidt tests/test_gram_schmidt.cpp)
target_link_libraries(test_gram_schmidt PRIVATE sns_eigen)

add_executable(test_projection tests/test_projection.cpp)
target_link_libraries(test_projection PRIVATE sns_eigen)

add_executable(test_eigen_svd tests/test_eigen_svd.cpp)
target_link_libraries(test_eigen_svd PRIVATE sns_eigen)

add_executable(test_eigen_reference_solvers tests/test_eigen_reference_solvers.cpp)
target_link_libraries(test_eigen_reference_solvers PRIVATE sns_eigen)

add_executable(test_cyclic_lanczos tests/test_cyclic_lanczos.cpp)
target_link_libraries(test_cyclic_lanczos PRIVATE sns_eigen)

add_executable(test_lobpcg tests/test_lobpcg.cpp)
target_link_libraries(test_lobpcg PRIVATE sns_eigen)

add_executable(test_universal_adapter tests/test_universal_adapter.cpp)
target_link_libraries(test_universal_adapter PRIVATE sns_eigen)

add_executable(test_subspace_compare tests/test_subspace_compare.cpp)
target_link_libraries(test_subspace_compare PRIVATE sns_eigen)

add_executable(test_all_solvers tests/test_all_solvers.cpp)
target_link_libraries(test_all_solvers PRIVATE sns_eigen)

# Convenience target to run the essential adapter/factory tests
add_custom_target(run_core_adapter_tests
  COMMAND test_vector_adapter
  COMMAND test_dense_adapter
  COMMAND test_sparse_adapter
  COMMAND test_vector_factory
  COMMAND test_adjoint_property
  COMMAND test_gram_schmidt
  COMMAND test_projection
  COMMAND test_eigen_svd
  DEPENDS
  test_vector_adapter
  test_dense_adapter
  test_sparse_adapter
  test_vector_factory
  test_adjoint_property
  test_gram_schmidt
  test_projection
  test_eigen_svd
)

add_executable(verification_suite benchmarks/workers/verification_suite.cpp)
target_link_libraries(verification_suite PRIVATE sns_eigen)

add_executable(memory_wall_suite benchmarks/workers/memory_wall_suite.cpp)
target_link_libraries(memory_wall_suite PRIVATE sns_eigen)

add_executable(hyperparam_worker benchmarks/workers/hyperparam_worker.cpp)
target_link_libraries(hyperparam_worker PRIVATE sns_eigen)

add_executable(conditioning_worker benchmarks/workers/conditioning_worker.cpp)
target_link_libraries(conditioning_worker PRIVATE sns_eigen)

add_executable(cluster_worker benchmarks/workers/cluster_worker.cpp)
target_link_libraries(cluster_worker PRIVATE sns_eigen)