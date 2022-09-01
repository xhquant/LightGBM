option(USE_MPI "Enable MPI-based distributed learning" OFF)
option(USE_OPENMP "Enable OpenMP" ON)
option(USE_GPU "Enable GPU-accelerated training" OFF)
option(USE_HDFS "Enable HDFS support (EXPERIMENTAL)" OFF)
option(USE_TIMETAG "Set to ON to output time costs" OFF)
option(USE_CUDA "Enable CUDA-accelerated training (EXPERIMENTAL)" OFF)
option(USE_CUDA_EXP "Enable CUDA-accelerated training with more acceleration (EXPERIMENTAL)" OFF)
option(USE_DEBUG "Set to ON for Debug mode" OFF)
option(USE_SANITIZER "Use santizer flags" OFF)
set(
        ENABLED_SANITIZERS
        "address" "leak" "undefined"
        CACHE
        STRING
        "Semicolon separated list of sanitizer names, e.g., 'address;leak'. \
Supported sanitizers are address, leak, undefined and thread."
)
option(BUILD_CPP_TEST "Build C++ tests with Google Test" OFF)
option(BUILD_STATIC_LIB "Build static library" OFF)
option(__INTEGRATE_OPENCL "Set to ON if building LightGBM with the OpenCL ICD Loader and its dependencies included" OFF)


if (__INTEGRATE_OPENCL)
    cmake_minimum_required(VERSION 3.11)
elseif (USE_GPU OR APPLE)
    cmake_minimum_required(VERSION 3.2)
elseif (USE_CUDA OR USE_CUDA_EXP)
    cmake_minimum_required(VERSION 3.16)
else ()
    cmake_minimum_required(VERSION 3.0)
endif ()

project(lightgbm LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

#-- Sanitizer
if (USE_SANITIZER)
    if (MSVC)
        message(FATAL_ERROR "Sanitizers are not supported with MSVC.")
    endif ()
    include(cmake/Sanitizer.cmake)
    enable_sanitizers("${ENABLED_SANITIZERS}")
endif ()

if (__INTEGRATE_OPENCL)
    set(__INTEGRATE_OPENCL ON CACHE BOOL "" FORCE)
    set(USE_GPU OFF CACHE BOOL "" FORCE)
    message(STATUS "Building library with integrated OpenCL components")
endif ()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.8.2")
        message(FATAL_ERROR "Insufficient gcc version")
    endif ()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.8")
        message(FATAL_ERROR "Insufficient Clang version")
    endif ()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.1.0")
        message(FATAL_ERROR "Insufficient AppleClang version")
    endif ()
    cmake_minimum_required(VERSION 3.16)
elseif (MSVC)
    if (MSVC_VERSION LESS 1900)
        message(
                FATAL_ERROR
                "The compiler ${CMAKE_CXX_COMPILER} doesn't support required C++11 features. Please use a newer MSVC."
        )
    endif ()
    cmake_minimum_required(VERSION 3.8)
endif ()

set(EIGEN_DIR "${PROJECT_SOURCE_DIR}/external_libs/eigen")
include_directories(${EIGEN_DIR})

# See https://gitlab.com/libeigen/eigen/-/blob/master/COPYING.README
add_definitions(-DEIGEN_MPL2_ONLY)
add_definitions(-DEIGEN_DONT_PARALLELIZE)

if (USE_TIMETAG)
    add_definitions(-DTIMETAG)
endif ()

if (USE_DEBUG)
    add_definitions(-DDEBUG)
endif ()

if (USE_MPI)
    find_package(MPI REQUIRED)
    add_definitions(-DUSE_MPI)
else ()
    add_definitions(-DUSE_SOCKET)
endif ()

if (USE_OPENMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

if (__INTEGRATE_OPENCL)
    if (WIN32)
        include(cmake/IntegratedOpenCL.cmake)
        add_definitions(-DUSE_GPU)
    else ()
        message(FATAL_ERROR "Integrated OpenCL build is available only for Windows")
    endif ()
endif ()

if (USE_HDFS)
    find_package(JNI REQUIRED)
    find_path(HDFS_INCLUDE_DIR hdfs.h REQUIRED)
    find_library(HDFS_LIB NAMES hdfs REQUIRED)
    include_directories(${HDFS_INCLUDE_DIR})
    add_definitions(-DUSE_HDFS)
    set(HDFS_CXX_LIBRARIES ${HDFS_LIB} ${JAVA_JVM_LIBRARY})
endif ()

include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#include <xmmintrin.h>
int main() {
  int a = 0;
  _mm_prefetch(&a, _MM_HINT_NTA);
  return 0;
}
" MM_PREFETCH)

if (${MM_PREFETCH})
    message(STATUS "Using _mm_prefetch")
    add_definitions(-DMM_PREFETCH)
endif ()

include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#include <mm_malloc.h>
int main() {
  char *a = (char*)_mm_malloc(8, 16);
  _mm_free(a);
  return 0;
}
" MM_MALLOC)

if (${MM_MALLOC})
    message(STATUS "Using _mm_malloc")
    add_definitions(-DMM_MALLOC)
endif ()

if (UNIX OR MINGW OR CYGWIN)
    set(
            CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -Wextra -Wall -Wno-ignored-attributes -Wno-unknown-pragmas -Wno-return-type"
    )
    if (USE_DEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    endif ()
    if (NOT USE_OPENMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-unused-private-field")
    endif ()
    if (__BUILD_FOR_R AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-cast-function-type")
    endif ()
endif ()

if (MSVC)
    set(
            variables
            CMAKE_C_FLAGS_DEBUG
            CMAKE_C_FLAGS_MINSIZEREL
            CMAKE_C_FLAGS_RELEASE
            CMAKE_C_FLAGS_RELWITHDEBINFO
            CMAKE_CXX_FLAGS_DEBUG
            CMAKE_CXX_FLAGS_MINSIZEREL
            CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /MP")
    if (USE_DEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Od")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Ob2 /Oi /Ot /Oy")
    endif ()
else ()
    if (NOT BUILD_STATIC_LIB)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    endif ()
    if (NOT USE_DEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
    endif ()
endif ()

set(LightGBM_HEADER_DIR ${PROJECT_SOURCE_DIR}/include)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

include_directories(${LightGBM_HEADER_DIR})

if (USE_MPI)
    include_directories(${MPI_CXX_INCLUDE_PATH})
endif ()

file(
        GLOB
        SOURCES
        src/boosting/*.cpp
        src/io/*.cpp
        src/metric/*.cpp
        src/objective/*.cpp
        src/network/*.cpp
        src/treelearner/*.cpp
        if (USE_CUDA OR USE_CUDA_EXP)
        src/treelearner/*.cu
        endif ()
        if (USE_CUDA_EXP)
        src/boosting/cuda/*.cpp
        src/boosting/cuda/*.cu
        src/treelearner/cuda/*.cpp
        src/treelearner/cuda/*.cu
        src/io/cuda/*.cu
        src/io/cuda/*.cpp
        src/cuda/*.cpp
        src/cuda/*.cu
        endif ()
)

add_library(lightgbm_objs OBJECT ${SOURCES})

add_executable(lightgbm src/main.cpp src/application/application.cpp)
target_link_libraries(lightgbm PRIVATE lightgbm_objs)

set(API_SOURCES "src/c_api.cpp")
# Only build the R part of the library if building for
# use with the R package
if (__BUILD_FOR_R)
    list(APPEND API_SOURCES "src/lightgbm_R.cpp")
endif ()

add_library(lightgbm_capi_objs OBJECT ${API_SOURCES})

if (BUILD_STATIC_LIB)
    add_library(_lightgbm STATIC)
else ()
    add_library(_lightgbm SHARED)
endif ()
# LightGBM headers include openmp, cuda, R etc. headers,
# thus PUBLIC is required for building _lightgbm_swig target.
target_link_libraries(_lightgbm PUBLIC lightgbm_capi_objs lightgbm_objs)

if (MSVC)
    set_target_properties(_lightgbm PROPERTIES OUTPUT_NAME "lib_lightgbm")
endif ()


if (USE_MPI)
    target_link_libraries(lightgbm_objs PUBLIC ${MPI_CXX_LIBRARIES})
endif ()

if (USE_OPENMP)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        target_link_libraries(lightgbm_objs PUBLIC OpenMP::OpenMP_CXX)
        # c_api headers also includes OpenMP headers, thus compiling
        # lightgbm_capi_objs needs include directory for OpenMP.
        # Specifying OpenMP in target_link_libraries will get include directory
        # requirements for compilation.
        # This uses CMake's Transitive Usage Requirements. Refer to CMake doc:
        # https://cmake.org/cmake/help/v3.16/manual/cmake-buildsystem.7.html#transitive-usage-requirements
        target_link_libraries(lightgbm_capi_objs PUBLIC OpenMP::OpenMP_CXX)
    endif ()
endif ()


if (__INTEGRATE_OPENCL)
    # targets OpenCL and Boost are added in IntegratedOpenCL.cmake
    add_dependencies(lightgbm_objs OpenCL Boost)
    # variables INTEGRATED_OPENCL_* are set in IntegratedOpenCL.cmake
    target_include_directories(lightgbm_objs PRIVATE ${INTEGRATED_OPENCL_INCLUDES})
    target_compile_definitions(lightgbm_objs PRIVATE ${INTEGRATED_OPENCL_DEFINITIONS})
    target_link_libraries(lightgbm_objs PUBLIC ${INTEGRATED_OPENCL_LIBRARIES})
endif ()

if (USE_HDFS)
    target_link_libraries(lightgbm_objs PUBLIC ${HDFS_CXX_LIBRARIES})
endif ()

#-- Google C++ tests
if (BUILD_CPP_TEST)
    find_package(GTest CONFIG)
    if (NOT GTEST_FOUND)
        message(STATUS "Did not find Google Test in the system root. Fetching Google Test now...")
        include(FetchContent)
        FetchContent_Declare(
                googletest
                GIT_REPOSITORY https://github.com/google/googletest.git
                GIT_TAG release-1.11.0
        )
        FetchContent_MakeAvailable(googletest)
        add_library(GTest::GTest ALIAS gtest)
    endif ()

    set(LightGBM_TEST_HEADER_DIR ${PROJECT_SOURCE_DIR}/tests/cpp_tests)
    include_directories(${LightGBM_TEST_HEADER_DIR})

    file(GLOB CPP_TEST_SOURCES tests/cpp_tests/*.cpp)
    if (MSVC)
        set(
                CompilerFlags
                CMAKE_CXX_FLAGS
                CMAKE_CXX_FLAGS_DEBUG
                CMAKE_CXX_FLAGS_RELEASE
                CMAKE_C_FLAGS
                CMAKE_C_FLAGS_DEBUG
                CMAKE_C_FLAGS_RELEASE
        )
        foreach (CompilerFlag ${CompilerFlags})
            string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
        endforeach ()
    endif ()
    add_executable(testlightgbm ${CPP_TEST_SOURCES})
    target_link_libraries(testlightgbm PRIVATE lightgbm_objs lightgbm_capi_objs GTest::GTest)
endif ()

install(
        TARGETS lightgbm _lightgbm
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

install(DIRECTORY ${LightGBM_HEADER_DIR}/LightGBM DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
