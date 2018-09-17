cmake_minimum_required(VERSION 3.10)
project(pfp_growth LANGUAGES CXX CUDA)
find_package(CUDA 9 REQUIRED)

set(CTest)
include_directories(${CUDA_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/include")
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/unit-test)
set(CMAKE_C_COMPILER /usr/bin/gcc-7 CACHE PATH "" FORCE)
set(CMAKE_CXX_COMPILER /usr/bin/g++-7 CACHE PATH "" FORCE)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -shared -gencode arch=compute_50,code=sm_50 -lineinfo")
set(CMAKE_C_FLAGS_DEBUG "-D_DEBUG")
add_definitions(-DUNIT_TEST)

add_library(gpu_library_fp_growth STATIC
        include/cudaHeaders.h
#        include/Kernel.h
#        src/Kernel.cu
        include/PFPGrowth.cu.h
        src/PFPGrowth.cu
        )

add_library(cpu_library_fp_growth STATIC
        include/PFPArray.h
        include/PFPTree.h
        src/PFPArray.cpp
        src/PFPTree.cpp
        )


# Prepare "Catch" library for other executables
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

target_compile_features(cpu_library_fp_growth PUBLIC cxx_std_11)

set_target_properties( cpu_library_fp_growth
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(fp_cuda main.cu include/Kernel.h src/Kernel.cu unit-test/catch.hpp)

set_property(TARGET fp_cuda
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries( fp_cuda cpu_library_fp_growth gpu_library_fp_growth Catch)


if(APPLE)
    # We need to add the path to the driver (libcuda.dylib) as an rpath,
    # so that the static cuda runtime can find it at runtime.
    set_property(TARGET fp_cuda
            PROPERTY
            BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
