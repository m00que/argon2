cmake_minimum_required(VERSION 3.7)

project(argon2-gpu CXX)
set(BINARY_INSTALL_DIR /usr/local/bin)
set(LIBRARY_INSTALL_DIR /usr/local/lib)
set(INCLUDE_INSTALL_DIR /usr/local/include)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

if(WIN32 AND NOT MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-multiple-definition")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-multiple-definition")
endif()

if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

set(CUDA_FOUND FALSE)

if(NOT NO_CUDA)
    find_package(CUDA)
endif()

if(CUDA_FOUND)
    message("INFO: Using CUDA version ${CUDA_VERSION}")
    add_definitions(-DHAVE_CUDA=1)
else()
    message("INFO: Building without CUDA support")
    add_definitions(-DHAVE_CUDA=0)
endif()

# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# Fermi†    Kepler† Maxwell‡	Pascal	Volta	Turing	Ampere	Ada (Lovelace)	Hopper
# sm_20     sm_30	sm_50	    sm_60	sm_70	sm_75	sm_80	sm_89	        sm_90
# 	        sm_35	sm_52	    sm_61	sm_72
#                                       (Xavier)		sm_86		            sm_90a (Thor)
#	        sm_37	sm_53	    sm_62			        sm_87 (Orin)
#
# 1080TI -> sm_60
# V100   -> sm_70
# 3080TI -> sm_80
#
if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -O3;--ptxas-options=-v;-arch sm_60
        ;-lineinfo;-allow-unsupported-compiler
        ;-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
    )
    if(NOT MSVC)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)
    endif()
endif()

if(NOT OPENCL_ROOT)
    if(CUDA_FOUND)
        set(OPENCL_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
    else()
        set(OPENCL_ROOT "F:/software/cuda")
    endif()
endif()
find_library(OpenCL_LIBRARY NAMES OpenCL
    PATHS
        ${OPENCL_ROOT}/lib/x64
        ${OPENCL_ROOT}/lib64
        /usr/lib
        /usr/local/lib
    NO_DEFAULT_PATH
)
if(NOT OpenCL_LIBRARY)
    set(OpenCL_LIBRARY "-lOpenCL")
endif()
find_path(OpenCL_INCLUDE_DIR NAMES CL/cl.h
    PATHS
        ${OPENCL_ROOT}/include
        /usr/include
        /usr/local/include
    NO_DEFAULT_PATH
)
message("INFO: OpenCL library: ${OpenCL_LIBRARY}")
message("INFO: OpenCL include: ${OpenCL_INCLUDE_DIR}")

add_subdirectory(ext/argon2)

add_library(argon2-gpu-common SHARED
    lib/argon2-gpu-common/argon2params.cpp
    lib/argon2-gpu-common/blake2b.cpp
)
target_include_directories(argon2-gpu-common INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(argon2-gpu-common PRIVATE
    include/argon2-gpu-common
    lib/argon2-gpu-common
)

if(CUDA_FOUND)
    cuda_add_library(argon2-cuda SHARED
        lib/argon2-cuda/cudaexception.cpp
        lib/argon2-cuda/device.cpp
        lib/argon2-cuda/globalcontext.cpp
        lib/argon2-cuda/programcontext.cpp
        lib/argon2-cuda/processingunit.cpp
        lib/argon2-cuda/kernelrunner.cu
    )
else()
    add_library(argon2-cuda SHARED
        lib/argon2-cuda/nocuda.cpp
    )
endif()

target_include_directories(argon2-cuda PRIVATE
    include/argon2-cuda
    lib/argon2-cuda
)
target_include_directories(argon2-cuda INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(argon2-cuda argon2-gpu-common)

add_library(argon2-opencl SHARED
    lib/argon2-opencl/device.cpp
    lib/argon2-opencl/globalcontext.cpp
    lib/argon2-opencl/kernelloader.cpp
    lib/argon2-opencl/programcontext.cpp
    lib/argon2-opencl/processingunit.cpp
    lib/argon2-opencl/kernelrunner.cpp
)
target_include_directories(argon2-opencl INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCL_INCLUDE_DIR}
)
target_include_directories(argon2-opencl PRIVATE
    include/argon2-opencl
    lib/argon2-opencl
    ${OpenCL_INCLUDE_DIR}
)
target_link_libraries(argon2-opencl
    argon2-gpu-common ${OpenCL_LIBRARY}
)

add_executable(argon2-gpu-test
    src/argon2-gpu-test/main.cpp
    src/argon2-gpu-test/testcase.cpp
)
target_include_directories(argon2-gpu-test PRIVATE src/argon2-gpu-test)
target_link_libraries(argon2-gpu-test
    argon2-cuda argon2-opencl argon2 ${OpenCL_LIBRARY}
)

add_executable(argon2-kraken
    src/argon2-kraken/main.cpp
    src/argon2-kraken/base64.cpp
    src/argon2-kraken/hash_parser.cpp
    src/argon2-kraken/strings_tools.cpp
)

add_library(kraken SHARED
    src/argon2-kraken/main.cpp
    src/argon2-kraken/base64.cpp
    src/argon2-kraken/hash_parser.cpp
    src/argon2-kraken/strings_tools.cpp
)
target_include_directories(kraken PRIVATE src/argon2-kraken)
target_link_libraries(kraken
    argon2-cuda argon2-opencl argon2 ${OpenCL_LIBRARY}
)

target_include_directories(argon2-kraken PRIVATE src/argon2-kraken)
target_link_libraries(argon2-kraken
    argon2-cuda argon2-opencl argon2 ${OpenCL_LIBRARY}
)

add_executable(argon2-gpu-bench
    src/argon2-gpu-bench/cpuexecutive.cpp
    src/argon2-gpu-bench/cudaexecutive.cpp
    src/argon2-gpu-bench/openclexecutive.cpp
    src/argon2-gpu-bench/benchmark.cpp
    src/argon2-gpu-bench/main.cpp
)
target_include_directories(argon2-gpu-bench PRIVATE src/argon2-gpu-bench)
target_link_libraries(argon2-gpu-bench
    argon2-cuda argon2-opencl argon2 ${OpenCL_LIBRARY}
)

add_test(argon2-gpu-test-opencl argon2-gpu-test -m opencl)
add_test(argon2-gpu-test-cuda argon2-gpu-test -m cuda)

install(
    TARGETS argon2-gpu-common argon2-opencl argon2-cuda
    DESTINATION ${LIBRARY_INSTALL_DIR}
)
install(FILES
    include/argon2-gpu-common/argon2-common.h
    include/argon2-gpu-common/argon2params.h
    include/argon2-opencl/cl.hpp
    include/argon2-opencl/opencl.h
    include/argon2-opencl/device.h
    include/argon2-opencl/globalcontext.h
    include/argon2-opencl/programcontext.h
    include/argon2-opencl/processingunit.h
    include/argon2-opencl/kernelrunner.h
    include/argon2-cuda/cudaexception.h
    include/argon2-cuda/kernelrunner.h
    include/argon2-cuda/device.h
    include/argon2-cuda/globalcontext.h
    include/argon2-cuda/programcontext.h
    include/argon2-cuda/processingunit.h
    DESTINATION ${INCLUDE_INSTALL_DIR}
)
install(
    TARGETS argon2-gpu-bench argon2-gpu-test argon2-kraken
    DESTINATION ${BINARY_INSTALL_DIR}
)
