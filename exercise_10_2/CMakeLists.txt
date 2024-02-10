cmake_minimum_required(VERSION 3.11)
include(FetchContent)
project(kokkos_matmul CXX)
FetchContent_Declare(kokkos
GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
)
FetchContent_MakeAvailable(kokkos)
add_executable(matmul matmul.cpp)
add_dependencies(matmul kokkos)
target_include_directories(matmul PRIVATE ${Kokkos_INCLUDE_DIRS_RET})
target_link_libraries(matmul PRIVATE kokkos)

FetchContent_Declare(kokkos-kernels
GIT_REPOSITORY "https://github.com/kokkos/kokkos-kernels.git"
)
FetchContent_MakeAvailable(kokkos-kernels)
target_link_libraries(matmul PRIVATE Kokkos::kokkoskernels)