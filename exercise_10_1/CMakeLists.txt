cmake_minimum_required(VERSION 3.11)
include(FetchContent)
project(kokkos_vecdot CXX)
FetchContent_Declare(kokkos
GIT_REPOSITORY "https://github.com/kokkos/kokkos.git"
)
FetchContent_MakeAvailable(kokkos)
add_executable(vecdot vecdot.cpp)
add_dependencies(vecdot kokkos)
target_include_directories(vecdot PRIVATE ${Kokkos_INCLUDE_DIRS_RET})
target_link_libraries(vecdot PRIVATE kokkos)


FetchContent_Declare(kokkos-kernels
GIT_REPOSITORY "https://github.com/kokkos/kokkos-kernels.git"
)
FetchContent_MakeAvailable(kokkos-kernels)
target_link_libraries(vecdot PRIVATE Kokkos::kokkoskernels)
