cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Name of your project
project(MyProject)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable profiling option
option(ENABLE_PROFILING "Enable detailed profiling output" OFF)

# BookSim2 build - using a custom target with its own Makefile
# Force C++14 for booksim2 build since it uses deprecated 'register' keyword
add_custom_target(booksim2 ALL
  COMMAND $(MAKE) CPPFLAGS="-Wall -I. -Iarbiters -Iallocators -Irouters -Inetworks -Ipower -O3 -g -std=c++14"
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/booksim2/src
  COMMENT "Building BookSim2 using its Makefile with C++14"
)

# Optional: copy booksim2 executable to our build directory
add_custom_command(
  TARGET booksim2
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/external/booksim2/src/booksim
    ${CMAKE_BINARY_DIR}/booksim
  COMMENT "Copying BookSim2 executable to build directory"
)

# Enable OpenMP
find_package(OpenMP REQUIRED)
add_library(digraph
  src/digraph.cc
  src/matrix.cc
  src/routing_optimizer.cc
)
target_link_libraries(digraph PRIVATE
  OpenMP::OpenMP_CXX
)
target_include_directories(digraph PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

# Add your main executable
add_executable(my_executable 
  src/main.cc
)

add_dependencies(my_executable booksim2)

target_link_libraries(my_executable PRIVATE
  OpenMP::OpenMP_CXX
  digraph
)
# Include directories
target_include_directories(my_executable PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/inc
  ${CMAKE_CURRENT_SOURCE_DIR}/external/booksim2/src
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -save-temps")

# Add profiling definitions if enabled
if(ENABLE_PROFILING)
  add_definitions(-DENABLE_PROFILING)
  message(STATUS "Profiling is enabled")
endif()


# Google Test setup
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Add the test executable
add_executable(unit_tests 
  tests/unit_tests.cc
  src/digraph.cc
)
target_link_libraries(unit_tests PRIVATE
  GTest::gtest_main
  OpenMP::OpenMP_CXX
  digraph
)
target_include_directories(unit_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/inc
  ${CMAKE_CURRENT_SOURCE_DIR}/external/booksim2/src
)

# Register tests
include(GoogleTest)
gtest_discover_tests(unit_tests)
