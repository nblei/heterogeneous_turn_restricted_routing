cmake_minimum_required(VERSION 3.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Name of your project
project(MyProject)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add your main executable
add_executable(my_executable src/main.cc)

# Enable OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(my_executable PRIVATE OpenMP::OpenMP_CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -save-temps")

# Include directories
target_include_directories(my_executable PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

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
add_executable(unit_tests tests/unit_tests.cc)
target_link_libraries(unit_tests PRIVATE
  GTest::gtest_main
  OpenMP::OpenMP_CXX
)
target_include_directories(unit_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

# Register tests
include(GoogleTest)
gtest_discover_tests(unit_tests)
