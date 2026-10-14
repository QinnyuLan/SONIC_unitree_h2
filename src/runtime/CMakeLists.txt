cmake_minimum_required(VERSION 3.18)
project(SONIC_unitree_h2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(SONIC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
option(BUILD_TESTING "Build offline runtime and numerical regression tests" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(CUDAToolkit 12 REQUIRED)
find_package(TensorRT 10 REQUIRED COMPONENTS nvinfer nvonnxparser)
find_package(Eigen3 REQUIRED)
find_path(ZMQ_INCLUDE_DIR zmq.hpp REQUIRED)
find_library(ZMQ_LIBRARY zmq REQUIRED)
find_path(JSON_INCLUDE_DIR nlohmann/json.hpp REQUIRED)

add_subdirectory(third_party/unitree_sdk2)

add_library(sonic_runtime_headers INTERFACE)
target_include_directories(sonic_runtime_headers INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${ZMQ_INCLUDE_DIR}" "${JSON_INCLUDE_DIR}")
target_link_libraries(sonic_runtime_headers INTERFACE Eigen3::Eigen ${ZMQ_LIBRARY})

add_library(sonic_tensorrt STATIC src/inference/InferenceEngine.cpp)
target_include_directories(sonic_tensorrt PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(sonic_tensorrt PUBLIC CUDA::cudart trt::nvinfer trt::nvonnxparser)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(sonic_tensorrt PRIVATE -Wno-deprecated-declarations)
endif()

add_executable(sonic_h2_runtime src/main.cpp)
target_link_libraries(sonic_h2_runtime PRIVATE
  sonic_runtime_headers sonic_tensorrt unitree_sdk2)

if(BUILD_TESTING)
  enable_testing()
  find_package(GTest REQUIRED)
  find_package(ZLIB REQUIRED)
  # CSV/NPZ readers and FK are retained for regression fixtures only. The
  # executable has no motion-library or observation-YAML dependency.
  add_library(sonic_test_support STATIC tests/support/cnpy.cpp tests/support/fk.cpp)
  target_include_directories(sonic_test_support PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/tests/support")
  target_link_libraries(sonic_test_support PUBLIC sonic_runtime_headers ZLIB::ZLIB)

  file(GLOB CONTROLLER_TESTS CONFIGURE_DEPENDS tests/controller/test_*.cpp)
  add_executable(controller_tests ${CONTROLLER_TESTS})
  target_link_libraries(controller_tests PRIVATE
    GTest::GTest GTest::Main sonic_test_support unitree_sdk2)
  add_test(NAME controller_tests COMMAND controller_tests)

  add_executable(runtime_tests tests/test_runtime.cpp)
  target_link_libraries(runtime_tests PRIVATE GTest::GTest GTest::Main sonic_runtime_headers)
  add_test(NAME runtime_tests COMMAND runtime_tests)
  set_tests_properties(controller_tests runtime_tests PROPERTIES
    WORKING_DIRECTORY "${SONIC_ROOT}")
endif()
