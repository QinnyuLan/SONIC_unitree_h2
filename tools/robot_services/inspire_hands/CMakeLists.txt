cmake_minimum_required(VERSION 3.16)
project(sonic_inspire_hands LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
option(BUILD_TESTING "Build and register hardware-free hand service checks" OFF)
set(UNITREE_SDK_ROOT "" CACHE PATH "Installed SDK root; empty uses the bundled SDK")
get_filename_component(SONIC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

if(UNITREE_SDK_ROOT)
  foreach(required include/unitree/robot/channel/channel_factory.hpp
      lib/libunitree_sdk2.a lib/libddsc.so lib/libddscxx.so)
    if(NOT EXISTS "${UNITREE_SDK_ROOT}/${required}")
      message(FATAL_ERROR "Missing SDK file: ${UNITREE_SDK_ROOT}/${required}")
    endif()
  endforeach()
  add_library(unitree_sdk2 STATIC IMPORTED)
  set_target_properties(unitree_sdk2 PROPERTIES
    IMPORTED_LOCATION "${UNITREE_SDK_ROOT}/lib/libunitree_sdk2.a"
    INTERFACE_INCLUDE_DIRECTORIES "${UNITREE_SDK_ROOT}/include;${UNITREE_SDK_ROOT}/include/ddscxx"
    INTERFACE_LINK_LIBRARIES "${UNITREE_SDK_ROOT}/lib/libddscxx.so;${UNITREE_SDK_ROOT}/lib/libddsc.so;Threads::Threads")
  set(DDS_LIB_DIR "${UNITREE_SDK_ROOT}/lib")
else()
  set(SDK_SOURCE "${SONIC_ROOT}/src/runtime/third_party/unitree_sdk2")
  add_subdirectory("${SDK_SOURCE}" "${CMAKE_BINARY_DIR}/unitree_sdk2" EXCLUDE_FROM_ALL)
  set(DDS_LIB_DIR "${SDK_SOURCE}/thirdparty/lib/${CMAKE_SYSTEM_PROCESSOR}")
endif()

add_executable(inspire_hand_service src/main.cpp)
target_include_directories(inspire_hand_service PRIVATE include)
target_link_libraries(inspire_hand_service PRIVATE unitree_sdk2 Eigen3::Eigen Boost::program_options)
add_executable(inspire_hand_client src/hand_client.cpp)
target_link_libraries(inspire_hand_client PRIVATE unitree_sdk2 Boost::program_options)

install(TARGETS inspire_hand_service inspire_hand_client RUNTIME DESTINATION bin)
# Keep the selected SDK's DDS libraries beside the installed service, including
# SONAME symlinks when an on-robot SDK provides versioned shared objects.
file(GLOB DDS_LIBRARIES "${DDS_LIB_DIR}/libddsc.so*" "${DDS_LIB_DIR}/libddscxx.so*")
install(FILES ${DDS_LIBRARIES} DESTINATION lib)
install(FILES "${SONIC_ROOT}/licenses/inspire_hands/LICENSE"
  DESTINATION share/licenses/inspire_hands)
install(FILES "${SONIC_ROOT}/src/runtime/third_party/unitree_sdk2/LICENSE"
  DESTINATION share/licenses/unitree_sdk2)

if(BUILD_TESTING)
  enable_testing()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(HAND_TEST_DIR "${SONIC_ROOT}/tests/tools/robot_services/inspire_hands")
  add_library(stall_serial MODULE "${HAND_TEST_DIR}/stall_serial.cpp")
  target_link_libraries(stall_serial PRIVATE ${CMAKE_DL_LIBS})
  add_test(NAME inspire_probe COMMAND "${Python3_EXECUTABLE}"
    "${HAND_TEST_DIR}/test_native485_probe.py" $<TARGET_FILE:inspire_hand_service>)
  add_test(NAME inspire_dds COMMAND "${Python3_EXECUTABLE}"
    "${HAND_TEST_DIR}/test_native485_dds.py" $<TARGET_FILE:inspire_hand_service>
    $<TARGET_FILE:inspire_hand_client> --network lo)
  add_test(NAME inspire_watchdog COMMAND "${Python3_EXECUTABLE}"
    "${HAND_TEST_DIR}/test_watchdog.py" $<TARGET_FILE:inspire_hand_service>
    $<TARGET_FILE:stall_serial>)
  set_tests_properties(inspire_probe inspire_dds inspire_watchdog PROPERTIES TIMEOUT 60)
endif()
