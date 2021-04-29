cmake_minimum_required(VERSION 3.12)

project(rwn_ndarray VERSION 0.1.1 LANGUAGES CXX)

include(GNUInstallDirs)

option(NDARRAY_TESTS "Build tests" OFF)

add_library(ndarray INTERFACE)
add_library(rwn::ndarray ALIAS ndarray)

target_compile_features(ndarray INTERFACE cxx_std_11)
target_include_directories(
  ndarray
  INTERFACE
  $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if (NDARRAY_TESTS)
  add_subdirectory(tests)
endif()

# Do CMake pacakge basics
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
  )

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
  )

# Install bits
install(
  TARGETS ndarray
  EXPORT ${PROJECT_NAME}_Targets
  )

install(
  EXPORT ${PROJECT_NAME}_Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE rwn::
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
  )

install(
  FILES
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
  )

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
