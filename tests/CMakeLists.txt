include(CTest)

find_package(Catch2 QUIET)
if (NOT Catch2_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v2.13.1)

  FetchContent_MakeAvailable(Catch2)
endif()

enable_testing()

add_executable(tests test.cpp)
target_link_libraries(
  tests rwn::ndarray Catch2::Catch2
  )

add_test(
  NAME "ndarray tests"
  COMMAND tests
  )


