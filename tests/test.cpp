#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <rwn/ndarray.hpp>

using rwn::ndarray;

TEMPLATE_TEST_CASE_SIG("Construction", "[ctor]",
		       ((int N), N),
		       (1), (2), (3), (4)) {
  SECTION("Default constructed arrays are valid but empty") {
    ndarray<int, N> empty;
    REQUIRE(empty.size() == 0);

    for (int i = 0; i < N; ++i) {
      REQUIRE(empty.shape()[i] == 0);
      REQUIRE(empty.strides()[i] == 0);
    }

    REQUIRE(empty.data() == nullptr);
  }

  SECTION("Default constructed const arrays are valid but empty") {
    ndarray<int, N> const empty;
    REQUIRE(empty.size() == 0);

    for (int i = 0; i < N; ++i) {
      REQUIRE(empty.shape()[i] == 0);
      REQUIRE(empty.strides()[i] == 0);
    }

    REQUIRE(empty.data() == nullptr);
  }

}
