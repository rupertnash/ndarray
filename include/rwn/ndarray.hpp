// -*- mode: C++; -*-
#ifndef RWN_NDARRAY_H
#define RWN_NDARRAY_H

// Copyright 2020 Rupert Nash, EPCC, University of Edinburgh

#include <array>
#include <memory>
#include <numeric>

namespace rwn {

// Simple multidimensional array class
// - Contained type must be trivial
// - Initial contents uninitialised
// - Uniquely owns its data
// - Element access via operator()
template <typename T, int N> class ndarray {
public:
  static constexpr int NDIMS = N;
  using value_type = T;
  using index_type = std::array<int, NDIMS>;

private:
  // Ensure value initialised to zero
  index_type m_dims = {};
  index_type m_strides = {};
  int m_size = 0;
  std::unique_ptr<T[]> m_data;

  // Set m_dims then call this to complete construction
  void init() {
    m_strides[NDIMS - 1] = 1;
    for (int i = int(NDIMS) - 2; i >= 0; --i) {
      m_strides[i] = m_strides[i + 1] * m_dims[i + 1];
    }
    m_size = m_strides[0] * m_dims[0];
    m_data = std::unique_ptr<T[]>(new T[m_size]);
  }

public:
  ndarray() = default;
  explicit ndarray(const index_type &xs) : m_dims{xs} { init(); }
  ndarray(const index_type &xs, const T &val) : ndarray(xs) {
    std::fill(m_data.get(), m_data.get() + m_size, val);
  }

  ndarray(ndarray &&) noexcept = default;
  ndarray &operator=(ndarray &&) = default;

  ndarray(const ndarray &src)
      : m_dims(src.m_dims), m_strides(src.m_strides), m_size(src.m_size),
        m_data(new T[src.m_size]) {
    std::copy(src.m_data.get(), src.m_data.get() + src.m_size, m_data.get());
  }

  ndarray &operator=(const ndarray &src) {
    if (m_size != src.m_size) {
      // If sizes differ, copy construct and swap
      ndarray tmp{src};
      std::swap(*this, tmp);
    } else {
      // Same size, don't 
      for (int i = 0; i < NDIMS; ++i) {
	m_dims[i] = src.m_dims[i];
	m_strides[i] = src.m_strides[i];
      }
      m_size = src.m_size;
      std::copy(src.m_data.get(), src.m_data.get() + src.m_size, m_data.get());
    }
    return *this;
  }

  int const& size() const { return m_size; }
  index_type const& shape() const { return m_dims; }
  index_type const& strides() const { return m_strides; }

  int n2one(const index_type& idx) const {
    return std::inner_product(m_strides.begin(), m_strides.end(), idx.begin(), 0);
  }
  index_type one2n(int ijk) const {
    index_type ans;
    // Assumes C-style array layout
    for (int i = 0; i < NDIMS; ++i) {
      ans[i] = ijk / m_strides[i];
      ijk %= m_strides[i];
    }
    return ans;
  }

  // Mutable element access
  template <typename... Ints>
  T& operator()(Ints... inds) {
    return m_data[n2one(index_type{static_cast<int>(inds)...})];
  }
  // Const element access
  template <typename... Ints>
  T const& operator()(Ints... inds) const {
    return m_data[n2one(index_type{static_cast<int>(inds)...})];
  }

  // Mutable raw data access
  T* data() {
    return m_data.get();
  }
  // Const raw data access
  T const* data() const {
    return m_data.get();
  }

  // Return a copy
  ndarray clone() const {
    auto ans = ndarray{m_dims};
    std::copy(m_data.get(), m_data.get() + m_size, ans.get());
    return ans;
  }
};

}

#endif
