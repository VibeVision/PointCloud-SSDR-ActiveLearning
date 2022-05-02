


#ifndef NANOFLANN_HPP_
#define NANOFLANN_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>   // for abs()
#include <cstdio>  // for fwrite()
#include <cstdlib> // for abs()
#include <functional>
#include <limits> // std::reference_wrapper
#include <stdexcept>
#include <vector>

/** Library version: 0xMmP (M=Major,m=minor,P=patch) */
#define NANOFLANN_VERSION 0x130

// Avoid conflicting declaration of min/max macros in windows headers
#if !defined(NOMINMAX) &&                                                      \
    (defined(_WIN32) || defined(_WIN32_) || defined(WIN32) || defined(_WIN64))
#define NOMINMAX
#ifdef max
#undef max
#undef min
#endif
#endif

namespace nanoflann {
/** @addtogroup nanoflann_grp nanoflann C++ library for ANN
 *  @{ */

/** the PI constant (required to avoid MSVC missing symbols) */
template <typename T> T pi_const() {
  return static_cast<T>(3.14159265358979323846);
}

/**
 * Traits if object is resizable and assignable (typically has a resize | assign
 * method)
 */
template <typename T, typename = int> struct has_resize : std::false_type {};

template <typename T>
struct has_resize<T, decltype((void)std::declval<T>().resize(1), 0)>
    : std::true_type {};

template <typename T, typename = int> struct has_assign : std::false_type {};

template <typename T>
struct has_assign<T, decltype((void)std::declval<T>().assign(1, 0), 0)>
    : std::true_type {};

/**
 * Free function to resize a resizable object
 */
template <typename Container>
inline typename std::enable_if<has_resize<Container>::value, void>::type
resize(Container &c, const size_t nElements) {
  c.resize(nElements);
}

/**
 * Free function that has no effects on non resizable containers (e.g.
 * std::array) It raises an exception if the expected size does not match
 */
template <typename Container>
inline typename std::enable_if<!has_resize<Container>::value, void>::type
resize(Container &c, const size_t nElements) {
  if (nElements != c.size())
    throw std::logic_error("Try to change the size of a std::array.");
}

/**
 * Free function to assign to a container
 */
template <typename Container, typename T>
inline typename std::enable_if<has_assign<Container>::value, void>::type
assign(Container &c, const size_t nElements, const T &value) {
  c.assign(nElements, value);
}

/**
 * Free function to assign to a std::array
 */
template <typename Container, typename T>
inline typename std::enable_if<!has_assign<Container>::value, void>::type
assign(Container &c, const size_t nElements, const T &value) {
  for (size_t i = 0; i < nElements; i++)
    c[i] = value;
}

/** @addtogroup result_sets_grp Result set classes
 *  @{ */
template <typename _DistanceType, typename _IndexType = size_t,
          typename _CountType = size_t>
class KNNResultSet {
public:
  typedef _DistanceType DistanceType;
  typedef _IndexType IndexType;
  typedef _CountType CountType;

private:
  IndexType *indices;
  DistanceType *dists;
  CountType capacity;
  CountType count;

public:
  inline KNNResultSet(CountType capacity_)
      : indices(0), dists(0), capacity(capacity_), count(0) {}

  inline void init(IndexType *indices_, DistanceType *dists_) {
    indices = indices_;
    dists = dists_;
    count = 0;
    if (capacity)
      dists[capacity - 1] = (std::numeric_limits<DistanceType>::max)();
  }

  inline CountType size() const { return count; }

  inline bool full() const { return count == capacity; }

  /**
   * Called during search to add an element matching the criteria.
   * @return true if the search should be continued, false if the results are
   * sufficient
   */
  inline bool addPoint(DistanceType dist, IndexType index) {
    CountType i;
    for (i = count; i > 0; --i) {
#ifdef NANOFLANN_FIRST_MATCH // If defined and two points have the same
                             // distance, the one with the lowest-index will be
                             // returned first.
      if ((dists[i - 1] > dist) ||
          ((dist == dists[i - 1]) && (indices[i - 1] > index))) {
#else
      if (dists[i - 1] > dist) {
#endif
        if (i < capacity) {
          dists[i] = dists[i - 1];
          indices[i] = indices[i - 1];
        }
      } else
        break;
    }
    if (i < capacity) {
      dists[i] = dist;
      indices[i] = index;
    }
    if (count < capacity)
      count++;

    // tell caller that the search shall continue
    return true;
  }

  inline DistanceType worstDist() const { return dists[capacity - 1]; }
};

/** operator "<" for std::sort() */
struct IndexDist_Sorter {
  /** PairType will be typically: std::pair<IndexType,DistanceType> */
  template <typename PairType>
  inline bool operator()(const PairType &p1, const PairType &p2) const {
    return p1.second < p2.second;
  }
};

/**
 * A result-set class used when performing a radius based search.
 */
template <typename _DistanceType, typename _IndexType = size_t>
class RadiusResultSet {
public:
  typedef _DistanceType DistanceType;
  typedef _IndexType IndexType;

public:
  const DistanceType radius;

  std::vector<std::pair<IndexType, DistanceType>> &m_indices_dists;

  inline RadiusResultSet(
      DistanceType radius_,
      std::vector<std::pair<IndexType, DistanceType>> &indices_dists)
      : radius(radius_), m_indices_dists(indices_dists) {
    init();
  }

  inline void init() { clear(); }
  inline void clear() { m_indices_dists.clear(); }

  inline size_t size() const { return m_indices_dists.size(); }

  inline bool full() const { return true; }

  /**
   * Called during search to add an element matching the criteria.
   * @return true if the search should be continued, false if the results are
   * sufficient
   */
  inline bool addPoint(DistanceType dist, IndexType index) {
    if (dist < radius)
      m_indices_dists.push_back(std::make_pair(index, dist));
    return true;
  }

  inline DistanceType worstDist() const { return radius; }

  /**
   * Find the worst result (furtherest neighbor) without copying or sorting
   * Pre-conditions: size() > 0
   */
  std::pair<IndexType, DistanceType> worst_item() const {
    if (m_indices_dists.empty())
      throw std::runtime_error("Cannot invoke RadiusResultSet::worst_item() on "
                               "an empty list of results.");
    typedef
        typename std::vector<std::pair<IndexType, DistanceType>>::const_iterator
            DistIt;
    DistIt it = std::max_element(m_indices_dists.begin(), m_indices_dists.end(),
                                 IndexDist_Sorter());
    return *it;
  }
};

/** @} */

/** @addtogroup loadsave_grp Load/save auxiliary functions
 * @{ */
template <typename T>
void save_value(FILE *stream, const T &value, size_t count = 1) {
  fwrite(&value, sizeof(value), count, stream);
}

template <typename T>
void save_value(FILE *stream, const std::vector<T> &value) {
  size_t size = value.size();
  fwrite(&size, sizeof(size_t), 1, stream);
  fwrite(&value[0], sizeof(T), size, stream);
}

template <typename T>
void load_value(FILE *stream, T &value, size_t count = 1) {