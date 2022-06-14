

#ifndef  NANOFLANN_HPP_
#define  NANOFLANN_HPP_

#include <vector>
#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <cstdio>  // for fwrite()
#define _USE_MATH_DEFINES // Required by MSVC to define M_PI,etc. in <cmath>
#include <cmath>   // for abs()
#include <cstdlib> // for abs()
#include <limits>

// Avoid conflicting declaration of min/max macros in windows headers
#if !defined(NOMINMAX) && (defined(_WIN32) || defined(_WIN32_)  || defined(WIN32) || defined(_WIN64))
# define NOMINMAX
# ifdef max
#  undef   max
#  undef   min
# endif
#endif

namespace nanoflann
{
/** @addtogroup nanoflann_grp nanoflann C++ library for ANN
  *  @{ */

  	/** Library version: 0xMmP (M=Major,m=minor,P=patch) */
	#define NANOFLANN_VERSION 0x123

	/** @addtogroup result_sets_grp Result set classes
	  *  @{ */
	template <typename DistanceType, typename IndexType = size_t, typename CountType = size_t>
	class KNNResultSet
	{
		IndexType * indices;
		DistanceType* dists;
		CountType capacity;
		CountType count;

	public:
		inline KNNResultSet(CountType capacity_) : indices(0), dists(0), capacity(capacity_), count(0)
		{
		}

		inline void init(IndexType* indices_, DistanceType* dists_)
		{
			indices = indices_;
			dists = dists_;
			count = 0;
            if (capacity)
                dists[capacity-1] = (std::numeric_limits<DistanceType>::max)();
		}

		inline CountType size() const
		{
			return count;
		}

		inline bool full() const
		{
			return count == capacity;
		}


                /**
                 * Called during search to add an e