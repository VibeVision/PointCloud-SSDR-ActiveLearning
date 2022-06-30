

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
                 * Called during search to add an element matching the criteria.
                 * @return true if the search should be continued, false if the results are sufficient
                 */
                inline bool addPoint(DistanceType dist, IndexType index)
		{
			CountType i;
			for (i = count; i > 0; --i) {
#ifdef NANOFLANN_FIRST_MATCH   // If defined and two points have the same distance, the one with the lowest-index will be returned first.
				if ( (dists[i-1] > dist) || ((dist == dists[i-1]) && (indices[i-1] > index)) ) {
#else
				if (dists[i-1] > dist) {
#endif
					if (i < capacity) {
						dists[i] = dists[i-1];
						indices[i] = indices[i-1];
					}
				}
				else break;
			}
			if (i < capacity) {
				dists[i] = dist;
				indices[i] = index;
			}
			if (count < capacity) count++;

                        // tell caller that the search shall continue
                        return true;
		}

		inline DistanceType worstDist() const
		{
			return dists[capacity-1];
		}
	};

	/** operator "<" for std::sort() */
	struct IndexDist_Sorter
	{
		/** PairType will be typically: std::pair<IndexType,DistanceType> */
		template <typename PairType>
		inline bool operator()(const PairType &p1, const PairType &p2) const {
			return p1.second < p2.second;
		}
	};

	/**
	 * A result-set class used when performing a radius based search.
	 */
	template <typename DistanceType, typename IndexType = size_t>
	class RadiusResultSet
	{
	public:
		const DistanceType radius;

		std::vector<std::pair<IndexType, DistanceType> > &m_indices_dists;

		inline RadiusResultSet(DistanceType radius_, std::vector<std::pair<IndexType,DistanceType> > &indices_dists) : radius(radius_), m_indices_dists(indices_dists)
		{
			init();
		}

		inline void init() { clear(); }
		inline void clear() { m_indices_dists.clear(); }

		inline size_t size() const { return m_indices_dists.size(); }

		inline bool full() const { return true; }

                /**
                 * Called during search to add an element matching the criteria.
                 * @return true if the search should be continued, false if the results are sufficient
                 */
                inline bool addPoint(DistanceType dist, IndexType index)
		{
			if (dist < radius)
				m_indices_dists.push_back(std::make_pair(index, dist));
                        return true;
		}

		inline DistanceType worstDist() const { return radius; }

		/**
		 * Find the worst result (furtherest neighbor) without copying or sorting
		 * Pre-conditions: size() > 0
		 */
		std::pair<IndexType,DistanceType> worst_item() const
		{
		   if (m_indices_dists.empty()) throw std::runtime_error("Cannot invoke RadiusResultSet::worst_item() on an empty list of results.");
		   typedef typename std::vector<std::pair<IndexType, DistanceType> >::const_iterator DistIt;
		   DistIt it = std::max_element(m_indices_dists.begin(), m_indices_dists.end(), IndexDist_Sorter());
		   return *it;
		}
	};


	/** @} */


	/** @addtogroup loadsave_grp Load/save auxiliary functions
	  * @{ */
	template<typename T>
	void save_value(FILE* stream, const T& value, size_t count = 1)
	{
		fwrite(&value, sizeof(value), count, stream);
	}

	template<typename T>
	void save_value(FILE* stream, const std::vector<T>& value)
	{
		size_t size = value.size();
		fwrite(&size, sizeof(size_t), 1, stream);
		fwrite(&value[0], sizeof(T), size, stream);
	}

	template<typename T>
	void load_value(FILE* stream, T& value, size_t count = 1)
	{
		size_t read_cnt = fread(&value, sizeof(value), count, stream);
		if (read_cnt != count) {
			throw std::runtime_error("Cannot read from file");
		}
	}


	template<typename T>
	void load_value(FILE* stream, std::vector<T>& value)
	{
		size_t size;
		size_t read_cnt = fread(&size, sizeof(size_t), 1, stream);
		if (read_cnt != 1) {
			throw std::runtime_error("Cannot read from file");
		}
		value.resize(size);
		read_cnt = fread(&value[0], sizeof(T), size, stream);
		if (read_cnt != size) {
			throw std::runtime_error("Cannot read from file");
		}
	}
	/** @} */


	/** @addtogroup metric_grp Metric (distance) classes
	  * @{ */

	struct Metric
	{
	};

	/** Manhattan distance functor (generic version, optimized for high-dimensionality data sets).
	  *  Corresponding distance traits: nanoflann::metric_L1
	  * \tparam T Type of the elements (e.g. double, float, uint8_t)
	  * \tparam _DistanceType Type of distance variables (must be signed) (e.g. float, double, int64_t)
	  */
	template<class T, class DataSource, typename _DistanceType = T>
	struct L1_Adaptor
	{
		typedef T ElementType;
		typedef _DistanceType DistanceType;

		const DataSource &data_source;

		L1_Adaptor(const DataSource &_data_source) : data_source(_data_source) { }

		inline DistanceType evalMetric(const T* a, const size_t b_idx, size_t size, DistanceType worst_dist = -1) const
		{
			DistanceType result = DistanceType();
			const T* last = a + size;
			const T* lastgroup = last - 3;
			size_t d = 0;

			/* Process 4 items with each loop for efficiency. */
			while (a < lastgroup) {
				const DistanceType diff0 = std::abs(a[0] - data_source.kdtree_get_pt(b_idx,d++));
				const DistanceType diff1 = std::abs(a[1] - data_source.kdtree_get_pt(b_idx,d++));
				const DistanceType diff2 = std::abs(a[2] - data_source.kdtree_get_pt(b_idx,d++));
				const DistanceType diff3 = std::abs(a[3] - data_source.kdtree_get_pt(b_idx,d++));
				result += diff0 + diff1 + diff2 + diff3;
				a += 4;
				if ((worst_dist > 0) && (result > worst_dist)) {
					return result;
				}
			}
			/* Process last 0-3 components.  Not needed for standard vector lengths. */
			while (a < last) {
				result += std::abs( *a++ - data_source.kdtree_get_pt(b_idx, d++) );
			}
			return result;
		}

		template <typename U, typename V>
		inline DistanceType accum_dist(const U a, const V b, int ) const
		{
			return std::abs(a-b);
		}
	};

	/** Squared Euclidean distance functor (generic version, optimized for high-dimensionality data sets).
	  *  Corresponding distance traits: nanoflann::metric_L2
	  * \tparam T Type of the elements (e.g. double, float, uint8_t)
	  * \tparam _DistanceType Type of distance variables (must be signed) (e.g. float, double, int64_t)
	  */
	template<class T, class DataSource, typename _DistanceType = T>
	struct L2_Adaptor
	{
		typedef T ElementType;
		typedef _DistanceType DistanceType;

		const DataSource &data_source;

		L2_Adaptor(const DataSource &_data_source) : data_source(_data_source) { }

		inline DistanceType evalMetric(const T* a, const size_t b_idx, size_t size, DistanceType worst_dist = -1) const
		{
			DistanceType result = DistanceType();
			const T* last = a + size;
			const T* lastgroup = last - 3;
			size_t d = 0;

			/* Process 4 items with each loop for efficiency. */
			while (a < lastgroup) {
				const DistanceType diff0 = a[0] - data_source.kdtree_get_pt(b_idx,d++);
				const DistanceType diff1 = a[1] - data_source.kdtree_get_pt(b_idx,d++);
				const DistanceType diff2 = a[2] - data_source.kdtree_get_pt(b_idx,d++);
				const DistanceType diff3 = a[3] - data_source.kdtree_get_pt(b_idx,d++);
				result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
				a += 4;
				if ((worst_dist > 0) && (result > worst_dist)) {
					return result;
				}
			}
			/* Process last 0-3 components.  Not needed for standard vector lengths. */
			while (a < last) {
				const DistanceType diff0 = *a++ - data_source.kdtree_get_pt(b_idx, d++);
				result += diff0 * diff0;
			}
			return result;
		}

		template <typename U, typename V>
		inline DistanceType accum_dist(const U a, const V b, int ) const
		{
			return (a - b) * (a - b);
		}
	};

	/** Squared Euclidean (L2) distance functor (suitable for low-dimensionality datasets, like 2D or 3D point clouds)
	  *  Corresponding distance traits: nanoflann::metric_L2_Simple
	  * \tparam T Type of the elements (e.g. double, float, uint8_t)
	  * \tparam _DistanceType Type of distance variables (must be signed) (e.g. float, double, int64_t)
	  */
	template<class T, class DataSource, typename _DistanceType = T>
	struct L2_Simple_Adaptor
	{
		typedef T ElementType;
		typedef _DistanceType DistanceType;

		const DataSource &data_source;

		L2_Simple_Adaptor(const DataSource &_data_source) : data_source(_data_source) { }

		inline DistanceType evalMetric(const T* a, const size_t b_idx, size_t size) const {
			DistanceType result = DistanceType();
			for (size_t i = 0; i < size; ++i) {
				const DistanceType diff = a[i] - data_source.kdtree_get_pt(b_idx, i);
				result += diff * diff;
			}
			return result;
		}

		template <typename U, typename V>
		inline DistanceType accum_dist(const U a, const V b, int ) const
		{
			return (a - b) * (a - b);
		}
	};

	/** SO2 distance functor
	  *  Corresponding distance traits: nanoflann::metric_SO2
	  * \tparam T Type of the elements (e.g. double, float)
	  * \tparam _DistanceType Type of distance variables (must be signed) (e.g. float, double)
	  * orientation is constrained to be in [-pi, pi]
	  */
	template<class T, class DataSource, typename _DistanceType = T>
	struct SO2_Adaptor
	{
		typedef T ElementType;
		typedef _DistanceType DistanceType;

		const DataSource &data_source;

		SO2_Adaptor(const DataSource &_data_source) : data_source(_data_source) { }

		inline DistanceType evalMetric(const T* a, const size_t b_idx, size_t size) const {
			return accum_dist(a[size-1], data_source.kdtree_get_pt(b_idx, size - 1) , size - 1);
		}

		template <typename U, typename V>
		inline DistanceType accum_dist(const U a, const V b, int ) const
		{
			DistanceType result = DistanceType();
			result = b - a;
			if (result > M_PI)
				result -= 2. * M_PI;
			else if (result < -M_PI)
				result += 2. * M_PI;
			return result;
		}
	};

	/** SO3 distance functor (Uses L2_Simple)
	  *  Corresponding distance traits: nanoflann::metric_SO3
	  * \tparam T Type of the elements (e.g. double, float)
	  * \tparam _DistanceType Type of distance variables (must be signed) (e.g. float, double)
	  */
	template<class T, class DataSource, typename _DistanceType = T>
	struct SO3_Adaptor
	{
		typedef T ElementType;
		typedef _DistanceType DistanceType;

		L2_Simple_Adaptor<T, DataSource > distance_L2_Simple;

		SO3_Adaptor(const DataSource &_data_source) : distance_L2_Simple(_data_source) { }

		inline DistanceType evalMetric(const T* a, const size_t b_idx, size_t size) const {
			return distance_L2_Simple.evalMetric(a, b_idx, size);
		}

		template <typename U, typename V>
		inline DistanceType accum_dist(const U a, const V b, int idx) const
		{
			return distance_L2_Simple.accum_dist(a, b, idx);
		}
	};

	/** Metaprogramming helper traits class for the L1 (Manhattan) metric */
	struct metric_L1 : public Metric
	{
		template<class T, class DataSource>
		struct traits {
			typedef L1_Adaptor<T, DataSource> distance_t;
		};
	};
	/** Metaprogramming helper traits class for the L2 (Euclidean) metric */
	struct metric_L2 : public Metric 
	{
		template<class T, class DataSource>
		struct traits {
			typedef L2_Adaptor<T, DataSource> distance_t;
		};
	};
	/** Metaprogramming helper traits class for the L2_simple (Euclidean) metric */
	struct metric_L2_Simple : public Metric
	{
		template<class T, class DataSource>
		struct traits {
			typedef L2_Simple_Adaptor<T, DataSource> distance_t;
		};
	};
	/** Metaprogramming helper traits class for the SO3_InnerProdQuat metric */
	struct metric_SO2 : public Metric 
	{
		template<class T, class DataSource>
		struct traits {
			typedef SO2_Adaptor<T, DataSource> distance_t;
		};
	};
	/** Metaprogramming helper traits class for the SO3_InnerProdQuat metric */
	struct metric_SO3 : public Metric
	{
		template<class T, class DataSource>
		struct traits {
			typedef SO3_Adaptor<T, DataSource> distance_t;
		};
	};

	/** @} */

	/** @addtogroup param_grp Parameter structs
	  * @{ */

	/**  Parameters (see README.md) */
	struct KDTreeSingleIndexAdaptorParams
	{
		KDTreeSingleIndexAdaptorParams(size_t _leaf_max_size = 10) :
			leaf_max_size(_leaf_max_size)
		{}

		size_t leaf_max_size;
	};

	/** Search options for KDTreeSingleIndexAdaptor::findNeighbors() */
	struct SearchParams
	{
		/** Note: The first argument (checks_IGNORED_) is ignored, but kept for compatibility with the FLANN interface */
		SearchParams(int checks_IGNORED_ = 32, float eps_ = 0, bool sorted_ = true ) :
			checks(checks_IGNORED_), eps(eps_), sorted(sorted_) {}

		int   checks;  //!< Ignored parameter (Kept for compatibility with the FLANN interface).
		float eps;  //!< search for eps-approximate neighbours (default: 0)
		bool sorted; //!< only for radius search, require neighbours sorted by distance (default: true)
	};
	/** @} */


	/** @addtogroup memalloc_grp Memory allocation
	  * @{ */

	/**
	 * Allocates (using C's malloc) a generic type T.
	 *
	 * Params:
	 *     count = number of instances to allocate.
	 * Returns: pointer (of type T*) to memory buffer
	 */
	template <typename T>
	inline T* allocate(size_t count = 1)
	{
		T* mem = static_cast<T*>( ::malloc(sizeof(T)*count));
		return mem;
	}


	/**
	 * Pooled storage allocator
	 *
	 * The following routines allow for the efficient allocation of storage in
	 * small chunks from a specified pool.  Rather than allowing each structure
	 * to be freed individually, an entire pool of storage is freed at once.
	 * This method has two advantages over just using malloc() and free().  First,
	 * it is far more efficient for allocating small objects, as there is
	 * no overhead for remembering all the information needed to free each
	 * object or consolidating fragmented memory.  Second, the decision about
	 * how long to keep an object is made at the time of allocation, and there
	 * is no need to track down all the objects to free them.
	 *
	 */

	const size_t     WORDSIZE = 16;
	const size_t     BLOCKSIZE = 8192;

	class PooledAllocator
	{
		/* We maintain memory alignment to word boundaries by requiring that all
		    allocations be in multiples of the machine wordsize.  */
		/* Size of machine word in bytes.  Must be power of 2. */
		/* Minimum number of bytes requested at a time from	the system.  Must be multiple of WORDSIZE. */


		size_t  remaining;  /* Number of bytes left in current block of storage. */
		void*   base;     /* Pointer to base of current block of storage. */
		void*   loc;      /* Current location in block to next allocate memory. */

		void internal_init()
		{
			remaining = 0;
			base = NULL;
			usedMemory = 0;
			wastedMemory = 0;
		}

	public:
		size_t  usedMemory;
		size_t  wastedMemory;

		/**
		    Default constructor. Initializes a new pool.
		 */
		PooledAllocator() {
			internal_init();
		}

		/**
		 * Destructor. Frees all the memory allocated in this pool.
		 */
		~PooledAllocator() {
			free_all();
		}

		/** Frees all allocated memory chunks */
		void free_all()
		{
			while (base != NULL) {
				void *prev = *(static_cast<void**>( base)); /* Get pointer to prev block. */
				::free(base);
				base = prev;
			}
			internal_init();
		}

		/**
		 * Returns a pointer to a piece of new memory of the given size in bytes
		 * allocated from the pool.
		 */
		void* malloc(const size_t req_size)
		{
			/* Round size up to a multiple of wordsize.  The following expression
			    only works for WORDSIZE that is a power of 2, by masking last bits of
			    incremented size to zero.
			 */
			const size_t size = (req_size + (WORDSIZE - 1)) & ~(WORDSIZE - 1);

			/* Check whether a new block must be allocated.  Note that the first word
			    of a block is reserved for a pointer to the previous block.
			 */
			if (size > remaining) {

				wastedMemory += remaining;

				/* Allocate new storage. */
				const size_t blocksize = (size + sizeof(void*) + (WORDSIZE - 1) > BLOCKSIZE) ?
							size + sizeof(void*) + (WORDSIZE - 1) : BLOCKSIZE;

				// use the standard C malloc to allocate memory
				void* m = ::malloc(blocksize);
				if (!m) {
					fprintf(stderr, "Failed to allocate memory.\n");
					return NULL;
				}

				/* Fill first word of new block with pointer to previous block. */
				static_cast<void**>(m)[0] = base;
				base = m;

				size_t shift = 0;
				//int size_t = (WORDSIZE - ( (((size_t)m) + sizeof(void*)) & (WORDSIZE-1))) & (WORDSIZE-1);

				remaining = blocksize - sizeof(void*) - shift;
				loc = (static_cast<char*>(m) + sizeof(void*) + shift);
			}
			void* rloc = loc;
			loc = static_cast<char*>(loc) + size;
			remaining -= size;

			usedMemory += size;

			return rloc;
		}

		/**
		 * Allocates (using this pool) a generic type T.
		 *
		 * Params:
		 *     count = number of instances to allocate.
		 * Returns: pointer (of type T*) to memory buffer
		 */
		template <typename T>
		T* allocate(const size_t count = 1)
		{
			T* mem = static_cast<T*>(this->malloc(sizeof(T)*count));
			return mem;
		}

	};

    template <typename T, std::size_t N>
    class CArray {
      public:
        T elems[N];    // fixed-size array of elements of type T

      public:
        // type definitions
        typedef T              value_type;
        typedef T*             iterator;
        typedef const T*       const_iterator;
        typedef T&             reference;
        typedef const T&       const_reference;
        typedef std::size_t    size_type;
        typedef std::ptrdiff_t difference_type;

        // iterator support
        inline iterator begin() { return elems; }
        inline const_iterator begin() const { return elems; }
        inline iterator end() { return elems+N; }
        inline const_iterator end() const { return elems+N; }

        // reverse iterator support
#if !defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION) && !defined(BOOST_MSVC_STD_ITERATOR) && !defined(BOOST_NO_STD_ITERATOR_TRAITS)
        typedef std::reverse_iterator<iterator> reverse_iterator;
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
#elif defined(_MSC_VER) && (_MSC_VER == 1300) && defined(BOOST_DINKUMWARE_STDLIB) && (BOOST_DINKUMWARE_STDLIB == 310)
        // workaround for broken reverse_iterator in VC7
        typedef std::reverse_iterator<std::_Ptrit<value_type, difference_type, iterator,
                                      reference, iterator, reference> > reverse_iterator;
        typedef std::reverse_iterator<std::_Ptrit<value_type, difference_type, const_iterator,
                                      const