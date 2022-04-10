

#include "../../cpp_utils/cloud/cloud.h"

#include <set>
#include <cstdint>

using namespace std;

class SampledData
{
public:

	// Elements
	// ********

	int count;
	PointXYZ point;
	vector<float> features;
	vector<unordered_map<int, int>> labels;


	// Methods
	// *******

	// Constructor
	SampledData() 
	{ 
		count = 0; 
		point = PointXYZ();
	}

	SampledData(const size_t fdim, const size_t ldim)
	{
		count = 0;
		point = PointXYZ();
	    features = vector<float>(fdim);
	    labels = vector<unordered_map<int, int>>(ldim);
	}

	// Method Update
	void update_all(const PointXYZ p, vector<float>::iterator f_begin, vector<int>::iterator l_begin)
	{
		count