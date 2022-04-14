#include <Python.h>
#include <numpy/arrayobject.h>
#include "grid_subsampling/grid_subsampling.h"
#include <string>



// docstrings for our module
// *************************

static char module_docstring[] = "This module provides an interface for the subsampling of a pointcloud";

static char compute_docstring[] = "function subsampling a pointcloud";


// Declare the functions
// *********************

static PyObject *grid_subsampling_compute(PyObject *self, PyObject *args, PyObject *keywds);


// Specify the members of the module
// *********************************

static PyMethodDef module_methods[] = 
{
	{ "compute", (PyCFunction)grid_subsampling_compute, METH_VARARGS | METH_KEYWORDS, compute_docstring },
    {NULL, NULL, 0, NULL}
};


// Initialize the module
// *********************

static struct PyModuleDef moduledef = 
{
    PyModuleDef_HEAD_INIT,
    "grid_subsampling",     // m_name
    module_docstring,       // m_doc
    -1,                     // m_size
    module_methods,         // m_methods
    NULL,                   // m_reload
    NULL,                   // m_traverse
    NULL,                   // m_clear
    NULL,                   // m_free
};

PyMODINIT_FUNC PyInit_grid_subsampling(void)
{
    import_array();
	return PyModule_Create(&moduledef);
}


// Actual wrapper
// **************

static PyObject *grid_subsampling_compute(PyObject *self, PyObject *args, PyObject *keywds)
{

    // Manage inputs
    // *************

	// Args containers
	PyObject *points_obj = NULL;
	PyObject *features_obj = NULL;
	PyObject *classes_obj = NULL;

	// Keywords containers
	static char *kwlist[] = {"points", "features", "classes", "sampleDl", "method", "verbose", NULL };
	float sampleDl = 0.1;
	const char *method_buffer = "barycenters";
	int verbose = 0;

	// Parse the input  
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|$OOfsi", kwlist, &points_obj, &features_obj, &classes_obj, &sampleDl, &method_buffer, &verbose))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing arguments");
		return NULL;
	}

	// Get the method argument
	string method(method_buffer);

	// Interpret method
	if (method.compare("barycenters") && method.compare("voxelcenters"))
	{
		PyErr_SetString(PyExc_RuntimeError, "Error parsing method. Valid method names are \"barycenters\" and \"voxelcenters\" ");
		return NULL;
	}

	// Check if using features or classes
	bool use_feature = true, use_classes = true;
	if (features_obj == NULL)
		use_feature = false;
	if (classes_obj == NULL)
		use_classes = false;

    // Interpret the input objects as numpy arrays.
	PyObject *points_array = PyArray_FROM_OTF(points_obj, NPY_FLOAT, NPY_IN_ARRAY);
	PyObject *features_array = NULL;
	PyObject *classes_array = NULL;
	if (use_feature)
		features_array = PyArray_FROM_OTF(features_obj, NPY_FLOAT, NPY_IN_ARRAY);
	if (use_classes)
		classes_array = PyArray_FROM_OT