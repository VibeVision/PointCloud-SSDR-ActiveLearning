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
    -1,                    