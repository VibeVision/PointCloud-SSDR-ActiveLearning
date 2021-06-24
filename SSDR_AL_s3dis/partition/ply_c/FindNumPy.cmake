
# - Try to find the Python module NumPy
#
# This module defines:
#  NUMPY_INCLUDE_DIR: include path for arrayobject.h

# Copyright (c) 2009-2012 Arnaud Barré <arnaud.barre@gmail.com>
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.

if (PYTHON_NUMPY_INCLUDE_DIR)
  set(PYTHON_NUMPY_FIND_QUIETLY TRUE)
endif()

if (NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "\"PYTHON_EXECUTABLE\" varabile not set before FindNumPy.cmake was run.")
endif()

# Look for the include path
# WARNING: The variable PYTHON_EXECUTABLE is defined by the script FindPythonInterp.cmake
execute_process(COMMAND "${PYTHON_EXECUTABLE}" -c "import numpy; print (numpy.get_include()); print (numpy.version.version)"
                 OUTPUT_VARIABLE NUMPY_OUTPUT
                 ERROR_