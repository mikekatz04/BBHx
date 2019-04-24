# from future.utils import iteritems
import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

lib_gsl_dir = "/opt/local/lib"
include_gsl_dir = "/opt/local/include"

extensions=[Extension('GPUPhenomHM',
        sources = ['src/globalPhenomHM.cpp', 'src/RingdownCW.cpp', 'src/IMRPhenomD_internals.cpp', 'src/IMRPhenomD.cpp', 'src/PhenomHM.cpp', 'src/c_manager.cpp', 'GPUPhenomHM.pyx'],
        library_dirs = [lib_gsl_dir],
        libraries = ["gsl", "gslcblas"],
        language = 'c++',
        #sruntime_library_dirs = [CUDA['lib64']],
        # This syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc
        # and not with gcc the implementation of this trick is in
        # customize_compiler()
        extra_compile_args= ["-O3"],
            include_dirs = [numpy_include, include_gsl_dir, 'src']
        )]

from Cython.Build import cythonize
extensions = cythonize(extensions, gdb_debug=True)

setup(name = 'GPUPhenomHM',
      # Random metadata. there's more you can supply
      author = 'Robert McGibbon',
      version = '0.1',

      ext_modules=extensions,


      # Since the package has c code, the egg cannot be zipped
      zip_safe = False)
