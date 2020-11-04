from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
#from Cython.Distutils import build_ext

import numpy

Options.buffer_max_dims = 3

extensions=[
    Extension("simpleRT.datatypes.MyVec3", ["simpleRT/datatypes/MyVec3.pyx"]),
    Extension("simpleRT.datatypes.MyArray3", ["simpleRT/datatypes/MyArray3.pyx"]),
    Extension("simpleRT.libmath.libmath", ["simpleRT/libmath/libmath.pyx"]),
    Extension("simpleRT.ray", ["simpleRT/ray.pyx"]),
    Extension("simpleRT.receiver3d", ["simpleRT/receiver3d.pyx"]),
    Extension("simpleRT.source3d", ["simpleRT/source3d.pyx"]),
    Extension("simpleRT.model3d", ["simpleRT/model3d.pyx"]),
    Extension("simpleRT.simulation3d", ["simpleRT/simulation3d.pyx"]),
]

setup(
    name = 'MyProject',
    ext_modules = cythonize(extensions, force=True,
    compiler_directives={'language_level': 3, 
    'boundscheck':True, 
    'cdivision':False,
    'profile':True}),
    include_dirs=[numpy.get_include()],
)
