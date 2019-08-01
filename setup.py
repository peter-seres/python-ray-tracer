from distutils.core import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy

Options.language_level = 3

setup(
    ext_modules=cythonize('main_cython.pyx'),
    include_dirs=[numpy.get_include()],
    requires=['Cython', 'numpy', 'PIL', 'matplotlib']
)
