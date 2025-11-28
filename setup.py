from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        "src/cython/negamax.pyx",
        "src/cython/cychess.pyx",
    ]),
    zip_safe=False,
)