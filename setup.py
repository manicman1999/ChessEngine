from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        "src/cython/board_to_tokens.pyx",
        "src/cython/cychess.pyx",
    ]),
    zip_safe=False,
)