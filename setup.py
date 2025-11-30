from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize([
        "src/cython/negamax.pyx",
        "src/cython/cychess.pyx",
        "src/cython/attn_model.pyx",
    ]),
    include_dirs=[np.get_include()],
    zip_safe=False,
)