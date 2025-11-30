from setuptools import setup
from Cython.Build import cythonize
import sys
import numpy as np

setup(
    ext_modules=cythonize([
        "src/cython/negamax.pyx",
        "src/cython/cychess.pyx",
        "src/cython/attn_model.pyx",
    ]),
    include_dirs=[np.get_include()],
    extra_compile_args=[
        '-O3',           # Max optimization
        '-march=native', # CPU-specific instrs (AVX2/SSE on your machine)
        '-ffast-math',   # Aggressive float math (faster expf/fmaxf, assumes no NaN/inf issues)
        '-funroll-loops',# Auto-unroll small loops (helps h linear, attn outer)
        '-ftree-vectorize', # Explicit SIMD vectorization
        '-fno-exceptions', # Disable exceptions for speed (Cython-safe here)
        '-DNDEBUG',      # Disable debug checks
    ] if 'posix' in sys.builtin_module_names else [  # Linux/Mac; for Windows/MSVC, use /O2 /arch:AVX2
        '/O2', '/arch:AVX2', '/DNDEBUG'
    ],
    zip_safe=False,
)