from setuptools import setup
from distutils.core import Extension
from Cython.Build import cythonize
import os

# File shortcuts
smm_core = os.path.join("em_methods", "smm_core")
py_smm_base = os.path.join(smm_core, "py_smm_base.pyx")
cpp_smm_base = os.path.join(smm_core, "smm_base.cpp")

ext = [
    Extension("em_methods.smm_core.py_smm_base", [py_smm_base, cpp_smm_base],
              language="c++",
              include_dirs=[smm_core])
]

setup(ext_modules=cythonize(ext))
