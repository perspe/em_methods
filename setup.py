from distutils.core import Extension
import os

from Cython.Build import cythonize
from setuptools import setup

# File shortcuts
# SMM
smm_core = os.path.join("em_methods", "smm_core")
py_smm_base = os.path.join(smm_core, "py_smm_base.pyx")
cpp_smm_base = os.path.join(smm_core, "smm_base.cpp")
# RCWA
# rcwa = os.path.join("em_methods", "rcwa_core")
# py_rcwa_base = os.path.join(rcwa, "rcwa_core.pyx")

ext = [
    Extension("em_methods.smm_core.py_smm_base", [py_smm_base, cpp_smm_base],
              include_dirs=[smm_core]),
    # Extension("em_methods.rcwa_core.rcwa_core", [py_rcwa_base],
    #           include_dirs=[np.get_include()])
]

setup(ext_modules=cythonize(ext, gdb_debug=False))
