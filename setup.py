from distutils.core import Extension
import os

from Cython.Build import cythonize
import Cython
from setuptools import setup
import subprocess

compilers = ["g++", "clang++", "cl"]

def check_compiler(compiler):
    """Check compiler availability"""
    try:
        result = subprocess.run(
            [compiler, "--version"], capture_output=True, text=True, check=True
        )
        print(f"{compiler} is installed:\n{result.stdout.splitlines()[0]}")
    except FileNotFoundError:
        print(f"{compiler} is NOT installed.")
        return False
    except subprocess.CalledProcessError:
        print(f"Error checking {compiler}.")
        return False
    return True

# File shortcuts
# SMM
smm_core = os.path.join("em_methods", "smm_core")
py_smm_base = os.path.join(smm_core, "py_smm_base.pyx")
cpp_smm_base = os.path.join(smm_core, "smm_base.cpp")
# Check for present binaries to avoid compiling

compiler_check = [check_compiler(compiler) for compiler in compilers]

if True in compiler_check:
    ext = [
        Extension(
            "em_methods.smm_core.py_smm_base",
            [py_smm_base, cpp_smm_base],
            include_dirs=[smm_core],
        )
    ]

# RCWA
# rcwa = os.path.join("em_methods", "rcwa_core")
# py_rcwa_base = os.path.join(rcwa, "rcwa_core.pyx")
# ext = [
# Extension("em_methods.rcwa_core.rcwa_core", [py_rcwa_base],
#           include_dirs=[np.get_include()])
# ]
setup(ext_modules=cythonize(ext, gdb_debug=False))
