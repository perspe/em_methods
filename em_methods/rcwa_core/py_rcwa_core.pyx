# distutils: language=c++
# cython: language_level=3
# cython: annotate=true
""" Main Cython functions implementing the RCWA algorithm """

from libcpp.complex cimport complex
cimport cython
cimport numpy as cnp

def test_func():
    print("HEllo")


cdef class SMatrix:
    def __cinit__(self):
        pass
