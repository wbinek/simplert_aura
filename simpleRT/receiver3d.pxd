# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 14:07:30 2018

@author: wojci
"""
cimport cython
from simpleRT.datatypes.MyVec3 cimport MyVec3

cdef class Receiver3D():
    cdef public MyVec3 position
    cdef public float radius