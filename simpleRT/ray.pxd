# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:10:08 2018

@author: wojci
"""
cimport cython
cimport numpy as np
from cpython cimport bool
from simpleRT.datatypes.MyVec3 cimport MyVec3


cdef enum RayStatus:
    OK = 0
    LOST = 1

cdef class Ray():
    cdef public np.ndarray energy;
    cdef public int refl_order;
    cdef public bool receiver_hit;

    cdef public MyVec3 position;
    cdef public MyVec3 direction;
    cdef public float total_dist;
    cdef public RayStatus status;
       
    cdef ray_lost(self)  
    cpdef bool is_lost(self)
        