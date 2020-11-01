# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:10:34 2018

@author: wojci
"""
import cython
from simpleRT.datatypes.MyVec3 cimport MyVec3
cimport numpy as np

cdef class Face3D:
    cpdef public list vertices
    cpdef public int normal_idx
    cpdef public str mat_name
    
cdef class Model3D:
    cpdef public np.ndarray faces
    cpdef public list vertices, normals
    cpdef public dict materials
    
    cpdef loadOBJ(self, str path)
        
        
        