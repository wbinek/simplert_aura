# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 11:10:34 2018

@author: wojci
"""
import cython
from simpleRT.datatypes.MyVec3 cimport MyVec3
cimport numpy as np

cdef class Face3D:
    cdef int vertices[3]
    cdef int normal_idx
    cdef str mat_name
    
cdef class Model3D:
    cpdef public np.ndarray _faces, _vertices, _normals
    cpdef public dict materials
    cdef MyVec3[:] vertices
    cdef MyVec3[:] normals
    cdef Face3D[:] faces
    
    cpdef loadOBJ(self, str path)
        
        
        